"""The hypothesis-generating agent.

Two interchangeable implementations behind one ``propose`` call:

``ClaudeAgent``
    Calls the Claude Messages API with the user profile, the baseline bio-eval
    readout, the mechanism-axis catalogue and every previously scored
    hypothesis. Structured output keeps the reply machine-readable.

``ReasonerAgent``
    A built-in optimiser that needs no API key. It probes single axes to
    measure their marginal value for this specific user, assigns credit by
    ridge regression over the scored history, then evolves combinations. It is
    what makes the loop demonstrably self-improving offline.

Both emit the same hypothesis shape, and both attach a concrete protocol drawn
from the evidence catalogue so the output is actionable rather than abstract.
"""

from . import compat  # noqa: F401

import json
import os
import random
import re

import numpy as np

from . import evidence, panels

MODEL = os.environ.get("LONGEVITY_LOOP_MODEL", "claude-opus-5")

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "mechanism_class": {"type": "string"},
        "rationale": {"type": "string"},
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": sorted(panels.AXES)},
                    "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["axis", "intensity"],
                "additionalProperties": False,
            },
        },
        "primary_endpoint": {"type": "string"},
        "falsifier": {"type": "string"},
    },
    "required": ["title", "mechanism_class", "rationale", "targets",
                 "primary_endpoint", "falsifier"],
    "additionalProperties": False,
}

SYSTEM = """You are the hypothesis generator inside a closed-loop longevity \
research system.

Each turn you propose ONE intervention hypothesis. It is scored by an \
evaluation harness built on the biolearn library: the harness applies your \
hypothesis to the subject's methylome and re-runs a twelve-model panel of \
epigenetic clocks and mortality predictors.

How the harness grades you:
- Improvement is measured in years-of-aging-equivalent across the whole panel, \
not on any single clock.
- Every CpG has a young-adult target. You can only close part of the distance \
to it, and never overshoot. If the subject has no burden on an axis, that axis \
has no headroom and targeting it earns nothing.
- Some panel members are held out of the reward on alternating iterations. \
Overfitting one clock shows up as a generalisation gap and is penalised.
- Penalties apply for incoherence between clock families, off-target \
methylation change, protocol burden, pulling more than four axes, and total \
methylation change beyond a plausible one-year budget.

Read the scored history carefully. Your job is to climb the reward, not to \
restate plausible biology. Prefer axes where the subject has real headroom, \
combine levers that showed independent gains, and abandon ones that scored flat.
Be specific and falsifiable. Never claim clinical proof for a mechanism that \
only has animal data."""


def available_backends():
    """Which agent backends this machine can actually run right now."""
    out = ["reasoner"]
    try:
        import anthropic  # noqa: F401
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            out.insert(0, "claude")
    except Exception:
        pass
    return out


def build(backend="auto", seed=0):
    if backend in ("auto", None):
        backend = available_backends()[0]
    if backend == "claude":
        return ClaudeAgent()
    return ReasonerAgent(seed=seed)


# ---------------------------------------------------------------------------
# Shared context rendering
# ---------------------------------------------------------------------------

def axis_brief(loads):
    lines = []
    for name, spec in panels.AXES.items():
        head = loads.get(name, 0.0)
        lines.append(
            f"- {name} (headroom {head:.2f}, burden {spec['burden']:.2f}): "
            f"{spec['label']}. {spec['biology']}"
        )
    return "\n".join(lines)


def history_brief(history, limit=8):
    if not history:
        return "No hypotheses scored yet. This is iteration 1."
    ranked = sorted(history, key=lambda h: -h["reward"])[:limit]
    lines = []
    for h in ranked:
        targets = ", ".join(f"{t['axis']}@{t['intensity']:.2f}" for t in h["targets"])
        pen = h.get("penalties", {})
        worst = max(pen, key=pen.get) if pen else "none"
        lines.append(
            f"- [{h['iteration']:02d}] reward {h['reward']:+.3f} "
            f"(score {h['score']:.1f}, {h['years_reversed']:+.2f} yr) "
            f"gap {h.get('generalisation_gap', 0):+.2f} | {targets} "
            f"| largest penalty: {worst} {pen.get(worst, 0):.2f} | {h['title']}"
        )
    return "\n".join(lines)


def profile_brief(profile, baseline, age, expected=None):
    from . import bioevals

    parts = [f"{k}: {v}" for k, v in profile.items() if v not in (None, "")]
    # Acceleration is measured against a neutral-lifestyle peer of the same age,
    # not against chronological age: several models carry a platform-wide offset
    # that would otherwise be handed to the agent as if it were signal.
    accel = bioevals.acceleration(baseline, expected or {})
    reads = []
    for name, value in baseline.items():
        if value is None:
            continue
        spec = panels.EVAL_PANEL[name]
        suffix = f" (accel {accel[name]:+.1f} yr)" if name in accel else ""
        reads.append(f"  {spec['label']}: {value:.3f} {spec['unit']}{suffix}")
    return "PROFILE\n  " + "; ".join(parts) + "\n\nBASELINE PANEL\n" + "\n".join(reads)


def attach_protocol(hypothesis, loads=None, profile=None):
    """Bolt a concrete, evidence-graded protocol onto an axis vector."""
    protocol = evidence.compose_protocol(hypothesis.get("targets", []), loads, profile)
    hypothesis["protocol"] = protocol
    hypothesis["protocol_quality"] = evidence.protocol_quality(protocol)
    return hypothesis


def sanitise(raw, fallback_title="Untitled hypothesis"):
    """Coerce whatever the model returned into a valid hypothesis."""
    targets = []
    seen = set()
    for item in (raw.get("targets") or [])[:6]:
        axis = str(item.get("axis", "")).strip()
        if axis not in panels.AXES or axis in seen:
            continue
        seen.add(axis)
        try:
            intensity = float(item.get("intensity", 0))
        except (TypeError, ValueError):
            continue
        intensity = float(np.clip(intensity, 0.0, 1.0))
        if intensity > 0.01:
            targets.append(dict(axis=axis, intensity=round(intensity, 3)))
    return dict(
        title=str(raw.get("title") or fallback_title)[:140],
        mechanism_class=str(raw.get("mechanism_class") or "unspecified")[:60],
        rationale=str(raw.get("rationale") or "")[:1400],
        targets=targets,
        primary_endpoint=str(raw.get("primary_endpoint") or "")[:200],
        falsifier=str(raw.get("falsifier") or "")[:300],
    )


# ---------------------------------------------------------------------------
# Claude-backed agent
# ---------------------------------------------------------------------------

class ClaudeAgent:
    name = "claude"

    def __init__(self, model=MODEL):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.label = f"Claude Messages API ({model})"

    def propose(self, ctx):
        prompt = self._prompt(ctx)
        raw = self._call(prompt)
        hypothesis = sanitise(raw, fallback_title=f"Hypothesis {ctx['iteration']}")
        if not hypothesis["targets"]:                 # never return an empty move
            hypothesis["targets"] = ReasonerAgent().propose(ctx)["targets"]
            hypothesis["rationale"] += (
                "\n\n(Model returned no usable axis targets; the built-in "
                "optimiser supplied the vector.)")
        hypothesis["source"] = self.label
        return attach_protocol(hypothesis, ctx.get("loads"), ctx.get("profile"))

    def _prompt(self, ctx):
        mode = ctx["mode"]
        steer = {
            "probe": "This is a probe iteration. Isolate ONE axis at high "
                     "intensity so the harness measures its marginal value cleanly.",
            "explore": "This is an exploration iteration. Try a combination "
                       "that the history has not tested. Novelty matters more "
                       "than safety here.",
            "exploit": "This is an exploitation iteration. Start from the best "
                       "scoring hypothesis so far and improve it. Change few "
                       "things and say what you changed and why.",
        }[mode]
        insights = "\n".join(f"- {i}" for i in ctx["insights"]) or "- none yet"
        return (
            f"{profile_brief(ctx['profile'], ctx['baseline'], ctx['age'], ctx.get('expected'))}\n\n"
            f"MECHANISM AXES AND THIS SUBJECT'S HEADROOM\n{axis_brief(ctx['loads'])}\n\n"
            f"SCORED HISTORY\n{history_brief(ctx['history'])}\n\n"
            f"ACCUMULATED INSIGHTS\n{insights}\n\n"
            f"ITERATION {ctx['iteration']} OF {ctx['total']}. {steer}\n\n"
            "Return one hypothesis. Intensities are 0 to 1. Use at most four axes."
        )

    def _call(self, prompt):
        kwargs = dict(
            model=self.model,
            max_tokens=8000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
        )
        try:
            response = self.client.messages.create(
                output_config={"format": {"type": "json_schema",
                                          "schema": HYPOTHESIS_SCHEMA}},
                **kwargs,
            )
        except TypeError:
            # Older SDK build without output_config; ask for JSON in the prompt.
            kwargs["messages"] = [{
                "role": "user",
                "content": prompt + "\n\nReply with JSON only, matching this "
                                    "schema:\n" + json.dumps(HYPOTHESIS_SCHEMA),
            }]
            response = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in response.content if b.type == "text")
        return _loads(text)


def _loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Built-in optimiser (no API key required)
# ---------------------------------------------------------------------------

class ReasonerAgent:
    """Probe, assign credit, then evolve.

    Not a language model. It reads the same context and produces the same
    hypothesis shape, and because it fits a credit model over the scored
    history it genuinely improves across iterations rather than resampling.
    """

    name = "reasoner"
    label = "Built-in credit-assignment optimiser"

    def __init__(self, seed=0):
        self.rng = random.Random(seed or 20260823)
        self.np_rng = np.random.default_rng(seed or 20260823)

    # -- credit assignment ------------------------------------------------
    def _credit(self, history, axes):
        """Ridge-regress reward on the intensity vectors seen so far."""
        if len(history) < 2:
            return {a: 0.0 for a in axes}
        X = np.zeros((len(history), len(axes)))
        y = np.zeros(len(history))
        index = {a: i for i, a in enumerate(axes)}
        for row, h in enumerate(history):
            for t in h["targets"]:
                if t["axis"] in index:
                    X[row, index[t["axis"]]] = t["intensity"]
            y[row] = h["reward"]
        X = np.column_stack([X, np.ones(len(history))])
        ridge = 0.6 * np.eye(X.shape[1])
        ridge[-1, -1] = 1e-6
        try:
            coef = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        except np.linalg.LinAlgError:
            return {a: 0.0 for a in axes}
        return {a: float(coef[i]) for a, i in index.items()}

    def propose(self, ctx):
        axes = sorted(panels.AXES)
        loads = ctx["loads"]
        history = ctx["history"]
        mode = ctx["mode"]
        credit = self._credit(history, axes)
        tried = {t["axis"] for h in history for t in h["targets"]}

        if mode == "probe":
            targets, why = self._probe(axes, loads, tried)
        elif mode == "exploit" and history:
            targets, why = self._exploit(history, credit)
        else:
            targets, why = self._explore(axes, loads, credit, history)

        targets = [t for t in targets if t["intensity"] > 0.05][:4]
        if not targets:
            best_axis = max(axes, key=lambda a: loads.get(a, 0.0) + credit.get(a, 0.0))
            targets = [dict(axis=best_axis, intensity=0.6)]
            why = f"Fell back to the axis with the most remaining headroom, {best_axis}."

        hypothesis = sanitise(dict(
            title=self._title(targets, mode),
            mechanism_class=self._mechanism_class(targets),
            rationale=why,
            targets=targets,
            primary_endpoint=self._endpoint(targets),
            falsifier=self._falsifier(targets),
        ))
        hypothesis["source"] = self.label
        hypothesis["credit"] = {k: round(v, 3) for k, v in credit.items()}
        return attach_protocol(hypothesis, loads, ctx.get("profile"))

    # -- the three move types ---------------------------------------------
    def _probe(self, axes, loads, tried):
        candidates = [a for a in axes if a not in tried]
        if not candidates:
            candidates = axes
        axis = max(candidates, key=lambda a: (loads.get(a, 0.0),
                                              -panels.AXES[a]["burden"]))
        spec = panels.AXES[axis]
        why = (
            f"Single-axis probe. This subject carries {loads.get(axis, 0.0):.2f} "
            f"units of measured burden on {spec['label'].lower()}, the largest "
            f"untested headroom in the catalogue. Running it alone at high "
            f"intensity lets the harness attribute the resulting panel movement "
            f"to this axis and nothing else, which is what the credit model "
            f"needs before combinations are worth trying. {spec['biology']}"
        )
        return [dict(axis=axis, intensity=0.85)], why

    def _exploit(self, history, credit):
        best = max(history, key=lambda h: h["reward"])
        targets = [dict(t) for t in best["targets"]]
        current = {t["axis"] for t in targets}
        changes = []

        for t in targets:
            nudge = float(np.clip(credit.get(t["axis"], 0.0) * 0.35, -0.2, 0.2))
            jitter = self.np_rng.normal(0, 0.07)
            new = float(np.clip(t["intensity"] + nudge + jitter, 0.0, 1.0))
            if abs(new - t["intensity"]) > 0.02:
                changes.append(f"{t['axis']} {t['intensity']:.2f}->{new:.2f}")
            t["intensity"] = round(new, 3)

        addable = [a for a, c in sorted(credit.items(), key=lambda kv: -kv[1])
                   if a not in current]
        if addable and len(targets) < 4 and credit[addable[0]] > 0:
            axis = addable[0]
            targets.append(dict(axis=axis, intensity=0.45))
            changes.append(f"added {axis} at 0.45 (credit {credit[axis]:+.2f})")

        drop = [t for t in targets if credit.get(t["axis"], 0.0) < -0.15]
        if drop and len(targets) > 1:
            targets.remove(drop[0])
            changes.append(f"dropped {drop[0]['axis']} (negative credit)")

        why = (
            f"Exploitation step from iteration {best['iteration']} "
            f"(reward {best['reward']:+.3f}). The credit model fitted over "
            f"{len(history)} scored hypotheses attributes the reward mostly to "
            + ", ".join(f"{a} ({c:+.2f})" for a, c in
                        sorted(credit.items(), key=lambda kv: -kv[1])[:3])
            + ". Changes this round: " + ("; ".join(changes) or "intensity jitter only")
            + "."
        )
        return targets, why

    def _explore(self, axes, loads, credit, history):
        scored = []
        for axis in axes:
            appeal = (1.15 * loads.get(axis, 0.0)
                      + 1.0 * credit.get(axis, 0.0)
                      - 0.55 * panels.AXES[axis]["burden"]
                      + self.np_rng.normal(0, 0.35))
            scored.append((appeal, axis))
        scored.sort(reverse=True)
        picked = [a for _, a in scored[:self.rng.choice([2, 3, 3, 4])]]
        targets = [dict(axis=a, intensity=round(float(
            np.clip(0.35 + 0.5 * self.np_rng.random(), 0.15, 0.95)), 3))
            for a in picked]
        why = (
            "Exploration step. The combination "
            + ", ".join(picked)
            + " has not been scored together. Axes were sampled by measured "
              "headroom and fitted credit, minus protocol burden, with noise "
              "added so the search does not collapse onto the incumbent. "
            + (f"{len(history)} hypotheses scored so far."
               if history else "No history yet.")
        )
        return targets, why

    # -- narrative helpers -------------------------------------------------
    def _title(self, targets, mode):
        lead = max(targets, key=lambda t: t["intensity"])["axis"]
        label = panels.AXES[lead]["label"]
        if len(targets) == 1:
            return f"Isolated {label.lower()} intervention"
        others = len(targets) - 1
        prefix = {"exploit": "Refined", "explore": "Combination"}.get(mode, "Combined")
        return f"{prefix} {label.lower()} protocol with {others} co-target" + ("s" if others > 1 else "")

    def _mechanism_class(self, targets):
        lead = max(targets, key=lambda t: t["intensity"])["axis"]
        mechanisms = panels.AXES[lead]["mechanisms"]
        return mechanisms[0] if mechanisms else "unspecified"

    def _endpoint(self, targets):
        lead = max(targets, key=lambda t: t["intensity"])["axis"]
        if lead in ("xenobiotic_smoking", "inflammatory_load", "metabolic_glycemia"):
            return "GrimAge V2 acceleration at 12 months"
        if lead in ("polycomb_hypermethylation", "pmd_hypomethylation"):
            return "Horvath and PhenoAge acceleration at 12 months"
        if lead in ("adiposity", "lipid_transport", "cardiometabolic_vascular"):
            return "DunedinPACE at 12 months"
        return "Composite panel year-equivalent at 12 months"

    def _falsifier(self, targets):
        names = ", ".join(t["axis"] for t in targets)
        return (
            f"If a 12-month protocol targeting {names} fails to move the primary "
            f"endpoint by more than the assay's test-retest error, the hypothesis "
            f"is wrong for this subject and the axis credit should go negative."
        )
