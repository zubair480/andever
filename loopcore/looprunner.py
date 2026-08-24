"""The self-improving loop.

    user input
        -> agent generates a longevity hypothesis
        -> the biolearn eval harness scores it
        -> hypothesis + score are stored
        -> the agent reads the high scorers and the credit they earned
        -> next hypothesis

Every step emits an event so the interface can watch it happen live, and the
whole run ends with a report plus a preference dataset for DPO.
"""

from . import compat  # noqa: F401

import time
import uuid

import numpy as np

from . import agent as agents
from . import bioevals, intervention, mortality, panels, reference, report, store

_REF = None
_REF_INFO = None
_RESPONSE = None


def bootstrap(progress=lambda m: None):
    """Load the reference epigenome and warm the panel. Safe to call repeatedly."""
    global _REF, _REF_INFO, _RESPONSE
    if _REF is None:
        _REF, _REF_INFO = reference.build(progress=progress)
        bioevals.warm_up(progress=progress)
        _RESPONSE = intervention.ResponseModel(_REF)
        progress(f"Mechanism axes resolved onto "
                 f"{sum(len(v) for v in _RESPONSE.members.values())} CpG assignments")
    return _REF, _REF_INFO, _RESPONSE


def reference_info():
    _, info, _ = bootstrap()
    return info


def mode_for(iteration, total, probes):
    if iteration <= probes:
        return "probe"
    return "exploit" if iteration % 3 else "explore"


def run(profile, iterations=10, backend="auto", emit=lambda e: None, seed=0,
        run_id=None, sink=None):
    """Execute the loop. ``emit`` receives dicts describing each step.

    ``sink`` defaults to the SQLite store. The hosted build passes a
    ``memstore.MemoryStore`` so that nothing a stranger submits is written to
    disk; everything comes back in the response instead.
    """
    sink = sink if sink is not None else store
    run_id = run_id or uuid.uuid4().hex[:12]
    started = time.time()

    emit(dict(type="status", stage="boot", message="Starting the loop"))
    ref, info, response = bootstrap(
        lambda m: emit(dict(type="status", stage="boot", message=m)))

    age, sex = float(profile["age"]), int(profile["sex"])
    brain = agents.build(backend, seed=seed)
    emit(dict(type="status", stage="boot",
              message=f"Hypothesis agent: {brain.label}"))

    # --- baseline --------------------------------------------------------
    beta, target, loads = response.personalise(
        profile, lambda m: emit(dict(type="status", stage="boot", message=m)))
    baseline, surrogates = bioevals.evaluate(beta, age, sex, want_subreadouts=True)
    scales = bioevals.calibrate_scales(
        ref, age, sex, progress=lambda m: emit(dict(type="status", stage="boot",
                                                    message=m)))
    expected = bioevals.expected_readout(ref, age, sex)
    clinical = report.clinical_phenoage(profile)
    lifespan_now = mortality.project(baseline, age, sex, expected)
    emit(dict(type="status", stage="boot",
              message=f"Current trajectory projects a median lifespan of "
                      f"{lifespan_now['median_age']:.0f}"))

    emit(dict(
        type="baseline",
        run_id=run_id,
        readout=baseline,
        surrogates=surrogates,
        acceleration=bioevals.acceleration(baseline, expected),
        expected=expected,
        lifespan=lifespan_now,
        loads=loads,
        clinical_phenoage=clinical,
        panel={k: dict(v) for k, v in panels.EVAL_PANEL.items()},
        axes={k: {kk: vv for kk, vv in v.items() if kk != "files"}
              for k, v in panels.AXES.items()},
        reference=info,
        headroom=float((target - beta).abs().mean()),
        agent=brain.label,
    ))
    sink.start_run(run_id, getattr(brain, "name", backend), iterations,
                   profile, baseline, info)

    # --- the loop --------------------------------------------------------
    history, insights = [], []
    probes = min(4, max(2, iterations // 3))
    best = None

    for iteration in range(1, int(iterations) + 1):
        mode = mode_for(iteration, iterations, probes)
        emit(dict(type="iteration_start", iteration=iteration, mode=mode,
                  total=iterations))

        ctx = dict(profile=profile, baseline=baseline, age=age, loads=loads,
                   expected=expected, history=history, insights=insights,
                   iteration=iteration, total=int(iterations), mode=mode)
        t0 = time.time()
        try:
            hypothesis = brain.propose(ctx)
        except Exception as exc:
            emit(dict(type="warning", iteration=iteration,
                      message=f"{type(exc).__name__} from the agent; "
                              f"the built-in optimiser took this iteration"))
            hypothesis = agents.ReasonerAgent(seed=seed + iteration).propose(ctx)
        think_ms = int((time.time() - t0) * 1000)

        emit(dict(type="hypothesis", iteration=iteration, mode=mode,
                  hypothesis=hypothesis, think_ms=think_ms))

        # --- evaluate against the biolearn panel -------------------------
        t0 = time.time()
        treated, applied = response.apply(beta, target, hypothesis["targets"])
        readout, treated_subs = bioevals.evaluate(treated, age, sex,
                                                  want_subreadouts=True)
        scored = bioevals.score_against(baseline, readout, applied, age, scales,
                                        iteration=iteration)
        scored["applied"] = applied
        scored["surrogates"] = treated_subs
        display = bioevals.display_score(scored["reward"])
        eval_ms = int((time.time() - t0) * 1000)

        sink.record(run_id, iteration, mode, hypothesis, scored, display)

        entry = dict(iteration=iteration, mode=mode, title=hypothesis["title"],
                     targets=hypothesis["targets"], reward=scored["reward"],
                     score=display, years_reversed=scored["years_reversed"],
                     generalisation_gap=scored["generalisation_gap"],
                     penalties=scored["penalties"], gains=scored["gains"])
        history.append(entry)

        improved = best is None or scored["reward"] > best["reward"]
        if improved:
            best = dict(entry, hypothesis=hypothesis, scored=scored,
                        surrogates=treated_subs)

        insight = _insight(entry, history, hypothesis)
        insights.append(insight)
        sink.add_insight(run_id, iteration, insight)
        insights[:] = insights[-8:]

        emit(dict(type="evaluation", iteration=iteration, score=display,
                  reward=scored["reward"], scored=scored, applied=applied,
                  surrogates=treated_subs, insight=insight,
                  new_best=improved, eval_ms=eval_ms,
                  best_so_far=(best["score"] if best else display)))

    # --- close the loop --------------------------------------------------
    emit(dict(type="status", stage="report", message="Composing the longevity report"))
    final = report.build(run_id, profile, baseline, surrogates, loads, history,
                         best, scales, info, clinical, expected, lifespan_now)

    # The prompt half of every preference pair: the context an agent would see
    # for this subject, so the exported dataset is trainable as-is.
    prompt_context = (
        agents.profile_brief(profile, baseline, age, expected)
        + "\n\nMECHANISM AXES AND THIS SUBJECT'S HEADROOM\n"
        + agents.axis_brief(loads)
        + "\n\nPropose one longevity intervention hypothesis for this subject."
    )
    exported = sink.export_preferences(run_id, prompt_context)
    sink.finish_run(run_id)

    emit(dict(type="complete", run_id=run_id, report=final, training=exported,
              elapsed_s=round(time.time() - started, 1),
              leaderboard=sink.leaderboard(run_id, limit=25)))
    return final


def _insight(entry, history, hypothesis):
    """One line of learning written back into the agent context."""
    rewards = [h["reward"] for h in history]
    axes = ", ".join(t["axis"] for t in entry["targets"])
    worst = max(entry["penalties"], key=entry["penalties"].get) if entry["penalties"] else None
    worst_txt = (f"; the dominant cost was {worst} at "
                 f"{entry['penalties'][worst]:.2f}") if worst else ""

    if len(rewards) == 1:
        return (f"Iteration 1 set the baseline at reward {entry['reward']:+.3f} "
                f"using {axes}{worst_txt}.")

    delta = entry["reward"] - max(rewards[:-1])
    if delta > 0.03:
        verdict = f"new best, up {delta:+.3f} on the previous leader"
    elif delta > -0.03:
        verdict = "level with the leader, so this direction is exhausted"
    else:
        verdict = f"worse by {delta:+.3f}, so this direction is not paying"

    gap = entry["generalisation_gap"]
    gap_txt = ""
    if abs(gap) > 0.4:
        gap_txt = (f" Held-out clocks disagreed by {gap:+.2f} year-equivalents, "
                   f"which reads as fitting the scored subset rather than "
                   f"general aging.")

    return (f"{axes} at {entry['mode']} gave {entry['reward']:+.3f} "
            f"({entry['years_reversed']:+.2f} yr): {verdict}{worst_txt}.{gap_txt}")
