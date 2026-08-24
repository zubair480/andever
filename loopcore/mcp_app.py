"""MCP server, so Claude Code or Codex can drive the loop.

Two modes, chosen by ``LONGEVITY_LOOP_HOSTED``:

**local** (default)
    Runs in the same process as the web server. An agent pushes a profile with
    ``set_subject_profile`` and it appears in the open browser tab; the run it
    starts streams into that same tab. Stateful, single user, full featured.

**hosted** (``LONGEVITY_LOOP_HOSTED=1``)
    For a public endpoint on serverless infrastructure, where every request is
    a separate invocation with no shared memory. There is one single-shot tool
    that takes the profile and returns the answer, and nothing is written down:
    no database, no profile persistence, no health values in logs.

Transports: streamable HTTP alongside the interface, or stdio for clients that
only speak stdio (``python -m loopcore.mcp_app``).
"""

from . import compat  # noqa: F401

import asyncio
import json
import os
import threading

from mcp.server.fastmcp import FastMCP

from . import evidence, panels, session

SERVER_NAME = "longevity-loop"
HOSTED = os.environ.get("LONGEVITY_LOOP_HOSTED") == "1"

MAX_ITERATIONS = 14 if HOSTED else 30
DEFAULT_ITERATIONS = 10 if HOSTED else 12

_SHARED_INSTRUCTIONS = """Estimate how long someone has, and how much of that
is still on the table, using a self-improving loop graded on biolearn
epigenetic clocks.

The answer is always two numbers:
1. Projected median age at death on the current trajectory.
2. The same figure if the loop's best-scoring protocol is followed and held.

Both are cohort-level actuarial projections from published mortality hazard
ratios, NOT predictions about an individual. Two people with identical
biomarker panels routinely die twenty years apart. Always report the quartile
band alongside the median, and never present either number as a personal
forecast. Nothing here is medical advice."""

INSTRUCTIONS_HOSTED = _SHARED_INSTRUCTIONS + """

Call `describe_inputs` to see what the profile accepts, then
`run_longevity_loop` with everything you know about the person. Only age and
sex are required; every other field sharpens the estimate. If the user has
health data in a file, a lab PDF or a wearable export, read it and map it
across rather than making them retype it.

This endpoint is stateless and stores nothing. The profile exists only for the
duration of the call."""

INSTRUCTIONS_LOCAL = _SHARED_INSTRUCTIONS + """

Normal flow:
1. `describe_inputs` to see what the profile accepts.
2. `set_subject_profile` with whatever you know. It fills the form in the
   user's open browser tab so they can see and correct it.
3. `run_longevity_loop`. The browser streams the run live while you wait."""


def _transport_security():
    """Allow the public hostname through the SDK's DNS-rebinding check.

    The MCP SDK ships with enable_dns_rebinding_protection on and an EMPTY host
    allowlist, which passes on localhost and returns 421 Misdirected Request for
    every real domain. LONGEVITY_LOOP_ALLOWED_HOSTS is a comma separated list;
    on Render, RENDER_EXTERNAL_HOSTNAME is set for us. With neither set the
    protection stays fully on, which is the right default for a local run.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = [h.strip() for h in
             os.environ.get("LONGEVITY_LOOP_ALLOWED_HOSTS", "").split(",")
             if h.strip()]
    render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    if render_host:
        hosts.append(render_host)
    if not hosts:
        return None

    # A host entry has to cover the port-qualified form too, and each host needs
    # a matching origin or a browser-issued request is refused separately.
    allowed = []
    for host in hosts:
        allowed += [host, f"{host}:443", f"{host}:80"]
    origins = [f"https://{h}" for h in hosts] + [f"http://{h}" for h in hosts]
    return TransportSecuritySettings(allowed_hosts=allowed,
                                     allowed_origins=origins)


mcp = FastMCP(SERVER_NAME,
              instructions=INSTRUCTIONS_HOSTED if HOSTED else INSTRUCTIONS_LOCAL,
              stateless_http=True,
              transport_security=_transport_security())


PROFILE_DOC = [
    ("age", "years", "required"),
    ("sex", "0 female, 1 male", "required"),
    ("bmi", "kg/m2", "23 if unknown"),
    ("smoking_status", "never | former | current", "never"),
    ("pack_years", "cigarette pack-years", "0"),
    ("exercise_minutes_per_week", "minutes of moderate-plus activity", "150"),
    ("sleep_hours", "hours per night", "7.25"),
    ("alcohol_units_per_week", "UK units", "4"),
    ("stress_level", "1 low to 10 high", "4"),
    ("diet_quality", "1 poor to 10 excellent", "6"),
    ("glucose", "fasting glucose mg/dL", "92"),
    ("c_reactive_protein", "hs-CRP mg/L", "1.0"),
    ("albumin", "g/L", "optional, enables clinical PhenoAge"),
    ("creatinine", "mg/dL", "optional"),
    ("lymphocyte_percent", "%", "optional"),
    ("mean_cell_volume", "fL", "optional"),
    ("red_blood_cell_distribution_width", "%", "optional"),
    ("alkaline_phosphate", "U/L", "optional"),
    ("white_blood_cell_count", "10^9/L", "optional"),
]


def _profile(values):
    """Drop unset fields and coerce sex to an int."""
    out = {k: v for k, v in values.items() if v is not None}
    if "sex" in out:
        out["sex"] = int(out["sex"])
    return out


def _describe_inputs():
    lines = ["field | unit | default", "----- | ---- | -------"]
    lines += [f"{n} | {u} | {d}" for n, u, d in PROFILE_DOC]
    lines.append("")
    lines.append("The nine optional blood values must all be present together "
                 "to compute the clinical PhenoAge; any missing one skips it.")
    return "\n".join(lines)


def _describe_harness():
    lines = ["EVAL PANEL (biolearn models)"]
    for name, spec in panels.EVAL_PANEL.items():
        held = (" [held out on alternating iterations]"
                if name in panels.GENERALISATION_HOLDOUT else "")
        lines.append(f"  {name}: {spec['label']}, {spec['family']}, "
                     f"weight {spec['weight']}{held}")
    lines.append("")
    lines.append("MECHANISM AXES (what a hypothesis may target)")
    for name, spec in panels.AXES.items():
        lines.append(f"  {name}: {spec['biology']}")
    lines.append("")
    lines.append("INTERVENTION CATALOGUE (evidence grade, risk 0-1)")
    for item in evidence.INTERVENTIONS:
        lines.append(f"  [{item['grade']}] {item['name']} "
                     f"(risk {item['risk']}) - {item['detail']}")
    return "\n".join(lines)


def _summarise(report, run_id, training=None):
    life = report.get("lifespan") or {}
    now, after = life.get("current", {}), life.get("treated", {})
    protocol = "\n".join(
        f"  [{p['grade']}] {p['name']} ({p['stance']}) - {p['detail']}"
        for p in report.get("protocol", [])) or "  none"
    head = report.get("headline", {})
    learning = report.get("learning", {})
    extra = ""
    if training and training.get("pairs"):
        extra = (f"\nThe run also produced {training['pairs']} DPO preference "
                 f"pairs from its own scored hypotheses.")
    return f"""Run {run_id} complete.

1. WHEN, ON THE CURRENT TRAJECTORY
   Median age at death {now.get('median_age')}.
   Middle half of a cohort like this: {now.get('quartile_low')} to {now.get('quartile_high')}.
   Mortality hazard ratio {now.get('hazard_ratio')} against an age-matched average.

2. WHEN, IF THE PROTOCOL BELOW IS FOLLOWED AND HELD
   Median age at death {after.get('median_age')}.
   Middle half: {after.get('quartile_low')} to {after.get('quartile_high')}.
   Difference: {life.get('years_gained')} years, hazard down {round((life.get('hazard_reduction') or 0) * 100)}%.

PROTOCOL (best-scoring hypothesis, iteration {report.get('best', {}).get('iteration')})
{protocol}

PANEL MOVEMENT
   GrimAge V2 {head.get('grimage_years')} yr, PhenoAge {head.get('phenoage_years')} yr,
   DunedinPACE {head.get('pace_before')} to {head.get('pace_after')}.
   Confidence: {head.get('confidence')}.

LOOP
   Reward climbed from {learning.get('first')} to {learning.get('best')}
   over {len(learning.get('per_iteration', []))} iterations.{extra}

Report both figures as cohort projections with their quartile bands, not as a
prediction about this person. Nothing here is medical advice.
"""


def _run_now(profile, iterations):
    """Run the loop synchronously in this process, storing nothing."""
    from . import looprunner, memstore

    sink = memstore.MemoryStore()
    report = looprunner.run(profile, iterations=iterations, backend="auto",
                            sink=sink)
    return report, sink.exported


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def describe_inputs() -> str:
    """List every field a subject profile accepts, with units and defaults.

    Call this first so values are sent in the units the harness expects.
    """
    return _describe_inputs()


@mcp.tool()
def describe_harness() -> str:
    """Explain what the loop is scored against: the panel, axes and catalogue."""
    return _describe_harness()


if HOSTED:

    @mcp.tool()
    def run_longevity_loop(
        age: float,
        sex: int,
        bmi: float | None = None,
        smoking_status: str | None = None,
        pack_years: float | None = None,
        exercise_minutes_per_week: float | None = None,
        sleep_hours: float | None = None,
        alcohol_units_per_week: float | None = None,
        stress_level: float | None = None,
        diet_quality: float | None = None,
        glucose: float | None = None,
        c_reactive_protein: float | None = None,
        albumin: float | None = None,
        creatinine: float | None = None,
        lymphocyte_percent: float | None = None,
        mean_cell_volume: float | None = None,
        red_blood_cell_distribution_width: float | None = None,
        alkaline_phosphate: float | None = None,
        white_blood_cell_count: float | None = None,
        iterations: int = DEFAULT_ITERATIONS,
    ) -> str:
        """Project a lifespan and search for the protocol that extends it most.

        Single-shot: pass everything you know about the person and get both
        numbers back. Only age and sex are required. Nothing is stored.
        """
        profile = _profile(locals())
        profile.pop("iterations", None)
        iterations = max(4, min(int(iterations), MAX_ITERATIONS))
        report, training = _run_now(profile, iterations)
        return _summarise(report, report.get("run_id", "hosted"), training)

else:

    @mcp.tool()
    def set_subject_profile(
        age: float,
        sex: int,
        bmi: float | None = None,
        smoking_status: str | None = None,
        pack_years: float | None = None,
        exercise_minutes_per_week: float | None = None,
        sleep_hours: float | None = None,
        alcohol_units_per_week: float | None = None,
        stress_level: float | None = None,
        diet_quality: float | None = None,
        glucose: float | None = None,
        c_reactive_protein: float | None = None,
        albumin: float | None = None,
        creatinine: float | None = None,
        lymphocyte_percent: float | None = None,
        mean_cell_volume: float | None = None,
        red_blood_cell_distribution_width: float | None = None,
        alkaline_phosphate: float | None = None,
        white_blood_cell_count: float | None = None,
    ) -> str:
        """Push a subject profile into the open Longevity AI Loop tab.

        Only age and sex are required. The browser form fills in live, so the
        person can see and correct what was sent before the loop runs.
        """
        from . import server as web

        session.note_client("agent", "mcp")
        profile = _profile(locals())
        session.set_profile(profile, source="MCP client")
        known = ", ".join(sorted(k for k in profile if k not in ("age", "sex")))
        return (f"Profile set for a {age:.0f} year old "
                f"{'male' if int(sex) == 1 else 'female'}. "
                f"Also supplied: {known or 'nothing else'}. "
                f"It is now showing in the browser at "
                f"{web.public_url()}. "
                f"Call run_longevity_loop next.")

    @mcp.tool()
    def run_longevity_loop(iterations: int = DEFAULT_ITERATIONS) -> str:
        """Run the loop on the current profile and return both lifespan figures.

        Blocks until the run finishes, typically ten to thirty seconds. The
        browser tab streams the same run live while this waits.
        """
        from . import server as web

        session.note_client("agent", "mcp")
        profile = session.take_profile()
        if not profile or not profile.get("age"):
            return "No subject profile set. Call set_subject_profile first."

        iterations = max(4, min(int(iterations), MAX_ITERATIONS))
        run_id = web.start_run(dict(profile=profile, iterations=iterations,
                                    backend="auto"))
        session.set_active_run(run_id)

        run = web.get_run(run_id)
        if run is None:
            return "Failed to start the run."
        run.done_event.wait(timeout=600)

        report = None
        for event in run.events:
            if event.get("type") == "complete":
                report = event["report"]
            elif event.get("type") == "error":
                return f"Run failed: {event.get('message')}"
        if report is None:
            return "The run did not produce a report before the timeout."
        return _summarise(report, run_id)

    @mcp.tool()
    def get_last_report() -> str:
        """Return the full report from the most recent run as JSON."""
        session.note_client("agent", "mcp")
        summary = session.last_summary()
        if not summary:
            return "No run has completed yet. Call run_longevity_loop."
        return json.dumps(summary, indent=2)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

def serve_http_in_thread(host="127.0.0.1", port=8771):
    """Run the streamable-HTTP MCP endpoint beside the web server."""
    mcp.settings.host = host
    mcp.settings.port = port

    def worker():
        try:
            asyncio.run(mcp.run_streamable_http_async())
        except Exception as exc:  # port taken, or no event loop support
            print(f"  MCP endpoint unavailable: {type(exc).__name__}: {exc}")

    thread = threading.Thread(target=worker, daemon=True, name="mcp-http")
    thread.start()
    return thread


def main():
    mcp.run("stdio")


if __name__ == "__main__":
    main()
