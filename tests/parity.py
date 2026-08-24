"""Does the slim build produce the same numbers as the full one?

Run it twice, once per mode, and diff. The slim build stubs torch, cvxpy,
scipy and sklearn to fit inside a 250 MB serverless bundle, and DunedinPACE
genuinely calls ``scipy.stats.rankdata``, so this is not a formality.

    python -m tests.parity full > full.json
    python -m tests.parity slim > slim.json
    python -m tests.parity compare full.json slim.json
"""

import json
import os
import sys


def readout(mode):
    if mode == "slim":
        os.environ["LONGEVITY_LOOP_SLIM"] = "1"
        from loopcore import slim
        slim.install()
    else:
        os.environ.pop("LONGEVITY_LOOP_SLIM", None)

    from loopcore import bioevals, intervention, reference

    ref, info = reference.build()
    response = intervention.ResponseModel(ref)

    profile = dict(age=54, sex=1, bmi=30.5, smoking_status="former",
                   pack_years=14, exercise_minutes_per_week=60, sleep_hours=6,
                   alcohol_units_per_week=12, stress_level=7, diet_quality=4,
                   glucose=104, c_reactive_protein=3.1)
    beta, target, loads = response.personalise(profile)
    baseline = bioevals.evaluate(beta, 54, 1)

    treated, applied = response.apply(beta, target, [
        dict(axis="inflammatory_load", intensity=0.8),
        dict(axis="metabolic_glycemia", intensity=0.7),
        dict(axis="adiposity", intensity=0.6),
    ])
    after = bioevals.evaluate(treated, 54, 1)

    return dict(
        mode=mode,
        stubbed=sorted(m for m in ("torch", "scipy", "sklearn", "cvxpy")
                       if getattr(sys.modules.get(m), "_longevity_stub", False)),
        loads=loads,
        applied={k: v for k, v in applied.items() if k != "per_axis"},
        baseline=baseline,
        treated=after,
    )


def compare(a_path, b_path, tolerance=1e-9):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    print(f"{a['mode']} (stubs: {a['stubbed'] or 'none'}) vs "
          f"{b['mode']} (stubs: {b['stubbed'] or 'none'})\n")

    worst = 0.0
    failures = []
    for section in ("baseline", "treated"):
        print(f"  {section}")
        for model in sorted(a[section]):
            x, y = a[section][model], b[section][model]
            if x is None or y is None:
                mark = "both none" if x == y else "MISMATCH (one is null)"
                print(f"    {model:16s} {mark}")
                if x != y:
                    failures.append(model)
                continue
            diff = abs(x - y)
            rel = diff / max(abs(x), 1e-12)
            worst = max(worst, rel)
            flag = "" if rel <= tolerance else "   <-- MISMATCH"
            if rel > tolerance:
                failures.append(f"{section}.{model}")
            print(f"    {model:16s} {x:14.6f}  {y:14.6f}  rel {rel:.2e}{flag}")
        print()

    print(f"largest relative difference: {worst:.3e}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print("PASS: the slim build is numerically identical")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    if argv[0] == "compare":
        return compare(argv[1], argv[2])
    print(json.dumps(readout(argv[0]), indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main(sys.argv[1:]))
