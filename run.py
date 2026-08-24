"""Entry point.

    python run.py                 start the interface on http://127.0.0.1:8770
    python run.py --port 9000     pick a port
    python run.py --headless      run one loop in the terminal and print the report
    python run.py --rebuild       refit the reference epigenome from the cohort
"""

import argparse
import json
import sys
import webbrowser

from loopcore import looprunner, reference, server


DEMO_PROFILE = dict(
    age=54, sex=1, bmi=30.5, smoking_status="former", pack_years=14,
    exercise_minutes_per_week=60, sleep_hours=6.0, stress_level=7,
    diet_quality=4, glucose=104, c_reactive_protein=3.1,
    alcohol_units_per_week=12, albumin=42, creatinine=0.95,
    lymphocyte_percent=28, mean_cell_volume=90,
    red_blood_cell_distribution_width=13.4, alkaline_phosphate=72,
    white_blood_cell_count=7.2,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--headless", action="store_true",
                        help="run one loop in the terminal instead of serving")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "claude", "reasoner"])
    parser.add_argument("--rebuild", action="store_true",
                        help="refit the reference epigenome before starting")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild:
        reference.build(force=True, progress=lambda m: print("  ", m))

    if args.headless:
        def emit(event):
            kind = event["type"]
            if kind == "status":
                print("  ..", event["message"])
            elif kind == "hypothesis":
                h = event["hypothesis"]
                print(f"\n[{event['iteration']:02d}] {event['mode']:8s} {h['title']}")
                for t in h["targets"]:
                    print(f"      {t['axis']:28s} {t['intensity']:.2f}")
            elif kind == "evaluation":
                flag = "  <- new best" if event["new_best"] else ""
                print(f"      score {event['score']:.1f}  "
                      f"reward {event['reward']:+.3f}{flag}")
            elif kind == "complete":
                r = event["report"]
                life = r["lifespan"]
                now, after = life["current"], life["treated"]
                print("\n" + "=" * 62)
                print(f"  1. On the current trajectory, median age at death is "
                      f"{now['median_age']:.0f}")
                print(f"     (middle half of such a cohort: "
                      f"{now['quartile_low']:.0f} to {now['quartile_high']:.0f}, "
                      f"hazard ratio {now['hazard_ratio']:.2f})")
                print(f"  2. Holding the protocol below, median age at death is "
                      f"{after['median_age']:.0f}")
                print(f"     (middle half: {after['quartile_low']:.0f} to "
                      f"{after['quartile_high']:.0f}, "
                      f"hazard ratio {after['hazard_ratio']:.2f})")
                print(f"\n     Difference: {life['years_gained']:+.1f} years, "
                      f"{life['hazard_reduction'] * 100:.0f}% lower mortality hazard")
                print("=" * 62)
                print("\nProtocol:")
                for item in r["protocol"]:
                    print(f"  [{item['grade']}] {item['name']} - {item['detail']}")
                print("\nPanel movement:", json.dumps(r["headline"], indent=2))
                print("\nTraining data:", event["training"])
        looprunner.run(DEMO_PROFILE, iterations=args.iterations,
                       backend=args.backend, emit=emit)
        return 0

    if not args.no_browser:
        url = f"http://{args.host}:{args.port}"
        try:
            import threading
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    server.serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
