"""Check the inlined Levine formula against biolearn's own implementation.

report.py transcribes biolearn.hematology.phenotypic_age so the deployed
service needs no biolearn. Transcription is exactly the kind of change that
looks right and is wrong, so this compares the two across random profiles.

    python -m tools.check_phenoage
"""

import sys

import numpy as np


def main():
    from loopcore import compat  # noqa: F401  (torch before biolearn)
    import pandas as pd
    from biolearn.hematology import phenotypic_age

    from loopcore.report import _phenotypic_age

    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(500):
        row = {
            "age": rng.uniform(20, 95),
            "albumin": rng.uniform(30, 55),
            "creatinine": rng.uniform(40, 160),
            "glucose": rng.uniform(3.5, 12.0),
            "c_reactive_protein": rng.uniform(0.01, 2.0),
            "lymphocyte_percent": rng.uniform(10, 50),
            "mean_cell_volume": rng.uniform(75, 105),
            "red_blood_cell_distribution_width": rng.uniform(11, 18),
            "alkaline_phosphate": rng.uniform(30, 160),
            "white_blood_cell_count": rng.uniform(3, 14),
        }
        theirs = float(phenotypic_age(pd.DataFrame([row])).iloc[0])
        ours = _phenotypic_age(row)
        if np.isfinite(theirs):
            worst = max(worst, abs(theirs - ours) / max(abs(theirs), 1e-12))

    print(f"max relative difference over 500 profiles: {worst:.3e}")
    ok = worst < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
