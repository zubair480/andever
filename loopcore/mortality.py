"""Turning a bio-eval panel readout into a projected lifespan.

This answers the two questions the report opens with: how long the current
trajectory runs, and how long it runs if the winning protocol is followed and
held.

The method is an ordinary actuarial one, not a new claim:

1. A Gompertz-Makeham baseline hazard, calibrated so that a person with average
   biomarkers for their age gets the median age at death a modern high-income
   life table gives them.
2. A proportional-hazards multiplier from the panel. The three inputs all have
   published all-cause mortality hazard ratios:

   ===================  ==========================  =========================
   Input                Reported HR                 Weight used here
   ===================  ==========================  =========================
   GrimAge accel        ~1.09 per year              0.075 per year (log HR)
   DunedinPACE          ~1.26 per SD (SD ~0.09)     2.00 per PACE unit
   PhenoAge accel       ~1.045 per year             0.030 per year (log HR)
   ===================  ==========================  =========================

   The weights are set below the published point estimates on purpose. The
   three predictors are strongly correlated, so summing their marginal hazard
   ratios would double-count shared signal.
3. Median, quartiles and mean remaining years read off the survival curve.

WHAT THIS IS NOT: an individual prediction. Epigenetic clocks are validated for
ranking mortality risk across groups, not for dating one person's death. Two
people with identical panels routinely die twenty years apart. The interquartile
band the report prints is the honest width of the answer, and it is wide.
"""

from . import compat  # noqa: F401

import math

import numpy as np

# Gompertz-Makeham: h(age) = makeham + a * exp(b * age)
DOUBLING_YEARS = 8.0                  # adult mortality doubles about every 8 yr
GOMPERTZ_B = math.log(2) / DOUBLING_YEARS
MAKEHAM = 0.0004                      # age-independent background hazard

# Median age at death for a 50-year-old with average biomarkers, which the
# baseline scale factor is solved against. Modern high-income life tables.
_MEDIAN_TARGET = {0: 86.0, 1: 82.0}   # 0 female, 1 male

# Log hazard ratio per unit of each panel input. See the table above.
W_GRIMAGE = 0.075
W_PACE = 2.00
W_PHENOAGE = 0.030

# Nothing in the panel justifies a hazard ratio outside this band.
LOG_HR_CAP = (-1.0, 1.4)

MAX_AGE = 120.0
_STEP = 0.05


def _cumulative_hazard(age_from, age_to, scale, frailty):
    """Integral of the hazard between two ages."""
    a = scale * frailty
    return (MAKEHAM * (age_to - age_from)
            + (a / GOMPERTZ_B) * (math.exp(GOMPERTZ_B * age_to)
                                  - math.exp(GOMPERTZ_B * age_from)))


def _survival_curve(age, scale, frailty):
    ages = np.arange(age, MAX_AGE + _STEP, _STEP)
    a = scale * frailty
    cum = (MAKEHAM * (ages - age)
           + (a / GOMPERTZ_B) * (np.exp(GOMPERTZ_B * ages)
                                 - math.exp(GOMPERTZ_B * age)))
    return ages, np.exp(-cum)


def _quantile_age(ages, survival, q):
    """Age at which the survival curve falls through ``1 - q``."""
    target = 1.0 - q
    below = np.nonzero(survival <= target)[0]
    if not len(below):
        return float(MAX_AGE)
    i = below[0]
    if i == 0:
        return float(ages[0])
    # Linear interpolation between the bracketing steps.
    s0, s1 = survival[i - 1], survival[i]
    t = (s0 - target) / (s0 - s1) if s0 > s1 else 0.0
    return float(ages[i - 1] + t * _STEP)


def _solve_scale(sex):
    """Find the Gompertz scale that reproduces the life-table median."""
    target = _MEDIAN_TARGET.get(int(sex), 82.0)
    lo, hi = 1e-8, 1e-2
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        ages, surv = _survival_curve(50.0, mid, 1.0)
        median = _quantile_age(ages, surv, 0.5)
        if median > target:
            lo = mid          # too healthy, raise the hazard
        else:
            hi = mid
    return math.sqrt(lo * hi)


_SCALE_CACHE = {}


def baseline_scale(sex):
    sex = int(sex)
    if sex not in _SCALE_CACHE:
        _SCALE_CACHE[sex] = _solve_scale(sex)
    return _SCALE_CACHE[sex]


# ---------------------------------------------------------------------------
# Panel readout -> hazard multiplier
# ---------------------------------------------------------------------------

def hazard_multiplier(readout, age, expected):
    """Proportional-hazards multiplier implied by one panel readout.

    ``expected`` is what the reference cohort reads at this age, so every input
    enters as an acceleration rather than an absolute value. That matters:
    biolearn's GrimAge sits about fifteen years above chronological age on this
    array platform for everyone, and only the residual is signal.
    """
    terms = {}

    grim = readout.get("GrimAgeV2")
    if grim is not None and "GrimAgeV2" in expected:
        accel = grim - expected["GrimAgeV2"]
        terms["GrimAge acceleration"] = (round(accel, 2), W_GRIMAGE * accel)

    pace = readout.get("DunedinPACE")
    if pace is not None and "DunedinPACE" in expected:
        excess = pace - expected["DunedinPACE"]
        terms["DunedinPACE excess"] = (round(excess, 3), W_PACE * excess)

    pheno = readout.get("PhenoAge")
    if pheno is not None and "PhenoAge" in expected:
        accel = pheno - expected["PhenoAge"]
        terms["PhenoAge acceleration"] = (round(accel, 2), W_PHENOAGE * accel)

    log_hr = sum(v[1] for v in terms.values())
    capped = float(np.clip(log_hr, *LOG_HR_CAP))
    return dict(
        multiplier=math.exp(capped),
        log_hr=capped,
        capped=bool(abs(capped - log_hr) > 1e-9),
        terms={k: dict(value=v[0], log_hr=round(v[1], 4)) for k, v in terms.items()},
    )


def project(readout, age, sex, expected):
    """Projected lifespan for a cohort with this panel readout."""
    age = float(age)
    frailty = hazard_multiplier(readout, age, expected)
    scale = baseline_scale(sex)
    ages, surv = _survival_curve(age, scale, frailty["multiplier"])

    median = _quantile_age(ages, surv, 0.50)
    lower = _quantile_age(ages, surv, 0.25)   # a quarter die by this age
    upper = _quantile_age(ages, surv, 0.75)
    mean_remaining = float(np.trapezoid(surv, ages)) if hasattr(np, "trapezoid") \
        else float(np.trapz(surv, ages))

    # The same computation with every acceleration zeroed, so the report can
    # say how far off the age-matched average this subject sits.
    _, ref_surv = _survival_curve(age, scale, 1.0)
    reference_median = _quantile_age(ages, ref_surv, 0.50)

    return dict(
        median_age=round(median, 1),
        quartile_low=round(lower, 1),
        quartile_high=round(upper, 1),
        years_remaining=round(median - age, 1),
        mean_years_remaining=round(mean_remaining, 1),
        reference_median_age=round(reference_median, 1),
        versus_average=round(median - reference_median, 1),
        hazard_ratio=round(frailty["multiplier"], 3),
        hazard_capped=frailty["capped"],
        drivers=frailty["terms"],
        survival_curve=[[round(float(a), 1), round(float(s), 4)]
                        for a, s in zip(ages[::20], surv[::20])],
    )


def compare(baseline_readout, treated_readout, age, sex, expected):
    """The two headline numbers, and the gap between them."""
    now = project(baseline_readout, age, sex, expected)
    after = project(treated_readout, age, sex, expected)
    return dict(
        current=now,
        treated=after,
        years_gained=round(after["median_age"] - now["median_age"], 1),
        hazard_reduction=round(1.0 - after["hazard_ratio"] / now["hazard_ratio"], 3)
        if now["hazard_ratio"] else 0.0,
    )
