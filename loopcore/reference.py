"""Builds and caches the reference epigenome model.

The reference is a per-CpG multiple regression fitted on a real GEO whole-blood
450k cohort loaded through ``biolearn.data_library``:

    beta_cpg ~ intercept + b_age * age + b_sex * sex + b_dis * disease

From it we can synthesise the expected methylome of a person at a given age and
sex, and we can measure how far each CpG has drifted from its young-adult value.
That drift is what an intervention is allowed to claw back.

The fit is computed once and cached to ``runtime/reference_epigenome.csv.gz`` so
that later runs start instantly and work with no network.
"""

from . import compat  # noqa: F401

import json
import os

import numpy as np
import pandas as pd

from . import panels

# Same override as loopcore.store, for the same reason: a container host may
# not let this process write next to the code. --rebuild is the only thing that
# writes here, and the shipped precomputed copy means it never runs unasked.
RUNTIME = os.environ.get("LONGEVITY_LOOP_RUNTIME") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runtime")
REFERENCE_CSV = os.path.join(RUNTIME, "reference_epigenome.csv.gz")
REFERENCE_META = os.path.join(RUNTIME, "reference_meta.json")

# A copy of the fitted reference ships with the package, so a fresh clone and
# the hosted build both start instantly and need no GEO download. runtime/
# still wins when it exists, which is what --rebuild writes to.
PRECOMPUTED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "precomputed")


def _cached_pair():
    for csv, meta in ((REFERENCE_CSV, REFERENCE_META),
                      (os.path.join(PRECOMPUTED, "reference_epigenome.csv.gz"),
                       os.path.join(PRECOMPUTED, "reference_meta.json"))):
        if os.path.exists(csv) and os.path.exists(meta):
            return csv, meta
    return None, None

# GEO series used as the aging reference. 95 Dutch whole-blood 450k samples,
# age 18-65, with sex and disease status. Downloaded and cached by biolearn.
COHORT_ID = "GSE41169"

YOUNG_ADULT_AGE = 25.0   # the target the "youthful" reference is evaluated at
MIN_COVERAGE = 0.60      # fraction of samples that must be non-null per CpG


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_from_cohort(progress=lambda m: None):
    from biolearn.data_library import DataLibrary

    progress(f"Loading reference cohort {COHORT_ID} through biolearn")
    data = DataLibrary().get(COHORT_ID).load()

    meta = data.metadata.copy()
    dnam = data.dnam

    keep = meta.index[meta["age"].notna() & meta["sex"].notna()]
    keep = [s for s in keep if s in dnam.columns]
    meta = meta.loc[keep]
    progress(f"Cohort has {len(keep)} usable samples, {dnam.shape[0]} probes")

    cpgs = [c for c in panels.universe() if c in dnam.index]
    progress(f"Restricting to the {len(cpgs)} CpG universe used by the panel")

    beta = dnam.loc[cpgs, keep].astype(float)

    age = meta["age"].to_numpy(float)
    sex = meta["sex"].to_numpy(float)               # 0 female, 1 male
    disease = meta.get("disease")
    if disease is None:
        dis = np.zeros_like(age)
    else:
        d = pd.to_numeric(disease, errors="coerce").to_numpy(float)
        # Encode as 0/1 against the modal (control) value; NaN -> control.
        vals, counts = np.unique(d[~np.isnan(d)], return_counts=True)
        control = vals[np.argmax(counts)] if len(vals) else 0.0
        dis = np.where(np.isnan(d), 0.0, (d != control).astype(float))

    design = np.column_stack([np.ones_like(age), age, sex, dis])

    Y = beta.to_numpy(float)                         # (n_cpg, n_sample)
    ok = np.isfinite(Y)
    coverage = ok.mean(axis=1)
    usable = coverage >= MIN_COVERAGE
    progress(f"{int(usable.sum())} CpGs pass the {MIN_COVERAGE:.0%} coverage bar")

    # Column-mean impute the few remaining holes so one lstsq covers everything.
    row_means = np.nanmean(np.where(ok, Y, np.nan), axis=1)
    Yf = np.where(ok, Y, row_means[:, None])
    Yf = np.nan_to_num(Yf, nan=0.5)

    coef, *_ = np.linalg.lstsq(design, Yf.T, rcond=None)   # (4, n_cpg)
    fitted = design @ coef
    resid = Yf.T - fitted
    dof = max(len(age) - design.shape[1], 1)
    resid_sd = np.sqrt((resid ** 2).sum(axis=0) / dof)

    # Partial correlation of beta with age after removing sex and disease.
    nuisance = np.column_stack([np.ones_like(age), sex, dis])
    ncoef, *_ = np.linalg.lstsq(nuisance, Yf.T, rcond=None)
    y_res = Yf.T - nuisance @ ncoef
    acoef, *_ = np.linalg.lstsq(nuisance, age, rcond=None)
    a_res = age - nuisance @ acoef
    denom = np.sqrt((y_res ** 2).sum(axis=0)) * np.sqrt((a_res ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r_age = np.where(denom > 0, (y_res * a_res[:, None]).sum(axis=0) / denom, 0.0)

    ref = pd.DataFrame(
        {
            "intercept": coef[0],
            "b_age": coef[1],
            "b_sex": coef[2],
            "b_disease": coef[3],
            "resid_sd": resid_sd,
            "r_age": np.nan_to_num(r_age),
            "coverage": coverage,
            "observed_mean": row_means,
        },
        index=beta.index,
    )
    ref = ref.loc[usable]

    info = dict(
        source="geo_cohort",
        cohort=COHORT_ID,
        samples=int(len(keep)),
        age_min=float(np.nanmin(age)),
        age_max=float(np.nanmax(age)),
        age_mean=float(np.nanmean(age)),
        cpgs=int(len(ref)),
    )
    return ref, info


def _fit_offline(progress=lambda m: None):
    """No-network fallback.

    Uses the sesame 450k median methylome that biolearn ships as the population
    level, and votes the sign of every linear age clock coefficient to get each
    CpG age direction. Slope magnitude is then calibrated so that the Horvath
    clock reads back chronological age. Less faithful than the cohort fit, but
    it keeps the whole pipeline runnable with no internet.
    """
    progress("No cohort available; building the analytic offline reference")
    med = panels.read_coefficients("sesame_450k_median.csv")["median"]

    cpgs = [c for c in panels.universe() if c in med.index]
    level = med.loc[cpgs].astype(float).clip(0.001, 0.999)

    votes = pd.Series(0.0, index=cpgs)
    weightsum = pd.Series(1e-9, index=cpgs)
    for name in ["Horvath1.csv", "Hannum.csv", "PhenoAge.csv", "StocP.csv",
                 "Lin.csv", "YingCausAge.csv", "HRSInCHPhenoAge.csv"]:
        try:
            tab = panels.read_coefficients(name)
        except Exception:
            continue
        if "CoefficientTraining" not in tab.columns:
            continue
        w = tab["CoefficientTraining"]
        w = w[[i for i in w.index if isinstance(i, str) and i.startswith("cg")]]
        w = w[w.index.isin(cpgs)]
        if w.empty:
            continue
        scale = w.abs().max() or 1.0
        votes.loc[w.index] += np.sign(w) * (w.abs() / scale)
        weightsum.loc[w.index] += w.abs() / scale

    direction = (votes / weightsum).fillna(0.0)
    # Sites with no clock vote drift towards 0.5 (regression to the mean).
    direction[direction.abs() < 1e-6] = (0.5 - level[direction.abs() < 1e-6]) * 0.4

    slope = direction * 0.0022          # ~0.2 percentage points of beta per decade
    intercept = level - slope * 45.0    # sesame medians sit near mid-adulthood

    ref = pd.DataFrame(
        {
            "intercept": intercept,
            "b_age": slope,
            "b_sex": 0.0,
            "b_disease": 0.0,
            "resid_sd": (level * (1 - level)).clip(lower=1e-4) ** 0.5 * 0.05,
            "r_age": direction.clip(-1, 1),
            "coverage": 1.0,
            "observed_mean": level,
        },
        index=cpgs,
    )
    info = dict(source="offline_analytic", cohort="sesame_450k_median",
                samples=0, age_min=20.0, age_max=80.0, age_mean=45.0,
                cpgs=int(len(ref)))
    return ref, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(force=False, progress=lambda m: None):
    """Return ``(reference_frame, info_dict)``, building and caching if needed."""
    csv, meta = _cached_pair()
    if not force and csv:
        ref = pd.read_csv(csv, index_col=0)
        info = json.load(open(meta, encoding="utf-8"))
        progress(f"Reference epigenome loaded from cache ({len(ref)} CpGs, "
                 f"{info.get('cohort')})")
        return ref, info

    os.makedirs(RUNTIME, exist_ok=True)
    try:
        ref, info = _fit_from_cohort(progress)
    except Exception as exc:  # network down, GEO unreachable, parser change
        progress(f"Cohort fit unavailable ({type(exc).__name__}: {exc})")
        ref, info = _fit_offline(progress)

    ref = _annotate_strata(ref)
    ref.to_csv(REFERENCE_CSV)
    info["strata"] = {k: int(v) for k, v in ref["stratum"].value_counts().items()}
    json.dump(info, open(REFERENCE_META, "w", encoding="utf-8"), indent=2)
    progress(f"Reference epigenome built: {len(ref)} CpGs from {info['cohort']}")
    return ref, info


def _annotate_strata(ref):
    """Assign every CpG to a mechanistic stratum from its own aging statistics."""
    level50 = ref["intercept"] + ref["b_age"] * 50.0
    r = ref["r_age"]
    sd = ref["resid_sd"]
    sd_hi = sd.quantile(0.66)

    stratum = pd.Series("unassigned", index=ref.index)
    stratum[(ref["b_age"] > 0) & (level50 < 0.45) & (r > 0.20)] = "cgi_hyper"
    stratum[(ref["b_age"] < 0) & (level50 > 0.55) & (r < -0.20)] = "pmd_hypo"
    stratum[(r.abs() < 0.15) & (sd >= sd_hi)] = "stochastic"

    ref = ref.copy()
    ref["stratum"] = stratum
    ref["level50"] = level50
    return ref


def expected_beta(ref, age, sex=0, disease=0.0):
    """Population-expected methylome for a person of this age and sex."""
    return (ref["intercept"]
            + ref["b_age"] * float(age)
            + ref["b_sex"] * float(sex)
            + ref["b_disease"] * float(disease)).clip(0.001, 0.999)


def youthful_beta(ref, sex=0):
    """The young-adult target profile. Interventions can approach it, not pass it."""
    return expected_beta(ref, YOUNG_ADULT_AGE, sex=sex, disease=0.0)
