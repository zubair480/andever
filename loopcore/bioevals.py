"""The bio-eval harness: biolearn clocks used as a graded test paper.

``evaluate`` runs a methylome through the whole panel. ``score_against`` turns a
baseline reading and a post-intervention reading into a single reward plus a
full breakdown, and it is deliberately hard to game:

* the panel spans five clock families, so pushing one clock alone scores badly;
* every model is auto-calibrated against its own drift per year of age, so all
  improvements are expressed in the same unit (years of biological age);
* three panel members are held out of the reward on alternating iterations and
  reported back as a generalisation gap;
* benefit is charged against intervention burden, off-target methylation change
  and an incoherence penalty when the clock families disagree.
"""

from . import compat  # noqa: F401

import math
import os
import warnings

import numpy as np
import pandas as pd

from . import panels

_GALLERY = None
_MODELS = {}

# Which column of a biolearn prediction frame carries the headline number.
_COLUMN = {
    "GrimAgeV2": "DNAmGrimAge",
}

# GrimAge also returns its DNAm surrogate biomarkers. They are the most
# interpretable readouts in the whole panel, so we keep them for the report.
GRIMAGE_SUBREADOUTS = {
    "DNAmlogCRP": "DNAm C-reactive protein (log)",
    "DNAmPACKYRS": "DNAm smoking pack-years",
    "DNAmlogA1C": "DNAm HbA1c (log)",
    "DNAmLeptin": "DNAm leptin",
    "DNAmB2M": "DNAm beta-2-microglobulin",
    "DNAmGDF15": "DNAm GDF15",
    "DNAmTIMP1": "DNAm TIMP1",
    "DNAmPAI1": "DNAm PAI-1",
    "DNAmADM": "DNAm adrenomedullin",
    "DNAmCystatinC": "DNAm cystatin C",
}


def gallery():
    global _GALLERY
    if _GALLERY is None:
        from biolearn.model_gallery import ModelGallery
        _GALLERY = ModelGallery()
    return _GALLERY


def model(name):
    if name not in _MODELS:
        _MODELS[name] = gallery().get(name)
    return _MODELS[name]


def warm_up(progress=lambda m: None):
    """Load the panel up front so the loop runs at speed."""
    panel = fast_panel()
    if panel is not None:
        progress(f"Bio-eval panel ready: {len(panels.EVAL_PANEL)} models from the "
                 f"precomputed numpy panel")
        return
    for name in panels.EVAL_PANEL:
        try:
            model(name)
        except Exception as exc:
            progress(f"panel model {name} unavailable: {exc}")
    progress(f"Bio-eval panel ready: {len(_MODELS)}/{len(panels.EVAL_PANEL)} biolearn models")


# ---------------------------------------------------------------------------
# Running the panel
# ---------------------------------------------------------------------------

def _geo_data(beta, age, sex):
    from biolearn.data_library import GeoData

    dnam = pd.DataFrame({"subject": beta.astype(float)})
    meta = pd.DataFrame({"age": [float(age)], "sex": [int(sex)]}, index=["subject"])
    return GeoData(metadata=meta, dnam=dnam)


def _pick(prediction, name):
    """Reduce a biolearn prediction frame to one number for our single sample."""
    if isinstance(prediction, pd.DataFrame):
        wanted = _COLUMN.get(name)
        if wanted and wanted in prediction.columns:
            series = prediction[wanted]
        elif "Predicted" in prediction.columns:
            series = prediction["Predicted"]
        else:
            numeric = prediction.select_dtypes("number")
            series = numeric.iloc[:, 0] if numeric.shape[1] else prediction.iloc[:, 0]
    else:
        series = prediction
    values = pd.to_numeric(pd.Series(series).squeeze(), errors="coerce")
    return float(np.atleast_1d(values)[0])


_FAST = None


def fast_panel():
    """The distilled numpy panel, or None when it is unavailable or disabled.

    ``tools/build_fastpanel.py`` bakes biolearn's coefficient tables into a
    single npz, which is what the deployed service loads. It matches biolearn to
    float64 rounding and runs about 136x faster, which is the difference between
    a host that fits this app and one that does not. Set LONGEVITY_LOOP_FAST=0
    to force the biolearn path, which is what the parity test does.
    """
    global _FAST
    if _FAST is None:
        if os.environ.get("LONGEVITY_LOOP_FAST", "1") == "0":
            _FAST = False
        else:
            try:
                from . import fastpanel
                _FAST = fastpanel.FastPanel.load()
            except Exception:
                _FAST = False       # fall back to biolearn rather than fail
    return _FAST or None


def _as_array(beta, panel):
    """Beta as a numpy vector in the panel's CpG order, or None if it cannot be."""
    values = getattr(beta, "to_numpy", None)
    if values is None:
        return np.asarray(beta, dtype=float)
    # The methylome is always built on the reference index, so the orders match
    # in practice. Check rather than assume: a silent misalignment here would
    # produce plausible numbers that are entirely wrong.
    index = np.asarray(beta.index)
    if index.shape != panel.cpgs.shape or not np.array_equal(index, panel.cpgs):
        return None
    return beta.to_numpy(dtype=float)


def evaluate(beta, age, sex, want_subreadouts=False):
    """Run the eval panel on one methylome.

    Returns ``{model: value}``, plus ``{surrogate: value}`` when asked.
    """
    panel = fast_panel()
    if panel is not None:
        array = _as_array(beta, panel)
        if array is not None:
            return panel.evaluate(array, age, sex, want_subreadouts=want_subreadouts)

    out, subs = {}, {}
    for name in panels.EVAL_PANEL:
        try:
            mdl = model(name)
        except Exception:
            out[name] = None
            continue
        # GrimAge appends Age/Female/Intercept rows to the frame it is handed,
        # so every model gets its own freshly built copy of the data.
        data = _geo_data(beta, age, sex)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = mdl.predict(data)
            out[name] = _pick(pred, name)
            if want_subreadouts and name == "GrimAgeV2" and isinstance(pred, pd.DataFrame):
                for col in GRIMAGE_SUBREADOUTS:
                    if col in pred.columns:
                        subs[col] = float(pd.to_numeric(pred[col]).iloc[0])
        except Exception:
            out[name] = None
    return (out, subs) if want_subreadouts else out


# ---------------------------------------------------------------------------
# Self-calibration: express every model in years-of-aging units
# ---------------------------------------------------------------------------

_MIN_SLOPE = 1e-9
_SLOPE_CACHE = None
_FIT_CACHE = None
_SLOPE_FILE = "panel_age_fit.json"


_REFERENCE_READOUT = {}


def expected_readout(ref, age, sex):
    """What the panel reads for someone this age with a neutral lifestyle.

    This has to be measured the same way the subject is, not read off the real
    cohort's regression line. Two reasons. Several panel models carry a large
    constant offset on this array platform, so subtracting chronological age
    would report that offset as the subject's age acceleration: biolearn's
    DNAmGrimAge sits about fifteen years high for every GSE41169 sample, young
    or old. And the reference methylome is a conditional mean, which compresses
    DunedinPACE relative to individual samples. Comparing a synthetic methylome
    against a real-sample fit would charge that compression to the subject.

    So the reference point is the same generator with every lifestyle offset at
    zero: the subject's own age and sex, neutral habits.
    """
    from . import reference as _ref

    key = (round(float(age), 1), int(sex))
    if key not in _REFERENCE_READOUT:
        neutral = _ref.expected_beta(ref, age, sex=sex)
        _REFERENCE_READOUT[key] = evaluate(neutral, age, sex)
    return _REFERENCE_READOUT[key]


def cohort_fit_readout(age):
    """What the real cohort's regression line predicts at this age."""
    fit = _FIT_CACHE or {}
    return {name: c["intercept"] + c["slope"] * float(age)
            for name, c in fit.items()}


def acceleration(readout, expected):
    """Age acceleration per panel model, against the neutral-lifestyle peer."""
    out = {}
    for name, value in readout.items():
        if value is None or panels.EVAL_PANEL.get(name, {}).get("kind") != "age":
            continue
        if expected.get(name) is None:
            continue
        out[name] = value - expected[name]
    return out


def calibrate_scales(ref, age, sex, span=8.0, progress=lambda m: None):
    """Measure how much each panel model moves per year of chronological age.

    Dividing an improvement by this slope converts it into "years of biological
    aging undone", which puts a telomere index, a mortality z-score and a
    methylation clock on one comparable axis.

    Slopes come from the real reference cohort where it is reachable, because
    the synthetic conditional-mean methylome compresses the spread of the
    pace-of-aging clocks. The synthetic fit is the fallback.
    """
    slopes = _cohort_slopes(progress)
    if slopes:
        progress(f"Panel calibrated on real cohort samples: {len(slopes)} models "
                 f"in year-equivalents")
        return slopes
    return _synthetic_slopes(ref, age, sex, span, progress)


def _cohort_slopes(progress, n_samples=24):
    """Regress each panel prediction on chronological age across real samples.

    Caches both halves of the fit: the slope is the year-equivalent scale, the
    intercept is what makes age acceleration a residual rather than a raw
    difference from chronological age.
    """
    global _SLOPE_CACHE, _FIT_CACHE
    if _SLOPE_CACHE is not None:
        return _SLOPE_CACHE

    import json
    import os
    from . import reference as _ref

    path = os.path.join(_ref.RUNTIME, _SLOPE_FILE)
    if not os.path.exists(path):
        path = os.path.join(_ref.PRECOMPUTED, _SLOPE_FILE)
    if os.path.exists(path):
        try:
            fit = json.load(open(path, encoding="utf-8"))
            _FIT_CACHE = fit
            _SLOPE_CACHE = {k: abs(v["slope"]) for k, v in fit.items()
                            if abs(v["slope"]) > _MIN_SLOPE}
            return _SLOPE_CACHE
        except Exception:
            pass

    try:
        from biolearn.data_library import DataLibrary, GeoData

        progress("Calibrating the panel on real cohort samples (first run only)")
        data = DataLibrary().get(_ref.COHORT_ID).load()
        meta = data.metadata
        usable = meta.index[meta["age"].notna() & meta["sex"].notna()]
        usable = [s for s in usable if s in data.dnam.columns]
        picked = list(pd.Series(usable).sample(
            n=min(n_samples, len(usable)), random_state=11))
        ages = meta.loc[picked, "age"].to_numpy(float)

        rows = {}
        for name in panels.EVAL_PANEL:
            try:
                mdl = model(name)
            except Exception:
                continue
            geo = GeoData(metadata=meta.loc[picked].copy(),
                          dnam=data.dnam[picked].copy())
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred = mdl.predict(geo)
                column = _COLUMN.get(name)
                if isinstance(pred, pd.DataFrame):
                    if column and column in pred.columns:
                        series = pred[column]
                    elif "Predicted" in pred.columns:
                        series = pred["Predicted"]
                    else:
                        series = pred.select_dtypes("number").iloc[:, 0]
                else:
                    series = pred
                values = pd.to_numeric(series, errors="coerce").reindex(picked)
                rows[name] = values.to_numpy(float)
            except Exception:
                continue

        fit = {}
        for name, values in rows.items():
            ok = np.isfinite(values) & np.isfinite(ages)
            if ok.sum() < 8:
                continue
            slope, intercept = np.polyfit(ages[ok], values[ok], 1)
            if math.isfinite(slope) and abs(slope) > _MIN_SLOPE:
                fit[name] = dict(slope=float(slope), intercept=float(intercept),
                                 n=int(ok.sum()))

        if fit:
            os.makedirs(_ref.RUNTIME, exist_ok=True)
            json.dump(fit, open(path, "w", encoding="utf-8"), indent=2)
            _FIT_CACHE = fit
            _SLOPE_CACHE = {k: abs(v["slope"]) for k, v in fit.items()}
            return _SLOPE_CACHE
    except Exception as exc:
        progress(f"Cohort calibration unavailable ({type(exc).__name__}); "
                 f"using the synthetic reference instead")
    return None


def _synthetic_slopes(ref, age, sex, span, progress):
    from . import reference as _ref

    low = max(age - span, 20.0)
    high = age + span
    a = evaluate(_ref.expected_beta(ref, low, sex=sex), low, sex)
    b = evaluate(_ref.expected_beta(ref, high, sex=sex), high, sex)

    scales = {}
    for name in panels.EVAL_PANEL:
        if a.get(name) is None or b.get(name) is None:
            continue
        slope = (b[name] - a[name]) / (high - low)
        if abs(slope) >= _MIN_SLOPE and math.isfinite(slope):
            scales[name] = abs(slope)
    progress(f"Panel calibrated on the synthetic reference: {len(scales)} models")
    return scales


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# No single clock is allowed to claim more than this many years of reversal.
# Without the clamp, a model with a shallow age slope turns a rounding-level
# change into a double-digit "gain" and dominates the average.
GAIN_CLAMP = 12.0


def _year_equivalent_gains(baseline, treated, scales):
    """Per-model improvement in years of biological aging undone."""
    gains = {}
    for name, spec in panels.EVAL_PANEL.items():
        before, after = baseline.get(name), treated.get(name)
        scale = scales.get(name)
        if before is None or after is None or not scale:
            continue
        if not (math.isfinite(before) and math.isfinite(after)):
            continue
        raw = (before - after) * spec["direction"] / scale
        gains[name] = float(np.clip(raw, -GAIN_CLAMP, GAIN_CLAMP))
    return gains


def score_against(baseline, treated, applied, age, scales, iteration=0,
                  use_holdout=True):
    """Grade one hypothesis. ``applied`` comes from ``intervention.apply``."""
    gains = _year_equivalent_gains(baseline, treated, scales)
    if not gains:
        return dict(reward=0.0, error="panel produced no comparable readouts",
                    gains={}, penalties={}, baseline=baseline, treated=treated)

    holdout = (set(panels.GENERALISATION_HOLDOUT)
               if (use_holdout and iteration % 2 == 1) else set())
    holdout &= set(gains)
    scored = {k: v for k, v in gains.items() if k not in holdout}
    held = {k: v for k, v in gains.items() if k in holdout}
    if not scored:                       # never hold out the whole panel
        scored, held = gains, {}

    weights = {k: panels.EVAL_PANEL[k]["weight"] for k in scored}
    total_w = sum(weights.values()) or 1.0
    benefit = sum(scored[k] * weights[k] for k in scored) / total_w

    held_mean = (sum(held.values()) / len(held)) if held else None
    plain_mean = sum(scored.values()) / len(scored)
    generalisation_gap = (plain_mean - held_mean) if held_mean is not None else 0.0

    # --- penalties, all in the same year-equivalent currency ---------------
    values = np.array(list(gains.values()), dtype=float)
    positive_fraction = float((values > 0).mean())
    # A real intervention moves the clock families together. Wide disagreement
    # is the signature of an agent that found one clock to push on.
    incoherence = float(np.std(values)) * (1.0 - positive_fraction)

    budget = applied.get("l1_budget", 1.0) or 1.0
    off_ratio = applied.get("off_target_l1", 0.0) / budget
    l1_ratio = applied.get("total_l1", 0.0) / budget
    burden = applied.get("burden", 0.0)
    polypharmacy = max(0, applied.get("axis_count", 0) - 4) * 0.18
    implausibility = max(0.0, l1_ratio - 1.0) * 3.0

    # Burden is charged gently: combining two levers that each work should beat
    # either alone, otherwise the loop can never discover combination protocols.
    penalty = (0.55 * incoherence + 1.20 * off_ratio + 0.22 * burden
               + polypharmacy + implausibility)
    reward = benefit - penalty

    age_models = [k for k, s in panels.EVAL_PANEL.items() if s["kind"] == "age"]
    years = [baseline[k] - treated[k] for k in age_models
             if baseline.get(k) is not None and treated.get(k) is not None]

    return dict(
        reward=float(reward),
        benefit=float(benefit),
        penalty=float(penalty),
        years_reversed=float(np.mean(years)) if years else 0.0,
        pace_before=baseline.get("DunedinPACE"),
        pace_after=treated.get("DunedinPACE"),
        gains={k: float(v) for k, v in gains.items()},
        held_out=sorted(holdout),
        generalisation_gap=float(generalisation_gap),
        coherence=positive_fraction,
        penalties=dict(
            incoherence=float(0.55 * incoherence),
            off_target=float(1.20 * off_ratio),
            burden=float(0.22 * burden),
            polypharmacy=float(polypharmacy),
            implausibility=float(implausibility),
        ),
        baseline=baseline,
        treated=treated,
        chronological_age=float(age),
    )


def display_score(reward):
    """Map an unbounded reward onto a 0-100 scale for the interface."""
    return round(100.0 / (1.0 + math.exp(-0.90 * (reward - 0.40))), 1)
