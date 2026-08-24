"""The in-silico response model.

This is the one part of the pipeline that is a *simulator* rather than measured
data. The clocks, the CpG sets and the cohort age regression are all real; how a
proposed intervention moves methylation is modelled, because no public dataset
maps arbitrary longevity protocols onto per-CpG deltas.

The model has three rules that keep it honest:

1. **Headroom.** Every CpG has a young-adult target from the cohort regression.
   An intervention can only close a fraction of the distance to that target and
   can never overshoot it, so the reward saturates instead of running away.
2. **User-specific headroom.** Lifestyle burden is written into the baseline
   methylome along real trait-signature directions. A never-smoker has no
   headroom on the smoking axis, so proposing smoking cessation earns nothing.
3. **Off-target cost.** Every axis has a selectivity below one. Pulling many
   levers hard sprays methylation change across sites nothing was aiming at,
   and the harness charges for it.
"""

from . import compat  # noqa: F401

import hashlib
import os

import numpy as np
import pandas as pd

from . import panels, reference

# The axis directions are a pure function of the reference epigenome and
# biolearn's static coefficient tables, so like the eval panel they are frozen
# at build time by ``tools/build_axes.py`` and reloaded here with numpy alone.
PRECOMPUTED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "precomputed")
AXES_NPZ = os.path.join(PRECOMPUTED, "axes.npz")

# Fraction of the remaining distance-to-young an axis can close at full
# intensity over one protocol year, and how cleanly it hits only its own sites.
AXIS_RESPONSE = {
    "polycomb_hypermethylation": dict(efficacy=0.34, selectivity=0.55),
    "pmd_hypomethylation":       dict(efficacy=0.28, selectivity=0.60),
    "stochastic_drift":          dict(efficacy=0.20, selectivity=0.45),
    "mitotic_burden":            dict(efficacy=0.30, selectivity=0.75),
    "immune_composition":        dict(efficacy=0.32, selectivity=0.70),
    "inflammatory_load":         dict(efficacy=0.45, selectivity=0.80),
    "metabolic_glycemia":        dict(efficacy=0.42, selectivity=0.82),
    "xenobiotic_smoking":        dict(efficacy=0.55, selectivity=0.90),
    "adiposity":                 dict(efficacy=0.40, selectivity=0.78),
    "lipid_transport":           dict(efficacy=0.38, selectivity=0.80),
    "telomere_maintenance":      dict(efficacy=0.18, selectivity=0.65),
    "cardiometabolic_vascular":  dict(efficacy=0.30, selectivity=0.76),
}

MAX_DELTA_BETA = 0.08     # hard cap on any single CpG change in one year
L1_BUDGET = 0.005         # mean |delta beta| an honest one-year protocol may use
OFF_TARGET_GAIN = 0.012   # how much spray an imperfectly selective axis makes
LIFESTYLE_GAIN = 0.020    # beta shift per unit of lifestyle burden on an axis


class ResponseModel:
    """Resolves mechanism axes onto CpGs and applies interventions."""

    def __init__(self, ref, precomputed=True):
        self.ref = ref
        self.index = ref.index
        self.resid_sd = ref["resid_sd"].reindex(self.index).fillna(0.02)
        self.directions = {}
        self.members = {}
        if precomputed and self._load_axes():
            return
        for name, spec in panels.AXES.items():
            direction = self._build_direction(name, spec)
            direction = direction[direction.abs() > 0]
            if direction.empty:
                continue
            # Normalise on a high quantile rather than the max: one extreme
            # coefficient must not shrink every other site in the axis down to
            # a few percent of its intended reach.
            scale = direction.abs().quantile(0.75) or direction.abs().max()
            self.directions[name] = (direction / scale).clip(-1.0, 1.0)
            self.members[name] = direction.index

    # -- precomputed axes -------------------------------------------------
    def _load_axes(self, path=AXES_NPZ):
        """Reload frozen axis directions. Needs numpy, not biolearn.

        What is stored is the *normalised* vector, the exact array that
        ``__init__`` would otherwise assign to ``self.directions``. Storing the
        raw one and re-deriving the 0.75-quantile scale here would put a
        pandas quantile and a float division between the build and the runtime
        for no gain, and any drift in either would land silently on the scores.
        """
        if os.environ.get("LONGEVITY_LOOP_FAST", "1") == "0":
            return False
        if not os.path.exists(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as handle:
                blob = {k: handle[k] for k in handle.files}
            stored = blob["cpgs"].astype(str)
            names = [str(n) for n in blob["axis_names"]]
        except Exception:
            return False

        # Built against a different reference epigenome: fall back rather than
        # hang the axes off the wrong CpGs.
        if not np.array_equal(stored, np.asarray(self.index.astype(str))):
            return False

        for name in names:
            members = self.index[blob["pos__" + name]]
            self.directions[name] = pd.Series(blob["dir__" + name],
                                              index=members)
            self.members[name] = members
        return True

    def save_axes(self, path=AXES_NPZ):
        """Freeze the resolved axes. Called by ``tools/build_axes.py``."""
        blob = {
            "cpgs": np.asarray(self.index.astype(str), dtype=np.str_),
            "axis_names": np.asarray(list(self.directions), dtype=np.str_),
        }
        for name, direction in self.directions.items():
            blob["pos__" + name] = self.index.get_indexer(
                direction.index).astype(np.int32)
            blob["dir__" + name] = direction.to_numpy(dtype=np.float64)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **blob)
        return path

    # -- axis construction ------------------------------------------------
    def _build_direction(self, name, spec):
        """Unit vector pointing the way methylation moves as things get worse."""
        source = spec["source"]
        parts = []

        if source == "cohort":
            stratum = self.ref["stratum"] == spec["stratum"]
            slope = self.ref.loc[stratum, "b_age"]
            parts.append(slope)

        if source in ("biolearn", "mixed"):
            for filename in spec.get("files", []):
                parts.append(self._direction_from_file(filename))

        if source in ("grimage", "mixed"):
            for component in spec.get("components", []):
                weights = panels.grimage_component_weights(component)
                weights = weights[weights.index.isin(self.index)]
                if weights.empty:
                    continue
                scaled = np.sign(weights) * self.resid_sd.reindex(weights.index)
                parts.append(scaled.groupby(level=0).mean())

        if not parts:
            return pd.Series(dtype=float)

        stacked = pd.concat(parts)
        stacked = stacked.groupby(level=0).mean()
        return stacked.reindex(self.index).fillna(0.0)

    def _direction_from_file(self, filename):
        table = panels.read_coefficients(filename)
        if "CoefficientTraining" in table.columns:
            weights = table["CoefficientTraining"]
            weights = weights[[i for i in weights.index
                               if isinstance(i, str) and i.startswith("cg")]]
            weights = weights[weights.index.isin(self.index)]
            if weights.empty:
                return pd.Series(dtype=float)
            signed = np.sign(weights) * self.resid_sd.reindex(weights.index)
            return signed.groupby(level=0).mean()

        if {"delta", "beta0"} <= set(table.columns):        # EpiTOC2 format
            sites = [i for i in table.index.astype(str) if i.startswith("cg")]
            sites = [s for s in sites if s in self.index]
            return self.resid_sd.reindex(sites)

        # Cell-type deconvolution reference: myeloid-minus-lymphoid skew is the
        # direction blood composition drifts with age.
        myeloid = [c for c in table.columns if c in ("neutrophil", "monocyte",
                                                     "granulocyte", "eosinophil")]
        lymphoid = [c for c in table.columns if c in ("cd4_t_cell", "cd8_t_cell",
                                                      "b_cell", "nk_cell")]
        if myeloid and lymphoid:
            skew = table[myeloid].mean(axis=1) - table[lymphoid].mean(axis=1)
            skew = skew[skew.index.isin(self.index)]
            return skew.groupby(level=0).mean() * 0.5

        return pd.Series(dtype=float)

    # -- baseline personalisation ----------------------------------------
    def personalise(self, profile, progress=lambda m: None):
        """Write the user lifestyle burden into their baseline methylome."""
        age, sex = profile["age"], profile["sex"]
        beta = reference.expected_beta(self.ref, age, sex=sex)
        target = reference.youthful_beta(self.ref, sex=sex)

        loads = lifestyle_loads(profile)
        total = pd.Series(0.0, index=self.index)
        for axis, severity in loads.items():
            direction = self.directions.get(axis)
            if direction is None or severity == 0:
                continue
            total = total.add(direction * severity * LIFESTYLE_GAIN, fill_value=0.0)

        beta = (beta + total.reindex(self.index).fillna(0.0)).clip(0.002, 0.998)
        progress("Baseline methylome personalised from the submitted profile")
        return beta, target, loads

    # -- applying a hypothesis -------------------------------------------
    def apply(self, beta, target, targets, seed_text=None):
        """Apply ``[{axis, intensity}, ...]`` to a methylome.

        Returns ``(treated_beta, applied_stats)``. The off-target spray is
        seeded from the intervention vector itself, so the harness is a
        deterministic function of the hypothesis: two hypotheses that differ
        only in their prose get identical scores, and a rerun reproduces.
        """
        if seed_text is None:
            seed_text = ";".join(
                f"{t.get('axis')}={float(t.get('intensity', 0)):.3f}"
                for t in sorted(targets, key=lambda t: str(t.get("axis"))))
        headroom = (target - beta).reindex(self.index).fillna(0.0)
        delta = pd.Series(0.0, index=self.index)
        touched = pd.Series(False, index=self.index)

        per_axis, burden, spray = {}, 0.0, 0.0
        for item in targets:
            axis = item.get("axis")
            direction = self.directions.get(axis)
            if direction is None:
                continue
            intensity = float(np.clip(item.get("intensity", 0.0), 0.0, 1.0))
            if intensity <= 0:
                continue
            response = AXIS_RESPONSE.get(axis, dict(efficacy=0.25, selectivity=0.6))
            reach = direction.abs()                       # 0..1 per member CpG
            move = headroom.reindex(direction.index).fillna(0.0) * (
                intensity * response["efficacy"] * reach)
            delta = delta.add(move.reindex(self.index).fillna(0.0), fill_value=0.0)
            touched.loc[direction.index] = True

            per_axis[axis] = dict(
                intensity=round(intensity, 3),
                headroom=float(headroom.reindex(direction.index).abs().mean()),
                moved=float(move.abs().mean()),
                sites=int(len(direction)),
            )
            burden += intensity * panels.AXES[axis]["burden"]
            spray += intensity * (1.0 - response["selectivity"])

        # Off-target spray: real interventions are not surgical.
        if spray > 0:
            rng = np.random.default_rng(_seed(seed_text))
            n = len(self.index)
            count = int(min(n, max(64, spray * 0.12 * n)))
            picks = rng.choice(n, size=count, replace=False)
            noise = rng.normal(0.0, OFF_TARGET_GAIN * spray, size=count)
            noise *= self.resid_sd.to_numpy()[picks] / max(self.resid_sd.mean(), 1e-6)
            off = pd.Series(0.0, index=self.index)
            off.iloc[picks] = noise
            delta = delta.add(off, fill_value=0.0)

        delta = delta.clip(-MAX_DELTA_BETA, MAX_DELTA_BETA)
        treated = (beta + delta).clip(0.002, 0.998)
        realised = treated - beta

        on_mask = touched.to_numpy()
        stats = dict(
            per_axis=per_axis,
            axis_count=len(per_axis),
            burden=float(burden),
            total_l1=float(realised.abs().mean()),
            on_target_l1=float(realised[on_mask].abs().mean()) if on_mask.any() else 0.0,
            off_target_l1=float(realised[~on_mask].abs().mean()) if (~on_mask).any() else 0.0,
            l1_budget=L1_BUDGET,
            sites_moved=int((realised.abs() > 1e-4).sum()),
        )
        return treated, stats


# ---------------------------------------------------------------------------
# Profile -> axis burden
# ---------------------------------------------------------------------------

#: The habits the reference methylome is defined against. Someone matching all
#: of these carries zero burden on every axis and reads exactly their age.
NEUTRAL = dict(exercise_minutes_per_week=150.0, sleep_hours=7.25, bmi=23.0,
               c_reactive_protein=1.0, diet_quality=6.0, stress_level=4.0,
               alcohol_units_per_week=4.0, glucose=92.0)

#: Burden below zero means the subject is doing better than the neutral peer.
#: It is floored well above the positive ceiling because the evidence for how
#: far good habits push a methylome down is far weaker than the evidence for
#: how far bad ones push it up.
LOAD_FLOOR = -0.60
LOAD_CEILING = 1.50


def _band(value, neutral, worse_at, better_at):
    """Signed distance from neutral, +1 at ``worse_at`` and -1 at ``better_at``."""
    value = float(value)
    if value >= neutral:
        span = worse_at - neutral
        return (value - neutral) / span if span else 0.0
    span = neutral - better_at
    return -((neutral - value) / span) if span else 0.0


def lifestyle_loads(profile):
    """Signed burden the submitted profile carries on each mechanism axis.

    Positive means damage the reference peer does not have, and it is what an
    intervention gets paid to undo. Negative means the subject is already ahead
    of the reference, which shows up as a younger baseline reading and as less
    headroom left for any protocol to claim.
    """
    get = profile.get

    def val(key):
        raw = get(key, NEUTRAL.get(key))
        return float(NEUTRAL.get(key, 0.0) if raw in (None, "") else raw)

    smoking = str(get("smoking_status", "never")).lower()
    pack_years = min(float(get("pack_years", 0) or 0), 40.0)
    smoke = {"current": 0.85, "former": 0.30, "never": 0.0}.get(smoking, 0.0)

    # Inactivity is positive when sedentary, negative when well above guideline.
    activity = _band(val("exercise_minutes_per_week"), 150.0, 0.0, 330.0)
    inactivity = -activity
    sleep_debt = -_band(val("sleep_hours"), 7.25, 9.5, 5.0)
    adiposity = _band(val("bmi"), 23.0, 32.0, 19.5)
    inflammation = _band(val("c_reactive_protein"), 1.0, 5.0, 0.3)
    poor_diet = -_band(val("diet_quality"), 6.0, 10.0, 1.0)
    strain = _band(val("stress_level"), 4.0, 9.0, 1.0)
    drink = _band(val("alcohol_units_per_week"), 4.0, 24.0, 0.0)
    glycemia = _band(val("glucose"), 92.0, 132.0, 78.0)

    loads = {
        "xenobiotic_smoking": smoke + pack_years / 40.0 * 0.65,
        "adiposity": adiposity,
        "inflammatory_load": (0.35 * inactivity + 0.45 * inflammation
                              + 0.30 * sleep_debt + 0.25 * strain),
        "metabolic_glycemia": glycemia + 0.35 * adiposity,
        "lipid_transport": 0.8 * poor_diet + 0.3 * adiposity,
        "cardiometabolic_vascular": (0.4 * inactivity + 0.6 * drink
                                     + 0.3 * adiposity),
        "immune_composition": 0.35 * inactivity + 0.3 * sleep_debt + 0.25 * smoke,
        "telomere_maintenance": (0.30 * sleep_debt + 0.25 * inactivity
                                 + 0.30 * strain),
        "stochastic_drift": 0.25 * inactivity + 0.2 * sleep_debt,
        "mitotic_burden": 0.30 * adiposity + 0.2 * smoke,
    }
    return {k: round(float(np.clip(v, LOAD_FLOOR, LOAD_CEILING)), 3)
            for k, v in loads.items() if abs(v) > 0.005}


def _seed(text):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)
