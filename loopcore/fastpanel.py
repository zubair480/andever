"""The eval panel distilled to arithmetic on one fixed-length numpy vector.

``loopcore.bioevals`` runs the twelve panel models through biolearn, which drags
pandas, torch, scipy, sklearn and cvxpy into the deployed image for what is, on
our code path, a fixed amount of linear algebra. Every methylome the loop ever
evaluates is indexed by ``reference.build()[0].index``, a frozen ordered set of
CpGs, so each coefficient lookup, each imputed constant and each rank table can
be resolved once and stored.

That is what this module does. ``FastPanel.build`` is the only thing here that
touches biolearn and it belongs in the build image; ``tools/build_fastpanel.py``
drives it. ``FastPanel.load`` and ``FastPanel.evaluate`` need numpy and nothing
else, and they return the same numbers to the last few bits.

How each panel member reduces:

* the nine plain ``LinearMethylationModel`` clocks are ``transform(c . beta + k)``
  where ``k`` folds in the coefficient table's own intercept row plus the fixed
  gold-standard values biolearn imputes for CpGs outside our index;
* ``DunedinPACE`` quantile-normalises the sample against its gold-standard means
  first, which on a single sample is a pure rank lookup, then does the dot;
* ``EpiTOC2`` has its own closed form over ``delta`` and ``beta0``;
* ``GrimAgeV2`` is ten linear sub-models feeding a COX combination and a final
  linear rescale. ``AgeAccelGrim`` needs an OLS fit across samples and is
  degenerate on one sample, so it is not reproduced here; we read DNAmGrimAge.
"""

import os

import numpy as np

PRECOMPUTED = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "precomputed")
PANEL_NPZ = os.path.join(PRECOMPUTED, "fastpanel.npz")

# The panel members that are plain linear clocks, in the order bioevals reports
# them. DunedinPACE is a LinearMethylationModel too but its preprocess step puts
# it in its own branch below.
LINEAR_MODELS = ["Horvathv1", "Hannum", "PhenoAge", "StocP", "StocZ",
                 "YingDamAge", "YingAdaptAge", "DNAmTL", "Zhang_10"]

#: biolearn stores its output transforms as lambdas in ``model_definitions``.
#: Pickling those would drag biolearn back into the runtime, so they are
#: restated here in closed form. ``build`` probes the real lambdas and refuses
#: to write the npz if any of these disagrees, which turns an upstream change
#: into a failed build rather than a silently wrong number.
TRANSFORMS = {
    "Horvathv1":    ("anti_trafo", 0.696),
    "Hannum":       ("linear", 0.0),
    "PhenoAge":     ("linear", 0.0),
    "StocP":        ("linear", 92.8310813279039),
    "StocZ":        ("linear", 64.8077188694894),
    "YingDamAge":   ("linear", 0.0),
    "YingAdaptAge": ("linear", 0.0),
    "DNAmTL":       ("linear", 0.0),
    "Zhang_10":     ("linear", 0.0),
    "DunedinPACE":  ("linear", 0.0),
}

# The GrimAge sub-models, in the order pandas' groupby hands them to biolearn.
# The COX combination is a plain sum over columns so the order is cosmetic, but
# keeping it means the two paths add the same numbers in the same sequence.
GRIMAGE_COMPONENTS = ["DNAmADM", "DNAmB2M", "DNAmCystatinC", "DNAmGDF15",
                      "DNAmLeptin", "DNAmPACKYRS", "DNAmPAI1", "DNAmTIMP1",
                      "DNAmlogA1C", "DNAmlogCRP"]


def anti_trafo(x, adult_age=20):
    """biolearn's inverse age transform, for one scalar."""
    if x < 0:
        return (1 + adult_age) * np.exp(x) - 1
    return (1 + adult_age) * x + adult_age


def rankdata(a):
    """Average ranks, matching ``scipy.stats.rankdata`` on ties.

    DunedinPACE maps every value to the gold-standard value at its rank, and a
    tie that lands on a half rank is treated differently from one that does not,
    so the tie handling has to be exact. Vectorised rather than the loop in
    ``loopcore.slim`` because this runs on 20k probes per evaluation.
    """
    arr = np.asarray(a, dtype=float).ravel()
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]

    starts = np.empty(n, dtype=bool)
    starts[0] = True
    np.not_equal(sorted_arr[1:], sorted_arr[:-1], out=starts[1:])

    group = np.cumsum(starts)                       # 1-based tie-group id
    bounds = np.append(np.nonzero(starts)[0], n)    # group start offsets, + n
    # A group spanning sorted positions [i, j] takes rank 0.5 * (i + j) + 1.
    average = 0.5 * (bounds[group - 1] + bounds[group] - 1) + 1.0

    ranks = np.empty(n, dtype=float)
    ranks[order] = average
    return ranks


def _apply_transform(kind, offset, value):
    if kind == "anti_trafo":
        return anti_trafo(value + offset)
    return value + offset


class FastPanel:
    """A frozen, numpy-only stand-in for ``loopcore.bioevals.evaluate``."""

    def __init__(self, blob):
        self._b = blob
        self.cpgs = blob["cpgs"]
        self.model_names = [str(n) for n in blob["model_names"]]

        self.lin_names = [str(n) for n in blob["lin_names"]]
        self.lin_kind = [str(k) for k in blob["lin_kind"]]
        self.lin_idx = blob["lin_idx"]
        self.lin_val = blob["lin_val"]
        self.lin_ptr = blob["lin_ptr"]
        self.lin_const = blob["lin_const"]
        self.lin_offset = blob["lin_offset"]

        self.e2_pos = blob["e2_pos"]
        self.e2_delta = blob["e2_delta"]
        self.e2_beta0 = blob["e2_beta0"]
        self.e2_const = float(blob["e2_const"])
        self.e2_k = float(blob["e2_k"])

        self.dp_base = blob["dp_base"]
        self.dp_target = blob["dp_target"]
        self.dp_src_pos = blob["dp_src_pos"]
        self.dp_src_row = blob["dp_src_row"]
        self.dp_coef_pos = blob["dp_coef_pos"]
        self.dp_coef_val = blob["dp_coef_val"]
        self.dp_intercept = float(blob["dp_intercept"])

        self.grim_names = [str(n) for n in blob["grim_names"]]
        self.grim_idx = blob["grim_idx"]
        self.grim_val = blob["grim_val"]
        self.grim_ptr = blob["grim_ptr"]
        self.grim_const = blob["grim_const"]
        self.grim_age = blob["grim_age"]
        self.grim_female = blob["grim_female"]
        self.grim_cox = blob["grim_cox"]
        self.grim_cox_age = float(blob["grim_cox_age"])
        self.grim_cox_female = float(blob["grim_cox_female"])
        self.grim_m_age = float(blob["grim_m_age"])
        self.grim_sd_age = float(blob["grim_sd_age"])
        self.grim_m_cox = float(blob["grim_m_cox"])
        self.grim_sd_cox = float(blob["grim_sd_cox"])

        # A model that could not be captured at build time is reported as None,
        # which is exactly what bioevals does when biolearn raises on it.
        self._captured = (set(self.lin_names)
                          | {"DunedinPACE", "EpiTOC2", "GrimAgeV2"})

    # -- evaluation -------------------------------------------------------

    def evaluate(self, beta, age, sex, want_subreadouts=False):
        """Run the panel on one methylome aligned to ``self.cpgs``.

        ``beta`` is a plain 1-D array in ``self.cpgs`` order. Returns
        ``{model: value}`` with the same keys as ``bioevals.evaluate``, plus
        ``{surrogate: value}`` when asked.
        """
        beta = np.asarray(beta, dtype=float)
        if beta.shape != self.cpgs.shape:
            raise ValueError(
                f"beta has {beta.shape[0] if beta.ndim else '?'} entries, "
                f"expected {self.cpgs.shape[0]} aligned to FastPanel.cpgs")

        age = float(age)
        # GrimAge encodes sex as a Female indicator; biolearn treats 0 as female.
        female = 1.0 if int(sex) == 0 else 0.0

        values = {}
        for i, name in enumerate(self.lin_names):
            lo, hi = int(self.lin_ptr[i]), int(self.lin_ptr[i + 1])
            total = float(np.dot(self.lin_val[lo:hi], beta[self.lin_idx[lo:hi]]))
            total += float(self.lin_const[i])
            values[name] = float(_apply_transform(
                self.lin_kind[i], float(self.lin_offset[i]), total))

        values["DunedinPACE"] = self._dunedin(beta)
        values["EpiTOC2"] = self._epitoc2(beta)

        subs = self._grimage(beta, age, female)
        values["GrimAgeV2"] = subs.pop("DNAmGrimAge")

        readout = {name: values.get(name) if name in self._captured else None
                   for name in self.model_names}
        return (readout, subs) if want_subreadouts else readout

    def _dunedin(self, beta):
        """Quantile-normalise against the gold means, then dot.

        On a single sample the normalisation collapses to: rank every probe in
        the gold-standard background, then read back the gold value sitting at
        that rank. Ranks whose fractional part exceeds 0.4 sit between two gold
        values and take their midpoint, which is biolearn's tie rule.
        """
        sample = self.dp_base.copy()
        sample[self.dp_src_pos] = beta[self.dp_src_row]

        ranks = rankdata(sample)
        floor = np.floor(ranks).astype(np.int64)
        straddles = (ranks - floor) > 0.4

        normalized = self.dp_target[floor - 1]
        if straddles.any():
            f = floor[straddles]
            normalized[straddles] = 0.5 * (self.dp_target[f - 1]
                                           + self.dp_target[f])

        total = float(np.dot(self.dp_coef_val,
                             normalized[self.dp_coef_pos]))
        return total + self.dp_intercept

    def _epitoc2(self, beta):
        b = beta[self.e2_pos]
        b = np.where(np.isnan(b), 0.0, b)
        contrib = (b - self.e2_beta0) / (self.e2_delta * (1.0 - self.e2_beta0))
        return float(2.0 * ((contrib.sum() + self.e2_const) / self.e2_k))

    def _grimage(self, beta, age, female):
        """Ten DNAm surrogate biomarkers, a COX score, then a linear rescale."""
        subs = {}
        cox = 0.0
        for i, name in enumerate(self.grim_names):
            lo, hi = int(self.grim_ptr[i]), int(self.grim_ptr[i + 1])
            value = float(np.dot(self.grim_val[lo:hi], beta[self.grim_idx[lo:hi]]))
            value += (float(self.grim_const[i])
                      + float(self.grim_age[i]) * age
                      + float(self.grim_female[i]) * female)
            subs[name] = value
            cox += value * float(self.grim_cox[i])
        cox += age * self.grim_cox_age + female * self.grim_cox_female

        y = (cox - self.grim_m_cox) / self.grim_sd_cox
        subs["DNAmGrimAge"] = y * self.grim_sd_age + self.grim_m_age
        return subs

    # -- persistence ------------------------------------------------------

    def save(self, path=PANEL_NPZ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **self._b)

    @classmethod
    def load(cls, path=PANEL_NPZ):
        with np.load(path, allow_pickle=False) as handle:
            blob = {k: handle[k] for k in handle.files}
        return cls(blob)

    # -- build time -------------------------------------------------------

    @classmethod
    def build(cls, ref):
        """Distil the biolearn panel against a fixed CpG index. Needs biolearn."""
        from . import compat  # noqa: F401  (torch must import before biolearn)

        import pandas as pd
        from biolearn.model import model_definitions
        from biolearn.util import get_data_file

        from . import panels

        cpgs = np.asarray(list(ref.index.astype(str)), dtype=np.str_)
        position = {c: i for i, c in enumerate(cpgs)}

        gold_files = {
            "sesame_450k": ("sesame_450k_median.csv", "median"),
            "dunedin": ("DunedinPACE_Gold_Means.csv", "mean"),
        }
        sources = {}

        def imputation_source(model_name):
            """The constant biolearn would fill a missing CpG with, per model."""
            method = model_definitions[model_name]["model"].get(
                "default_imputation", "sesame_450k")
            if method == "none":
                return None
            if method not in gold_files:
                raise ValueError(
                    f"{model_name} uses imputation '{method}', which fastpanel "
                    f"does not know how to freeze")
            if method not in sources:
                filename, column = gold_files[method]
                sources[method] = pd.read_csv(get_data_file(filename),
                                              index_col=0)[column]
            return sources[method]

        _verify_transforms(model_definitions)

        blob = {
            "cpgs": cpgs,
            "model_names": np.asarray(list(panels.EVAL_PANEL), dtype=object
                                      ).astype(str),
        }
        blob.update(_build_linear(model_definitions, get_data_file, pd,
                                  position, imputation_source))
        blob.update(_build_epitoc2(get_data_file, pd, position,
                                   imputation_source("EpiTOC2")))
        blob.update(_build_dunedin(get_data_file, pd, cpgs, position))
        blob.update(_build_grimage(get_data_file, pd, position,
                                   imputation_source("GrimAgeV2")))
        return cls(blob)


# ---------------------------------------------------------------------------
# Build helpers. Everything below runs once, in the build image.
# ---------------------------------------------------------------------------

_PROBES = (-40.0, -3.0, -0.5, 0.0, 0.7, 2.5, 40.0)


def _verify_transforms(model_definitions):
    """Check the closed forms in TRANSFORMS against biolearn's own lambdas."""
    for name, (kind, offset) in TRANSFORMS.items():
        theirs = model_definitions[name]["model"].get("transform")
        for probe in _PROBES:
            mine = _apply_transform(kind, offset, probe)
            expected = probe if theirs is None else float(
                np.asarray(theirs(probe)))
            if not np.isclose(mine, expected, rtol=1e-13, atol=1e-13):
                raise ValueError(
                    f"transform for {name} drifted: fastpanel gives {mine} at "
                    f"{probe}, biolearn gives {expected}")


def _fold_coefficients(series, position, source, model_name):
    """Split a coefficient table into (index, value) pairs and a constant.

    Anything the fixed CpG index cannot supply is either the table's own
    intercept row or a CpG biolearn would impute from a gold-standard table.
    Both are constants for every sample we will ever score, so they collapse
    into one number here.
    """
    idx, val, const = [], [], 0.0
    for name, weight in series.items():
        name = str(name)
        weight = float(weight)
        if name == "intercept":
            const += weight
            continue
        slot = position.get(name)
        if slot is not None:
            idx.append(slot)
            val.append(weight)
            continue
        if source is None:       # imputation "none": biolearn drops the site
            continue
        if name not in source.index:
            raise ValueError(
                f"{model_name} needs {name}, which is neither in the CpG index "
                f"nor in its gold-standard imputation table")
        const += weight * float(source[name])
    return idx, val, const


def _build_linear(model_definitions, get_data_file, pd, position, imputation_source):
    names, kinds, offsets, consts = [], [], [], []
    idx, val, ptr = [], [], [0]

    for name in LINEAR_MODELS:
        spec = model_definitions[name]["model"]
        table = pd.read_csv(get_data_file(spec["file"]), index_col=0)
        column = ("CoefficientTraining" if "CoefficientTraining" in table.columns
                  else "Weight")
        model_idx, model_val, const = _fold_coefficients(
            table[column], position, imputation_source(name), name)

        kind, offset = TRANSFORMS[name]
        names.append(name)
        kinds.append(kind)
        offsets.append(offset)
        consts.append(const)
        idx.extend(model_idx)
        val.extend(model_val)
        ptr.append(len(idx))

    return {
        "lin_names": np.asarray(names, dtype=object).astype(str),
        "lin_kind": np.asarray(kinds, dtype=object).astype(str),
        "lin_offset": np.asarray(offsets, dtype=np.float64),
        "lin_const": np.asarray(consts, dtype=np.float64),
        "lin_idx": np.asarray(idx, dtype=np.int32),
        "lin_val": np.asarray(val, dtype=np.float64),
        "lin_ptr": np.asarray(ptr, dtype=np.int64),
    }


def _build_epitoc2(get_data_file, pd, position, source):
    table = pd.read_csv(get_data_file("EpiTOC2.csv"), index_col=0)
    pos, delta, beta0 = [], [], []
    const, found = 0.0, 0

    for name, row in table.iterrows():
        name = str(name)
        d, b0 = float(row["delta"]), float(row["beta0"])
        slot = position.get(name)
        if slot is not None:
            pos.append(slot)
            delta.append(d)
            beta0.append(b0)
            found += 1
            continue
        if source is None or name not in source.index:
            continue             # biolearn's rep_mask drops it and shrinks k
        const += (float(source[name]) - b0) / (d * (1.0 - b0))
        found += 1

    return {
        "e2_pos": np.asarray(pos, dtype=np.int32),
        "e2_delta": np.asarray(delta, dtype=np.float64),
        "e2_beta0": np.asarray(beta0, dtype=np.float64),
        "e2_const": np.float64(const),
        "e2_k": np.float64(found),
    }


def _build_dunedin(get_data_file, pd, cpgs, position):
    """Freeze the gold-standard background DunedinPACE normalises against.

    biolearn builds the background by taking the gold-standard probe list,
    filling in whatever the sample provides and imputing the rest from the gold
    means, then sorting by probe name. Only the sample's own probes vary, so the
    background vector, the sorted target and the probe-to-row map are constants.
    """
    gold = pd.read_csv(get_data_file("DunedinPACE_Gold_Means.csv"),
                       index_col=0)["mean"]
    order = gold.index.astype(str).sort_values()      # hybrid_impute sorts
    base = gold.reindex(order).to_numpy(float)

    slot_of = {c: i for i, c in enumerate(order)}
    src_pos, src_row = [], []
    for cpg in order:
        row = position.get(str(cpg))
        if row is not None:
            src_pos.append(slot_of[cpg])
            src_row.append(row)

    table = pd.read_csv(get_data_file("DunedinPACE.csv"), index_col=0)
    coef_pos, coef_val, intercept = [], [], 0.0
    for name, weight in table["CoefficientTraining"].items():
        name = str(name)
        if name == "intercept":
            intercept += float(weight)
            continue
        slot = slot_of.get(name)
        if slot is None:         # not in the background: biolearn's join drops it
            continue
        coef_pos.append(slot)
        coef_val.append(float(weight))

    return {
        "dp_base": base,
        "dp_target": np.sort(base),
        "dp_src_pos": np.asarray(src_pos, dtype=np.int32),
        "dp_src_row": np.asarray(src_row, dtype=np.int32),
        "dp_coef_pos": np.asarray(coef_pos, dtype=np.int32),
        "dp_coef_val": np.asarray(coef_val, dtype=np.float64),
        "dp_intercept": np.float64(intercept),
    }


def _build_grimage(get_data_file, pd, position, source):
    table = pd.read_csv(get_data_file("GrimAgeV2.csv"), index_col=0)
    cox = table.loc["COX"].set_index("var")["beta"]
    transform = table.loc["transform"].set_index("var")["beta"]

    names, consts, age_terms, female_terms, cox_terms = [], [], [], [], []
    idx, val, ptr = [], [], [0]

    for name in GRIMAGE_COMPONENTS:
        group = table.loc[[name]].set_index("var")["beta"]
        const = float(group.get("Intercept", 0.0))
        age_terms.append(float(group.get("Age", 0.0)))
        female_terms.append(float(group.get("Female", 0.0)))

        cpg_only = group[[v for v in group.index
                          if v not in ("Intercept", "Age", "Female")]]
        sub_idx, sub_val, folded = _fold_coefficients(
            cpg_only, position, source, f"GrimAgeV2/{name}")

        names.append(name)
        consts.append(const + folded)
        cox_terms.append(float(cox[name]))
        idx.extend(sub_idx)
        val.extend(sub_val)
        ptr.append(len(idx))

    return {
        "grim_names": np.asarray(names, dtype=object).astype(str),
        "grim_idx": np.asarray(idx, dtype=np.int32),
        "grim_val": np.asarray(val, dtype=np.float64),
        "grim_ptr": np.asarray(ptr, dtype=np.int64),
        "grim_const": np.asarray(consts, dtype=np.float64),
        "grim_age": np.asarray(age_terms, dtype=np.float64),
        "grim_female": np.asarray(female_terms, dtype=np.float64),
        "grim_cox": np.asarray(cox_terms, dtype=np.float64),
        "grim_cox_age": np.float64(cox["Age"]),
        "grim_cox_female": np.float64(cox["Female"]),
        "grim_m_age": np.float64(transform["m_age"]),
        "grim_sd_age": np.float64(transform["sd_age"]),
        "grim_m_cox": np.float64(transform["m_cox"]),
        "grim_sd_cox": np.float64(transform["sd_cox"]),
    }
