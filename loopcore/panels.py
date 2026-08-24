"""Catalogue of the biolearn assets this project uses.

Everything referenced here is a real biolearn artefact: coefficient tables that
ship inside the installed package. Nothing is invented.

The catalogue itself (``EVAL_PANEL``, ``AXES``) is plain Python and has to stay
importable on the deployed host, which ships numpy and pandas only. So biolearn
is imported inside the handful of functions that read its coefficient tables,
and those functions only ever run in the build image behind ``tools/``.
"""

from . import compat  # noqa: F401  (import-order shim, must be first)

import pandas as pd


# ---------------------------------------------------------------------------
# Evaluation panel: the "test paper" that hypotheses are graded against.
# ---------------------------------------------------------------------------
# direction = +1 when a *higher* prediction is worse (we want it to go down),
#             -1 when a higher prediction is better (e.g. telomere length).

EVAL_PANEL = {
    # --- first-generation chronological clocks -----------------------------
    # First-generation clocks were trained on chronological age and are known to
    # be nearly inert to intervention, so they carry the least reward weight.
    "Horvathv1": dict(kind="age", unit="years", direction=+1, weight=0.6,
                      label="Horvath multi-tissue clock", family="chronological"),
    "Hannum": dict(kind="age", unit="years", direction=+1, weight=0.4,
                   label="Hannum blood clock", family="chronological"),
    # --- second-generation phenotypic / mortality clocks -------------------
    "PhenoAge": dict(kind="age", unit="years", direction=+1, weight=1.6,
                     label="Levine PhenoAge", family="phenotypic"),
    "GrimAgeV2": dict(kind="age", unit="years", direction=+1, weight=1.8,
                      label="GrimAge V2", family="mortality"),
    # --- third-generation pace / stochastic --------------------------------
    "DunedinPACE": dict(kind="pace", unit="years/year", direction=+1, weight=1.6,
                        label="DunedinPACE", family="pace"),
    "StocP": dict(kind="age", unit="years", direction=+1, weight=1.0,
                  label="StocP stochastic PhenoAge", family="stochastic"),
    "StocZ": dict(kind="risk", unit="z", direction=+1, weight=1.0,
                  label="StocZ mortality risk", family="stochastic"),
    # --- causal decomposition (Ying 2024) ----------------------------------
    "YingDamAge": dict(kind="age", unit="years", direction=+1, weight=1.2,
                       label="YingDamAge (damage)", family="causal"),
    "YingAdaptAge": dict(kind="age", unit="years", direction=-1, weight=0.6,
                         label="YingAdaptAge (adaptation)", family="causal"),
    # --- mechanistic biomarkers -------------------------------------------
    # biolearn returns DNAmTL and EpiTOC2 on their raw model scales rather than
    # in kb / divisions-per-year, so both are read as relative indices.
    "DNAmTL": dict(kind="telomere", unit="index", direction=-1, weight=1.0,
                   label="DNAm telomere length index", family="mechanistic"),
    "EpiTOC2": dict(kind="mitotic", unit="divisions", direction=+1, weight=0.8,
                    label="EpiTOC2 cumulative stem-cell divisions",
                    family="mechanistic"),
    "Zhang_10": dict(kind="risk", unit="score", direction=+1, weight=0.8,
                     label="Zhang 10-CpG mortality score", family="mortality"),
}

# Held out of the reward on alternating iterations to measure generalisation.
GENERALISATION_HOLDOUT = ["GrimAgeV2", "DunedinPACE", "StocZ"]


# ---------------------------------------------------------------------------
# Mechanism axes: the levers a hypothesis is allowed to pull.
# ---------------------------------------------------------------------------
# Axes with source="cohort" are defined statistically from the reference
# cohort per-CpG age regression; the rest are literal biolearn CpG sets.

AXES = {
    "polycomb_hypermethylation": dict(
        source="cohort", stratum="cgi_hyper",
        label="Polycomb / CpG-island hypermethylation",
        biology="Age-related gain of methylation at low-baseline, polycomb-target "
                "promoters. The dominant signal in Horvath-family clocks.",
        burden=0.55,
        mechanisms=["partial_reprogramming", "epigenetic_editing", "dnmt3a_modulation"],
    ),
    "pmd_hypomethylation": dict(
        source="cohort", stratum="pmd_hypo",
        label="Partially-methylated-domain erosion",
        biology="Progressive loss of methylation across late-replicating solo-WCGW "
                "domains; tracks cumulative mitotic and replicative history.",
        burden=0.45,
        mechanisms=["one_carbon_metabolism", "dnmt1_support", "partial_reprogramming"],
    ),
    "stochastic_drift": dict(
        source="cohort", stratum="stochastic",
        label="Stochastic epigenetic drift",
        biology="High-variance sites with weak age correlation. Accumulated "
                "maintenance error rather than programmed change; what the "
                "StocZ and StocP clocks were built to read.",
        burden=0.40,
        mechanisms=["proteostasis", "antioxidant", "dna_repair", "autophagy"],
    ),
    "mitotic_burden": dict(
        source="biolearn", files=["EpiTOC1.csv", "EpiTOC2.csv"],
        label="Mitotic / stem-cell division burden",
        biology="EpiTOC CpGs act as a mitotic counter in blood stem cells. "
                "A lower division rate means lower replicative and oncogenic load.",
        burden=0.50,
        mechanisms=["mtor_inhibition", "caloric_restriction", "senolytic"],
    ),
    "immune_composition": dict(
        source="biolearn", files=["450K_reinius_12_reference.csv"],
        label="Immune cell composition remodelling",
        biology="Leukocyte-discriminating CpGs. Shifting these changes the "
                "inferred naive-T to myeloid balance that drives immunosenescence.",
        burden=0.50,
        mechanisms=["thymic_regeneration", "senolytic", "exercise", "hematopoietic"],
    ),
    "inflammatory_load": dict(
        source="grimage",
        components=["DNAmlogCRP", "DNAmB2M", "DNAmTIMP1", "DNAmGDF15"],
        label="Inflammaging / SASP load",
        biology="DNAm surrogates for CRP, beta-2-microglobulin, TIMP1 and GDF15, "
                "the inflammatory arm of GrimAge and the strongest single "
                "mortality signal in the panel.",
        burden=0.35,
        mechanisms=["inflammation_resolution", "senolytic", "exercise", "omega3"],
    ),
    "metabolic_glycemia": dict(
        source="grimage", components=["DNAmlogA1C", "DNAmLeptin", "DNAmADM"],
        label="Glycaemic and adipokine load",
        biology="DNAm surrogates for HbA1c, leptin and adrenomedullin. Reads "
                "insulin resistance and adipose signalling.",
        burden=0.30,
        mechanisms=["caloric_restriction", "glp1", "metformin", "exercise",
                    "time_restricted_eating"],
    ),
    "xenobiotic_smoking": dict(
        source="mixed", files=["Smoking.csv"], components=["DNAmPACKYRS"],
        label="Xenobiotic / smoke damage",
        biology="AHRR and related loci. The single largest modifiable term in "
                "GrimAge for anyone with tobacco exposure.",
        burden=0.20,
        mechanisms=["smoking_cessation", "nrf2_activation", "air_quality"],
    ),
    "adiposity": dict(
        source="biolearn",
        files=["BMI_McCartney.csv", "BodyFatMcCartney.csv", "BMI_Reed.csv"],
        label="Adiposity signature",
        biology="Blood methylation signature of BMI and body-fat percentage; "
                "overlaps heavily with the DunedinPACE site set.",
        burden=0.35,
        mechanisms=["caloric_restriction", "glp1", "exercise", "time_restricted_eating"],
    ),
    "lipid_transport": dict(
        source="biolearn",
        files=["TotalCholesterolMcCartney.csv", "HDLCholesterolMcCartney.csv",
               "LDLCholesterolMcCartney.csv"],
        label="Lipid transport signature",
        biology="Methylation signature of total, HDL and remnant-LDL cholesterol.",
        burden=0.30,
        mechanisms=["lipid_lowering", "diet_quality", "exercise"],
    ),
    "telomere_maintenance": dict(
        source="biolearn", files=["DNAmTL.csv"],
        label="Telomere maintenance",
        biology="DNAmTL site set. Longer estimated telomeres track lower "
                "all-cause mortality independently of the age clocks.",
        burden=0.60,
        mechanisms=["telomerase_activation", "exercise", "stress_reduction"],
    ),
    "cardiometabolic_vascular": dict(
        source="biolearn", files=["CVD_Westermann.csv"],
        label="Cardiovascular risk signature",
        biology="Westerman coronary-heart-disease methylation signature.",
        burden=0.35,
        mechanisms=["blood_pressure_control", "exercise", "lipid_lowering",
                    "diet_quality"],
    ),
}


# ---------------------------------------------------------------------------
# CpG universe
# ---------------------------------------------------------------------------

_ALL_COEF_FILES = [
    "Horvath1.csv", "Horvath2.csv", "Hannum.csv", "PhenoAge.csv", "Lin.csv",
    "DunedinPACE.csv", "DunedinPoAm38.csv", "GrimAgeV2.csv", "GrimAgeV1.csv",
    "StocP.csv", "StocZ.csv", "StocH.csv", "YingCausAge.csv", "YingDamAge.csv",
    "YingAdaptAge.csv", "Zhang_10.csv", "HRSInCHPhenoAge.csv", "DNAmTL.csv",
    "EpiTOC1.csv", "EpiTOC2.csv", "VidalBralo.csv", "Weidner.csv",
    "450K_reinius_12_reference.csv", "Smoking.csv", "Alcohol.csv",
    "BMI_McCartney.csv", "BMI_Reed.csv", "BodyFatMcCartney.csv",
    "TotalCholesterolMcCartney.csv", "HDLCholesterolMcCartney.csv",
    "LDLCholesterolMcCartney.csv", "CVD_Westermann.csv", "EducationMcCartney.csv",
]


def read_coefficients(filename):
    """Load a biolearn coefficient table by filename. Build time only."""
    from biolearn.util import get_data_file

    return pd.read_csv(get_data_file(filename), index_col=0)


def cpgs_in(filename):
    """CpG identifiers referenced by a biolearn coefficient table."""
    df = read_coefficients(filename)
    if "var" in df.columns:  # GrimAge-style long format
        ids = df["var"].astype(str)
    else:
        ids = pd.Series(df.index.astype(str))
    return {c for c in ids if c.startswith("cg")}


def grimage_component_weights(component):
    """Signed coefficients of one GrimAge sub-model, indexed by CpG."""
    df = read_coefficients("GrimAgeV2.csv")
    rows = df[df.index.astype(str) == component]
    rows = rows[rows["var"].astype(str).str.startswith("cg")]
    return pd.Series(rows["beta"].to_numpy(),
                     index=rows["var"].astype(str).to_numpy())


def grimage_component_cpgs(component):
    return set(grimage_component_weights(component).index)


def universe():
    """Union of every CpG the panel or any axis can touch."""
    out = set()
    for name in _ALL_COEF_FILES:
        try:
            out |= cpgs_in(name)
        except Exception:
            continue
    return sorted(out)
