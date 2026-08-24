"""Curated longevity intervention catalogue.

Each entry maps a real, named intervention onto the mechanism axes it acts on,
with an evidence grade and a practical risk weight. The agent uses this to turn
an abstract axis vector into a protocol a person could actually follow, and the
report uses it to attach caveats.

Evidence grades:
    A  randomised human trials with hard or validated surrogate endpoints
    B  human trials with biomarker endpoints, or large consistent cohorts
    C  early human data, small or short trials, strong mechanism
    D  animal or in-vitro only

Nothing here is medical advice. Several entries are investigational and some
are prescription-only; the report surfaces that.
"""

INTERVENTIONS = [
    dict(name="Structured aerobic exercise", category="lifestyle", grade="A", risk=0.05,
         detail="150-300 min/week moderate plus 2 sessions of vigorous intervals",
         axes={"inflammatory_load": 0.55, "metabolic_glycemia": 0.50,
               "immune_composition": 0.45, "cardiometabolic_vascular": 0.60,
               "adiposity": 0.45, "telomere_maintenance": 0.30},
         note="The most consistently replicated modifier of DunedinPACE and GrimAge."),
    dict(name="Resistance training", category="lifestyle", grade="A", risk=0.06,
         detail="2-3 full-body sessions per week, progressive load",
         axes={"metabolic_glycemia": 0.45, "adiposity": 0.40,
               "immune_composition": 0.25, "inflammatory_load": 0.25},
         note="Adds insulin sensitivity and lean mass that aerobic work alone misses."),
    dict(name="Smoking cessation", category="lifestyle", grade="A", risk=0.02,
         requires="xenobiotic_smoking",
         # A former smoker still carries burden on this axis, but cessation is
         # not the lever for them; residual AHRR methylation resolves with time.
         requires_profile=("smoking_status", ("current",)),
         detail="Full cessation, with pharmacological support if needed",
         axes={"xenobiotic_smoking": 1.00, "inflammatory_load": 0.35,
               "cardiometabolic_vascular": 0.45, "mitotic_burden": 0.25},
         note="AHRR methylation recovers measurably within 1-5 years of quitting."),
    dict(name="Sleep extension and regularity", category="lifestyle", grade="B", risk=0.02,
         requires="inflammatory_load",
         detail="7-8.5 h with a fixed wake time and morning light exposure",
         axes={"inflammatory_load": 0.45, "telomere_maintenance": 0.35,
               "stochastic_drift": 0.30, "metabolic_glycemia": 0.25},
         note="Short sleep raises CRP and IL-6 and accelerates GrimAge in cohorts."),
    dict(name="Mediterranean-pattern diet", category="lifestyle", grade="A", risk=0.03,
         detail="High legume, fish, olive oil and fibre; low refined carbohydrate",
         axes={"lipid_transport": 0.55, "inflammatory_load": 0.40,
               "cardiometabolic_vascular": 0.45, "metabolic_glycemia": 0.35},
         note="The only dietary pattern with hard-endpoint randomised evidence."),
    dict(name="Time-restricted eating", category="lifestyle", grade="B", risk=0.08,
         requires="metabolic_glycemia",
         detail="8-10 h daily eating window, earlier in the day",
         axes={"metabolic_glycemia": 0.45, "adiposity": 0.40,
               "mitotic_burden": 0.20},
         note="Improves glycaemia largely through the calorie deficit it creates."),
    dict(name="Sustained caloric restriction (CALERIE-style)", category="lifestyle",
         grade="B", risk=0.20, requires="adiposity",
         detail="~12% energy restriction with protein and micronutrient adequacy",
         axes={"metabolic_glycemia": 0.55, "adiposity": 0.60,
               "mitotic_burden": 0.40, "inflammatory_load": 0.30},
         note="CALERIE-2 slowed DunedinPACE by ~2-3% over two years."),
    dict(name="Stress reduction / mindfulness practice", category="lifestyle",
         grade="C", risk=0.02, detail="Daily 15-20 min practice plus social connection",
         axes={"inflammatory_load": 0.30, "telomere_maintenance": 0.30,
               "stochastic_drift": 0.20},
         note="Effect sizes are small and trials are mostly unblinded."),
    dict(name="Air quality control", category="lifestyle", grade="B", risk=0.03,
         detail="HEPA filtration at home, avoid high-PM2.5 exposure windows",
         axes={"xenobiotic_smoking": 0.35, "inflammatory_load": 0.25,
               "cardiometabolic_vascular": 0.25},
         note="PM2.5 exposure tracks epigenetic age acceleration in cohorts."),

    dict(name="Metformin", category="pharmacological", grade="B", risk=0.35,
         requires="metabolic_glycemia",
         detail="500-1500 mg/day, prescription only",
         axes={"metabolic_glycemia": 0.60, "mitotic_burden": 0.35,
               "inflammatory_load": 0.25},
         note="TAME trial still unreported; may blunt exercise adaptations."),
    dict(name="GLP-1 receptor agonist", category="pharmacological", grade="A", risk=0.45,
         requires="adiposity",
         detail="Semaglutide or tirzepatide, prescription only, for BMI >= 30 or >= 27 with comorbidity",
         axes={"adiposity": 0.75, "metabolic_glycemia": 0.70,
               "inflammatory_load": 0.35, "cardiometabolic_vascular": 0.45},
         note="Strong hard-endpoint data; requires lean-mass protection and monitoring."),
    dict(name="Statin or equivalent lipid lowering", category="pharmacological",
         grade="A", risk=0.25, requires="lipid_transport", detail="Dose to an ApoB target, prescription only",
         axes={"lipid_transport": 0.75, "cardiometabolic_vascular": 0.55},
         note="Largest absolute mortality effect of anything in this table for high-ApoB people."),
    dict(name="Blood pressure control", category="pharmacological", grade="A", risk=0.20,
         requires="cardiometabolic_vascular",
         detail="Target systolic < 120 mmHg where tolerated",
         axes={"cardiometabolic_vascular": 0.70, "inflammatory_load": 0.20},
         note="SPRINT-grade evidence for cardiovascular and all-cause mortality."),
    dict(name="Low-dose rapamycin", category="pharmacological", grade="C", risk=0.65,
         detail="Intermittent weekly dosing, investigational, prescription only",
         axes={"mitotic_burden": 0.65, "inflammatory_load": 0.35,
               "immune_composition": 0.40, "metabolic_glycemia": -0.15},
         note="Best-evidenced geroprotector in mammals; human longevity data absent, "
              "and it can worsen glucose tolerance."),
    dict(name="Senolytic dasatinib + quercetin", category="pharmacological",
         grade="C", risk=0.70, detail="Intermittent hit-and-run dosing, investigational",
         axes={"inflammatory_load": 0.55, "immune_composition": 0.45,
               "mitotic_burden": 0.30},
         note="Small human trials in IPF and diabetic kidney disease only."),
    dict(name="Omega-3 (EPA/DHA)", category="supplement", grade="B", risk=0.10,
         detail="1-2 g/day combined EPA and DHA",
         axes={"inflammatory_load": 0.40, "lipid_transport": 0.30,
               "cardiometabolic_vascular": 0.25},
         note="DO-HEALTH reported a small slowing of several epigenetic clocks."),
    dict(name="Vitamin D repletion", category="supplement", grade="B", risk=0.08,
         detail="Dose to a 25-OH-D of 30-50 ng/mL",
         axes={"immune_composition": 0.30, "inflammatory_load": 0.25},
         note="Benefit is largely confined to people who start deficient."),
    dict(name="One-carbon / methyl donor support", category="supplement", grade="C",
         risk=0.20, detail="Folate, B12, betaine and choline adequacy",
         axes={"pmd_hypomethylation": 0.45, "stochastic_drift": 0.20},
         note="Corrects a deficit; supraphysiological dosing has no support."),
    dict(name="Spermidine", category="supplement", grade="C", risk=0.20,
         detail="1-6 mg/day, or wheat germ and fermented foods",
         axes={"stochastic_drift": 0.30, "mitotic_burden": 0.20,
               "inflammatory_load": 0.20},
         note="Autophagy induction; human data limited to small cognition trials."),
    dict(name="Taurine", category="supplement", grade="D", risk=0.15,
         detail="1-3 g/day",
         axes={"inflammatory_load": 0.25, "stochastic_drift": 0.20,
               "metabolic_glycemia": 0.15},
         note="Striking mouse lifespan data; no human longevity trial yet."),
    dict(name="NAD+ precursor (NR / NMN)", category="supplement", grade="C", risk=0.25,
         detail="250-1000 mg/day",
         axes={"stochastic_drift": 0.25, "metabolic_glycemia": 0.20,
               "mitotic_burden": 0.15},
         note="Raises NAD+ reliably; downstream clinical benefit remains unproven."),
    dict(name="Sulforaphane / NRF2 activation", category="supplement", grade="C",
         risk=0.15, detail="Broccoli sprout extract, standardised",
         axes={"xenobiotic_smoking": 0.35, "stochastic_drift": 0.25,
               "inflammatory_load": 0.20},
         note="Accelerates airborne-toxicant clearance in controlled exposure studies."),

    dict(name="Partial epigenetic reprogramming (OSK)", category="experimental",
         grade="D", risk=0.95,
         detail="Cyclic transient OSK expression; no approved human protocol exists",
         axes={"polycomb_hypermethylation": 0.85, "pmd_hypomethylation": 0.55,
               "stochastic_drift": 0.35},
         note="The only lever that meaningfully moves first-generation clocks, and "
              "the only one with a real teratoma risk. Preclinical only."),
    dict(name="Targeted epigenetic editing (dCas9-TET/DNMT)", category="experimental",
         grade="D", risk=0.90,
         detail="Locus-specific methylation editing; preclinical",
         axes={"polycomb_hypermethylation": 0.70, "pmd_hypomethylation": 0.45},
         note="Higher precision than OSK, far less in-vivo delivery evidence."),
    dict(name="Thymic regeneration (rTh-GH/DHEA protocol)", category="experimental",
         grade="C", risk=0.70, detail="TRIIM-style protocol, investigational",
         axes={"immune_composition": 0.65, "inflammatory_load": 0.25},
         note="TRIIM/TRIIM-X are tiny, unblinded and uncontrolled."),
    dict(name="Telomerase activation", category="experimental", grade="D", risk=0.85,
         detail="TA-65 and gene-therapy approaches; preclinical or unproven",
         axes={"telomere_maintenance": 0.60},
         note="Plausible oncogenic trade-off; avoid outside a trial."),
    dict(name="Therapeutic plasma exchange", category="experimental", grade="C",
         risk=0.60, detail="Periodic albumin-replaced exchange, investigational",
         axes={"inflammatory_load": 0.50, "immune_composition": 0.35,
               "stochastic_drift": 0.25},
         note="Early human biomarker data only; invasive and expensive."),
]

GRADE_SCORE = {"A": 1.0, "B": 0.78, "C": 0.50, "D": 0.25}

BY_NAME = {item["name"]: item for item in INTERVENTIONS}


def axis_catalogue():
    """Which interventions are available per axis, best evidence first."""
    out = {}
    for item in INTERVENTIONS:
        for axis, strength in item["axes"].items():
            if strength <= 0:
                continue
            out.setdefault(axis, []).append((item["name"], strength, item["grade"]))
    for axis in out:
        out[axis].sort(key=lambda t: (-GRADE_SCORE[t[2]], -t[1]))
    return out


# An intervention with a ``requires`` axis is only meaningful when the subject
# actually carries burden there. Telling a never-smoker to quit smoking is the
# failure mode this threshold exists to prevent.
REQUIRED_BURDEN = 0.08
MAINTENANCE_BURDEN = 0.10
RESIDUAL_DEMAND = 0.15


def _is_eligible(item, loads, profile):
    """Whether this intervention makes sense for this particular subject."""
    if "requires" in item and loads.get(item["requires"], 0.0) < REQUIRED_BURDEN:
        return False
    gate = item.get("requires_profile")
    if gate:
        field, allowed = gate
        if str(profile.get(field, "")).lower() not in allowed:
            return False
    return True


def compose_protocol(targets, loads=None, profile=None, max_items=6):
    """Pick a concrete protocol that covers the requested axis intensities.

    Greedy set cover weighted by evidence grade, so the loop always hands back
    something a person could act on rather than an abstract vector. ``loads``
    is the subject's measured burden per axis; it gates interventions that only
    make sense for someone carrying that burden, and it decides whether each
    remaining item is something to start or something to keep doing.
    """
    loads = loads or {}
    profile = profile or {}
    demand = {t["axis"]: float(t.get("intensity", 0.0))
              for t in targets if float(t.get("intensity", 0.0)) > 0}
    eligible = [i for i in INTERVENTIONS if _is_eligible(i, loads, profile)]

    chosen = []
    remaining = dict(demand)
    while remaining and len(chosen) < max_items:
        best, best_value = None, 0.0
        for item in eligible:
            if item["name"] in {c["name"] for c in chosen}:
                continue
            covered = sum(min(remaining.get(a, 0.0), max(s, 0.0))
                          for a, s in item["axes"].items())
            if covered <= 0:
                continue
            value = covered * GRADE_SCORE[item["grade"]] / (1.0 + item["risk"])
            if value > best_value:
                best, best_value = item, value
        if best is None:
            break
        for axis, strength in best["axes"].items():
            if strength > 0 and axis in remaining:
                remaining[axis] = max(0.0, remaining[axis] - strength)
                # Drop an axis once it is mostly covered, so a second item is
                # not pulled in to mop up a sliver of residual demand.
                if remaining[axis] <= RESIDUAL_DEMAND:
                    remaining.pop(axis)
        acts_on = [a for a, s in best["axes"].items() if s > 0 and a in demand]
        stance = ("start" if any(loads.get(a, 0.0) >= MAINTENANCE_BURDEN
                                 for a in acts_on) else "maintain")
        chosen.append(dict(name=best["name"], category=best["category"],
                           grade=best["grade"], risk=best["risk"],
                           detail=best["detail"], note=best["note"],
                           stance=stance, axes=acts_on))
    return chosen


def protocol_quality(protocol):
    """Mean evidence grade and peak risk of a composed protocol."""
    if not protocol:
        return dict(evidence=0.0, risk=0.0, experimental=0)
    evidence = sum(GRADE_SCORE[p["grade"]] for p in protocol) / len(protocol)
    return dict(
        evidence=round(evidence, 3),
        risk=round(max(p["risk"] for p in protocol), 3),
        experimental=sum(1 for p in protocol if p["category"] == "experimental"),
    )
