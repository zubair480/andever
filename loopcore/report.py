"""Turns a finished loop into a longevity report.

The report is deliberately conservative: it separates what the biolearn panel
actually measured from what the response model simulated, grades every
recommendation by evidence, and states the assay noise floor so a reader can
tell a real signal from a rounding error.
"""

from . import compat  # noqa: F401

import numpy as np

from . import bioevals, evidence, mortality, panels

# Test-retest error of a 450k/EPIC methylation array, in years of clock output.
# Anything smaller than this is not a measurable effect in a real subject.
ASSAY_NOISE_YEARS = 1.6


def clinical_phenoage(profile):
    """Levine phenotypic age from blood chemistry, straight out of biolearn.

    The published coefficients are fitted on NHANES SI units. The interface
    collects the units a lab report actually prints, so the conversions happen
    here rather than being silently assumed.

    Returns ``None`` when the profile does not carry a full blood panel.
    """
    import pandas as pd
    from biolearn.hematology import phenotypic_age

    needed = ["albumin", "creatinine", "glucose", "c_reactive_protein",
              "lymphocyte_percent", "mean_cell_volume",
              "red_blood_cell_distribution_width", "alkaline_phosphate",
              "white_blood_cell_count"]
    if not all(profile.get(f) not in (None, "") for f in needed):
        return None

    try:
        row = {
            "age": float(profile["age"]),
            "albumin": float(profile["albumin"]),                        # g/L
            "creatinine": float(profile["creatinine"]) * 88.4,           # mg/dL -> umol/L
            "glucose": float(profile["glucose"]) / 18.0182,              # mg/dL -> mmol/L
            "c_reactive_protein": max(float(profile["c_reactive_protein"]) / 10.0,
                                      0.01),                             # mg/L -> mg/dL
            "lymphocyte_percent": float(profile["lymphocyte_percent"]),
            "mean_cell_volume": float(profile["mean_cell_volume"]),      # fL
            "red_blood_cell_distribution_width":
                float(profile["red_blood_cell_distribution_width"]),     # %
            "alkaline_phosphate": float(profile["alkaline_phosphate"]),  # U/L
            "white_blood_cell_count": float(profile["white_blood_cell_count"]),
        }
        value = float(phenotypic_age(pd.DataFrame([row])).iloc[0])
    except Exception:
        return None
    if not np.isfinite(value) or not 10.0 <= value <= 130.0:
        return None
    return dict(value=round(value, 2),
                acceleration=round(value - float(profile["age"]), 2),
                source="biolearn.hematology.phenotypic_age (Levine 2018)")


def build(run_id, profile, baseline, surrogates, loads, history, best, scales,
          reference_info, clinical, expected, lifespan_now):
    age = float(profile["age"])
    sex = int(profile["sex"])
    rewards = [h["reward"] for h in history]
    running_best = np.maximum.accumulate(rewards).tolist() if rewards else []

    learning = dict(
        first=round(rewards[0], 4) if rewards else None,
        best=round(max(rewards), 4) if rewards else None,
        improvement=round(max(rewards) - rewards[0], 4) if rewards else None,
        curve=[round(v, 4) for v in running_best],
        per_iteration=[round(v, 4) for v in rewards],
        improved_over_first=bool(rewards and max(rewards) > rewards[0] + 1e-9),
    )

    if best is None:
        return dict(run_id=run_id, error="no hypothesis completed", learning=learning)

    scored = best["scored"]
    lifespan = mortality.compare(baseline, scored["treated"], age, sex, expected)
    protocol = best["hypothesis"].get("protocol", [])
    quality = best["hypothesis"].get("protocol_quality", {})

    movers = sorted(
        ((name, gain) for name, gain in scored["gains"].items()),
        key=lambda kv: -abs(kv[1]))[:6]

    panel_rows = []
    for name, spec in panels.EVAL_PANEL.items():
        before, after = baseline.get(name), scored["treated"].get(name)
        if before is None or after is None:
            continue
        panel_rows.append(dict(
            model=name, label=spec["label"], family=spec["family"],
            unit=spec["unit"], before=round(before, 3), after=round(after, 3),
            delta=round(after - before, 3),
            years_equivalent=round(scored["gains"].get(name, 0.0), 2),
            held_out=name in scored.get("held_out", []),
        ))

    surrogate_rows = []
    after_subs = best.get("surrogates", {})
    for key, label in bioevals.GRIMAGE_SUBREADOUTS.items():
        if key not in surrogates:
            continue
        before = surrogates[key]
        after = after_subs.get(key, before)
        surrogate_rows.append(dict(key=key, label=label,
                                   before=round(before, 3), after=round(after, 3),
                                   percent=(round(100 * (after - before) / before, 1)
                                            if before else 0.0)))

    years = scored["years_reversed"]
    confidence = _confidence(years, scored, learning)

    actions = []
    for item in protocol:
        entry = evidence.BY_NAME.get(item["name"], {})
        actions.append(dict(
            name=item["name"], category=item["category"], grade=item["grade"],
            risk=item["risk"], detail=item["detail"], note=item["note"],
            stance=item.get("stance", "start"),
            axes=[panels.AXES[a]["label"] for a in item.get("axes", [])
                  if a in panels.AXES],
            prescription_only=("prescription only" in entry.get("detail", "").lower()),
        ))

    monitoring = _monitoring(best["hypothesis"], loads)

    return dict(
        run_id=run_id,
        chronological_age=age,
        # The two questions the report opens with.
        lifespan=dict(
            current=lifespan["current"],
            treated=lifespan["treated"],
            years_gained=lifespan["years_gained"],
            hazard_reduction=lifespan["hazard_reduction"],
            method="Gompertz-Makeham baseline with a proportional-hazards "
                   "multiplier from GrimAge acceleration, DunedinPACE and "
                   "PhenoAge acceleration",
        ),
        clinical_phenoage=clinical,
        best=dict(
            iteration=best["iteration"], title=best["hypothesis"]["title"],
            mechanism=best["hypothesis"]["mechanism_class"],
            rationale=best["hypothesis"]["rationale"],
            targets=best["hypothesis"]["targets"],
            primary_endpoint=best["hypothesis"].get("primary_endpoint"),
            falsifier=best["hypothesis"].get("falsifier"),
            score=best["score"], reward=round(best["reward"], 4),
        ),
        headline=dict(
            years_reversed=round(years, 2),
            # The first-generation clocks barely respond to intervention, so the
            # panel mean understates what actually moved. Surface the two
            # intervention-responsive clocks alongside it.
            grimage_years=_clock_delta(baseline, scored["treated"], "GrimAgeV2"),
            phenoage_years=_clock_delta(baseline, scored["treated"], "PhenoAge"),
            above_noise=bool(abs(years) >= ASSAY_NOISE_YEARS),
            noise_floor=ASSAY_NOISE_YEARS,
            pace_before=(round(scored["pace_before"], 3)
                         if scored.get("pace_before") else None),
            pace_after=(round(scored["pace_after"], 3)
                        if scored.get("pace_after") else None),
            benefit=round(scored["benefit"], 3),
            penalty=round(scored["penalty"], 3),
            generalisation_gap=round(scored["generalisation_gap"], 3),
            coherence=round(scored["coherence"], 3),
            confidence=confidence,
        ),
        panel=panel_rows,
        surrogates=surrogate_rows,
        movers=[dict(model=m, label=panels.EVAL_PANEL[m]["label"],
                     years_equivalent=round(g, 2)) for m, g in movers],
        protocol=actions,
        protocol_quality=quality,
        monitoring=monitoring,
        burden=loads,
        learning=learning,
        reference=reference_info,
        caveats=_caveats(quality, reference_info, years, lifespan),
    )


def _clock_delta(baseline, treated, name):
    """Years the named clock moved, positive meaning younger."""
    before, after = baseline.get(name), treated.get(name)
    if before is None or after is None:
        return None
    return round(before - after, 2)


def _confidence(years, scored, learning):
    if abs(years) < ASSAY_NOISE_YEARS * 0.5:
        return "below the assay noise floor"
    if abs(scored["generalisation_gap"]) > 0.8:
        return "held-out clocks disagree; treat as clock-specific"
    if scored["coherence"] < 0.6:
        return "clock families disagree on direction"
    if not learning["improved_over_first"]:
        return "the loop did not improve on its first attempt"
    if abs(years) >= ASSAY_NOISE_YEARS:
        return "coherent across families and above the noise floor"
    return "directionally coherent but small"


def _monitoring(hypothesis, loads):
    """Which biolearn readouts to re-measure, and when."""
    axes = {t["axis"] for t in hypothesis.get("targets", [])}
    plan = [dict(what="Full biolearn panel on a fresh 450k/EPIC sample",
                 when="baseline, 6 months, 12 months",
                 why="DunedinPACE and GrimAge V2 are the two panel members that "
                     "move fastest under lifestyle change")]
    if axes & {"inflammatory_load", "immune_composition"}:
        plan.append(dict(what="hs-CRP and full blood count with differential",
                         when="quarterly",
                         why="the DNAm CRP surrogate should track the measured one; "
                             "if it does not, the axis assignment is wrong"))
    if axes & {"metabolic_glycemia", "adiposity"}:
        plan.append(dict(what="HbA1c, fasting insulin, DEXA or bioimpedance",
                         when="baseline and 6 months",
                         why="the harness scored this on a DNAm proxy for glycaemia, "
                             "which needs a direct measurement to confirm"))
    if axes & {"lipid_transport", "cardiometabolic_vascular"}:
        plan.append(dict(what="ApoB, Lp(a) once, and ambulatory blood pressure",
                         when="baseline and 3 months after any change",
                         why="ApoB is the causal lipid measure; the methylation "
                             "signature is downstream of it"))
    if axes & {"xenobiotic_smoking"}:
        plan.append(dict(what="Exhaled CO or urinary cotinine",
                         when="monthly during cessation",
                         why="AHRR methylation lags actual cessation by months"))
    if axes & {"polycomb_hypermethylation", "pmd_hypomethylation",
               "telomere_maintenance"}:
        plan.append(dict(what="No validated human protocol exists for this axis",
                         when="n/a",
                         why="treat the score as a research direction, not a plan"))
    return plan


def _caveats(quality, reference_info, years, lifespan=None):
    out = [
        "The two lifespan figures are cohort projections, not predictions about "
        "one person. They say what median age at death a large group with this "
        "biomarker profile would reach. Epigenetic clocks are validated for "
        "ranking mortality risk across groups; two people with identical panels "
        "routinely die twenty years apart, which is why the quartile band is as "
        "wide as it is.",
        "The treated figure assumes the modelled biomarker change is actually "
        "achieved and then held for the rest of life. A one-year improvement "
        "that lapses buys far less than the number shown.",
    ]
    if lifespan and lifespan["current"].get("hazard_capped"):
        out.append(
            "The hazard multiplier hit its cap. The panel readout is extreme "
            "enough that the projection is pinned at the edge of what these "
            "biomarkers can support, so treat the figure as a floor or ceiling "
            "rather than an estimate.")
    out += [
        "The clocks, CpG sets and cohort age model are real biolearn artefacts. "
        "How an intervention moves methylation is simulated, because no public "
        "dataset maps arbitrary longevity protocols onto per-CpG deltas. Treat "
        "the scores as hypothesis ranking, not as predicted clinical effect.",
        f"The reference epigenome was fitted on {reference_info.get('cohort')} "
        f"(n={reference_info.get('samples')}, ages "
        f"{reference_info.get('age_min')}-{reference_info.get('age_max')}). "
        f"Subjects outside that age range are extrapolated.",
        "This is not medical advice and nothing here is a diagnosis. Anything "
        "prescription-only requires a clinician.",
    ]
    if quality.get("experimental"):
        out.append(
            f"{quality['experimental']} item(s) in the protocol are experimental "
            f"with animal-only or preclinical evidence. They are listed because "
            f"the harness rewarded the mechanism, not because they are ready to use.")
    if abs(years) < ASSAY_NOISE_YEARS:
        out.append(
            f"The headline effect is smaller than the {ASSAY_NOISE_YEARS} year "
            f"test-retest error of a methylation array, so it would not be "
            f"distinguishable from assay noise in a single subject.")
    return out
