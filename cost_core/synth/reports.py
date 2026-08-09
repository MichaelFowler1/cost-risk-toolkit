"""
reports.py - The six CSDR/SRDR report shapes, emitted from one truth table.

Each builder here is a *view* of ``program.truth.cells`` plus whatever mess the
:class:`~cost_core.synth.spec.PathologyConfig` calls for. They reconcile to
each other and to the truth by construction when the pathologies are switched
off, which is what makes exact-recovery assertions possible downstream.

One deliberate choice: **no report carries the WBS code.** Real submissions do
carry numbers, but contractors renumber between periods about as often as they
rename, and if a reliable key were present in every file the crosswalk would be
decoration rather than the load-bearing artifact it is in practice. Keying on
drifting names is the problem an analyst actually has, so it is the problem
this generator poses.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cost_core.synth.generator import (DIRECT_LABOR_FRACTION, Submission,
                                       SyntheticProgram, dollar_basis,
                                       plan_submissions, reported_name)
from cost_core.synth.spec import (FUNCTIONAL_CATEGORIES, SRDR_ACTIVITY_MIX,
                                  TRUE_SOFTWARE_CER)

logger = logging.getLogger(__name__)

REPORT_NAMES: tuple[str, ...] = (
    "dd1921",
    "dd1921_1",
    "dd1921_2",
    "flexfile",
    "quantity_report",
    "srdr",
)

#: Categories that report hours as well as dollars.
_HOURS_CATEGORIES = {name for name, has_hours in FUNCTIONAL_CATEGORIES if has_hours}


def _escalate(value: float, basis: str, fiscal_year: int, index: dict[int, float]) -> float:
    """Convert a base-year amount to the reported basis."""
    return value * index[fiscal_year] if basis == "TY" else value


def _dollar_year(basis: str, fiscal_year: int, base_year: int) -> int:
    """The year the reported dollars are stated in."""
    return fiscal_year if basis == "TY" else base_year


# --------------------------------------------------------------------------
# DD 1921 - Cost Data Summary Report
# --------------------------------------------------------------------------
def build_dd1921(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """WBS-level recurring/nonrecurring cost, to date and at completion.

    ``cost_to_date`` is cumulative *in the basis of the row*, which means a
    then-year row's to-date figure is a sum of dollars from several different
    years. That is what real submissions do and it is why the ingest pipeline
    normalises from the incremental period column and treats to-date as a
    reconciliation check rather than a source.
    """
    truth, spec = program.truth, program.spec
    p = spec.pathologies
    index, base_year = truth.inflation_index, truth.base_year

    by_lot = (
        truth.cells.groupby(["lot", "wbs_name", "recurring_flag"])["dollars_by"]
        .sum()
        .reset_index()
    )
    element_totals = (
        truth.cells.groupby(["wbs_name", "recurring_flag"])["dollars_by"]
        .sum()
        .to_dict()
    )

    # Cumulative-through-lot, precomputed from the truth rather than
    # accumulated while walking submissions: a resubmitted period appears in
    # the submission list twice, and an accumulator would count it twice.
    ordered = by_lot.sort_values("lot")
    ordered["cum"] = ordered.groupby(["wbs_name", "recurring_flag"])[
        "dollars_by"
    ].cumsum()
    cumulative = {
        (str(r["wbs_name"]), bool(r["recurring_flag"]), int(r["lot"])): float(r["cum"])
        for _, r in ordered.iterrows()
    }

    rows = []
    n_sub = len(submissions)

    for order, sub in enumerate(submissions):
        basis = dollar_basis(rng, p.then_year_prob)
        # Early at-completion estimates are optimistic and converge late.
        progress = (order + 1) / max(n_sub, 1)
        optimism = 1.0 - p.eac_optimism * (1.0 - progress)

        period = by_lot[by_lot["lot"] == sub.lot]
        for _, row in period.iterrows():
            canonical = str(row["wbs_name"])
            recurring = bool(row["recurring_flag"])
            key = (canonical, recurring)
            incurred_by = float(row["dollars_by"]) * sub.error_factor
            # A wrong submission is wrong in its cumulative column too.
            to_date_by = (
                cumulative.get((canonical, recurring, int(sub.lot)), 0.0)
                * sub.error_factor
            )

            incurred = _escalate(incurred_by, basis, sub.fiscal_year, index)
            to_date = _escalate(to_date_by, basis, sub.fiscal_year, index)
            at_completion = _escalate(
                element_totals.get(key, 0.0) * optimism,
                basis,
                sub.fiscal_year,
                index,
            )

            rows.append(
                {
                    "program": spec.program,
                    "contractor": spec.contractor,
                    "report_type": "DD1921",
                    "report_date": sub.report_date.isoformat(),
                    "period_fy": sub.fiscal_year,
                    "lot": sub.lot,
                    "wbs_element_name": reported_name(
                        canonical, rng, p.name_drift_prob
                    ),
                    "recurring_flag": bool(row["recurring_flag"]),
                    "cost_incurred_period": incurred,
                    "cost_to_date": to_date,
                    "cost_at_completion": at_completion,
                    "basis": basis,
                    "dollar_year": _dollar_year(basis, sub.fiscal_year, base_year),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DD 1921-1 - Functional Cost-Hour Report
# --------------------------------------------------------------------------
def build_dd1921_1(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Functional categories split into hours and dollars per WBS element.

    Material carries dollars but no hours. Anything that divides total dollars
    by total hours to recover a blended wrap rate will be wrong by the material
    share, which is roughly a third of recurring cost -- the reconciliation the
    ingest layer has to get right.
    """
    truth, spec = program.truth, program.spec
    p = spec.pathologies
    index, base_year = truth.inflation_index, truth.base_year

    grain = (
        truth.cells.groupby(
            ["lot", "wbs_name", "recurring_flag", "functional_category"]
        )[["hours", "dollars_by"]]
        .sum()
        .reset_index()
    )

    rows = []
    for sub in submissions:
        basis = dollar_basis(rng, p.then_year_prob)
        period = grain[grain["lot"] == sub.lot]
        for _, row in period.iterrows():
            canonical = str(row["wbs_name"])
            category = str(row["functional_category"])
            dollars_by = float(row["dollars_by"]) * sub.error_factor
            hours = float(row["hours"]) * sub.error_factor
            rows.append(
                {
                    "program": spec.program,
                    "contractor": spec.contractor,
                    "report_type": "DD1921-1",
                    "report_date": sub.report_date.isoformat(),
                    "period_fy": sub.fiscal_year,
                    "lot": sub.lot,
                    "wbs_element_name": reported_name(
                        canonical, rng, p.name_drift_prob
                    ),
                    "functional_category": category,
                    "recurring_flag": bool(row["recurring_flag"]),
                    # Hours are hours: they are never escalated.
                    "hours": hours if category in _HOURS_CATEGORIES else np.nan,
                    "dollars": _escalate(dollars_by, basis, sub.fiscal_year, index),
                    "basis": basis,
                    "dollar_year": _dollar_year(basis, sub.fiscal_year, base_year),
                }
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# DD 1921-2 - Progress Curve Report
# --------------------------------------------------------------------------
def build_dd1921_2(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Recurring cost by lot -- the natural input to a learning-curve fit.

    Recurring only, by definition of a progress curve: nonrecurring cost does
    not follow the curve and folding it in is the single most common way to
    fit a slope that is too steep.
    """
    truth, spec = program.truth, program.spec
    p = spec.pathologies
    index, base_year = truth.inflation_index, truth.base_year

    lot_costs = truth.recurring_lot_costs().set_index("lot")

    first_unit, cum = {}, 0
    for i, q in enumerate(truth.lot_quantities, start=1):
        first_unit[i] = cum + 1
        cum += q

    rows = []
    for sub in submissions:
        basis = dollar_basis(rng, p.then_year_prob)
        if sub.lot not in lot_costs.index:
            continue
        rec = lot_costs.loc[sub.lot]
        qty = int(rec["quantity"])
        lot_cost_by = float(rec["lot_cost_by"]) * sub.error_factor
        lot_cost = _escalate(lot_cost_by, basis, sub.fiscal_year, index)
        rows.append(
            {
                "program": spec.program,
                "contractor": spec.contractor,
                "report_type": "DD1921-2",
                "report_date": sub.report_date.isoformat(),
                "period_fy": sub.fiscal_year,
                "lot": sub.lot,
                "lot_quantity": qty,
                "first_unit": first_unit[sub.lot],
                "last_unit": first_unit[sub.lot] + qty - 1,
                "cumulative_quantity": first_unit[sub.lot] + qty - 1,
                "recurring_lot_cost": lot_cost,
                "unit_cost": lot_cost / qty,
                "basis": basis,
                "dollar_year": _dollar_year(basis, sub.fiscal_year, base_year),
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Cost and Hour Report (FlexFile)
# --------------------------------------------------------------------------
def build_flexfile(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """The modern flat/normalised structure that replaced the 1921 series.

    One row per (WBS element, functional category, cost element, period), with
    a unit column, because hours and dollars share the same value column. This
    is why the FlexFile is easier to ingest and harder to eyeball: nothing is
    pre-summed, so totals must be derived rather than read off.
    """
    truth, spec = program.truth, program.spec
    p = spec.pathologies
    index, base_year = truth.inflation_index, truth.base_year

    grain = (
        truth.cells.groupby(
            ["lot", "wbs_name", "recurring_flag", "functional_category"]
        )[["hours", "dollars_by"]]
        .sum()
        .reset_index()
    )

    rows = []
    for sub in submissions:
        basis = dollar_basis(rng, p.then_year_prob)
        period = grain[grain["lot"] == sub.lot]
        for _, row in period.iterrows():
            canonical = str(row["wbs_name"])
            category = str(row["functional_category"])
            dollars_by = float(row["dollars_by"]) * sub.error_factor
            hours = float(row["hours"]) * sub.error_factor
            dollars = _escalate(dollars_by, basis, sub.fiscal_year, index)

            base = {
                "program": spec.program,
                "contractor": spec.contractor,
                "report_type": "FLEXFILE",
                "report_date": sub.report_date.isoformat(),
                "period_fy": sub.fiscal_year,
                "lot": sub.lot,
                "wbs_element_name": reported_name(canonical, rng, p.name_drift_prob),
                "functional_category": category,
                "recurring_flag": bool(row["recurring_flag"]),
                "basis": basis,
                "dollar_year": _dollar_year(basis, sub.fiscal_year, base_year),
            }

            if category in _HOURS_CATEGORIES:
                rows.append(
                    {**base, "cost_element": "direct_labor_hours",
                     "value": hours, "unit": "hours"}
                )
                rows.append(
                    {**base, "cost_element": "direct_labor_dollars",
                     "value": dollars * DIRECT_LABOR_FRACTION, "unit": "dollars"}
                )
                rows.append(
                    {**base, "cost_element": "overhead_dollars",
                     "value": dollars * (1.0 - DIRECT_LABOR_FRACTION),
                     "unit": "dollars"}
                )
            else:
                rows.append(
                    {**base, "cost_element": "material_dollars",
                     "value": dollars, "unit": "dollars"}
                )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Quantity Data Report
# --------------------------------------------------------------------------
def build_quantity_report(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Lot quantities and delivery periods, including any mid-program change.

    Emitted once per submission because the quantity report is resubmitted
    alongside the cost reports, and a rebaseline shows up as the same lot
    carrying a different quantity in a later submission -- which is exactly
    the case a naive join on lot number gets wrong.
    """
    truth, spec = program.truth, program.spec
    rows = []
    for sub in submissions:
        cum = 0
        for lot_idx, qty in enumerate(truth.lot_quantities, start=1):
            if lot_idx > sub.lot:
                break
            cum += qty
            rows.append(
                {
                    "program": spec.program,
                    "contractor": spec.contractor,
                    "report_type": "QUANTITY",
                    "report_date": sub.report_date.isoformat(),
                    "period_fy": sub.fiscal_year,
                    "lot": lot_idx,
                    "lot_quantity": qty,
                    "planned_quantity": truth.planned_quantities[lot_idx - 1],
                    "cumulative_quantity": cum,
                    "delivery_fy": spec.fiscal_year(lot_idx - 1),
                    "rebaselined": qty != truth.planned_quantities[lot_idx - 1],
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# SRDR (DD 2630) - Software Resources Data Report
# --------------------------------------------------------------------------
def build_srdr(
    program: SyntheticProgram,
    submissions: list[Submission],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Development and maintenance software report: size, effort, schedule.

    Effort is generated from :data:`TRUE_SOFTWARE_CER` on adaptation-adjusted
    size, so a CER fitted across a generated portfolio has a known right
    answer. Note this report deliberately does *not* reconcile to the CSDR
    dollars: SRDR is a separate collection with its own boundaries, and a
    pipeline that assumes the two tie out is making an assumption the real
    data does not support.
    """
    spec = program.spec
    sw = spec.software
    p = spec.pathologies

    # One SRDR per software build, on the earlier submissions.
    build_periods = [s for s in submissions if not s.is_resubmission][:4]
    if not build_periods:
        build_periods = submissions[:1]

    total_effort = sw.true_effort_hours()
    if p.noise_cv > 0.0:
        total_effort *= float(rng.lognormal(0.0, p.noise_cv))

    # Split the programme's software across builds, front-loaded.
    weights = np.array([0.40, 0.28, 0.20, 0.12])[: len(build_periods)]
    weights = weights / weights.sum()

    rows = []
    for build_no, (sub, w) in enumerate(zip(build_periods, weights), start=1):
        effort = total_effort * float(w)
        record = {
            "program": spec.program,
            "contractor": spec.contractor,
            "report_type": "SRDR",
            "report_date": sub.report_date.isoformat(),
            "period_fy": sub.fiscal_year,
            "build": f"Build {build_no}",
            "wbs_element_name": reported_name(
                "Air Vehicle Software", rng, p.name_drift_prob
            ),
            "sloc_new": int(sw.sloc_new * w),
            "sloc_modified": int(sw.sloc_modified * w),
            "sloc_reused": int(sw.sloc_reused * w),
            "sloc_autogen": int(sw.sloc_autogen * w),
            "equivalent_sloc": sw.equivalent_sloc * float(w),
            "total_effort_hours": effort,
            "schedule_start": f"{sub.fiscal_year - 1}-10-01",
            "schedule_end": f"{sub.fiscal_year}-09-30",
            "duration_months": 12,
            "primary_language": sw.primary_language,
            "application_domain": sw.application_domain,
            "development_process": sw.development_process,
            "cmmi_level": sw.cmmi_level,
            "peak_staff": int(sw.peak_staff * w),
            "team_experience": sw.team_experience,
            "requirements_volatility": sw.requirements_volatility,
        }
        for activity, share in SRDR_ACTIVITY_MIX.items():
            record[f"hours_{activity}"] = effort * share
        rows.append(record)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
_BUILDERS = {
    "dd1921": build_dd1921,
    "dd1921_1": build_dd1921_1,
    "dd1921_2": build_dd1921_2,
    "flexfile": build_flexfile,
    "quantity_report": build_quantity_report,
    "srdr": build_srdr,
}


def build_all(
    program: SyntheticProgram, rng: np.random.Generator
) -> dict[str, pd.DataFrame]:
    """Build every report shape for one program.

    All six share a single submission plan, so a period missing from the
    DD 1921 is missing from the FlexFile too -- which is how a real gap
    presents itself, and what stops the pipeline quietly filling one report
    from another.
    """
    submissions = plan_submissions(program, rng)
    if not submissions:
        raise ValueError(
            "Every reporting period was dropped as missing. Lower "
            "PathologyConfig.missing_period_prob or use a different seed."
        )

    out: dict[str, pd.DataFrame] = {}
    for name, builder in _BUILDERS.items():
        out[name] = builder(program, submissions, rng)
        logger.debug("Built %s: %d rows", name, len(out[name]))
    return out
