# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""The lot cost model: the ported engine, and the statistics added to it.

This module is held to a different standard than the rest of the suite, and
deliberately so. The engine came from a working spreadsheet-replacement tool
whose numbers analysts have already used, so the first duty here is that
nothing moved. The golden-master tests below fit the reference program and
assert the projections, the analyst summary and the fit chart data against
values captured from the original script. If a refactor ever shifts a slope in
the fourth decimal, these fail.

Everything after that covers the layer added on top -- unbiased refits,
influence, prediction intervals and buy risk -- and those are held to the
usual standard for this repository: closed-form identities rather than
recorded output.

The reference program below is invented, not taken from any real or supplied
data. The values it is asserted against were produced by running the original
script on it, so the golden master still proves the port is faithful.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from cost_core.lotmodel import (SETTINGS, EnrichmentError, compare_fitting_methods,
                                enrich_run, generate_analyst_summary,
                                generate_fit_chart_data, influence_diagnostics,
                                lmp_func, projection_intervals,
                                run_lot_cost_model, selected_model_name,
                                simulate_buy, track_units)
from cost_core.lotmodel import models
from cost_core.lotmodel.models import ols_via_fitting

ANALOGY = pd.DataFrame({
    "Lot": [1, 2, 3, 4, 5, 6],
    "Lot FY": [2018, 2019, 2020, 2021, 2022, 2023],
    "Qty": [8, 16, 24, 24, 18, 18],
    "AUC ($K)": [3120.00, 2585.50, 2402.75, 2438.10, 2310.40, 2266.85],
})

ESTIMATE = pd.DataFrame({
    "Lot": list(range(1, 9)),
    "Lot FY": [2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037],
    "Qty": [6, 12, 12, 12, 12, 12, 12, 6],
    "Complexity": [1.0] * 8,
})

RUN_INFO = {"RunID": "R001", "Program": "TEST", "RunLabel": "golden",
            "BaseYear": ""}


@pytest.fixture(scope="module")
def run():
    projections, ctx = run_lot_cost_model(ANALOGY, ESTIMATE, {})
    summary = generate_analyst_summary(ctx, RUN_INFO)
    return projections, ctx, summary


def summary_value(summary: pd.DataFrame, item: str, column: str) -> str:
    row = summary[summary["Item"] == item]
    assert not row.empty, f"no summary row {item!r}"
    return str(row.iloc[0][column])


# ======================================================= golden master
def test_the_engine_selects_the_learning_curve_on_the_reference_program(run):
    """The reference answer. If selection ever changes on this data, the rule
    changed, and that is a decision rather than a refactor."""
    _, _, summary = run
    assert summary_value(summary, "SELECTED", "LC") == "YES"
    assert summary_value(summary, "SELECTED", "Rate").strip() == ""
    assert summary_value(summary, "SELECTED", "LC+Rate").strip() == ""


def test_the_rate_t_is_withheld_when_there_is_no_residual_scale_to_form_it(run):
    """Costs priced straight off the curve leave no scatter, and a t-statistic
    needs scatter to be a statement about anything.

    On such a series the fit reproduces every point, so the residual scale sits
    at the floating-point floor and the rate coefficient and its standard error
    are both rounding errors. Their ratio comes out near 14 here, well past the
    2.0 gate, and it would decide which model the tool recommends on the last
    bit of the solve rather than on the data. summary.py applies the test
    cost_core.cer.diagnostics already applies before dividing by a residual
    scale -- sigma at or below 1e-10 of the fitted values -- and reports the t
    as not available instead, which leaves LC selected.

    The second half of this test is as much the point as the first. The rule
    has to be inert on data that has a residual scale, so the reference
    programme is checked to still form and print its t; and the pure Rate model
    on the synthetic series, which does not reproduce it exactly, keeps its own
    t as well. The threshold sits in an empty band: across every lot fit this
    suite performs the ratio is either at or below 1.1e-14 or at or above
    2.6e-4, so nothing is within six orders of magnitude of it either way.
    """
    qty = np.array([10, 15, 20, 25, 30, 35])
    t1, b = 900.0, np.log2(0.85)
    spans = track_units(qty, 0)
    midpoints = np.array([lmp_func(s["S"], s["E"], q, b)
                          for s, q in zip(spans, qty)])
    analogy = pd.DataFrame({
        "Lot": range(1, 7),
        "Lot FY": range(2018, 2024),
        "Qty": qty,
        "AUC ($K)": t1 * midpoints ** b,
    })
    estimate = pd.DataFrame({"Lot": [1], "Lot FY": [2030], "Qty": [12],
                             "Complexity": [1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, exact_ctx = run_lot_cost_model(analogy, estimate, {})
        exact = generate_analyst_summary(exact_ctx, RUN_INFO)

    # The model is fitted and reported; it is only the significance test that
    # declines to run, so this is not the singular guard by another name.
    assert exact_ctx["mdl_lcr"] is not None
    assert summary_value(exact, "t (rate coefficient)", "LC+Rate") == "n/a"
    assert selected_model_name(exact) == "LC"
    # The pure Rate model does not reproduce this series, so it has a residual
    # scale and keeps its t.
    assert summary_value(exact, "t (rate coefficient)", "Rate") != "n/a"

    _, _, summary = run
    assert summary_value(summary, "t (rate coefficient)", "LC+Rate") == "-0.90"


def test_the_fitted_coefficients_are_unchanged(run):
    """Captured by running the original script on this reference programme,
    before the port. If a refactor shifts one of these, the estimate moved."""
    _, ctx, _ = run
    assert ctx["t1_lc"] == pytest.approx(3433.6272850614305, rel=1e-12)
    assert ctx["b_lc"] == pytest.approx(-0.09122296708531767, rel=1e-12)
    assert ctx["t1_rt"] == pytest.approx(5039.891205103033, rel=1e-12)
    assert ctx["b_rt"] == pytest.approx(-0.24680319530865424, rel=1e-12)
    assert ctx["t1_br"] == pytest.approx(3795.081707201815, rel=1e-12)
    assert ctx["b_br"] == pytest.approx(-0.07757032066816473, rel=1e-12)
    assert ctx["c_br"] == pytest.approx(-0.051981406149173966, rel=1e-12)
    assert ctx["n_keep"] == 6


def test_the_reported_statistics_are_unchanged(run):
    """R2, adjusted R2, SEE, CV, MAPE and bias, as the summary prints them."""
    _, _, summary = run
    expected = {
        "R2 (log)": ("0.9487", "0.7291", "0.9595"),
        "Adj R2": ("0.9359", "0.6614", "0.9324"),
        "SEE (log)": ("0.0296", "0.0680", "0.0304"),
        "CV": ("2.96%", "6.80%", "3.04%"),
        "MAPE": ("2.10%", "5.12%", "1.77%"),
        "Learning curve slope": ("93.87%", "-", "94.77%"),
    }
    for item, (lc, rate, lcr) in expected.items():
        assert summary_value(summary, item, "LC") == lc, item
        assert summary_value(summary, item, "Rate") == rate, item
        assert summary_value(summary, item, "LC+Rate") == lcr, item


def test_the_projection_table_shape_and_totals_are_unchanged(run):
    projections, _, _ = run
    assert projections.shape == (8, 62)
    total = projections["LC Lot Cost After Complexity ($)"].sum()
    assert total == pytest.approx(210_987_213.17, rel=1e-6)


def test_the_fit_chart_data_is_one_row_per_analogy_lot(run):
    _, ctx, _ = run
    chart = generate_fit_chart_data(ctx)
    assert len(chart) == 6


# ============================================== the maths, on its own terms
@pytest.mark.parametrize("qty", [1, 2, 7, 25])
def test_the_lot_midpoint_falls_inside_its_own_lot(qty):
    """A midpoint outside the lot it describes would be meaningless."""
    start, end = 10, 10 + qty - 1
    mid = lmp_func(start, end, qty, -0.15)
    assert start - 0.5 <= mid <= end + 0.5


def test_a_single_unit_lot_is_its_own_midpoint():
    assert lmp_func(7, 7, 1, -0.15) == pytest.approx(7.0)


def test_a_flat_curve_puts_the_midpoint_at_the_arithmetic_centre():
    """With b = 0 there is no learning, so every unit costs the same and the
    midpoint is the plain average of the endpoints."""
    assert lmp_func(1, 11, 11, 0.0) == pytest.approx(6.0)


def test_unit_tracking_tiles_the_production_run():
    spans = track_units(np.array([9, 21, 22]), prior=0)
    assert [s["S"] for s in spans] == [1, 10, 31]
    assert [s["E"] for s in spans] == [9, 30, 52]


def test_prior_units_shift_the_whole_run():
    spans = track_units(np.array([5, 5]), prior=40)
    assert spans[0]["S"] == 41 and spans[-1]["E"] == 50


def test_ols_matches_the_normal_equations():
    rng = np.random.default_rng(0)
    x = np.linspace(1.0, 4.0, 12)
    y = 2.0 + 0.7 * x + rng.normal(0, 0.05, 12)
    design = np.column_stack([np.ones(12), x])
    # The models fit the response in levels and take their own logarithm, so
    # the log-space response the normal equations solve for is exp()d going in.
    fit = ols_via_fitting("LC", design, np.exp(y), SETTINGS["SingularTol"])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.asarray(fit["Beta"]) == pytest.approx(beta, rel=1e-10)
    assert fit["DF"] == 10


def test_a_singular_design_is_refused_rather_than_inverted():
    """Two identical predictors have no unique solution. Returning None beats
    returning whatever a pseudo-inverse happens to pick."""
    x = np.linspace(1.0, 4.0, 8)
    design = np.column_stack([np.ones(8), x, x])
    y = np.exp(2.0 + 0.5 * x)
    assert ols_via_fitting("LC+Rate", design, y, SETTINGS["SingularTol"]) is None



def test_a_design_with_no_spare_observations_reports_its_own_emptiness():
    """n == p leaves nothing to estimate a spread from.

    The engine refuses fewer than three costed lots long before this, so no run
    reaches it, but the fit-space dict is the contract the summary reads and it
    has to say "no degrees of freedom" rather than raise. The shared estimator
    refuses such a fit outright, so the wrapper answers for it.
    """
    x = np.array([1.0, 2.0])
    y = np.array([10.0, 20.0])
    fit = models.ols_via_fitting("LC", np.column_stack([np.ones_like(x), x]),
                                 y, 1e-12)
    assert fit is not None
    assert fit["DF"] == 0
    assert fit["N"] == 2 and fit["K"] == 2
    assert np.isnan(fit["SEy"])
    assert all(np.isnan(se) for se in fit["SE"])

def test_a_near_constant_column_is_refused_at_the_shipped_tolerance():
    """A column that is constant to twelve decimal places carries nothing the
    intercept does not already carry, and the design is refused.

    SingularTol ships at 1e-12 and any caller can override it, so it is worth
    pinning that the guard does its job where the tool actually runs.
    models.is_singular's docstring says what happens to a caller who loosens it
    below about 1e-16, where the scaled determinant stops being a number at
    all, and why neither this solver nor the one it replaced returns anything
    worth having past that point.
    """
    near_constant = np.full(6, 2.5) + 1e-12 * np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    design = np.column_stack([np.ones(6), near_constant])
    assert models.is_singular(design, SETTINGS["SingularTol"]) is True
    y = np.exp(1.0 + 0.3 * near_constant)
    assert ols_via_fitting("LC", design, y, SETTINGS["SingularTol"]) is None


def test_a_design_with_no_spare_observations_still_reports():
    """n == p keeps the old contract: NaN where a variance is needed, not a
    refusal. cost_core.fitting declines to qualify an interpolating fit, and
    swapping that in for the engine's older behaviour would be a change of
    substance inside a refactor. The engine cannot build such a design -- it
    needs three lots for LC and four for LC+Rate -- so this is the only thing
    that exercises the branch.
    """
    x = np.array([1.0, 2.0, 3.0])
    design = np.column_stack([np.ones(3), x, x ** 2])
    y_log = np.array([0.5, 0.9, 1.6])
    fit = ols_via_fitting("LC+Rate", design, np.exp(y_log), SETTINGS["SingularTol"])
    assert fit is not None
    assert fit["N"] == 3 and fit["K"] == 3 and fit["DF"] == 0
    # It interpolates, so the fitted values are the data and SSE is zero.
    assert fit["Fitted"] == pytest.approx(y_log, abs=1e-12)
    assert fit["SSE"] == pytest.approx(0.0, abs=1e-24)
    # Nothing that needs a residual variance is reported.
    assert all(pd.isna(v) for v in fit["SE"])
    assert pd.isna(fit["SEy"])


# ================================================ engine input validation
def test_an_empty_analogy_table_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        run_lot_cost_model(ANALOGY.iloc[0:0], ESTIMATE, {})


def test_an_empty_estimate_table_is_refused():
    with pytest.raises(ValueError, match="no rows"):
        run_lot_cost_model(ANALOGY, ESTIMATE.iloc[0:0], {})


def test_a_missing_cost_column_is_refused_with_the_columns_listed():
    bad = ANALOGY.drop(columns=["AUC ($K)"])
    with pytest.raises(ValueError, match="quantity column and a unit-cost"):
        run_lot_cost_model(bad, ESTIMATE, {})


def test_column_names_are_matched_case_insensitively():
    """Real files say 'Qty', 'QUANTITY', 'Unit Cost'. The engine should not
    care which."""
    renamed = ANALOGY.rename(columns={"Qty": "QUANTITY", "AUC ($K)": "Unit Cost"})
    _, ctx = run_lot_cost_model(renamed, ESTIMATE, {})
    assert ctx["t1_lc"] == pytest.approx(3433.6272851, rel=1e-9)


# ============================ the projections have to satisfy the equation
def equation_unit_costs(projections: pd.DataFrame, ctx: dict, model: str):
    """Re-evaluate the fitted equation by hand for every projected lot.

    Midpoints are recomputed with ``lmp_func`` rather than read from the
    projections table, because that column is rounded to four decimals for
    display and this comparison is meant to be exact.
    """
    qty = projections["Lot Quantity"].to_numpy(dtype=float)
    first = projections["First Unit in Lot"].to_numpy(dtype=float)
    last = projections["Last Unit in Lot"].to_numpy(dtype=float)
    if model == "LC":
        b = ctx["b_lc"]
        mid = np.array([lmp_func(f, l, q, b) for f, l, q in zip(first, last, qty)])
        return ctx["t1_lc"] * mid ** b
    if model == "Rate":
        return ctx["t1_rt"] * qty ** ctx["b_rt"]
    b = ctx["b_br"]
    mid = np.array([lmp_func(f, l, q, b) for f, l, q in zip(first, last, qty)])
    return ctx["t1_br"] * mid ** b * qty ** ctx["c_br"]


@pytest.mark.parametrize("model", ["LC", "Rate", "LC+Rate"])
def test_the_projections_satisfy_the_equation_the_tool_prints(run, model):
    """The defect this exists to prevent.

    The engine used to project Rate on the lot midpoint, which is not the
    variable that model regresses on, and LC+Rate without its rate factor at
    all. Both came across from the original tool. The result was one run
    reporting two different formulas: the residual columns showed the model
    tracking its own lots to about 1% while the projections built from the
    same coefficients were tens of percent away.

    Tolerance is the cent the column is rounded to, not a judgement call.
    """
    projections, ctx, _ = run
    printed = projections[f"{model} Unit Cost ($K)"].to_numpy(dtype=float)
    assert printed == pytest.approx(
        equation_unit_costs(projections, ctx, model), abs=0.01)


@pytest.mark.parametrize("model", ["LC", "Rate", "LC+Rate"])
def test_the_lot_cost_columns_follow_from_the_same_equation(run, model):
    """The rest of the chain, so a corrected unit cost cannot be undone by the
    columns built from it. Everything is checked back to the equation rather
    than to the unit cost column, which is rounded to the cent for display."""
    projections, ctx, _ = run
    qty = projections["Lot Quantity"].to_numpy(dtype=float)
    cf = projections["Complexity Factor"].to_numpy(dtype=float)
    unit = equation_unit_costs(projections, ctx, model)

    before = unit * qty * SETTINGS["TotalScale"]
    assert projections[f"{model} Lot Cost Before Complexity ($)"].to_numpy(
        dtype=float) == pytest.approx(before, abs=0.01)
    assert projections[f"{model} Lot Cost After Complexity ($)"].to_numpy(
        dtype=float) == pytest.approx(before * cf, abs=0.01)


def test_a_back_cast_of_the_fitted_lots_recovers_their_actual_total(run):
    """Price the analogy lots as though they were the estimate. The answer is
    known, so the projection has nowhere to hide: it has to land on the total
    that was fitted, within the scatter the fit reports."""
    _, ctx, summary = run
    model = selected_model_name(summary)
    back = ANALOGY[["Lot", "Lot FY", "Qty"]].copy()
    back["Complexity"] = 1.0
    projections, _ = run_lot_cost_model(ANALOGY, back, {})
    total = (projections[f"{model} Unit Cost ($K)"]
             * projections["Lot Quantity"]).sum()
    actual = float((ANALOGY["Qty"] * ANALOGY["AUC ($K)"]).sum())
    assert total == pytest.approx(actual, rel=0.01)


def test_the_legacy_switch_still_reproduces_the_original_tool(run):
    """Kept so a legacy workbook can be reproduced on request, and so the gap
    between the two is a measured number rather than a claim."""
    reference, ctx, _ = run
    legacy, _ = run_lot_cost_model(ANALOGY, ESTIMATE, {"LegacyRateOmission": True})

    qty = reference["Lot Quantity"].to_numpy(dtype=float)
    first = reference["First Unit in Lot"].to_numpy(dtype=float)
    last = reference["Last Unit in Lot"].to_numpy(dtype=float)

    def midpoints(b):
        return np.array([lmp_func(f, l, q, b)
                         for f, l, q in zip(first, last, qty)])

    # LC has no rate term, so it is untouched either way.
    assert legacy["LC Unit Cost ($K)"].to_numpy(dtype=float) == pytest.approx(
        reference["LC Unit Cost ($K)"].to_numpy(dtype=float), rel=1e-12)

    # Legacy LC+Rate is the fitted equation with its rate factor deleted.
    assert legacy["LC+Rate Unit Cost ($K)"].to_numpy(dtype=float) == pytest.approx(
        ctx["t1_br"] * midpoints(ctx["b_br"]) ** ctx["b_br"], abs=0.01)

    # Legacy Rate substitutes the lot midpoint for the lot quantity, which is
    # the variable that model was actually regressed on.
    assert legacy["Rate Unit Cost ($K)"].to_numpy(dtype=float) == pytest.approx(
        ctx["t1_rt"] * midpoints(ctx["b_rt"]) ** ctx["b_rt"], abs=0.01)

    # The LC+Rate omission overstates, always, never understates: the rate
    # exponent is negative and lot quantities are greater than one.
    assert (legacy["LC+Rate Unit Cost ($K)"].to_numpy(dtype=float)
            > reference["LC+Rate Unit Cost ($K)"].to_numpy(dtype=float)).all()


def test_the_old_setting_name_is_refused_rather_than_ignored():
    """Silently dropping an unknown key would leave a caller who asked for the
    legacy behaviour getting the corrected one without being told."""
    with pytest.raises(ValueError, match="LegacyRateOmission"):
        run_lot_cost_model(ANALOGY, ESTIMATE, {"ToolMatchProjection": True})


# ======================================== added statistics: unbiased refits
def test_mupe_and_zmpe_drive_the_mean_percentage_error_to_zero(run):
    """The reason they exist. OLS in log space leaves a positive bias; these
    do not."""
    _, ctx, _ = run
    comparison = compare_fitting_methods(ctx, "LC")
    frame = comparison.frame.set_index("Method")
    assert abs(frame.loc["MUPE", "Mean % error"]) < 1e-9
    assert abs(frame.loc["ZMPE", "Mean % error"]) < 1e-9
    assert frame.loc["OLS", "Mean % error"] > 0


def test_the_retransformation_bias_matches_its_closed_form(run):
    """exp(s^2/2) under lognormal log-space errors, and Duan's smearing
    estimate as a nonparametric check on the same quantity."""
    _, ctx, _ = run
    c = compare_fitting_methods(ctx, "LC")
    assert c.theoretical_factor == pytest.approx(
        np.exp(c.log_residual_variance / 2.0), rel=1e-12)
    assert c.percent_understated == pytest.approx(
        (c.theoretical_factor - 1.0) * 100.0, rel=1e-12)
    assert c.smearing_factor > 1.0
    # MUPE lifts the curve relative to naive OLS, which is the whole point.
    assert c.mupe_over_ols > 1.0


def test_the_refits_do_not_touch_the_engine(run):
    """The added statistics must never change the estimate. Fitting them and
    then re-running the engine has to give the same coefficients."""
    _, ctx, _ = run
    before = (ctx["t1_lc"], ctx["b_lc"])
    compare_fitting_methods(ctx, "LC")
    influence_diagnostics(ctx, "LC")
    assert (ctx["t1_lc"], ctx["b_lc"]) == before


def test_an_unfitted_model_is_refused_by_name(run):
    _, ctx, _ = run
    with pytest.raises(EnrichmentError, match="Unknown model"):
        compare_fitting_methods(ctx, "Quadratic")


# ============================================ added statistics: influence
def test_leverages_sum_to_the_number_of_parameters(run):
    """The hat matrix is a projection, so its trace is its rank. An exact
    identity, and a check that the design matrix rebuilt here is the one the
    engine actually fitted."""
    _, ctx, _ = run
    for model, params in (("LC", 2), ("Rate", 2), ("LC+Rate", 3)):
        diag = influence_diagnostics(ctx, model)
        assert diag["Leverage"].sum() == pytest.approx(params, rel=1e-9), model


def test_the_influence_table_reproduces_the_engines_fitted_values(run):
    """If the rebuilt design were wrong, these would not match."""
    projections, ctx, _ = run
    diag = influence_diagnostics(ctx, "LC")
    chart = generate_fit_chart_data(ctx)
    # The chart sheet rounds to cents for display, so compare at that scale.
    assert diag["Fitted ($K)"].to_numpy() == pytest.approx(
        chart["LC Estimated AUC ($K)"].to_numpy(), abs=0.01)
    assert diag["Actual ($K)"].to_numpy() == pytest.approx(
        chart["Actual AUC ($K)"].to_numpy(), rel=1e-12)


def test_the_first_analogy_lot_is_flagged_as_influential(run):
    """On this reference program lot 1 carries leverage above 0.8 and sets the
    slope. Nothing in the original summary said so."""
    _, ctx, _ = run
    diag = influence_diagnostics(ctx, "LC")
    first = diag.iloc[0]
    assert first["Leverage"] > 0.5
    assert bool(first["Influential"])


def test_cooks_distance_matches_a_leave_one_out_refit(run):
    """The closed form is exact, so it is checked against the thing it is a
    closed form for: drop each lot, refit, measure how far the fitted surface
    moved."""
    from cost_core.lotmodel.enrich import fit_on_design

    _, ctx, _ = run
    _, design, y = fit_on_design(ctx, "LC")
    n, p = design.shape
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    sigma2 = float(np.sum((y - fitted) ** 2) / (n - p))

    loo = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        b_i = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        loo.append(float(np.sum((fitted - design @ b_i) ** 2) / (p * sigma2)))

    diag = influence_diagnostics(ctx, "LC")
    assert diag["Cook's D"].to_numpy() == pytest.approx(np.array(loo), rel=1e-8)


# ==================================== added statistics: prediction intervals
def test_every_projected_lot_gets_an_interval_that_brackets_it(run):
    projections, ctx, _ = run
    out = projection_intervals(ctx, projections, "LC", level=0.80)
    assert len(out) == len(projections)
    assert (out["Lot Cost Lower"] < out["Lot Cost ($)"]).all()
    assert (out["Lot Cost Upper"] > out["Lot Cost ($)"]).all()
    assert (out["Kind"] == "prediction").all()


def test_the_interval_point_estimate_is_the_engines_own_number(run):
    """The interval wraps the estimate; it does not replace it."""
    projections, ctx, _ = run
    out = projection_intervals(ctx, projections, "LC")
    assert out["Lot Cost ($)"].to_numpy() == pytest.approx(
        projections["LC Lot Cost After Complexity ($)"].to_numpy(), rel=1e-12)


def test_a_wider_level_gives_a_wider_interval(run):
    projections, ctx, _ = run
    narrow = projection_intervals(ctx, projections, "LC", level=0.50)
    wide = projection_intervals(ctx, projections, "LC", level=0.95)
    assert ((wide["Lot Cost Upper"] - wide["Lot Cost Lower"]) >
            (narrow["Lot Cost Upper"] - narrow["Lot Cost Lower"])).all()


def test_an_impossible_level_is_refused(run):
    projections, ctx, _ = run
    with pytest.raises(EnrichmentError, match="between 0 and 1"):
        projection_intervals(ctx, projections, "LC", level=1.4)


# ============================================= added statistics: buy risk
def test_the_simulation_centres_on_the_engines_total(run):
    projections, ctx, _ = run
    risk = simulate_buy(ctx, projections, "LC", n_iter=20_000, seed=1)
    engine_total = projections["LC Lot Cost After Complexity ($)"].sum()
    assert risk.point_estimate == pytest.approx(engine_total, rel=1e-12)
    assert 30.0 < risk.point_estimate_percentile < 70.0
    assert risk.p50 <= risk.p80 <= risk.p90


def test_the_simulation_agrees_with_the_analytic_interval_on_one_lot(run):
    """Two independent routes to the same answer. For a single lot there is no
    correlation assumption separating them, so a disagreement would mean one
    of the two is wrong."""
    projections, ctx, _ = run
    one = projections.iloc[[3]].copy()
    analytic = projection_intervals(ctx, one, "LC", level=0.80)
    risk = simulate_buy(ctx, one, "LC", n_iter=120_000, seed=2)
    assert np.percentile(risk.totals, 10) == pytest.approx(
        analytic["Lot Cost Lower"].iloc[0], rel=0.02)
    assert np.percentile(risk.totals, 90) == pytest.approx(
        analytic["Lot Cost Upper"].iloc[0], rel=0.02)


def test_correlated_lot_residuals_widen_the_total(run):
    """The same lesson as the WBS-level simulator: treating consecutive lots as
    independent lets their shocks cancel and understates the spread."""
    projections, ctx, _ = run
    independent = simulate_buy(ctx, projections, "LC", n_iter=30_000, seed=3,
                               lot_correlation=0.0)
    correlated = simulate_buy(ctx, projections, "LC", n_iter=30_000, seed=3,
                              lot_correlation=0.6)
    assert correlated.std > independent.std
    assert correlated.p80 > independent.p80


def test_the_simulation_is_seed_deterministic(run):
    """A P80 that moves between runs is not a number anyone can defend."""
    projections, ctx, _ = run
    a = simulate_buy(ctx, projections, "LC", n_iter=5_000, seed=42)
    b = simulate_buy(ctx, projections, "LC", n_iter=5_000, seed=42)
    assert np.array_equal(a.totals, b.totals)
    assert a.p80 == b.p80
    assert simulate_buy(ctx, projections, "LC", n_iter=5_000, seed=43).p80 != a.p80


def test_lot_level_draws_add_up_to_the_total(run):
    projections, ctx, _ = run
    risk = simulate_buy(ctx, projections, "LC", n_iter=2_000, seed=1)
    assert risk.per_lot.shape == (2_000, len(projections))
    assert risk.totals == pytest.approx(risk.per_lot.sum(axis=1), rel=1e-12)


def test_too_few_iterations_are_refused(run):
    projections, ctx, _ = run
    with pytest.raises(EnrichmentError, match="at least 2 iterations"):
        simulate_buy(ctx, projections, "LC", n_iter=1)


# =================================================== the whole added layer
def test_enrich_run_assembles_everything_and_names_the_selected_model(run):
    projections, ctx, summary = run
    en = enrich_run(ctx, projections, summary, n_iter=5_000, seed=0)
    assert en.selected_model == "LC"
    assert set(en.sheets()) == {"Fit_Methods", "Influence",
                                "Prediction_Intervals", "Buy_Risk"}
    assert len(en.influence) == 6
    assert len(en.intervals) == 8
    # Lot 1 dominating the fit is worth saying out loud.
    assert any("setting this fit" in w for w in en.warnings_raised)


def test_the_selected_model_is_read_from_the_summary(run):
    _, _, summary = run
    assert selected_model_name(summary) == "LC"


def test_a_summary_with_nothing_selected_is_refused():
    empty = pd.DataFrame({"Item": ["SELECTED"], "Value": [""], "LC": [""],
                          "Rate": [""], "LC+Rate": [""]})
    with pytest.raises(EnrichmentError, match="nothing to add statistics to"):
        selected_model_name(empty)


def test_enrichment_leaves_the_estimate_untouched(run):
    """The guarantee the whole design rests on: the added statistics read the
    fitted models and never write back."""
    projections, ctx, summary = run
    before = projections.copy(deep=True)
    enrich_run(ctx, projections, summary, n_iter=2_000, seed=0)
    pd.testing.assert_frame_equal(projections, before)
