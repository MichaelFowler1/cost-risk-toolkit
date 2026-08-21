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
                                lmp_func, ols_fit, projection_intervals,
                                run_lot_cost_model, selected_model_name,
                                simulate_buy, track_units)

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
    fit = ols_fit([x], y)
    design = np.column_stack([np.ones(12), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.asarray(fit["Beta"]) == pytest.approx(beta, rel=1e-10)
    assert fit["DF"] == 10


def test_a_singular_design_is_refused_rather_than_inverted():
    """Two identical predictors have no unique solution. Returning None beats
    returning whatever a pseudo-inverse happens to pick."""
    x = np.linspace(1.0, 4.0, 8)
    assert ols_fit([x, x], 2.0 + 0.5 * x) is None


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
    from cost_core.lotmodel.enrich import _design

    _, ctx, _ = run
    design, y, _ = _design(ctx, "LC")
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
