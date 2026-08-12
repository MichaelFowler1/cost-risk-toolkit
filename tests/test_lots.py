"""The simple lot-entry front door: units and cost, one row per lot.

This is the path a user takes with real production history, so the tests care
most about the ways that input goes wrong quietly. Three of them move the
fitted slope by several points while leaving a fit that looks entirely healthy:
nonrecurring cost folded into the lot totals, escalation left in dollars
declared constant, and lots that do not start at unit 1.

The round-trip tests use the same standard as the rest of the suite: build lot
costs from a curve of known slope and first-unit cost, feed them in as the two
columns a user would type, and require the fit to return the generating
parameters exactly.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from cost_core.learning_curve import CurveModel, Theory
from cost_core.lots import (COMFORTABLE_LOTS, LotInputError, LotSeries,
                            analyse_lots, build_assumption_log)

PROFILE = [20, 20, 25, 25, 30, 30]


def lot_costs_from(theory=Theory.CRAWFORD, t1=5_000_000.0, slope=0.85,
                   profile=None, first_unit=1):
    """Exact lot costs from a known curve, in the two columns a user types."""
    profile = profile or PROFILE
    truth = CurveModel(theory, t1, np.log2(slope))
    cursor, spans = first_unit, []
    for q in profile:
        spans.append((cursor, cursor + q - 1))
        cursor += q
    spans = np.array(spans)
    return np.array(profile), truth.lot_cost(spans[:, 0], spans[:, 1])


def series(**kwargs):
    quantities, costs = lot_costs_from(**{
        k: v for k, v in kwargs.items()
        if k in {"theory", "t1", "slope", "profile", "first_unit"}
    })
    kwargs.setdefault("dollar_year", 2026)
    for key in ("theory", "t1", "slope", "profile"):
        kwargs.pop(key, None)
    return LotSeries(quantities=quantities, costs=costs, **kwargs)


# ============================================================== round trips
@pytest.mark.parametrize("slope", [0.75, 0.85, 0.92])
@pytest.mark.parametrize("theory", list(Theory))
def test_the_fit_recovers_the_curve_the_lots_were_built_from(theory, slope):
    """Two columns in, the generating slope and first-unit cost back out."""
    quantities, costs = lot_costs_from(theory=theory, slope=slope, t1=5e6)
    fit = LotSeries(
        quantities=quantities, costs=costs, dollar_year=2026
    ).fit(theory=theory)
    assert fit.slope == pytest.approx(slope, rel=1e-6)
    assert fit.t1 == pytest.approx(5e6, rel=1e-6)


def test_lots_tile_the_unit_sequence_without_gaps_or_overlap():
    s = series()
    ranges = s.unit_ranges()
    assert ranges[0, 0] == 1
    assert list(ranges[1:, 0]) == list(ranges[:-1, 1] + 1)
    assert ranges[-1, 1] == sum(PROFILE)
    assert list(ranges[:, 1] - ranges[:, 0] + 1) == PROFILE


def test_a_prior_buy_shifts_every_lot_and_changes_the_answer():
    """If the programme already built 40 units, lot 1 is not unit 1. Getting
    this wrong makes T1 the cost of a unit nobody ever built."""
    quantities, costs = lot_costs_from(first_unit=41, t1=5e6, slope=0.85)

    right = LotSeries(quantities=quantities, costs=costs, dollar_year=2026,
                      first_unit=41).fit()
    wrong = LotSeries(quantities=quantities, costs=costs, dollar_year=2026).fit()

    assert right.t1 == pytest.approx(5e6, rel=1e-6)
    assert right.slope == pytest.approx(0.85, rel=1e-6)
    # Assuming the run starts at unit 1 badly misstates the first unit cost.
    assert wrong.t1 < 0.75 * 5e6


def test_the_derived_table_matches_the_input():
    s = series()
    frame = s.to_frame()
    assert list(frame["units"]) == PROFILE
    assert frame["cumulative_units"].iloc[-1] == sum(PROFILE)
    assert frame["lot_average_cost"].to_numpy() == pytest.approx(
        (s.costs / s.quantities)
    )
    assert frame["cumulative_average_cost"].to_numpy() == pytest.approx(
        np.cumsum(s.costs) / np.cumsum(s.quantities)
    )


# ============================================================== dollar year
def test_the_base_year_is_required():
    """Constant dollars are constant relative to a year. Without it the fitted
    first-unit cost cannot be escalated, compared or reused."""
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="dollar_year is required"):
        LotSeries(quantities=quantities, costs=costs)


@pytest.mark.parametrize("bad", ["not a year", None, 12, 3500])
def test_an_implausible_base_year_is_refused(bad):
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="dollar_year"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=bad)


def test_the_log_records_that_no_index_was_applied_and_why():
    """The decision has to read as a decision, not a step that was skipped."""
    report = analyse_lots(series(program="X"))
    note = report.series.dollar_basis_note()
    assert "FY2026" in note
    assert "No inflation index was applied" in note
    text = build_assumption_log(report).render()
    assert "constant FY2026" in text
    assert "no inflation index applied" in text.lower()
    assert "Dollar basis" in text


# ==================================================== escalation detection
def escalated(rate):
    """A true 85% curve with `rate` a year of escalation left in it."""
    quantities, costs = lot_costs_from(slope=0.85, t1=5e6)
    return LotSeries(
        quantities=quantities,
        costs=costs * np.array([(1.0 + rate) ** i for i in range(len(costs))]),
        dollar_year=2026,
    )


def test_a_rising_cumulative_average_is_flagged():
    """Severe escalation turns the cumulative average upward, which cannot
    happen on a learning curve."""
    findings = escalated(0.15).check_constant_dollars(warn=False)
    assert findings
    assert "RISES" in findings[0]
    assert "then-year" in findings[0]


def test_a_clean_series_raises_no_escalation_finding():
    assert series().check_constant_dollars(warn=False) == []


def test_the_level_check_alone_misses_realistic_escalation():
    """Documents the detection floor honestly. At 4% a year -- squarely in the
    realistic range -- learning still outpaces escalation, the cumulative
    average keeps falling, and the level check stays silent while the fitted
    slope is several points wrong."""
    s = escalated(0.04)
    assert s.check_constant_dollars(warn=False) == []
    fit = s.fit()
    assert fit.slope > 0.87            # true slope is 0.85
    assert fit.r_squared > 0.98        # and nothing looks wrong


def test_the_curvature_test_catches_what_the_level_check_misses():
    """Escalation compounds with time, learning with log quantity, so the
    mismatch bends the residuals upward. Detectable from about 2% a year."""
    report = analyse_lots(escalated(0.04))
    coefficient, t_stat = report.curvature()
    assert coefficient > 0                       # convex
    assert abs(t_stat) > 2.5
    findings = report.check_curve_shape(warn=False)
    assert findings and "convex" in findings[0]
    assert "escalation" in findings[0]


def test_the_curvature_test_is_quiet_on_honest_scatter():
    """It must not fire on ordinary noise, or it is useless."""
    quantities, costs = lot_costs_from()
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noisy = LotSeries(
            quantities=quantities, costs=costs * rng.lognormal(0.0, 0.04, len(costs)),
            dollar_year=2026,
        )
        report = analyse_lots(noisy)
        assert report.check_curve_shape(warn=False) == [], seed


def test_curvature_cannot_be_tested_on_three_lots():
    """A quadratic needs four points to leave any residual degrees of freedom.
    The report says so rather than returning a meaningless number."""
    report = analyse_lots(series(profile=[20, 25, 30]))
    coefficient, t_stat = report.curvature()
    assert np.isnan(t_stat)
    findings = report.check_curve_shape(warn=False)
    assert findings and "too few to test the shape" in findings[0]


def test_fitting_warns_about_escalation_at_the_point_of_fitting():
    with pytest.warns(RuntimeWarning, match="RISES"):
        escalated(0.15).fit()


# ================================================================ cost basis
def test_the_cost_basis_must_be_declared():
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="cost_basis must be"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=2026,
                  cost_basis="whatever")


def test_declaring_total_cost_warns_loudly():
    """Nonrecurring is front-loaded, so including it reads as steeper learning
    and overstates future savings."""
    with pytest.warns(RuntimeWarning, match="TOTAL cost"):
        series(cost_basis="total").fit()


def test_recurring_cost_fits_without_that_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        series(cost_basis="recurring").fit()


def test_nonrecurring_in_the_totals_really_does_steepen_the_slope():
    """The reason the warning exists, demonstrated rather than asserted."""
    quantities, costs = lot_costs_from(slope=0.85, t1=5e6)
    nonrecurring = 40e6 * (0.55 ** np.arange(len(costs)))   # front-loaded
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = LotSeries(quantities=quantities, costs=costs,
                          dollar_year=2026).fit()
        polluted = LotSeries(quantities=quantities, costs=costs + nonrecurring,
                             dollar_year=2026, cost_basis="total").fit()
    assert clean.slope == pytest.approx(0.85, rel=1e-6)
    assert polluted.slope < clean.slope - 0.02


# ============================================================== small samples
def test_two_lots_are_refused():
    """Two points and two parameters interpolate: a perfect fit with no
    estimable uncertainty."""
    with pytest.raises(LotInputError, match="cannot support a learning curve"):
        series(profile=[20, 25]).fit()


def test_three_lots_fit_but_warn():
    with pytest.warns(RuntimeWarning, match="degree"):
        fit = series(profile=[20, 25, 30]).fit()
    assert fit.df == 1


def test_a_small_sample_can_be_made_fatal():
    with pytest.raises(LotInputError, match="degree"):
        series(profile=[20, 25, 30]).fit(allow_small_sample=False)


def test_degrees_of_freedom_are_lots_minus_two():
    for n in (3, 4, 6):
        s = series(profile=PROFILE[:n])
        assert s.df == n - 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert s.fit().df == n - 2


def test_a_comfortable_sample_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        series(profile=[20, 20, 25, 25, 30]).fit()
    assert len([20, 20, 25, 25, 30]) == COMFORTABLE_LOTS


# ================================================================ bad input
def test_mismatched_column_lengths_are_refused():
    with pytest.raises(LotInputError, match="Every lot needs both"):
        LotSeries(quantities=[10, 20, 30], costs=[1e6, 2e6], dollar_year=2026)


def test_no_lots_at_all_is_refused():
    with pytest.raises(LotInputError, match="No lots supplied"):
        LotSeries(quantities=[], costs=[], dollar_year=2026)


@pytest.mark.parametrize("bad", [[20, 0, 30], [20, -5, 30]])
def test_non_positive_quantities_are_refused_by_lot_number(bad):
    with pytest.raises(LotInputError, match="lot 2"):
        LotSeries(quantities=bad, costs=[1e6, 2e6, 3e6], dollar_year=2026)


def test_fractional_quantities_are_refused():
    """Half an aircraft is not a data point."""
    with pytest.raises(LotInputError, match="whole units"):
        LotSeries(quantities=[20, 22.5, 30], costs=[1e6, 2e6, 3e6],
                  dollar_year=2026)


def test_non_positive_costs_are_refused_by_lot_number():
    with pytest.raises(LotInputError, match="lot 3"):
        LotSeries(quantities=[20, 25, 30], costs=[1e6, 2e6, 0.0],
                  dollar_year=2026)


def test_a_first_unit_below_one_is_refused():
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="first_unit must be"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=2026,
                  first_unit=0)


def test_mismatched_labels_are_refused():
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="labels for"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=2026,
                  labels=("only", "two"))


# ================================================================ file input
def test_a_two_column_csv_reads_straight_in(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"units": PROFILE, "cost": lot_costs_from()[1]}).to_csv(
        path, index=False
    )
    s = LotSeries.read(path, dollar_year=2026)
    assert s.n_lots == len(PROFILE)
    assert list(s.quantities) == PROFILE
    assert s.fit().slope == pytest.approx(0.85, rel=1e-6)


@pytest.mark.parametrize(
    "headers", [("Qty", "Total Cost"), ("QUANTITY", "amount"),
                ("lot_quantity", "lot_cost"), ("Units", "Dollars")]
)
def test_common_column_spellings_are_recognised(tmp_path, headers):
    path = tmp_path / "lots.csv"
    pd.DataFrame({headers[0]: PROFILE, headers[1]: lot_costs_from()[1]}).to_csv(
        path, index=False
    )
    assert LotSeries.read(path, dollar_year=2026).n_lots == len(PROFILE)


def test_an_unrecognisable_header_is_refused_with_the_columns_listed(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"widgets": PROFILE, "spend": lot_costs_from()[1]}).to_csv(
        path, index=False
    )
    with pytest.raises(LotInputError, match="Could not identify"):
        LotSeries.read(path, dollar_year=2026)
    # But naming them explicitly works.
    s = LotSeries.read(path, dollar_year=2026, units_col="widgets",
                       cost_col="spend")
    assert s.n_lots == len(PROFILE)


def test_naming_a_column_that_is_not_there_is_refused(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"units": PROFILE, "cost": lot_costs_from()[1]}).to_csv(
        path, index=False
    )
    with pytest.raises(LotInputError, match="No column 'nope'"):
        LotSeries.read(path, dollar_year=2026, units_col="nope")


def test_currency_formatting_from_a_spreadsheet_is_parsed(tmp_path):
    """Spreadsheets export dollars as text more often than not, and a column
    that silently becomes NaN is worse than one that fails."""
    path = tmp_path / "lots.csv"
    pd.DataFrame({
        "units": [20, 25, 30],
        "cost": ["$1,200,000", "1,400,000", "$1,500,000.50"],
    }).to_csv(path, index=False)
    s = LotSeries.read(path, dollar_year=2026)
    assert s.costs == pytest.approx([1_200_000.0, 1_400_000.0, 1_500_000.50])


def test_unparseable_cost_text_is_refused_with_examples(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"units": [20, 25, 30],
                  "cost": ["1200000", "see note 4", "1500000"]}).to_csv(
        path, index=False)
    with pytest.raises(LotInputError, match="see note 4"):
        LotSeries.read(path, dollar_year=2026)


def test_a_missing_file_and_a_bad_extension_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="No lot data file"):
        LotSeries.read(tmp_path / "nope.csv", dollar_year=2026)
    (tmp_path / "lots.json").write_text("{}")
    with pytest.raises(LotInputError, match="Unsupported file type"):
        LotSeries.read(tmp_path / "lots.json", dollar_year=2026)


def test_a_lot_label_column_is_carried_through(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"lot": ["LRIP 1", "LRIP 2", "FRP 1", "FRP 2"],
                  "units": [10, 12, 20, 25],
                  "cost": [5e7, 5.4e7, 8e7, 9.4e7]}).to_csv(path, index=False)
    s = LotSeries.read(path, dollar_year=2026)
    assert s.labels == ("LRIP 1", "LRIP 2", "FRP 1", "FRP 2")
    assert "LRIP 1" in analyse_lots(s).per_lot["lot"].tolist()


# ================================================================= reporting
def test_per_lot_errors_identify_which_lot_the_curve_misses():
    quantities, costs = lot_costs_from()
    costs = costs.copy()
    costs[3] *= 1.25                                   # one bad lot
    report = analyse_lots(LotSeries(quantities=quantities, costs=costs,
                                    dollar_year=2026))
    per_lot = report.per_lot
    assert len(per_lot) == len(PROFILE)
    assert per_lot["percent_error"].abs().idxmax() == 3
    assert "Lot 4" in report.narrative()


def test_the_summary_puts_r_squared_last_with_a_caveat():
    report = analyse_lots(series())
    stats = report.summary()["statistic"].tolist()
    assert stats[-1] == "r_squared_read_last"
    assert stats.index("slope") < stats.index("r_squared_read_last")
    assert "R squared" in build_assumption_log(report).render()


def test_the_forecast_continues_from_the_last_unit_built():
    report = analyse_lots(series())
    out = report.forecast([30, 40])
    assert out["first_unit"].iloc[0] == sum(PROFILE) + 1
    assert out["last_unit"].iloc[0] == sum(PROFILE) + 30
    assert out["first_unit"].iloc[1] == sum(PROFILE) + 31
    # Prediction intervals, and lot cost is the average times the quantity.
    assert (out["kind"] == "prediction").all()
    assert out["lot_cost"].to_numpy() == pytest.approx(
        (out["lot_average"] * out["quantity"]).to_numpy(), rel=1e-12
    )


def test_a_non_positive_forecast_quantity_is_refused():
    with pytest.raises(LotInputError, match="must be positive"):
        analyse_lots(series()).forecast([10, 0])


def test_all_three_methods_and_both_theories_are_reported():
    rng = np.random.default_rng(1)
    quantities, costs = lot_costs_from()
    report = analyse_lots(LotSeries(
        quantities=quantities, costs=costs * rng.lognormal(0.0, 0.08, len(costs)),
        dollar_year=2026,
    ))
    assert set(report.method_comparison()["method"]) == {"OLS", "MUPE", "ZMPE"}
    assert set(report.theory_comparison()["theory"]) == {"wright", "crawford"}
    bias = report.retransformation()
    assert bias.theoretical_factor >= 1.0


def test_the_assumption_log_records_every_declared_choice():
    s = series(program="PROGRAM Z", quantity_definition="accepted",
               cost_basis="total", first_unit=7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = build_assumption_log(analyse_lots(s), source="history.csv").render()
    assert "PROGRAM Z" in text
    assert "accepted" in text
    assert "history.csv" in text
    assert "unit 7" in text
    assert "nonrecurring" in text.lower()
    for characteristic in ("Comprehensive", "Well-documented", "Accurate", "Credible"):
        assert f"### {characteristic}" in text


# =============================================================== determinism
def test_the_whole_path_is_deterministic():
    a = analyse_lots(series())
    b = analyse_lots(series())
    assert a.fit.slope == b.fit.slope
    assert a.fit.t1 == b.fit.t1
    pd.testing.assert_frame_equal(a.per_lot, b.per_lot)


# ====================================================== forecast simulation
def noisy_series(scatter=0.08, seed=0, **kwargs):
    """A series with realistic scatter.

    The simulation tests deliberately do *not* use the noiseless helper: lots
    lying exactly on a curve give sigma of zero and a covariance of zero, so
    every draw returns the same number and the "distribution" is a point. That
    is the right answer for perfect data, and a useless basis for testing a
    simulator.
    """
    quantities, costs = lot_costs_from()
    rng = np.random.default_rng(seed)
    kwargs.setdefault("dollar_year", 2026)
    return LotSeries(
        quantities=quantities,
        costs=costs * rng.lognormal(0.0, scatter, len(costs)),
        **kwargs,
    )


def test_a_perfect_fit_forecasts_with_no_uncertainty():
    """The degenerate case, asserted rather than left as a surprise. If the
    lots lie exactly on a curve there is nothing left to be uncertain about,
    and the simulation must say so instead of manufacturing spread."""
    sim = analyse_lots(series()).simulate_forecast([40], n_iter=2_000, seed=1)
    assert sim.std == pytest.approx(0.0, abs=1e-6)
    assert sim.p80 == pytest.approx(sim.point_estimate, rel=1e-9)


def test_the_simulation_centres_on_the_deterministic_forecast():
    """The point estimate should sit near the middle of the simulated
    distribution, because the simulation is that same curve with its
    uncertainty attached rather than a different model."""
    sim = analyse_lots(noisy_series()).simulate_forecast([30, 40], n_iter=20_000, seed=1)
    assert sim.point_estimate_percentile == pytest.approx(50.0, abs=4.0)
    assert sim.mean == pytest.approx(sim.point_estimate, rel=0.02)
    assert sim.p50 <= sim.p80 <= sim.p90


def test_the_simulation_agrees_with_the_analytic_interval_on_one_lot():
    """Two independent routes to the same answer. For a single forecast lot
    the delta-method prediction interval and the Monte Carlo have to agree --
    there is no correlation assumption in play to separate them, so a
    disagreement would mean one of the two is wrong."""
    report = analyse_lots(noisy_series())
    analytic = report.forecast([40], level=0.80)
    sim = report.simulate_forecast([40], n_iter=80_000, seed=3)

    assert np.percentile(sim.totals, 10) == pytest.approx(
        analytic["lot_cost_lower"].iloc[0], rel=0.02
    )
    assert np.percentile(sim.totals, 90) == pytest.approx(
        analytic["lot_cost_upper"].iloc[0], rel=0.02
    )


def test_correlated_lot_residuals_widen_the_total():
    """The same lesson as the WBS-level simulator: treating consecutive lots
    as independent lets their shocks cancel and understates the spread of the
    whole buy."""
    report = analyse_lots(noisy_series())
    independent = report.simulate_forecast(
        [30, 40, 40], n_iter=40_000, seed=5, residual_correlation=0.0
    )
    correlated = report.simulate_forecast(
        [30, 40, 40], n_iter=40_000, seed=5, residual_correlation=0.5
    )
    assert correlated.std > independent.std
    assert correlated.p80 > independent.p80


def test_dropping_the_residual_narrows_it_to_a_confidence_statement():
    report = analyse_lots(noisy_series())
    with_residual = report.simulate_forecast([40], n_iter=20_000, seed=7)
    curve_only = report.simulate_forecast(
        [40], n_iter=20_000, seed=7, include_residual=False
    )
    assert curve_only.std < with_residual.std
    assert "confidence statement" in curve_only.narrative()


def test_a_looser_fit_produces_a_wider_forecast_distribution():
    """The whole point of propagating the fit: a curve estimated from scattered
    lots must forecast less confidently than one estimated from clean lots."""
    quantities, costs = lot_costs_from()
    rng = np.random.default_rng(11)
    tight = analyse_lots(LotSeries(
        quantities=quantities, costs=costs * rng.lognormal(0, 0.01, len(costs)),
        dollar_year=2026)).simulate_forecast([40], n_iter=20_000, seed=1)
    loose = analyse_lots(LotSeries(
        quantities=quantities, costs=costs * rng.lognormal(0, 0.15, len(costs)),
        dollar_year=2026)).simulate_forecast([40], n_iter=20_000, seed=1)
    assert loose.cv > tight.cv * 3


def test_the_simulation_is_seed_deterministic():
    report = analyse_lots(noisy_series())
    a = report.simulate_forecast([30, 40], n_iter=5_000, seed=42)
    b = report.simulate_forecast([30, 40], n_iter=5_000, seed=42)
    assert np.array_equal(a.totals, b.totals)
    assert a.p80 == b.p80
    assert report.simulate_forecast([30, 40], n_iter=5_000, seed=43).p80 != a.p80


def test_the_simulation_reports_the_history_it_came_from_not_the_forecast():
    sim = analyse_lots(noisy_series()).simulate_forecast([30, 40], n_iter=2_000, seed=1)
    assert sim.n_history_lots == len(PROFILE)
    assert f"{len(PROFILE)}-lot history" in sim.narrative()


def test_lot_level_samples_add_up_to_the_total():
    sim = analyse_lots(noisy_series()).simulate_forecast([30, 40], n_iter=2_000, seed=1)
    assert sim.per_lot.shape == (2_000, 2)
    assert sim.totals == pytest.approx(sim.per_lot.sum(axis=1), rel=1e-12)


@pytest.mark.parametrize("bad", [[0], [30, -5], []])
def test_bad_forecast_quantities_are_refused(bad):
    with pytest.raises(LotInputError, match="positive"):
        analyse_lots(noisy_series()).simulate_forecast(bad, n_iter=1_000, seed=1)


def test_too_few_iterations_are_refused():
    with pytest.raises(LotInputError, match="at least 2 iterations"):
        analyse_lots(noisy_series()).simulate_forecast([40], n_iter=1, seed=1)
