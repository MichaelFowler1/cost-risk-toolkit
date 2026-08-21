"""The lot-entry front door: units and cost, one row per lot.

This is the path a user takes with real production history, so the tests care
most about the ways that input goes wrong quietly: nonrecurring cost folded
into the lot totals, escalation left in dollars declared constant, and lots
that do not start at unit 1. Each of those moves the fitted slope by several
points while leaving a fit that looks entirely healthy.

The fitting itself is the lot cost model in :mod:`cost_core.lotmodel` -- three
candidate models against the lot midpoint, one selected on the significance of
the rate term with an AICc tiebreak. The reference program below is the one the
desktop tool ships as its example, so these tests double as a check that the
command line and the window give the same answer for the same lots.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from cost_core.lots import (COMFORTABLE_LOTS, LotInputError, LotSeries,
                            analyse_lots, build_assumption_log)


# The input-layer tests below were written against the earlier fitting code and
# are unchanged: they check validation, file reading and the escalation checks,
# none of which moved. These three helpers give them their data, now drawn from
# the same reference program the fitting tests use.
PROFILE = [8, 16, 24, 24, 18, 18]


def lot_costs_from():
    """The reference program as (quantities, lot totals)."""
    return (np.array(REF_QTY, dtype=int),
            np.array([q * a for q, a in zip(REF_QTY, REF_AUC)], dtype=float))


def series(**kwargs) -> LotSeries:
    return reference_series(**kwargs)


def clean_series(**kwargs) -> LotSeries:
    """Lots sitting exactly on an 85% curve, priced at their own midpoints.

    Monotone by construction, so the escalation checks have nothing to find.
    """
    from cost_core.lotmodel.mathx import lmp_func

    t1, b = 5_000.0, np.log2(0.85)
    quantities = [20, 20, 25, 25, 30, 30]
    cursor, costs = 1, []
    for q in quantities:
        mid = lmp_func(cursor, cursor + q - 1, q, b)
        costs.append(t1 * mid ** b * q)
        cursor += q
    kwargs.setdefault("dollar_year", 2026)
    return LotSeries(quantities=quantities, costs=costs, **kwargs)


def escalated(rate: float) -> LotSeries:
    """The reference program with `rate` a year of escalation left in it."""
    quantities, costs = lot_costs_from()
    return LotSeries(
        quantities=quantities,
        costs=costs * np.array([(1.0 + rate) ** i for i in range(len(costs))]),
        dollar_year=2026,
    )



# ============================================================== round trips


def test_lots_tile_the_unit_sequence_without_gaps_or_overlap():
    s = series()
    ranges = s.unit_ranges()
    assert ranges[0, 0] == 1
    assert list(ranges[1:, 0]) == list(ranges[:-1, 1] + 1)
    assert ranges[-1, 1] == sum(PROFILE)
    assert list(ranges[:, 1] - ranges[:, 0] + 1) == PROFILE


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


def test_a_rising_cumulative_average_is_flagged():
    """Severe escalation turns the cumulative average upward, which cannot
    happen on a learning curve."""
    findings = escalated(0.15).check_constant_dollars(warn=False)
    assert findings
    assert "RISES" in findings[0]
    assert "then-year" in findings[0]


def test_a_clean_series_raises_no_escalation_finding():
    assert clean_series().check_constant_dollars(warn=False) == []


def test_the_reference_programme_is_not_clean_and_says_so():
    """Real production history is bumpy. Two lots in the reference programme
    tick upward in unit cost, and the check notices -- which is the point of
    having it, and why the clean fixture above exists separately."""
    findings = series().check_constant_dollars(warn=False)
    assert findings
    assert "individual lot average cost rises" in findings[0]


def test_the_cost_basis_must_be_declared():
    quantities, costs = lot_costs_from()
    with pytest.raises(LotInputError, match="cost_basis must be"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=2026,
                  cost_basis="whatever")


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


def test_the_priced_plan_says_which_program_it_borrowed_from():
    """Two tables in one folder, one fitted to six lots and one pricing five,
    with nothing to distinguish them, reads as a truncation bug."""
    report = analyse_lots(series(program="SOURCE PROGRAM"))
    priced = report.price_lot_plan([10, 15, 20, 25, 30])
    assert (priced["priced_by_analogy_from"] == "SOURCE PROGRAM").all()
    assert (priced["source_lots_fitted"] == len(PROFILE)).all()
    assert len(priced) == 5          # the plan, not the source programme


@pytest.mark.parametrize("scatter", [0.05, 0.10, 0.30])
def test_mupe_and_zmpe_stay_at_zero_however_scattered_the_data(scatter):
    """The distinction that matters: OLS bias grows with scatter, theirs does
    not move off zero. At 30% scatter OLS is around 3% and MUPE is 1e-10%."""
    from cost_core.learning_curve import compare_methods

    quantities, costs = lot_costs_from()
    rng = np.random.default_rng(4)
    lots = LotSeries(
        quantities=quantities,
        costs=costs * rng.lognormal(0.0, scatter, len(costs)),
        dollar_year=2026,
    )
    fits = compare_methods(lots=lots.unit_ranges(), lot_costs=lots.costs)

    ols = abs(fits["ols"].result.mean_percent_error)
    for method in ("mupe", "zmpe"):
        bias = abs(fits[method].result.mean_percent_error)
        assert bias < 1e-8, method
        # Orders of magnitude apart at every scatter level, so the claim that
        # MUPE drives the bias to zero is not an artefact of tight data.
        assert ols > bias * 1_000




# ==========================================================================
# The three-model fit on the lot midpoint
# ==========================================================================
# The reference program is the invented one the desktop tool ships as its
# example, so these double as a check that the command line and the window
# agree. The analogy lots are given as unit costs there and as lot totals here,
# which is the same data: cost = quantity x unit cost.
REF_QTY = [8, 16, 24, 24, 18, 18]
REF_AUC = [3120.00, 2585.50, 2402.75, 2438.10, 2310.40, 2266.85]


def reference_series(**kwargs) -> LotSeries:
    kwargs.setdefault("dollar_year", 2026)
    kwargs.setdefault("program", "TEST")
    return LotSeries(
        quantities=REF_QTY,
        costs=[q * a for q, a in zip(REF_QTY, REF_AUC)],
        **kwargs,
    )


@pytest.fixture(scope="module")
def reference():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analyse_lots(reference_series())


def test_the_command_line_reproduces_the_desktop_tool(reference):
    """The same lots through this module and through the desktop tool have to
    give the same answer, because they are the same engine. These are the
    numbers the original spreadsheet-replacement script produced."""
    fit = reference.fit
    assert reference.selected_model == "LC"
    # T1 and b are the raw coefficients the original script held in memory.
    assert fit.t1 == pytest.approx(3433.6272850614305, rel=1e-12)
    assert fit.b == pytest.approx(-0.09122296708531767, rel=1e-12)
    # The rest are what it printed, so they are asserted to the precision it
    # printed them at rather than to a precision only this port can supply.
    assert fit.slope == pytest.approx(0.9387, abs=5e-5)
    assert fit.sigma == pytest.approx(0.0296, abs=5e-5)
    assert fit.r_squared == pytest.approx(0.9487, abs=5e-5)
    assert fit.df == 4


def test_all_three_models_are_fitted_and_reported(reference):
    """Selection picks one, but the alternatives stay on the record: the first
    question a reviewer asks is what the other two said."""
    table = reference.model_comparison().set_index("Item")
    assert set(table.columns) == {"LC", "Rate", "LC+Rate"}
    assert table.loc["Fitted"].tolist() == ["Yes", "Yes", "Yes"]
    assert table.loc["SELECTED", "LC"] == "YES"
    # Rate alone explains far less of this data than the learning curve does.
    assert table.loc["R2 (log)", "LC"] == "0.9487"
    assert table.loc["R2 (log)", "Rate"] == "0.7291"


def test_the_selection_says_why(reference):
    assert "not significant" in reference.fit.selection_note
    assert reference.fit.selection_note.startswith("Default model")


def test_the_fit_recovers_a_curve_built_on_lot_midpoints():
    """Round trip through the engine's own geometry: price lots from a known
    LC model at their midpoints, feed the lot totals back in, and the slope and
    first-unit cost have to return."""
    from cost_core.lotmodel.mathx import lmp_func

    t1, b = 5_000.0, np.log2(0.85)
    quantities = [10, 15, 20, 25, 30, 35]
    cursor, costs = 1, []
    for q in quantities:
        mid = lmp_func(cursor, cursor + q - 1, q, b)
        costs.append(t1 * mid ** b * q)          # unit cost at the midpoint x qty
        cursor += q

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = LotSeries(quantities=quantities, costs=costs,
                        dollar_year=2026).fit()
    assert fit.selected_model == "LC"
    assert fit.t1 == pytest.approx(t1, rel=1e-6)
    assert fit.slope == pytest.approx(0.85, rel=1e-6)


def test_a_rate_driven_programme_selects_a_rate_term():
    """Where cost depends on how many units are bought at once rather than on
    how many have been built, the selection rule should say so."""
    rng = np.random.default_rng(3)
    quantities = [40, 5, 38, 6, 36, 8, 34, 10]
    # Unit cost driven purely by lot size, with no learning at all.
    costs = [q * (3_000.0 * q ** -0.35) * rng.lognormal(0, 0.01)
             for q in quantities]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = LotSeries(quantities=quantities, costs=costs,
                        dollar_year=2026).fit()
    assert fit.selected_model in ("Rate", "LC+Rate")
    assert fit.c is not None
    assert fit.rate_slope is not None and fit.rate_slope < 1.0


def test_a_stricter_t_gate_pushes_selection_back_to_the_learning_curve():
    """The gate is a real control, not decoration: raise it far enough and a
    marginal rate term stops qualifying."""
    rng = np.random.default_rng(3)
    quantities = [40, 5, 38, 6, 36, 8, 34, 10]
    costs = [q * (3_000.0 * q ** -0.35) * rng.lognormal(0, 0.01)
             for q in quantities]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        series = LotSeries(quantities=quantities, costs=costs, dollar_year=2026)
        loose = series.fit(t_gate=2.0)
        strict = series.fit(t_gate=500.0)
    assert loose.selected_model in ("Rate", "LC+Rate")
    assert strict.selected_model == "LC"


def test_the_lot_midpoint_lies_inside_its_own_lot(reference):
    """A midpoint outside the lot it prices would be meaningless."""
    per_lot = reference.per_lot
    assert (per_lot["lot_midpoint"] >= per_lot["first_unit"] - 0.5).all()
    assert (per_lot["lot_midpoint"] <= per_lot["last_unit"] + 0.5).all()
    # And it advances with the production run.
    assert per_lot["lot_midpoint"].is_monotonic_increasing


def test_a_prior_buy_shifts_every_midpoint_and_changes_the_answer():
    """If the programme already built units, lot 1 is not unit 1. Getting that
    wrong makes T1 the cost of a unit nobody built."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = reference_series().fit()
        experienced = reference_series(first_unit=101).fit()
    assert experienced.t1 > fresh.t1 * 1.2
    assert experienced.slope != pytest.approx(fresh.slope, rel=1e-6)


def test_the_per_lot_table_names_the_lot_the_model_misses(reference):
    per_lot = reference.per_lot
    assert len(per_lot) == 6
    # Lots 2 and 4 sit either side of the curve about 3.2% out, and within a
    # rounding error of each other, so which of the two is nominally "worst"
    # is not a distinction worth asserting on. What has to hold is that the
    # table separates the lots the model misses from the ones it gets right.
    ranked = per_lot.reindex(
        per_lot["percent_error"].abs().sort_values(ascending=False).index)
    assert set(ranked["lot"].head(2)) == {"Lot 2", "Lot 4"}
    assert ranked["percent_error"].abs().head(2).min() > 3.0
    assert ranked["percent_error"].abs().tail(2).max() < 1.0
    assert ranked["lot"].iloc[0] in reference.narrative()
    # Fitted lot cost is the fitted unit cost times the quantity.
    assert per_lot["fitted_lot_cost"].to_numpy() == pytest.approx(
        (per_lot["fitted_unit_cost"] * per_lot["units"]).to_numpy(), rel=1e-12)


def test_the_equation_reproduces_the_model_when_evaluated_by_hand(reference):
    """The equation is only worth publishing if someone can retype it and get
    the same numbers, so the test does exactly that."""
    import re

    text = reference.equation()
    t1 = float(re.search(r"= ([\d,\.]+) \*", text).group(1).replace(",", ""))
    b = float(re.search(r"midpoint\^\(([-\d\.]+)\)", text).group(1))

    per_lot = reference.per_lot
    by_hand = t1 * per_lot["lot_midpoint"].to_numpy() ** b
    assert by_hand == pytest.approx(
        per_lot["fitted_unit_cost"].to_numpy(), rel=1e-4)


def test_the_summary_puts_r_squared_last(reference):
    stats_listed = reference.summary()["statistic"].tolist()
    assert stats_listed[-1] == "r_squared_read_last"
    assert stats_listed.index("selected_model") == 0
    assert "ols_understates_mean_pct" in stats_listed


# ============================================ the statistics layered on top
def test_mupe_and_zmpe_drive_the_percentage_bias_to_zero(reference):
    frame = reference.methods().frame.set_index("Method")
    assert abs(frame.loc["MUPE", "Mean % error"]) < 1e-9
    assert abs(frame.loc["ZMPE", "Mean % error"]) < 1e-9
    assert frame.loc["OLS", "Mean % error"] > 0


def test_the_retransformation_bias_is_measured_on_this_data(reference):
    methods = reference.methods()
    assert methods.theoretical_factor == pytest.approx(
        np.exp(methods.log_residual_variance / 2.0), rel=1e-12)
    assert methods.percent_understated > 0
    assert methods.mupe_over_ols > 1.0


def test_the_influence_table_flags_the_lot_that_sets_the_fit(reference):
    """On this reference programme the nine-unit first lot carries leverage
    above 0.8. Nothing in the original summary said so."""
    influence = reference.influence()
    assert len(influence) == 6
    assert influence["Leverage"].sum() == pytest.approx(2.0, rel=1e-9)
    first = influence.iloc[0]
    assert first["Leverage"] > 0.5
    assert bool(first["Influential"])
    assert any("setting this fit" in note for note in reference.diagnostics())


def test_every_priced_lot_gets_a_prediction_interval(reference):
    intervals = reference.intervals()
    assert len(intervals) == 6            # no forecast given: the fitted lots
    assert (intervals["Lot Cost Lower"] < intervals["Lot Cost ($)"]).all()
    assert (intervals["Lot Cost Upper"] > intervals["Lot Cost ($)"]).all()
    assert (intervals["Kind"] == "prediction").all()


def test_a_forecast_continues_from_the_last_unit_built(reference):
    forecast = reference.forecast([14, 14])
    assert len(forecast) == 2
    # Later lots are cheaper per unit, because learning continues.
    assert forecast["Unit Cost ($K)"].iloc[1] < forecast["Unit Cost ($K)"].iloc[0]
    # And below what the model says the last historical lot cost, which is
    # the like-for-like comparison. The last lot's *actual* came in under the
    # curve, so comparing against that would be comparing a fitted value to a
    # residual.
    last_fitted = reference.per_lot["fitted_unit_cost"].iloc[-1]
    assert forecast["Unit Cost ($K)"].iloc[0] < last_fitted


def test_the_buy_simulation_centres_on_the_priced_total(reference):
    risk = reference.simulate(n_iter=20_000, seed=1)
    assert risk.point_estimate == pytest.approx(
        reference.fit.projections["LC Lot Cost After Complexity ($)"].sum(),
        rel=1e-12)
    assert 30.0 < risk.point_estimate_percentile < 70.0
    assert risk.p50 <= risk.p80 <= risk.p90


def test_the_buy_simulation_is_seed_deterministic(reference):
    """A P80 that moves between runs is not a number anyone can defend."""
    a = reference.simulate(n_iter=5_000, seed=42)
    b = reference.simulate(n_iter=5_000, seed=42)
    assert np.array_equal(a.totals, b.totals)
    assert reference.simulate(n_iter=5_000, seed=43).p80 != a.p80


def test_pricing_a_lot_plan_starts_at_unit_one(reference):
    """The model used as an estimating relationship rather than as a forecast
    of its own programme."""
    priced = reference.price_lot_plan([10, 15, 20])
    assert priced["first_unit"].iloc[0] == 1
    assert list(priced["first_unit"][1:]) == list(priced["last_unit"][:-1] + 1)
    assert priced["cumulative_units"].iloc[-1] == 45
    assert (priced["priced_by_analogy_from"] == "TEST").all()
    assert (priced["source_lots_fitted"] == 6).all()


def test_pricing_from_a_later_unit_is_cheaper(reference):
    fresh = reference.price_lot_plan([20], first_unit=1)
    experienced = reference.price_lot_plan([20], first_unit=201)
    assert experienced["lot_cost"].iloc[0] < fresh["lot_cost"].iloc[0]


@pytest.mark.parametrize("bad", [[0], [10, -5], []])
def test_a_bad_lot_plan_is_refused(reference, bad):
    with pytest.raises(LotInputError, match="positive whole units"):
        reference.price_lot_plan(bad)


def test_pricing_from_unit_zero_is_refused(reference):
    with pytest.raises(LotInputError, match="first_unit must be"):
        reference.price_lot_plan([10], first_unit=0)


# ==================================================== small-sample handling
def test_two_lots_are_refused():
    with pytest.raises(LotInputError, match="cannot support a learning curve"):
        LotSeries(quantities=[20, 25], costs=[1e6, 1.2e6],
                  dollar_year=2026).fit()


def test_three_lots_fit_but_warn():
    with pytest.warns(RuntimeWarning, match="degree"):
        fit = LotSeries(quantities=[20, 25, 30],
                        costs=[6.0e4, 7.0e4, 8.0e4], dollar_year=2026).fit()
    assert fit.df == 1


def test_a_small_sample_can_be_made_fatal():
    with pytest.raises(LotInputError, match="degree"):
        LotSeries(quantities=[20, 25, 30], costs=[6.0e4, 7.0e4, 8.0e4],
                  dollar_year=2026).fit(allow_small_sample=False)


def test_a_comfortable_sample_does_not_warn_about_size():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clean_series().fit()


def test_declaring_total_cost_warns_loudly():
    """Nonrecurring is front-loaded, so including it reads as steeper learning
    and overstates future savings."""
    with pytest.warns(RuntimeWarning, match="TOTAL cost"):
        reference_series(cost_basis="total").fit()


def test_recurring_cost_fits_without_that_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        clean_series(cost_basis="recurring").fit()


# ================================================== escalation diagnostics
def test_fitting_warns_when_the_cumulative_average_rises():
    quantities = [20, 20, 25, 25, 30, 30]
    costs = [q * 1_000.0 * (1.20 ** i) for i, q in enumerate(quantities)]
    with pytest.warns(RuntimeWarning, match="RISES"):
        LotSeries(quantities=quantities, costs=costs, dollar_year=2026).fit()


def escalate(series_in: LotSeries, rate: float) -> LotSeries:
    """The same lots with `rate` a year of escalation left in the costs."""
    return LotSeries(
        quantities=series_in.quantities,
        costs=series_in.costs * np.array(
            [(1.0 + rate) ** i for i in range(series_in.n_lots)]),
        dollar_year=series_in.dollar_year,
    )


@pytest.mark.parametrize("rate", [0.02, 0.04, 0.06])
def test_moderate_escalation_goes_undetected_under_a_midpoint_fit(rate):
    """Documents a real limit rather than a capability.

    Under an exact-lot-average fit, escalation left in constant dollars bends
    the residuals and is detectable from around 2% a year. Under a midpoint
    fit it is not: the fitted slope moves with the escalation, the midpoint
    moves with the slope, and the trend is absorbed instead of being left in
    the residuals. Both checks stay silent while the slope is several points
    wrong, and the assumptions log says so.
    """
    base = clean_series()
    dirty = escalate(base, rate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyse_lots(dirty)

    assert dirty.check_constant_dollars(warn=False) == []     # level: silent
    assert report.check_curve_shape(warn=False) == []         # curvature: silent
    # And the damage it does meanwhile: the true slope is 85%.
    assert report.fit.slope > 0.855


def test_severe_escalation_is_caught_by_the_level_check():
    """What does work. Once escalation overwhelms learning the cumulative
    average turns upward, which cannot happen on a learning curve."""
    dirty = escalate(clean_series(), 0.15)
    findings = dirty.check_constant_dollars(warn=False)
    assert findings
    assert "RISES" in findings[0]
    assert "then-year" in findings[0]


def test_a_perfectly_fitted_series_has_nothing_to_test_for_curvature():
    """Residuals at the floating-point floor carry only rounding. Fitting a
    quadratic to those returns a large t from nothing at all, so the check
    reports that there is nothing to test instead."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyse_lots(clean_series())
    assert not np.isfinite(report.curvature()[1])
    assert report.check_curve_shape(warn=False) == []


def test_the_curvature_test_is_quiet_on_the_reference_programme(reference):
    """Bumpy real data, but not systematically bent."""
    assert reference.check_curve_shape(warn=False) == []


def test_curvature_cannot_be_tested_on_three_lots():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyse_lots(LotSeries(quantities=[20, 25, 30],
                                        costs=[6.0e4, 7.0e4, 8.0e4],
                                        dollar_year=2026))
    assert not np.isfinite(report.curvature()[1])
    findings = report.check_curve_shape(warn=False)
    assert findings and "too few to test the shape" in findings[0]


# =============================================================== the log
def test_the_log_records_the_dollar_basis_and_the_selection(reference):
    text = build_assumption_log(reference, source="history.csv").render()
    assert "constant FY2026" in text
    assert "no inflation index applied" in text.lower()
    assert "history.csv" in text
    assert "LC" in text
    assert "Retransformation bias" in text
    assert "Influence" in text
    for characteristic in ("Comprehensive", "Well-documented", "Accurate",
                           "Credible"):
        assert f"### {characteristic}" in text


def test_a_priced_plan_is_recorded_in_the_log_as_an_assumption(reference):
    """Analogy is a judgement, not a result. The log has to say so."""
    priced = reference.price_lot_plan([10, 20, 30])
    text = build_assumption_log(reference, priced_plan=priced).render()
    assert "analogy" in text.lower()
    assert "Analyst judgement" in text
    assert "not included in any interval" in text


def test_the_whole_path_is_deterministic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = analyse_lots(reference_series())
        b = analyse_lots(reference_series())
    assert a.fit.t1 == b.fit.t1
    assert a.fit.b == b.fit.b
    pd.testing.assert_frame_equal(a.per_lot, b.per_lot)


def test_a_two_column_csv_reads_straight_in(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"units": REF_QTY,
                  "cost": [q * a for q, a in zip(REF_QTY, REF_AUC)]}).to_csv(
        path, index=False)
    series = LotSeries.read(path, dollar_year=2026)
    assert series.n_lots == 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert series.fit().slope == pytest.approx(0.9387, abs=5e-5)


def test_a_lot_label_column_is_carried_through(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"lot": ["LRIP 1", "LRIP 2", "FRP 1", "FRP 2"],
                  "units": [10, 12, 20, 25],
                  "cost": [5e4, 5.4e4, 8e4, 9.4e4]}).to_csv(path, index=False)
    series = LotSeries.read(path, dollar_year=2026)
    assert series.labels == ("LRIP 1", "LRIP 2", "FRP 1", "FRP 2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyse_lots(series)
    assert "LRIP 1" in report.per_lot["lot"].tolist()
    assert "LRIP 1" in report.influence()["Lot"].tolist()
