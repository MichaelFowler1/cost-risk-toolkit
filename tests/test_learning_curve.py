"""Learning curves: does the maths actually hold?

These are property tests against known answers, not smoke tests. The strongest
one generates data from a curve whose slope we chose, fits it, and checks the
fit recovers that slope -- if the log-log regression or the exponent
conversion were wrong, a round trip like that cannot come back clean.

The first half covers the original :class:`LearningCurveModel` API. The second
covers the two-theory model added alongside it: Wright's cumulative-average
form and Crawford's unit form, fitted by OLS, MUPE or ZMPE, with rate breaks
and prediction intervals. The identities asserted there -- that doubling
quantity multiplies the *right* quantity by the slope under each theory, that
Wright's unit costs telescope back to its cumulative total, that Crawford's
lot cost is the exact sum of its units -- are definitional. A model that fails
them is not the theory it claims to be.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.fitting import FitError
from cost_core.learning_curve import (CurveModel, LearningCurveModel,
                                      RateBreak, Theory, compare_methods,
                                      compare_theories, comparison_table,
                                      detect_rate_breaks, fit_curve,
                                      fit_from_progress_report,
                                      fit_learning_curve, forecast_costs,
                                      retransformation_report)


def synthetic(slope, t1=1000.0, n=24):
    """Units 1..n priced by an exact Wright curve: cost(x) = T1 * x^log2(slope)."""
    q = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({"unit_quantity": q, "unit_cost": t1 * q ** np.log2(slope)})


# --------------------------------------------------------------- the model
@pytest.mark.parametrize("slope", [0.70, 0.80, 0.85, 0.90, 0.95])
def test_doubling_quantity_multiplies_cost_by_the_slope(slope):
    """This is the definition of a learning curve, so it must hold exactly.

    An 85% curve means the 2nd unit costs 85% of the 1st, the 4th costs 85%
    of the 2nd, and so on. If this fails the model isn't a Wright curve
    whatever the parameters say.
    """
    m = LearningCurveModel(slope=slope, reference_quantity=1.0,
                           reference_cost=1000.0)
    for q in (1.0, 3.0, 7.5, 40.0, 250.0):
        assert m.predict_unit_cost(2 * q) / m.predict_unit_cost(q) == \
            pytest.approx(slope, rel=1e-12)


def test_learning_exponent_is_log2_of_the_slope():
    assert LearningCurveModel(0.85, 1.0, 1.0).learning_exponent == \
        pytest.approx(np.log2(0.85))
    # A 100% curve means no learning: the exponent is zero and cost is flat.
    flat = LearningCurveModel(1.0, 1.0, 500.0)
    assert flat.learning_exponent == pytest.approx(0.0)
    assert flat.predict_unit_cost(1000.0) == pytest.approx(500.0)


def test_the_reference_point_reproduces_its_own_cost():
    m = LearningCurveModel(slope=0.85, reference_quantity=50.0,
                           reference_cost=1234.5)
    assert m.predict_unit_cost(50.0) == pytest.approx(1234.5, rel=1e-12)


# ----------------------------------------------------------------- fitting
@pytest.mark.parametrize("slope", [0.72, 0.85, 0.93])
def test_fitting_recovers_the_slope_it_was_generated_from(slope):
    """Round trip: known curve -> data -> fit -> same slope back."""
    fitted = fit_learning_curve(synthetic(slope))
    assert fitted.slope == pytest.approx(slope, rel=1e-9)


def test_fitting_recovers_the_cost_curve_itself():
    """Not just the slope -- the fitted model must reprice the input data."""
    df = synthetic(0.85, t1=2500.0, n=30)
    m = fit_learning_curve(df)
    predicted = np.array([m.predict_unit_cost(q) for q in df["unit_quantity"]])
    assert np.allclose(predicted, df["unit_cost"].to_numpy(), rtol=1e-9)


def test_fit_is_insensitive_to_row_order():
    df = synthetic(0.85)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert fit_learning_curve(shuffled).slope == \
        pytest.approx(fit_learning_curve(df).slope, rel=1e-12)


def test_noisy_data_still_fits_close_to_the_underlying_curve():
    df = synthetic(0.85, n=60)
    rng = np.random.default_rng(7)
    df["unit_cost"] *= rng.normal(1.0, 0.03, len(df))       # 3% scatter
    assert fit_learning_curve(df).slope == pytest.approx(0.85, abs=0.02)


# -------------------------------------------------------------- validation
def test_a_single_point_cannot_define_a_curve():
    one = pd.DataFrame({"unit_quantity": [1.0], "unit_cost": [100.0]})
    with pytest.raises(ValueError, match="At least 2"):
        fit_learning_curve(one)


@pytest.mark.parametrize("bad", [
    {"unit_quantity": [0.0, 2.0], "unit_cost": [100.0, 85.0]},     # zero qty
    {"unit_quantity": [1.0, 2.0], "unit_cost": [100.0, -85.0]},    # negative cost
])
def test_non_positive_values_are_rejected(bad):
    """log-log regression is undefined at or below zero, so refuse rather
    than quietly emitting nan."""
    with pytest.raises(ValueError, match="positive"):
        fit_learning_curve(pd.DataFrame(bad))


# -------------------------------------------------------------- forecasting
def test_forecast_totals_are_quantity_times_unit_cost():
    m = LearningCurveModel(0.85, 1.0, 1000.0)
    out = forecast_costs(m, [10, 50, 100])
    assert np.allclose(out["total_cost"], out["quantity"] * out["unit_cost"])


def test_forecast_is_sorted_and_unit_costs_fall():
    m = LearningCurveModel(0.85, 1.0, 1000.0)
    out = forecast_costs(m, [100, 10, 50])              # deliberately unsorted
    assert list(out["quantity"]) == sorted(out["quantity"])
    # Learning means later units are cheaper.
    assert out["unit_cost"].is_monotonic_decreasing


# ==========================================================================
# The two theories
# ==========================================================================
SLOPES = [0.72, 0.80, 0.85, 0.90, 0.95]


def lots_from_quantities(quantities):
    """Turn a buy profile into contiguous (first_unit, last_unit) ranges."""
    out, cursor = [], 1
    for q in quantities:
        out.append((cursor, cursor + q - 1))
        cursor += q
    return np.array(out, dtype=int)


PROFILE = (4, 6, 10, 12, 12, 16, 18, 20)
LOTS = lots_from_quantities(PROFILE)


# ------------------------------------------------------------- definitions
@pytest.mark.parametrize("slope", SLOPES)
def test_crawford_doubling_multiplies_the_unit_cost_by_the_slope(slope):
    """Unit theory's defining property, asserted to machine precision."""
    m = CurveModel(Theory.CRAWFORD, 1000.0, np.log2(slope))
    for q in (1.0, 3.0, 7.0, 64.0, 500.0):
        assert m.unit_cost(2 * q)[0] / m.unit_cost(q)[0] == pytest.approx(
            slope, rel=1e-12
        )


@pytest.mark.parametrize("slope", SLOPES)
def test_wright_doubling_multiplies_the_cumulative_average_by_the_slope(slope):
    """Cumulative-average theory's defining property. Note it is a different
    quantity being multiplied -- that difference is the whole distinction."""
    m = CurveModel(Theory.WRIGHT, 1000.0, np.log2(slope))
    for q in (1.0, 3.0, 7.0, 64.0, 500.0):
        assert m.cum_average(2 * q)[0] / m.cum_average(q)[0] == pytest.approx(
            slope, rel=1e-12
        )


def test_both_theories_agree_that_the_first_unit_costs_t1():
    for theory in Theory:
        m = CurveModel(theory, 4_250_000.0, np.log2(0.85))
        assert m.unit_cost(1)[0] == pytest.approx(4_250_000.0, rel=1e-12)
        assert m.cum_average(1)[0] == pytest.approx(4_250_000.0, rel=1e-12)
        assert m.cum_total(1)[0] == pytest.approx(4_250_000.0, rel=1e-12)


def test_wright_total_is_quantity_times_cumulative_average():
    m = CurveModel(Theory.WRIGHT, 1000.0, np.log2(0.85))
    q = np.array([1.0, 5.0, 40.0, 250.0])
    assert m.cum_total(q) == pytest.approx(q * m.cum_average(q), rel=1e-12)


def test_wright_unit_costs_telescope_back_to_the_cumulative_total():
    """Unit cost is defined as the increment in cumulative total, so summing
    the increments must return the total exactly."""
    m = CurveModel(Theory.WRIGHT, 1000.0, np.log2(0.85))
    assert float(np.sum(m.unit_cost(np.arange(1, 61)))) == pytest.approx(
        m.cum_total(60)[0], rel=1e-10
    )


def test_crawford_lot_cost_is_the_exact_sum_of_its_units():
    m = CurveModel(Theory.CRAWFORD, 1000.0, np.log2(0.85))
    assert m.lot_cost([11], [30])[0] == pytest.approx(
        float(np.sum(m.unit_cost(np.arange(11, 31)))), rel=1e-12
    )


def test_wright_closed_form_lot_cost_agrees_with_summing_units():
    """Wright takes a closed-form shortcut for lot cost. It must agree with
    the long way round, or the shortcut is wrong."""
    m = CurveModel(Theory.WRIGHT, 1000.0, np.log2(0.82))
    for first, last in ((1, 4), (5, 10), (33, 71)):
        summed = float(np.sum(m.unit_cost(np.arange(first, last + 1))))
        assert m.lot_cost([first], [last])[0] == pytest.approx(summed, rel=1e-9)


def test_lots_tile_the_whole_production_run():
    """Every lot's cost must add up to the cumulative total, for both theories."""
    for theory in Theory:
        m = CurveModel(theory, 1000.0, np.log2(0.85))
        total = float(np.sum(m.lot_cost(LOTS[:, 0], LOTS[:, 1])))
        assert total == pytest.approx(m.cum_total(sum(PROFILE))[0], rel=1e-9)


def test_the_two_theories_are_genuinely_different():
    """If they agreed, letting the caller choose would be pointless. For the
    same nominal slope Wright's unit costs fall faster, because its *average*
    is what obeys the slope."""
    b = np.log2(0.85)
    crawford = CurveModel(Theory.CRAWFORD, 1000.0, b)
    wright = CurveModel(Theory.WRIGHT, 1000.0, b)
    units = np.arange(2, 101)
    assert np.all(wright.unit_cost(units) < crawford.unit_cost(units))
    # And the totals differ substantially over a real production run.
    ratio = wright.cum_total(100)[0] / crawford.cum_total(100)[0]
    assert ratio < 0.85


def test_a_hundred_percent_curve_means_no_learning():
    for theory in Theory:
        flat = CurveModel(theory, 500.0, 0.0)
        assert flat.slope == pytest.approx(1.0, rel=1e-15)
        assert flat.unit_cost([1, 10, 1000]) == pytest.approx(
            [500.0, 500.0, 500.0], rel=1e-12
        )


def test_unit_indices_below_one_are_refused():
    m = CurveModel(Theory.CRAWFORD, 1000.0, np.log2(0.85))
    with pytest.raises(FitError, match="start at 1"):
        m.unit_cost([0])


def test_a_lot_that_ends_before_it_begins_is_refused():
    m = CurveModel(Theory.CRAWFORD, 1000.0, np.log2(0.85))
    with pytest.raises(FitError, match="cannot end before it begins"):
        m.lot_cost([10], [4])


# ---------------------------------------------------------- round trips
@pytest.mark.parametrize("theory", list(Theory))
@pytest.mark.parametrize("method", ["ols", "mupe", "zmpe"])
def test_lot_fitting_recovers_the_curve_it_was_generated_from(theory, method):
    """The central round trip: build lots from a known curve, fit them back,
    and both parameters have to return. Exact data, so exact recovery."""
    truth = CurveModel(theory, 4_250_000.0, np.log2(0.85))
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])

    fit = fit_curve(theory=theory, method=method, lots=LOTS, lot_costs=totals)
    assert fit.slope == pytest.approx(0.85, rel=1e-6)
    assert fit.t1 == pytest.approx(4_250_000.0, rel=1e-6)


@pytest.mark.parametrize("theory", list(Theory))
def test_unit_fitting_recovers_the_curve_it_was_generated_from(theory):
    """Under Crawford the observations are unit costs; under Wright they are
    cumulative averages. Each theory must recover from its own scale."""
    truth = CurveModel(theory, 1000.0, np.log2(0.88))
    units = np.arange(1, 41, dtype=float)
    costs = (
        truth.unit_cost(units)
        if theory is Theory.CRAWFORD
        else truth.cum_average(units)
    )
    fit = fit_curve(theory=theory, method="ols", units=units, costs=costs)
    assert fit.slope == pytest.approx(0.88, rel=1e-9)
    assert fit.t1 == pytest.approx(1000.0, rel=1e-9)


def test_fitting_the_wrong_theory_gives_a_different_and_wrong_answer():
    """Data generated under Crawford, fitted as Wright. The fit will look
    perfectly healthy and the forecast will be wrong -- which is why the
    theory is an explicit argument rather than a default."""
    truth = CurveModel(Theory.CRAWFORD, 1_000_000.0, np.log2(0.85))
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])

    right = fit_curve(theory=Theory.CRAWFORD, method="ols", lots=LOTS, lot_costs=totals)
    wrong = fit_curve(theory=Theory.WRIGHT, method="ols", lots=LOTS, lot_costs=totals)

    assert right.slope == pytest.approx(0.85, rel=1e-6)
    assert abs(wrong.slope - 0.85) > 0.01

    future = np.array([[109, 160]])
    truth_cost = float(truth.lot_cost(future[:, 0], future[:, 1])[0])
    right_cost = float(right.forecast_lots(future)["lot_cost"].iloc[0])
    wrong_cost = float(wrong.forecast_lots(future)["lot_cost"].iloc[0])
    assert right_cost == pytest.approx(truth_cost, rel=1e-5)
    assert abs(wrong_cost / truth_cost - 1.0) > 0.02


def test_noisy_lot_data_still_recovers_the_slope_closely():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    rng = np.random.default_rng(11)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.05, len(LOTS))
    fit = fit_curve(theory=Theory.CRAWFORD, method="mupe", lots=LOTS, lot_costs=totals)
    assert fit.slope == pytest.approx(0.85, abs=0.02)


# ------------------------------------------------------- methods and bias
def test_mupe_and_zmpe_have_zero_percentage_bias_on_curve_data():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    rng = np.random.default_rng(4)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.18, len(LOTS))

    fits = compare_methods(lots=LOTS, lot_costs=totals)
    for method in ("mupe", "zmpe"):
        assert fits[method].result.mean_percent_error == pytest.approx(
            0.0, abs=1e-8
        ), method
    assert abs(fits["ols"].result.mean_percent_error) > 1e-3


def test_the_retransformation_report_quantifies_the_ols_understatement():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    rng = np.random.default_rng(21)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.25, len(LOTS))

    fits = compare_methods(lots=LOTS, lot_costs=totals)
    bias = retransformation_report(fits)
    assert bias.theoretical_factor > 1.0
    assert bias.percent_understated > 0.0
    assert bias.mupe_ratio > 1.0


def test_the_retransformation_report_needs_an_ols_fit():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])
    fits = {"mupe": fit_curve(method="mupe", lots=LOTS, lot_costs=totals)}
    with pytest.raises(FitError, match="none was"):
        retransformation_report(fits)


def test_comparison_tables_cover_every_variant():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(2)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.1, len(LOTS))

    theories = comparison_table(compare_theories(lots=LOTS, lot_costs=totals))
    assert set(theories["theory"]) == {"wright", "crawford"}

    methods = comparison_table(compare_methods(lots=LOTS, lot_costs=totals))
    assert set(methods["method"]) == {"OLS", "MUPE", "ZMPE"}
    # SE and CV are the headline quality numbers, and both are populated.
    assert (methods["std_error"] > 0).all()
    assert (methods["cv"] > 0).all()


def test_a_high_r_squared_does_not_mean_the_model_is_right():
    """The concrete reason SE and CV lead the reporting instead of R^2.

    Data generated under Crawford, fitted as Wright: the wrong theory, and a
    forecast that is demonstrably off (see the wrong-theory test above). R^2
    still comes out around 0.995, which would pass any eyeball review. R^2
    measures how tightly the points hug the fitted line, and a wrong model can
    do that beautifully -- so it cannot be the number that decides whether a
    model is fit to brief."""
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])

    wrong = fit_curve(theory=Theory.WRIGHT, method="ols", lots=LOTS, lot_costs=totals)
    assert wrong.r_squared > 0.99          # flattering
    assert wrong.slope != pytest.approx(0.85, rel=1e-3)   # and wrong


def test_the_cv_tracks_the_scatter_actually_present():
    """Unlike R^2, the CV is in a unit a reviewer can argue with, and it moves
    with the thing it claims to measure."""
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(6)
    cvs = []
    for noise in (0.03, 0.20):
        totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(
            0.0, noise, len(LOTS)
        )
        cvs.append(fit_curve(lots=LOTS, lot_costs=totals).cv)
    assert cvs[0] < cvs[1]
    assert cvs[1] > 0.10


# ------------------------------------------------------------ rate breaks
def test_a_modelled_rate_break_recovers_its_step_factor():
    """Data with a known 30% step at unit 45. Modelling the break must recover
    both the step and the underlying slope."""
    step, at = 1.30, 45
    truth = CurveModel(
        Theory.CRAWFORD, 4_250_000.0, np.log2(0.85),
        breaks=(RateBreak(at_unit=at, step_factor=step),),
    )
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])

    fit = fit_curve(
        theory=Theory.CRAWFORD, method="ols", lots=LOTS, lot_costs=totals,
        breaks=[RateBreak(at_unit=at)],          # step estimated
    )
    assert fit.model.breaks[0].step_factor == pytest.approx(step, rel=1e-4)
    assert fit.slope == pytest.approx(0.85, rel=1e-4)


def test_ignoring_a_rate_break_biases_the_slope():
    """The reason breaks are worth modelling: an unmodelled step does not
    average out, it tilts the slope, and the tilt propagates into every
    forecast lot."""
    truth = CurveModel(
        Theory.CRAWFORD, 4_250_000.0, np.log2(0.85),
        breaks=(RateBreak(at_unit=45, step_factor=1.30),),
    )
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])

    modelled = fit_curve(lots=LOTS, lot_costs=totals, breaks=[RateBreak(at_unit=45)])
    ignored = fit_curve(lots=LOTS, lot_costs=totals)

    assert modelled.slope == pytest.approx(0.85, rel=1e-4)
    assert ignored.slope > 0.87, "an unmodelled step should flatten the curve"
    assert ignored.cv > modelled.cv


def test_a_production_gap_raises_cost_by_backing_up_the_curve():
    smooth = CurveModel(Theory.CRAWFORD, 1000.0, np.log2(0.85))
    gapped = CurveModel(
        Theory.CRAWFORD, 1000.0, np.log2(0.85),
        breaks=(RateBreak(at_unit=50, learning_loss=0.40),),
    )
    # Before the gap the curves are identical; after it, cost steps back up.
    assert gapped.unit_cost([49])[0] == pytest.approx(smooth.unit_cost([49])[0])
    assert gapped.unit_cost([50])[0] > smooth.unit_cost([50])[0]
    # And the loss is bounded: it can never cost more than the first unit.
    assert gapped.unit_cost([50])[0] < 1000.0


def test_a_break_before_unit_two_is_refused():
    with pytest.raises(FitError, match="unit 2 or later"):
        RateBreak(at_unit=1)


def test_an_impossible_learning_loss_is_refused():
    with pytest.raises(FitError, match="learning_loss"):
        RateBreak(at_unit=10, learning_loss=1.5)


def test_a_negative_step_factor_is_refused():
    with pytest.raises(FitError, match="step_factor must be positive"):
        RateBreak(at_unit=10, step_factor=-1.0)


def test_a_break_beyond_the_data_is_refused():
    """There is no information on both sides of it, so a step there is
    unidentifiable and would silently absorb the last lot."""
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])
    with pytest.raises(FitError, match="beyond the last observed unit"):
        fit_curve(lots=LOTS, lot_costs=totals, breaks=[RateBreak(at_unit=5000)])


def test_break_detection_finds_a_real_step_and_stays_quiet_on_smooth_data():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    smooth_totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1])
    table = pd.DataFrame(
        {"first_unit": LOTS[:, 0], "last_unit": LOTS[:, 1], "lot_cost": smooth_totals}
    )
    assert detect_rate_breaks(table) == []

    shocked = CurveModel(
        Theory.CRAWFORD, 4_250_000.0, np.log2(0.85),
        breaks=(RateBreak(at_unit=45, step_factor=1.35),),
    )
    table["lot_cost"] = shocked.lot_cost(LOTS[:, 0], LOTS[:, 1])
    found = detect_rate_breaks(table)
    assert found, "a 35% step should be visible in the residuals"
    assert any(abs(b.at_unit - 45) <= 20 for b in found)


# --------------------------------------------------------------- intervals
def test_forecast_prediction_intervals_are_wider_than_confidence_intervals():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    rng = np.random.default_rng(3)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.15, len(LOTS))
    fit = fit_curve(lots=LOTS, lot_costs=totals)

    future = np.array([[109, 130], [131, 160]])
    pred = fit.forecast_lots(future, kind="prediction")
    conf = fit.forecast_lots(future, kind="confidence")
    assert np.all(
        (pred["lot_cost_upper"] - pred["lot_cost_lower"])
        > (conf["lot_cost_upper"] - conf["lot_cost_lower"])
    )


def test_forecast_lot_cost_is_lot_average_times_quantity():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(8)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.1, len(LOTS))
    fit = fit_curve(lots=LOTS, lot_costs=totals)

    out = fit.forecast_lots(np.array([[109, 130], [131, 160]]))
    assert out["lot_cost"].to_numpy() == pytest.approx(
        (out["lot_average"] * out["quantity"]).to_numpy(), rel=1e-12
    )
    assert (out["lot_cost_lower"] < out["lot_cost"]).all()
    assert (out["lot_cost_upper"] > out["lot_cost"]).all()


def test_forecast_intervals_widen_further_into_the_future():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(9)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.15, len(LOTS))
    fit = fit_curve(lots=LOTS, lot_costs=totals)

    near = fit.forecast_lots(np.array([[109, 120]]), kind="confidence")
    far = fit.forecast_lots(np.array([[900, 911]]), kind="confidence")
    near_width = float(
        near["lot_average_upper"].iloc[0] / near["lot_average_lower"].iloc[0]
    )
    far_width = float(
        far["lot_average_upper"].iloc[0] / far["lot_average_lower"].iloc[0]
    )
    assert far_width > near_width


def test_the_slope_interval_brackets_the_point_estimate():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(10)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.12, len(LOTS))
    fit = fit_curve(lots=LOTS, lot_costs=totals)
    lo, hi = fit.slope_interval
    assert lo < fit.slope < hi


# ------------------------------------------------------------- integration
def test_fitting_straight_from_the_synthetic_progress_report():
    """The whole path: generate a clean program, ingest it, fit the 1921-2.
    The generator prices lots with exact Crawford unit theory, so the fit has
    to come back with the generating slope exactly."""
    from cost_core.ingest import normalize_program
    from cost_core.synth import PathologyConfig, generate_program

    program = generate_program(seed=3, pathologies=PathologyConfig.clean())
    data = normalize_program(program)
    fit = fit_from_progress_report(data.learning_curve_input(), theory=Theory.CRAWFORD)

    assert fit.slope == pytest.approx(program.truth.learning_slope, rel=1e-6)
    assert fit.t1 == pytest.approx(program.truth.t1_cost, rel=1e-6)


def test_fitting_a_messy_program_lands_near_the_truth():
    """Scatter, rate breaks and a quantity rebaseline in the data. MUPE should
    still land close, and the CV should be honest about the spread."""
    from cost_core.ingest import normalize_program
    from cost_core.synth import generate_program

    program = generate_program(seed=3)
    data = normalize_program(program)
    fit = fit_from_progress_report(
        data.learning_curve_input(), theory=Theory.CRAWFORD, method="mupe"
    )
    assert fit.slope == pytest.approx(program.truth.learning_slope, abs=0.06)
    assert fit.cv > 0.0


def test_a_progress_table_missing_its_columns_is_refused():
    with pytest.raises(FitError, match="missing column"):
        fit_from_progress_report(pd.DataFrame({"lot": [1, 2], "cost": [1.0, 2.0]}))


# ------------------------------------------------------------- bad input
def test_supplying_both_unit_and_lot_data_is_refused():
    with pytest.raises(FitError, match="exactly one of"):
        fit_curve(
            units=[1, 2, 3], costs=[100.0, 85.0, 80.0],
            lots=LOTS, lot_costs=np.ones(len(LOTS)),
        )


def test_supplying_neither_is_refused():
    with pytest.raises(FitError, match="exactly one of"):
        fit_curve(theory=Theory.CRAWFORD, method="ols")


def test_mismatched_lengths_are_refused():
    with pytest.raises(FitError, match="one to one"):
        fit_curve(units=[1, 2, 3], costs=[100.0, 85.0])
    with pytest.raises(FitError, match="lots and"):
        fit_curve(lots=LOTS, lot_costs=[1.0, 2.0])


def test_an_unknown_theory_is_refused():
    with pytest.raises(FitError, match="Unknown learning-curve theory"):
        fit_curve(theory="boeing", units=[1, 2, 3], costs=[100.0, 85.0, 80.0])


def test_an_unknown_method_is_refused():
    with pytest.raises(FitError, match="Unknown method"):
        fit_curve(method="eyeball", units=[1, 2, 3], costs=[100.0, 85.0, 80.0])


def test_a_malformed_lot_table_is_refused():
    with pytest.raises(FitError, match="n, 2"):
        fit_curve(lots=np.array([1, 2, 3]), lot_costs=[1.0])
    with pytest.raises(FitError, match="missing column"):
        fit_curve(lots=pd.DataFrame({"lot": [1, 2]}), lot_costs=[1.0, 2.0])


# ------------------------------------------------------- legacy interop
def test_the_new_fit_converts_to_the_original_model_type():
    truth = CurveModel(Theory.CRAWFORD, 4_250_000.0, np.log2(0.85))
    units = np.arange(1, 31, dtype=float)
    fit = fit_curve(theory=Theory.CRAWFORD, units=units, costs=truth.unit_cost(units))

    legacy = fit.to_legacy_model()
    assert isinstance(legacy, LearningCurveModel)
    assert legacy.slope == pytest.approx(0.85, rel=1e-9)
    # And it prices units the same way the new model does.
    assert legacy.predict_unit_cost(20.0) == pytest.approx(
        fit.model.unit_cost([20.0])[0], rel=1e-9
    )


def test_the_original_api_still_behaves_as_it_did():
    """The new code must not have disturbed the old entry points."""
    data = synthetic(0.85, t1=2500.0, n=30)
    model = fit_learning_curve(data)
    assert model.slope == pytest.approx(0.85, rel=1e-9)
    assert len(forecast_costs(model, [32, 64])) == 2


def test_curve_fitting_is_deterministic():
    truth = CurveModel(Theory.CRAWFORD, 1e6, np.log2(0.85))
    rng = np.random.default_rng(1)
    totals = truth.lot_cost(LOTS[:, 0], LOTS[:, 1]) * rng.lognormal(0.0, 0.15, len(LOTS))
    for method in ("ols", "mupe", "zmpe"):
        a = fit_curve(method=method, lots=LOTS, lot_costs=totals)
        b = fit_curve(method=method, lots=LOTS, lot_costs=totals)
        assert a.slope == b.slope, method
        assert a.t1 == b.t1, method
