"""Wright's learning curve: does the maths actually hold?

These are property tests against known answers, not smoke tests. The strongest
one generates data from a curve whose slope we chose, fits it, and checks the
fit recovers that slope -- if the log-log regression or the exponent
conversion were wrong, a round trip like that cannot come back clean.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.learning_curve import (LearningCurveModel, fit_learning_curve,
                                      forecast_costs)


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
