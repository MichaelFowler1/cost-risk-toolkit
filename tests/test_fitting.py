"""The shared estimation engine: OLS, MUPE, ZMPE, and the intervals.

Everything a learning curve or a CER reports about its own uncertainty comes
from this module, so the tests here are identity checks against closed forms
rather than comparisons to recorded output.

The three that matter most:

* our log-log OLS *is* the textbook OLS -- same coefficients, same sigma, same
  standard errors as ``scipy.stats.linregress`` and the normal equations;
* MUPE and ZMPE drive the mean percentage error to exactly zero, which is the
  entire reason for using them;
* the delta-method prediction interval reduces algebraically to the textbook
  ``s * sqrt(1 + 1/n + (x0-xbar)^2/Sxx)`` formula, so the generic machinery is
  provably not an approximation in the case where an approximation would be
  noticed.
"""
import numpy as np
import pytest
from scipy import stats

from cost_core.fitting import (MIN_COMFORTABLE_DF, FitError, ModelSpec, fit,
                               fit_all_methods, predict_with_interval,
                               retransformation_bias)

# --------------------------------------------------------------- fixtures
def _power_predict(th, X):
    return np.exp(th[0]) * np.asarray(X, dtype=float) ** th[1]


POWER = ModelSpec(
    name="power",
    param_names=("log_a", "b"),
    predict=_power_predict,
    link="log",
    log_scale_index=0,
    initial=lambda X, y: np.array([np.log(np.mean(y)), -0.2]),
    # d f / d(log a) = f ;  d f / db = f * ln(x)
    jacobian=lambda th, X: np.column_stack(
        [_power_predict(th, X), _power_predict(th, X) * np.log(np.asarray(X, float))]
    ),
)

POWER_NUMERIC = ModelSpec(  # same model, no analytic derivative supplied
    name="power-numeric",
    param_names=POWER.param_names,
    predict=_power_predict,
    link="log",
    log_scale_index=0,
    initial=POWER.initial,
)


def linear_spec(k: int) -> ModelSpec:
    """y = b0 + sum b_j x_j, additive errors."""
    return ModelSpec(
        name="linear",
        param_names=("b0", *(f"b{i + 1}" for i in range(k))),
        predict=lambda th, X: th[0] + np.asarray(X, dtype=float) @ np.asarray(th[1:]),
        link="identity",
        initial=lambda X, y: np.concatenate([[np.mean(y)], np.zeros(k)]),
        jacobian=lambda th, X: np.column_stack(
            [np.ones(np.asarray(X, dtype=float).shape[0]), np.asarray(X, dtype=float)]
        ),
    )


def power_data(n=20, t1=1000.0, slope=0.85, sigma=0.25, seed=0):
    """A power curve with exact lognormal multiplicative scatter."""
    rng = np.random.default_rng(seed)
    x = np.arange(1, n + 1, dtype=float)
    y = t1 * x ** np.log2(slope) * np.exp(rng.normal(0.0, sigma, n))
    return x, y


def closed_form_loglog(x, y):
    """Textbook log-log OLS, computed independently of the module under test."""
    lx, ly = np.log(x), np.log(y)
    reg = stats.linregress(lx, ly)
    n = len(y)
    resid = ly - (reg.intercept + reg.slope * lx)
    s = np.sqrt(np.sum(resid**2) / (n - 2))
    sxx = np.sum((lx - lx.mean()) ** 2)
    return reg, s, sxx, lx.mean(), n


# ============================================================== OLS is OLS
def test_log_log_ols_reproduces_the_textbook_regression_exactly():
    """If the generic engine did not reduce to linregress on a log-linear
    model, nothing else it reports could be trusted."""
    x, y = power_data()
    reg, s, sxx, _, _ = closed_form_loglog(x, y)
    f = fit(POWER, x, y, "ols")

    assert f.theta[1] == pytest.approx(reg.slope, rel=1e-12)
    assert f.theta[0] == pytest.approx(reg.intercept, rel=1e-12)
    assert f.sigma == pytest.approx(s, rel=1e-12)
    assert f.param_se["b"] == pytest.approx(s / np.sqrt(sxx), rel=1e-10)


def test_linear_ols_reproduces_the_normal_equations():
    rng = np.random.default_rng(3)
    x = np.arange(1, 21, dtype=float)
    design = np.column_stack([x, np.sqrt(x)])
    y = 50.0 + 3.0 * x - 2.0 * np.sqrt(x) + rng.normal(0.0, 4.0, x.size)

    f = fit(linear_spec(2), design, y, "ols")

    full = np.column_stack([np.ones(x.size), design])
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    s = np.sqrt(np.sum((y - full @ beta) ** 2) / (x.size - 3))

    assert f.theta == pytest.approx(beta, rel=1e-10)
    assert f.sigma == pytest.approx(s, rel=1e-12)
    # Covariance must be s^2 (X'X)^-1, not merely something of the right size.
    assert f.cov == pytest.approx(s**2 * np.linalg.inv(full.T @ full), rel=1e-8)


def test_the_numerical_jacobian_agrees_with_the_analytic_one():
    """Most specs in the library supply a derivative; anything a caller builds
    by hand falls back to central differences. The two paths must not give
    materially different standard errors."""
    x, y = power_data()
    exact = fit(POWER, x, y, "ols")
    numeric = fit(POWER_NUMERIC, x, y, "ols")
    assert numeric.theta == pytest.approx(exact.theta, rel=1e-8)
    assert numeric.param_se["b"] == pytest.approx(exact.param_se["b"], rel=1e-8)


def test_fitting_recovers_parameters_from_noiseless_data():
    """Round trip: exact power curve in, same exponent out, for all methods."""
    x = np.arange(1, 21, dtype=float)
    y = 1000.0 * x ** np.log2(0.85)
    for method in ("ols", "mupe", "zmpe"):
        f = fit(POWER, x, y, method)
        assert 2.0 ** f.theta[1] == pytest.approx(0.85, rel=1e-8), method
        assert np.exp(f.theta[0]) == pytest.approx(1000.0, rel=1e-8), method


# ================================================ the defining MUPE/ZMPE property
@pytest.mark.parametrize("method", ["mupe", "zmpe"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mupe_and_zmpe_drive_mean_percentage_error_to_zero(method, seed):
    """This is what 'unbiased percentage error' and 'zero percentage bias'
    literally mean, and it is the whole reason to prefer them over OLS."""
    x, y = power_data(sigma=0.30, seed=seed)
    f = fit(POWER, x, y, method)
    assert f.mean_percent_error == pytest.approx(0.0, abs=1e-9)
    # Equivalently: the fitted values are a mean-preserving fit of the ratios.
    assert np.mean(y / f.fitted) == pytest.approx(1.0, abs=1e-9)


def test_naive_ols_does_not_have_that_property():
    """The contrast is the point: OLS in log space leaves a positive mean
    percentage error, which is the retransformation bias."""
    x, y = power_data(sigma=0.30, seed=0)
    assert fit(POWER, x, y, "ols").mean_percent_error > 0.01


def test_zmpe_beats_mupe_on_its_own_objective():
    """ZMPE minimises squared percentage error subject to zero bias; MUPE
    satisfies the same constraint without minimising that objective. So ZMPE's
    sum of squared percentage errors must be the smaller of the two -- a
    theorem, not a tuning outcome."""
    x, y = power_data(sigma=0.30, seed=5)
    sspe = lambda f: np.sum(f.percent_errors**2)  # noqa: E731
    fits = fit_all_methods(POWER, x, y)
    assert sspe(fits["zmpe"]) <= sspe(fits["mupe"]) + 1e-12


def test_the_three_methods_disagree_on_noisy_data():
    """If they returned the same curve the comparison would be theatre."""
    x, y = power_data(sigma=0.35, seed=9)
    fits = fit_all_methods(POWER, x, y)
    slopes = {m: 2.0 ** f.theta[1] for m, f in fits.items()}
    assert len(set(np.round(list(slopes.values()), 6))) == 3


# ====================================================== intervals vs closed form
def test_prediction_interval_matches_the_textbook_formula():
    """s * sqrt(1 + 1/n + (x0 - xbar)^2 / Sxx), computed independently."""
    x, y = power_data()
    reg, s, sxx, lxbar, n = closed_form_loglog(x, y)
    f = fit(POWER, x, y, "ols")

    x0 = np.array([30.0, 50.0, 100.0])
    out = predict_with_interval(f, x0, level=0.90, kind="prediction")

    lx0 = np.log(x0)
    se_expected = s * np.sqrt(1.0 + 1.0 / n + (lx0 - lxbar) ** 2 / sxx)
    assert out["se"].to_numpy() == pytest.approx(se_expected, rel=1e-9)

    tcrit = stats.t.ppf(0.95, n - 2)
    centre = reg.intercept + reg.slope * lx0
    assert out["lower"].to_numpy() == pytest.approx(
        np.exp(centre - tcrit * se_expected), rel=1e-8
    )
    assert out["upper"].to_numpy() == pytest.approx(
        np.exp(centre + tcrit * se_expected), rel=1e-8
    )


def test_confidence_interval_matches_the_textbook_formula():
    """Same expression without the leading 1 -- that 1 is the new observation."""
    x, y = power_data()
    _, s, sxx, lxbar, n = closed_form_loglog(x, y)
    f = fit(POWER, x, y, "ols")

    x0 = np.array([30.0, 50.0])
    out = predict_with_interval(f, x0, level=0.90, kind="confidence")
    se_expected = s * np.sqrt(1.0 / n + (np.log(x0) - lxbar) ** 2 / sxx)
    assert out["se"].to_numpy() == pytest.approx(se_expected, rel=1e-9)


def test_prediction_variance_is_confidence_variance_plus_residual_variance():
    """Var_pred = Var_mean + sigma^2 exactly. This is the algebraic statement
    of why a prediction interval is the right one for a cost estimate: the
    extra term is the scatter of a single new program about the line, and it
    does not go away with more data."""
    x, y = power_data()
    f = fit(POWER, x, y, "ols")
    x0 = np.array([5.0, 30.0, 200.0])
    pred = predict_with_interval(f, x0, kind="prediction")
    conf = predict_with_interval(f, x0, kind="confidence")
    assert pred["se"].to_numpy() ** 2 == pytest.approx(
        conf["se"].to_numpy() ** 2 + f.sigma**2, rel=1e-12
    )


def test_prediction_intervals_are_always_wider_than_confidence_intervals():
    x, y = power_data()
    for method in ("ols", "mupe", "zmpe"):
        f = fit(POWER, x, y, method)
        x0 = np.array([5.0, 40.0])
        pred = predict_with_interval(f, x0, kind="prediction")
        conf = predict_with_interval(f, x0, kind="confidence")
        assert np.all(
            (pred["upper"] - pred["lower"]) > (conf["upper"] - conf["lower"])
        ), method


def test_linear_prediction_interval_is_additive_and_matches_closed_form():
    """An additive-error model must not get a multiplicative interval."""
    rng = np.random.default_rng(3)
    x = np.arange(1, 21, dtype=float)
    design = np.column_stack([x, np.sqrt(x)])
    y = 50.0 + 3.0 * x - 2.0 * np.sqrt(x) + rng.normal(0.0, 4.0, x.size)
    f = fit(linear_spec(2), design, y, "ols")

    full = np.column_stack([np.ones(x.size), design])
    beta = np.linalg.lstsq(full, y, rcond=None)[0]
    s = np.sqrt(np.sum((y - full @ beta) ** 2) / (x.size - 3))
    xtxi = np.linalg.inv(full.T @ full)

    new = np.array([[25.0, 5.0]])
    out = predict_with_interval(f, new, level=0.90, kind="prediction")
    d0 = np.array([1.0, 25.0, 5.0])
    se_expected = s * np.sqrt(1.0 + d0 @ xtxi @ d0)

    assert out["se"].iloc[0] == pytest.approx(se_expected, rel=1e-8)
    # Additive: the point estimate sits exactly at the midpoint.
    assert (out["lower"].iloc[0] + out["upper"].iloc[0]) / 2.0 == pytest.approx(
        out["fit"].iloc[0], rel=1e-10
    )


def test_a_wider_confidence_level_gives_a_wider_interval():
    x, y = power_data()
    f = fit(POWER, x, y, "ols")
    widths = [
        float(
            predict_with_interval(f, np.array([50.0]), level=lv)["upper"].iloc[0]
            - predict_with_interval(f, np.array([50.0]), level=lv)["lower"].iloc[0]
        )
        for lv in (0.50, 0.80, 0.95)
    ]
    assert widths == sorted(widths)


def test_intervals_widen_away_from_the_centre_of_the_data():
    """Extrapolation is genuinely less certain, and the formula must say so."""
    x, y = power_data(n=20)
    f = fit(POWER, x, y, "ols")
    inside = predict_with_interval(f, np.array([10.0]), kind="confidence")
    outside = predict_with_interval(f, np.array([2000.0]), kind="confidence")
    assert outside["se"].iloc[0] > inside["se"].iloc[0]


# ================================================== retransformation bias
def test_retransformation_bias_matches_the_lognormal_factor():
    """With errors generated as exactly lognormal(0, s), the mean-to-median
    ratio is exp(s^2/2). The reported factor has to land on that."""
    sigma = 0.30
    x, y = power_data(n=200, sigma=sigma, seed=4)
    f = fit(POWER, x, y, "ols")
    bias = retransformation_bias(f)
    # Recovered log-residual variance should be close to the true sigma^2 ...
    assert np.sqrt(bias.log_residual_variance) == pytest.approx(sigma, rel=0.10)
    # ... and the factor is exactly exp(s^2/2) of whatever variance was found.
    assert bias.theoretical_factor == pytest.approx(
        np.exp(bias.log_residual_variance / 2.0), rel=1e-12
    )
    assert bias.percent_understated == pytest.approx(
        (bias.theoretical_factor - 1.0) * 100.0, rel=1e-12
    )


def test_duan_smearing_equals_the_mean_ratio_identically():
    """These two are the same quantity -- exp(log y - log f) is y/f -- so this
    asserts an identity. Reported separately only because reviewers ask for
    each by name; their agreement is arithmetic, never corroboration."""
    x, y = power_data(sigma=0.4, seed=6)
    bias = retransformation_bias(fit(POWER, x, y, "ols"))
    assert bias.smearing_factor == pytest.approx(bias.observed_mean_ratio, rel=1e-14)


def test_mupe_sits_above_naive_ols_by_roughly_the_bias_factor():
    """The practical claim of the whole section: MUPE recovers the mean that
    naive retransformation loses."""
    x, y = power_data(n=120, sigma=0.30, seed=8)
    fits = fit_all_methods(POWER, x, y)
    bias = retransformation_bias(fits["ols"], fits["mupe"], fits["zmpe"])
    assert bias.mupe_ratio > 1.0
    assert bias.mupe_ratio == pytest.approx(bias.theoretical_factor, rel=0.05)


def test_bias_is_negligible_when_the_data_is_noiseless():
    x = np.arange(1, 21, dtype=float)
    y = 1000.0 * x ** np.log2(0.85)
    bias = retransformation_bias(fit(POWER, x, y, "ols"))
    assert bias.theoretical_factor == pytest.approx(1.0, abs=1e-9)
    assert bias.percent_understated == pytest.approx(0.0, abs=1e-6)


def test_retransformation_bias_refuses_a_non_ols_fit():
    """The quantity is defined for exponentiating a log-space regression; on a
    MUPE fit it would be meaningless, so it raises rather than returning ~1."""
    x, y = power_data()
    with pytest.raises(FitError, match="log-space"):
        retransformation_bias(fit(POWER, x, y, "mupe"))


# =================================================================== guardrails
def test_zero_degrees_of_freedom_is_refused():
    """Two points and two parameters interpolate exactly. A perfect fit with
    no estimable uncertainty is the most dangerous number in cost analysis."""
    with pytest.raises(FitError, match="zero degrees of freedom"):
        fit(POWER, np.array([1.0, 2.0]), np.array([100.0, 85.0]), "ols")


def test_more_parameters_than_observations_is_refused():
    with pytest.raises(FitError, match="cannot identify"):
        fit(linear_spec(4), np.ones((3, 4)), np.array([1.0, 2.0, 3.0]), "ols")


def test_low_degrees_of_freedom_warns_by_default():
    x = np.arange(1, 5, dtype=float)          # n=4, p=2 -> df=2 < 3
    y = 1000.0 * x ** np.log2(0.85)
    with pytest.warns(RuntimeWarning, match="degree"):
        f = fit(POWER, x, y, "ols")
    assert f.df < MIN_COMFORTABLE_DF
    assert not f.df_is_adequate


def test_low_degrees_of_freedom_can_be_made_fatal():
    x = np.arange(1, 5, dtype=float)
    y = 1000.0 * x ** np.log2(0.85)
    with pytest.raises(FitError, match="degree"):
        fit(POWER, x, y, "ols", allow_low_df=False)


def test_an_unknown_method_is_refused_by_name():
    x, y = power_data()
    with pytest.raises(FitError, match="Unknown fitting method"):
        fit(POWER, x, y, "wls")


@pytest.mark.parametrize("method", ["ols", "mupe", "zmpe"])
def test_non_positive_costs_are_refused_for_multiplicative_methods(method):
    """A percentage error against zero is undefined; refuse rather than
    emitting inf."""
    x = np.arange(1, 6, dtype=float)
    y = np.array([100.0, 90.0, 0.0, 70.0, 65.0])
    with pytest.raises(FitError, match="non-positive"):
        fit(POWER, x, y, method)


def test_an_invalid_confidence_level_is_refused():
    x, y = power_data()
    f = fit(POWER, x, y, "ols")
    with pytest.raises(FitError, match="level must be"):
        predict_with_interval(f, np.array([10.0]), level=1.5)


def test_an_unknown_interval_kind_is_refused():
    """Silently defaulting to a confidence interval would understate the risk
    of the estimate by exactly the amount that matters."""
    x, y = power_data()
    f = fit(POWER, x, y, "ols")
    with pytest.raises(FitError, match="prediction' or 'confidence"):
        predict_with_interval(f, np.array([10.0]), kind="interval")


# =============================================================== bookkeeping
def test_degrees_of_freedom_and_shapes_are_consistent():
    x, y = power_data(n=25)
    f = fit(POWER, x, y, "mupe")
    assert f.n_obs == 25
    assert f.n_params == 2
    assert f.df == 23
    assert f.fitted.shape == f.observed.shape == (25,)
    assert f.cov.shape == (2, 2)
    assert set(f.params) == {"log_a", "b"}


def test_cv_of_a_multiplicative_fit_is_its_relative_residual_spread():
    x, y = power_data(sigma=0.2, seed=2)
    f = fit(POWER, x, y, "mupe")
    assert f.cv == pytest.approx(f.sigma, rel=1e-12)
    # And the standard error is that proportion carried back into dollars.
    assert f.standard_error == pytest.approx(f.sigma * np.mean(f.fitted), rel=1e-12)


def test_summary_table_has_one_row_per_parameter():
    x, y = power_data()
    summary = fit(POWER, x, y, "ols").summary()
    assert list(summary["parameter"]) == ["log_a", "b"]
    assert (summary["std_error"] > 0).all()


def test_fitting_is_deterministic():
    """No random starts anywhere: the same data must give identical numbers."""
    x, y = power_data(seed=12)
    for method in ("ols", "mupe", "zmpe"):
        a = fit(POWER, x, y, method)
        b = fit(POWER, x, y, method)
        assert np.array_equal(a.theta, b.theta), method
        assert a.sigma == b.sigma, method
