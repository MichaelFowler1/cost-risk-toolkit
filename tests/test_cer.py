"""Parametric cost estimating relationships.

The synthetic portfolio is generated from two exact power laws -- one relating
first-unit cost to airframe weight, one relating software effort to equivalent
source lines. With the scatter turned off, a correct CER fit has to return
those coefficients to machine precision. That is the closed-form answer this
module is held to, and it is a stronger check than any goodness-of-fit
statistic, because a wrong estimator can still produce a high R^2.

Beyond recovery, three things get the most attention here:

**Prediction versus confidence intervals.** ``Var_pred = Var_mean + sigma^2``
is an identity, and it is asserted as one. The gap between the two is the
scatter of programs about the line, which does not shrink with sample size --
which is exactly why quoting a confidence interval for a cost estimate
understates the risk no matter how much data you have.

**Cook's distance against an actual leave-one-out refit.** The closed form is
exact, so the test drops each program, refits, and checks the formula
reproduces the movement in the fitted surface. On a fourteen-program sample
one program can set the slope, and this is the diagnostic that finds it.

**Refusing to answer.** Zero degrees of freedom, too few programs per
parameter, a prediction well outside the fitting data, a log-log form asked to
take the log of a negative number. Each of these has a plausible-looking
number waiting on the other side of it, which is why each one raises or warns.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from cost_core.cer import (CER, MIN_OBS_PER_PARAM, ExtrapolationWarning, Form,
                           cer_comparison_table, compare_cer_forms,
                           compare_cer_methods, fit_cer)
from cost_core.fitting import FitError
from cost_core.synth import (TRUE_AIRFRAME_CER, TRUE_SOFTWARE_CER,
                             PathologyConfig, generate_portfolio)

CLEAN = PathologyConfig.clean()


def portfolio_table(n=14, seed=1, scatter=0.18):
    """A CER fitting table: one row per program, drivers and outcomes.

    Built without the report shapes -- a CER needs one row per program, and
    generating thousands of submission rows per program to throw them away
    would dominate the runtime of this file.
    """
    portfolio = generate_portfolio(
        n_programs=n, seed=seed, pathologies=CLEAN, scatter_cv=scatter,
        with_reports=False,
    )
    table = portfolio.cer_table()
    table["weight_klb"] = table["empty_weight_lb"] / 1000.0
    return table


@pytest.fixture(scope="module")
def noiseless():
    return portfolio_table(n=8, seed=1, scatter=0.0)


@pytest.fixture(scope="module")
def scattered():
    return portfolio_table(n=14, seed=1, scatter=0.18)


# ==================================================== recovery of known CERs
def test_a_log_log_cer_recovers_the_true_airframe_relationship(noiseless):
    """The portfolio's first-unit costs are priced from an exact power law in
    weight. With no scatter, the fit must return that law exactly."""
    true_a, true_b = TRUE_AIRFRAME_CER
    cer = fit_cer(noiseless, "t1_cost_observed", ["weight_klb"], form=Form.LOG_LOG)
    assert np.exp(cer.result.theta[0]) == pytest.approx(true_a, rel=1e-9)
    assert cer.result.theta[1] == pytest.approx(true_b, rel=1e-9)


def test_a_log_log_cer_recovers_the_true_software_relationship(noiseless):
    true_a, true_b = TRUE_SOFTWARE_CER
    cer = fit_cer(
        noiseless, "software_effort_observed", ["equivalent_ksloc"], form=Form.LOG_LOG
    )
    assert np.exp(cer.result.theta[0]) == pytest.approx(true_a, rel=1e-9)
    assert cer.result.theta[1] == pytest.approx(true_b, rel=1e-9)


@pytest.mark.parametrize("method", ["ols", "mupe", "zmpe"])
def test_every_method_recovers_the_truth_from_noiseless_data(noiseless, method):
    true_a, true_b = TRUE_AIRFRAME_CER
    cer = fit_cer(
        noiseless, "t1_cost_observed", ["weight_klb"], form=Form.LOG_LOG, method=method
    )
    assert np.exp(cer.result.theta[0]) == pytest.approx(true_a, rel=1e-6)
    assert cer.result.theta[1] == pytest.approx(true_b, rel=1e-6)


def test_a_scattered_sample_still_lands_near_the_truth(scattered):
    _, true_b = TRUE_AIRFRAME_CER
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method="mupe")
    assert cer.result.theta[1] == pytest.approx(true_b, abs=0.10)
    assert cer.cv > 0.05           # and the CV is honest about the scatter


def test_a_linear_cer_reproduces_the_normal_equations(scattered):
    cer = fit_cer(
        scattered, "t1_cost_observed", ["weight_klb"], form=Form.LINEAR, method="ols"
    )
    x = scattered["weight_klb"].to_numpy()
    y = scattered["t1_cost_observed"].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    assert cer.result.theta == pytest.approx(beta, rel=1e-9)


def test_a_multi_predictor_cer_fits_and_reports_each_coefficient(scattered):
    cer = fit_cer(
        scattered,
        "t1_cost_observed",
        ["weight_klb", "max_speed_kts"],
        form=Form.LOG_LOG,
    )
    assert set(cer.coefficients) == {"log_a", "b_weight_klb", "b_max_speed_kts"}
    assert cer.result.n_params == 3
    assert cer.df == len(scattered) - 3
    # Weight drives cost here and speed does not, so weight's coefficient must
    # be the one that is clearly non-zero.
    summary = cer.summary().set_index("parameter")
    assert summary.loc["b_weight_klb", "p_value"] < 0.01


# ================================================ prediction vs confidence
def test_prediction_variance_is_confidence_variance_plus_residual_variance(scattered):
    """The identity that makes the distinction concrete. The extra term is the
    scatter of a single new program about the line."""
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    point = {"weight_klb": [20.0, 30.0, 40.0]}
    pred = cer.predict(point, kind="prediction")
    conf = cer.predict(point, kind="confidence")
    assert pred["se"].to_numpy() ** 2 == pytest.approx(
        conf["se"].to_numpy() ** 2 + cer.result.sigma**2, rel=1e-12
    )


def test_a_prediction_interval_is_always_wider_than_a_confidence_interval(scattered):
    for form in Form:
        cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], form=form)
        point = {"weight_klb": [25.0]}
        pred = cer.predict(point, kind="prediction")
        conf = cer.predict(point, kind="confidence")
        assert (pred["upper"] - pred["lower"]).iloc[0] > (
            conf["upper"] - conf["lower"]
        ).iloc[0], form


def test_the_prediction_interval_is_the_default(scattered):
    """Because a cost estimate forecasts one new program, not the mean of a
    line. Getting this default wrong would understate every estimate."""
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    default = cer.predict({"weight_klb": [25.0]})
    explicit = cer.predict({"weight_klb": [25.0]}, kind="prediction")
    assert default["kind"].iloc[0] == "prediction"
    assert default["se"].iloc[0] == pytest.approx(explicit["se"].iloc[0], rel=1e-15)


def test_confidence_intervals_shrink_with_sample_size_but_prediction_ones_do_not():
    """The practical consequence of the identity: more programs pin down the
    line, but never the scatter of the next program about it."""
    small = fit_cer(portfolio_table(n=8, seed=4), "t1_cost_observed", ["weight_klb"])
    large = fit_cer(portfolio_table(n=40, seed=4), "t1_cost_observed", ["weight_klb"])
    point = {"weight_klb": [25.0]}

    def width(cer, kind):
        out = cer.predict(point, kind=kind, warn_on_extrapolation=False)
        return float(out["upper"].iloc[0] / out["lower"].iloc[0])

    assert width(large, "confidence") < width(small, "confidence")
    # The prediction interval stays wide: it is dominated by sigma, not by
    # parameter uncertainty.
    assert width(large, "prediction") > 1.3


def test_an_unknown_interval_kind_is_refused(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    with pytest.raises(FitError, match="not interchangeable"):
        cer.predict({"weight_klb": [25.0]}, kind="ci")


def test_a_wider_level_gives_a_wider_interval(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    widths = [
        float(
            (lambda o: o["upper"].iloc[0] - o["lower"].iloc[0])(
                cer.predict({"weight_klb": [25.0]}, level=lv)
            )
        )
        for lv in (0.50, 0.80, 0.95)
    ]
    assert widths == sorted(widths)


# ================================================================ diagnostics
def test_leverages_sum_to_the_number_of_parameters(scattered):
    """The hat matrix is a projection onto the column space of the design, so
    its trace is the rank. An exact identity, and a strong check that the
    design matrix used for diagnostics is the one the fit actually used."""
    for predictors in (["weight_klb"], ["weight_klb", "max_speed_kts"]):
        cer = fit_cer(scattered, "t1_cost_observed", predictors)
        diag = cer.diagnostics()
        assert diag.leverage.sum() == pytest.approx(cer.result.n_params, rel=1e-10)
        assert np.all((diag.leverage >= 0) & (diag.leverage <= 1))


def test_cooks_distance_matches_an_actual_leave_one_out_refit(scattered):
    """The closed form is exact, so it is checked against the thing it is a
    closed form *for*: drop each program, refit, and measure how far the
    fitted surface moved."""
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method="ols")
    diag = cer.diagnostics()

    x = np.log(scattered["weight_klb"].to_numpy())
    y = np.log(scattered["t1_cost_observed"].to_numpy())
    design = np.column_stack([np.ones(len(x)), x])
    n, p = design.shape
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    sigma2 = np.sum((y - fitted) ** 2) / (n - p)

    loo = []
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        beta_i = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        loo.append(np.sum((fitted - design @ beta_i) ** 2) / (p * sigma2))

    assert diag.cooks_distance == pytest.approx(np.array(loo), rel=1e-8)


def test_an_extreme_program_is_flagged_as_influential():
    """A CER on fourteen programs can be set by one of them. The diagnostics
    have to say which."""
    table = portfolio_table(n=12, seed=2, scatter=0.05).copy()
    table.loc[table.index[0], "t1_cost_observed"] *= 3.0     # one bad program

    cer = fit_cer(
        table, "t1_cost_observed", ["weight_klb"], label_col="program"
    )
    diag = cer.diagnostics()
    assert diag.influential, "a 3x outlier should be flagged"
    assert table.loc[table.index[0], "program"] in diag.influential
    assert "Influential" in diag.narrative()


def test_diagnostics_are_labelled_by_program(scattered):
    cer = fit_cer(
        scattered, "t1_cost_observed", ["weight_klb"], label_col="program"
    )
    frame = cer.diagnostics().to_frame()
    assert set(frame["observation"]) == set(scattered["program"])
    assert len(frame) == len(scattered)
    # Sorted most influential first, so the eye lands on the right row.
    assert frame["cooks_distance"].is_monotonic_decreasing


def test_an_exact_fit_flags_nothing_rather_than_flagging_rounding_error():
    """With noiseless data the residuals and sigma are both at the
    floating-point floor, so standardising one by the other is 0/0. The
    influence measures must report zero -- a fit that passes through every
    point has no influential observations -- instead of ratios of rounding
    error, which would be O(1) and would flag half the sample."""
    cer = fit_cer(portfolio_table(n=8, seed=1, scatter=0.0),
                  "t1_cost_observed", ["weight_klb"])
    diag = cer.diagnostics()
    assert np.all(diag.cooks_distance == 0.0)
    assert np.all(diag.dffits == 0.0)
    assert diag.influential == []
    # Leverage is a property of the predictors alone, so it is unaffected and
    # a genuinely unusual weight is still flagged -- correctly.
    assert diag.leverage.sum() == pytest.approx(cer.result.n_params, rel=1e-10)
    assert "Influential" not in diag.narrative()


def test_mismatched_diagnostic_labels_are_refused(scattered):
    from cost_core.cer import compute_diagnostics

    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    with pytest.raises(ValueError, match="labels for"):
        compute_diagnostics(cer.result, ["only", "two"])


# ============================================================ extrapolation
def test_predicting_outside_the_fitting_range_warns_and_says_where(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    hi = cer.predictor_ranges["weight_klb"][1]

    with pytest.warns(ExtrapolationWarning, match="outside the fitting data"):
        out = cer.predict({"weight_klb": [hi * 2.0]})

    assert out["outside_fitting_range"].iloc[0]
    assert "above observed max" in out["extrapolation_note"].iloc[0]


def test_predicting_inside_the_fitting_range_is_silent(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    lo, hi = cer.predictor_ranges["weight_klb"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        out = cer.predict({"weight_klb": [(lo + hi) / 2.0]})
    assert not out["outside_fitting_range"].iloc[0]


def test_hidden_extrapolation_is_caught_by_leverage():
    """Every predictor inside its own range, but the *combination* unobserved.
    A per-column range check passes this; only leverage catches it."""
    rng = np.random.default_rng(0)
    weight = np.linspace(10.0, 40.0, 14)
    # Speed is tightly coupled to weight in the sample: no light-and-fast
    # programs exist, so that corner is unobserved.
    speed = 300.0 + 8.0 * weight + rng.normal(0.0, 5.0, weight.size)
    cost = 400_000.0 * weight**0.72 * rng.lognormal(0.0, 0.05, weight.size)
    table = pd.DataFrame(
        {"weight_klb": weight, "speed": speed, "cost": cost}
    )
    cer = fit_cer(table, "cost", ["weight_klb", "speed"], form=Form.LOG_LOG)

    # Light and fast: both values are inside their own observed ranges.
    point = {"weight_klb": [12.0], "speed": [610.0]}
    assert 10.0 < 12.0 < 40.0
    assert speed.min() < 610.0 < speed.max()

    with pytest.warns(ExtrapolationWarning):
        out = cer.predict(point)
    assert out["outside_fitting_range"].iloc[0]
    assert "hidden extrapolation" in out["extrapolation_note"].iloc[0]
    assert out["leverage_ratio"].iloc[0] > 1.0


def test_the_extrapolation_warning_can_be_escalated_to_an_error(scattered):
    """Worth doing in an automated pipeline, where nobody reads the log."""
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExtrapolationWarning)
        with pytest.raises(ExtrapolationWarning):
            cer.predict({"weight_klb": [500.0]})


# ======================================================= small-sample guards
def test_zero_degrees_of_freedom_is_refused():
    """Two programs and two parameters interpolate exactly: a perfect fit with
    no estimable uncertainty."""
    table = pd.DataFrame({"cost": [1e6, 2e6], "weight_klb": [10.0, 20.0]})
    with pytest.raises(FitError, match="zero degrees of freedom"):
        fit_cer(table, "cost", ["weight_klb"])


def test_more_parameters_than_programs_is_refused():
    table = pd.DataFrame(
        {"cost": [1e6, 2e6, 3e6], "a": [1.0, 2.0, 3.0],
         "b": [2.0, 1.0, 4.0], "c": [3.0, 5.0, 1.0]}
    )
    with pytest.raises(FitError, match="cannot identify"):
        fit_cer(table, "cost", ["a", "b", "c"])


def test_too_few_programs_per_parameter_warns():
    """Five programs and three parameters is 1.7 per parameter: the overall
    fit can look fine while no individual coefficient means anything."""
    table = portfolio_table(n=5, seed=3, scatter=0.1)
    with pytest.warns(RuntimeWarning, match="per parameter"):
        cer = fit_cer(table, "t1_cost_observed", ["weight_klb", "max_speed_kts"])
    assert cer.obs_per_param < MIN_OBS_PER_PARAM


def test_inadequate_degrees_of_freedom_can_be_made_fatal():
    table = portfolio_table(n=5, seed=3, scatter=0.1)
    with pytest.raises(FitError):
        fit_cer(
            table,
            "t1_cost_observed",
            ["weight_klb", "max_speed_kts"],
            allow_low_df=False,
        )


def test_an_adequate_sample_warns_about_nothing(scattered):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    assert cer.obs_per_param >= MIN_OBS_PER_PARAM
    assert cer.result.df_is_adequate


# ================================================================= bad input
def test_a_missing_column_is_named(scattered):
    with pytest.raises(FitError, match="wingspan"):
        fit_cer(scattered, "t1_cost_observed", ["wingspan"])
    with pytest.raises(FitError, match="not_a_cost"):
        fit_cer(scattered, "not_a_cost", ["weight_klb"])


def test_a_cer_with_no_predictors_is_refused(scattered):
    with pytest.raises(FitError, match="at least one predictor"):
        fit_cer(scattered, "t1_cost_observed", [])


def test_a_log_log_cer_refuses_non_positive_values():
    table = pd.DataFrame(
        {"cost": [1e6, 2e6, 3e6, 4e6], "x": [1.0, 2.0, 0.0, 4.0]}
    )
    with pytest.raises(FitError, match="undefined at or below zero"):
        fit_cer(table, "cost", ["x"], form=Form.LOG_LOG)

    negative_cost = pd.DataFrame(
        {"cost": [1e6, -2e6, 3e6, 4e6], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(FitError, match="undefined at or below zero"):
        fit_cer(negative_cost, "cost", ["x"], form=Form.LOG_LOG)


def test_a_linear_cer_tolerates_what_log_log_cannot():
    """The reason both forms exist: a zero-valued predictor has no logarithm,
    but is perfectly ordinary in an additive model."""
    table = pd.DataFrame(
        {
            "cost": [1.0e6, 2.0e6, 3.1e6, 3.9e6, 5.2e6, 5.8e6, 7.1e6, 8.0e6],
            "x": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    with pytest.raises(FitError, match="undefined at or below zero"):
        fit_cer(table, "cost", ["x"], form=Form.LOG_LOG)

    cer = fit_cer(table, "cost", ["x"], form=Form.LINEAR, method="ols")
    assert cer.result.n_obs == 8
    assert cer.coefficients["b_x"] > 0


def test_an_unknown_form_or_method_is_refused(scattered):
    with pytest.raises(FitError, match="Unknown CER form"):
        fit_cer(scattered, "t1_cost_observed", ["weight_klb"], form="quadratic")
    with pytest.raises(FitError, match="Unknown method"):
        fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method="eyeball")


def test_predicting_with_the_wrong_columns_is_refused(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    with pytest.raises(FitError, match="missing predictor column"):
        cer.predict(pd.DataFrame({"wingspan": [30.0]}))
    with pytest.raises(FitError, match="Expected 1 predictor"):
        cer.predict(np.array([[1.0, 2.0, 3.0]]))


def test_predicting_a_non_positive_value_from_a_log_log_cer_is_refused(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"])
    with pytest.raises(FitError, match="undefined at or below zero"):
        cer.predict({"weight_klb": [-5.0]})


def test_rows_with_missing_values_are_dropped_not_imputed(scattered):
    table = scattered.copy()
    table.loc[table.index[0], "weight_klb"] = np.nan
    cer = fit_cer(table, "t1_cost_observed", ["weight_klb"])
    assert cer.result.n_obs == len(scattered) - 1


# ============================================================== comparison
def test_mupe_and_zmpe_have_zero_percentage_bias(scattered):
    fits = compare_cer_methods(scattered, "t1_cost_observed", ["weight_klb"])
    for method in ("mupe", "zmpe"):
        assert fits[method].result.mean_percent_error == pytest.approx(
            0.0, abs=1e-8
        ), method
    assert abs(fits["ols"].result.mean_percent_error) > 1e-4


def test_the_comparison_table_covers_methods_and_forms(scattered):
    methods = cer_comparison_table(
        compare_cer_methods(scattered, "t1_cost_observed", ["weight_klb"])
    )
    assert set(methods["method"]) == {"OLS", "MUPE", "ZMPE"}

    forms = cer_comparison_table(
        compare_cer_forms(scattered, "t1_cost_observed", ["weight_klb"])
    )
    assert set(forms["form"]) == {"log_log", "linear"}
    # SE and CV are populated for every variant, since they are what the
    # comparison is meant to be made on.
    for table in (methods, forms):
        assert (table["std_error"] > 0).all()
        assert (table["cv"] > 0).all()


def test_the_equation_reads_back_the_fitted_relationship(scattered):
    loglog = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], form=Form.LOG_LOG)
    assert "weight_klb^" in loglog.equation()

    linear = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], form=Form.LINEAR)
    assert "*weight_klb" in linear.equation()


def test_describe_carries_everything_needed_for_the_log(scattered):
    cer = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method="mupe")
    described = cer.describe()
    for key in (
        "form", "method", "equation", "n_obs", "df", "obs_per_param",
        "standard_error", "cv", "predictor_ranges", "coefficients",
    ):
        assert key in described, key
    assert described["predictor_ranges"]["weight_klb"][0] < (
        described["predictor_ranges"]["weight_klb"][1]
    )


# ============================================================= determinism
def test_cer_fitting_is_deterministic(scattered):
    for method in ("ols", "mupe", "zmpe"):
        a = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method=method)
        b = fit_cer(scattered, "t1_cost_observed", ["weight_klb"], method=method)
        assert np.array_equal(a.result.theta, b.result.theta), method


def test_the_whole_chain_from_synthetic_data_to_a_cer_holds_together():
    """Generate a portfolio, fit a learning curve to each program's 1921-2 to
    recover its first-unit cost, then fit a CER across those recovered values.
    That is the real workflow, and it has to arrive at the relationship the
    portfolio was generated from."""
    from cost_core.ingest import normalize_program
    from cost_core.learning_curve import Theory, fit_from_progress_report

    portfolio = generate_portfolio(
        n_programs=6, seed=5, pathologies=CLEAN, scatter_cv=0.0
    )
    rows = []
    for program in portfolio:
        data = normalize_program(program)
        curve = fit_from_progress_report(
            data.learning_curve_input(), theory=Theory.CRAWFORD
        )
        rows.append(
            {
                "program": program.spec.program,
                "weight_klb": program.spec.empty_weight_lb / 1000.0,
                "t1_recovered": curve.t1,
            }
        )
    table = pd.DataFrame(rows)

    cer = fit_cer(table, "t1_recovered", ["weight_klb"], form=Form.LOG_LOG)
    true_a, true_b = TRUE_AIRFRAME_CER
    assert np.exp(cer.result.theta[0]) == pytest.approx(true_a, rel=1e-5)
    assert cer.result.theta[1] == pytest.approx(true_b, rel=1e-5)
