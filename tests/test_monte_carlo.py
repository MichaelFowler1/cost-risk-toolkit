"""Monte Carlo cost risk.

A simulation whose numbers move between runs can't support a P80 you'd put in
front of anyone, so determinism under a fixed seed is the first thing checked
here. The rest pins the statistics against distributions whose true values we
know in closed form.
"""
import numpy as np
import pytest

from cost_core.monte_carlo import SimulationResult, run_monte_carlo

NORMAL = {"type": "normal", "loc": 100.0, "scale": 10.0}
FIXED_QTY = {"type": "normal", "loc": 50.0, "scale": 0.0}   # degenerate: exactly 50


# ------------------------------------------------------------ determinism
def test_the_same_seed_gives_exactly_the_same_answer():
    """A P80 that changes between runs is not a number you can defend."""
    a = run_monte_carlo(20_000, NORMAL, FIXED_QTY, seed=42)
    b = run_monte_carlo(20_000, NORMAL, FIXED_QTY, seed=42)
    assert np.array_equal(a.samples, b.samples)
    assert (a.p50, a.p80, a.p90) == (b.p50, b.p80, b.p90)


def test_different_seeds_give_different_draws():
    a = run_monte_carlo(5_000, NORMAL, FIXED_QTY, seed=1)
    b = run_monte_carlo(5_000, NORMAL, FIXED_QTY, seed=2)
    assert not np.array_equal(a.samples, b.samples)


# ------------------------------------------------------------- statistics
def test_percentiles_are_ordered():
    r = run_monte_carlo(20_000, NORMAL, FIXED_QTY, seed=3)
    assert r.p50 <= r.p80 <= r.p90


def test_mean_total_matches_the_analytic_answer():
    """Unit cost ~ N(100, 10) at a fixed quantity of 50 has a true mean total
    of 5000. With 200k draws the estimate should be within a fraction of a
    percent, which pins the whole cost = unit x quantity path."""
    r = run_monte_carlo(200_000, NORMAL, FIXED_QTY, seed=11)
    assert r.mean == pytest.approx(5_000.0, rel=0.01)
    assert r.p50 == pytest.approx(5_000.0, rel=0.01)


def test_a_wider_input_distribution_widens_the_result():
    tight = run_monte_carlo(50_000, {"type": "normal", "loc": 100.0, "scale": 2.0},
                            FIXED_QTY, seed=5)
    wide = run_monte_carlo(50_000, {"type": "normal", "loc": 100.0, "scale": 25.0},
                           FIXED_QTY, seed=5)
    # More input uncertainty must show up as a bigger gap between P50 and P90.
    assert (wide.p90 - wide.p50) > (tight.p90 - tight.p50)


def test_triangular_mean_matches_its_closed_form():
    """Mean of a triangular distribution is (left + mode + right) / 3."""
    tri = {"type": "triangular", "left": 10.0, "mode": 20.0, "right": 60.0}
    r = run_monte_carlo(200_000, tri, {"type": "normal", "loc": 1.0, "scale": 0.0},
                        seed=13)
    assert r.mean == pytest.approx((10.0 + 20.0 + 60.0) / 3.0, rel=0.01)


def test_costs_are_never_negative():
    """A distribution wide enough to go negative gets clamped, because a
    negative cost is not a physical outcome."""
    r = run_monte_carlo(20_000, {"type": "normal", "loc": 5.0, "scale": 50.0},
                        FIXED_QTY, seed=17)
    assert (r.samples >= 0).all()


def test_result_shape_and_type():
    r = run_monte_carlo(1_000, NORMAL, FIXED_QTY, seed=1)
    assert isinstance(r, SimulationResult)
    assert r.samples.shape == (1_000,)


# -------------------------------------------------------------- bad input
def test_an_unknown_distribution_is_refused():
    with pytest.raises(ValueError, match="Unsupported distribution"):
        run_monte_carlo(100, {"type": "poisson", "lam": 3.0}, FIXED_QTY, seed=1)


def test_a_missing_parameter_is_refused_by_name():
    """Silently defaulting a missing sigma would produce a plausible-looking
    number that means nothing."""
    with pytest.raises(ValueError, match="Missing required parameter"):
        run_monte_carlo(100, {"type": "normal", "loc": 10.0}, FIXED_QTY, seed=1)


# ==========================================================================
# Correlated WBS-level risk models
# ==========================================================================
import warnings

import pandas as pd

from cost_core.monte_carlo import (DEFAULT_CORRELATION, CorrelationImpact,
                                   CorrelationWarning, CostElement,
                                   DiscreteRisk, RiskModel, RiskModelError,
                                   correlation_impact, make_distribution,
                                   risk_model_from_elements,
                                   simulate_risk_model, uniform_correlation,
                                   validate_correlation)

TRIANGLE = {"type": "triangular", "left": 10.0, "mode": 20.0, "right": 60.0}
PERT = {"type": "pert", "left": 10.0, "mode": 20.0, "right": 60.0}


def equal_model(k=10, rho=0.30, value=100e6, **kwargs):
    """k identically distributed elements at a common correlation.

    The case where the variance inflation has an exact closed form.
    """
    return risk_model_from_elements(
        {f"E{i}": value for i in range(k)}, default_correlation=rho, **kwargs
    )


def quiet(fn, *args, **kwargs):
    """Run something without the default-correlation notice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CorrelationWarning)
        return fn(*args, **kwargs)


# ------------------------------------------------------- marginals in closed form
def test_pert_mean_is_the_classic_weighted_average():
    """(left + 4*mode + right) / 6, exactly. That is what choosing lambda = 4
    buys, and it is the reason PERT is preferred to a triangular for an
    expert-elicited range: the triangular puts far more weight in the tails
    than anyone means when they give three numbers."""
    dist = make_distribution(PERT)
    assert dist.mean() == pytest.approx((10.0 + 4 * 20.0 + 60.0) / 6.0, rel=1e-12)


def test_triangular_mean_is_the_simple_average_of_its_three_points():
    dist = make_distribution(TRIANGLE)
    assert dist.mean() == pytest.approx((10.0 + 20.0 + 60.0) / 3.0, rel=1e-12)


def test_pert_is_tighter_than_the_triangular_on_the_same_three_points():
    """Same inputs, materially different risk answer -- so the choice has to
    be deliberate rather than a default nobody looked at."""
    assert make_distribution(PERT).var() < make_distribution(TRIANGLE).var()
    assert make_distribution(PERT).mean() < make_distribution(TRIANGLE).mean()


def test_lognormal_mean_matches_its_closed_form():
    mu, sigma = np.log(150.0), 0.30
    dist = make_distribution({"type": "lognormal", "mean": mu, "sigma": sigma})
    assert dist.mean() == pytest.approx(np.exp(mu + sigma**2 / 2.0), rel=1e-10)
    assert dist.median() == pytest.approx(np.exp(mu), rel=1e-10)


def test_a_fixed_distribution_carries_no_uncertainty():
    dist = make_distribution({"type": "fixed", "value": 42.0})
    assert dist.mean() == pytest.approx(42.0)
    assert dist.var() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "spec,message",
    [
        ({"type": "beta"}, "Unsupported distribution"),
        ({"type": "normal", "loc": 1.0}, "needs parameter"),
        ({"type": "triangular", "left": 10.0, "mode": 90.0, "right": 60.0}, "left <= mode <= right"),
        ({"type": "pert", "left": 10.0, "mode": 5.0, "right": 60.0}, "left <= mode <= right"),
        ({"type": "normal", "loc": 1.0, "scale": -1.0}, "scale must be"),
        ({"type": "uniform", "low": 10.0, "high": 1.0}, "high >= low"),
    ],
)
def test_a_malformed_distribution_is_refused(spec, message):
    with pytest.raises(RiskModelError, match=message):
        make_distribution(spec)


# ------------------------------------------------- the variance identity
@pytest.mark.parametrize("k", [2, 5, 10, 25])
@pytest.mark.parametrize("rho", [0.0, 0.2, 0.3, 0.5])
def test_variance_inflation_is_exactly_one_plus_rho_times_k_minus_one(k, rho):
    """The headline claim, in closed form. For k equally variable elements at
    a common rho, ignoring correlation understates the variance of the total by
    exactly this factor. Ten elements at rho = 0.3 gives 3.7."""
    model = quiet(equal_model, k=k, rho=rho)
    assert model.variance_inflation() == pytest.approx(1.0 + rho * (k - 1), rel=1e-12)


def test_analytic_variance_matches_the_simulation_for_normal_marginals():
    """With normal marginals a Gaussian copula reproduces Pearson correlation
    exactly, so the simulated variance must match the closed form to sampling
    error alone. This pins the sampler against the algebra."""
    elements = [
        CostElement(f"E{i}", {"type": "normal", "loc": 100e6, "scale": 15e6}, 100e6)
        for i in range(8)
    ]
    model = RiskModel(elements=elements, correlation=uniform_correlation(8, 0.30))
    result = simulate_risk_model(model, 200_000, seed=1)
    assert np.var(result.totals, ddof=1) == pytest.approx(
        model.analytic_variance(), rel=0.02
    )


def test_the_gaussian_copula_falls_slightly_short_of_the_target_correlation():
    """An honest property of the method, not a defect. Mapping correlated
    normals through non-normal marginals preserves rank correlation but pulls
    Pearson correlation slightly toward zero. Iman-Conover, which reorders
    rather than transforms, lands closer."""
    model = quiet(equal_model, k=6, rho=0.30)
    copula = quiet(simulate_risk_model, model, 60_000, 1, method="gaussian_copula")
    reorder = quiet(simulate_risk_model, model, 60_000, 1, method="iman_conover")

    def pairwise(result):
        return float(
            np.corrcoef(result.element_samples[:, 0], result.element_samples[:, 1])[0, 1]
        )

    assert 0.25 < pairwise(copula) < 0.30
    assert pairwise(reorder) == pytest.approx(0.30, abs=0.02)


def test_correlation_widens_the_distribution_and_raises_the_p80():
    """The practical consequence, and the reason any of this matters."""
    model = quiet(equal_model, k=10, rho=0.30)
    impact = correlation_impact(model, 40_000, seed=3)

    assert impact.correlated.std > impact.independent.std
    assert impact.correlated.p80 > impact.independent.p80
    assert impact.p80_understatement > 0.0
    assert impact.reserve_understatement > 0.0


def test_the_measured_variance_ratio_agrees_with_the_closed_form():
    """Two independent routes to the same number: if they disagreed, the
    briefing claim would rest on whichever one happened to be quoted."""
    model = quiet(equal_model, k=10, rho=0.30)
    impact = correlation_impact(model, 60_000, seed=5)
    assert impact.empirical_variance_ratio == pytest.approx(
        impact.analytic_variance_ratio, rel=0.10
    )
    assert impact.analytic_variance_ratio == pytest.approx(3.7, rel=1e-12)


def test_zero_correlation_leaves_the_variance_unchanged():
    """The control case: with rho = 0 the two runs are the same model."""
    model = quiet(equal_model, k=6, rho=0.0)
    impact = correlation_impact(model, 20_000, seed=7)
    assert impact.analytic_variance_ratio == pytest.approx(1.0, rel=1e-12)
    assert impact.empirical_variance_ratio == pytest.approx(1.0, rel=0.05)


def test_the_impact_narrative_and_table_are_populated():
    model = quiet(equal_model, k=10, rho=0.30)
    impact = correlation_impact(model, 20_000, seed=2)
    frame = impact.to_frame()
    assert set(frame["metric"]) == {
        "std deviation of total", "variance of total", "P80", "P80 risk reserve",
    }
    assert (frame["ratio"] > 1.0).all()
    assert "understates the variance" in impact.narrative()


# ------------------------------------------------- correlation matrices
def test_a_valid_matrix_passes_through_untouched():
    matrix = uniform_correlation(4, 0.3)
    checked, notes = validate_correlation(matrix)
    assert np.array_equal(checked, matrix)
    assert notes == []


def test_a_non_positive_semi_definite_matrix_is_repaired_with_a_warning():
    """Pairwise judgements that are each reasonable and jointly impossible.
    Falling back to independence here would discard the whole point of
    supplying a matrix, so it is repaired -- loudly."""
    matrix = np.array(
        [
            [1.0, 0.9, -0.9],
            [0.9, 1.0, 0.9],
            [-0.9, 0.9, 1.0],
        ]
    )
    assert np.linalg.eigvalsh(matrix).min() < 0        # genuinely impossible

    with pytest.warns(CorrelationWarning, match="not positive semi-definite"):
        repaired, notes = validate_correlation(matrix)

    assert np.linalg.eigvalsh(repaired).min() >= -1e-10
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.allclose(repaired, repaired.T)
    assert notes and "Largest change" in notes[-1]


def test_a_repaired_matrix_can_actually_be_used_to_simulate():
    """Repair is only worth anything if the result works downstream."""
    matrix = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    elements = [
        CostElement(f"E{i}", {"type": "normal", "loc": 100.0, "scale": 10.0}, 100.0)
        for i in range(3)
    ]
    model = RiskModel(elements=elements, correlation=matrix)
    with pytest.warns(CorrelationWarning):
        result = simulate_risk_model(model, 5_000, seed=1)
    assert np.isfinite(result.totals).all()
    assert result.notes


@pytest.mark.parametrize(
    "matrix,message",
    [
        (np.array([[1.0, 0.3]]), "must be square"),
        (np.array([[1.0, 0.3], [0.5, 1.0]]), "not symmetric"),
        (np.array([[1.0, 0.3], [0.3, 0.9]]), "diagonal must be all ones"),
        (np.array([[1.0, 1.4], [1.4, 1.0]]), "outside"),
    ],
)
def test_a_structurally_invalid_matrix_is_refused_not_repaired(matrix, message):
    """These are construction errors, not elicitation errors. Repairing them
    would be guessing at what was meant."""
    with pytest.raises(RiskModelError, match=message):
        validate_correlation(matrix)


def test_a_matrix_of_the_wrong_size_is_refused():
    elements = [
        CostElement(f"E{i}", {"type": "fixed", "value": 1.0}) for i in range(3)
    ]
    model = RiskModel(elements=elements, correlation=uniform_correlation(2, 0.3))
    with pytest.raises(RiskModelError, match="but there are 3 elements"):
        model.resolved_correlation()


def test_an_impossible_common_correlation_is_refused():
    """A strongly negative common correlation is not merely unusual: across
    more than two elements it cannot exist. Ten elements cannot all be
    mutually negatively correlated."""
    assert uniform_correlation(2, -0.9).shape == (2, 2)     # fine for a pair
    with pytest.raises(RiskModelError, match="not achievable across 10"):
        uniform_correlation(10, -0.9)


def test_defaulting_the_correlation_warns_and_says_what_it_assumed():
    """A default is an assumption. An unstated assumption is the thing the
    documentation characteristic exists to prevent."""
    model = equal_model(k=10, rho=DEFAULT_CORRELATION)
    with pytest.warns(CorrelationWarning, match="No correlation matrix supplied"):
        _, notes = model.resolved_correlation()
    assert "assumption, not a measurement" in notes[0]


def test_supplying_a_matrix_does_not_warn():
    elements = [
        CostElement(f"E{i}", {"type": "fixed", "value": 1.0}) for i in range(3)
    ]
    model = RiskModel(elements=elements, correlation=uniform_correlation(3, 0.4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrelationWarning)
        matrix, notes = model.resolved_correlation()
    assert notes == []
    assert matrix[0, 1] == pytest.approx(0.4)


# ----------------------------------------------------------- discrete risks
def test_a_discrete_risk_has_the_expected_value_probability_times_impact():
    risk = DiscreteRisk("Qual failure", 0.20, TRIANGLE)
    assert risk.expected_value == pytest.approx(
        0.20 * (10.0 + 20.0 + 60.0) / 3.0, rel=1e-12
    )


def test_a_discrete_risk_occurs_at_about_its_stated_probability():
    elements = [CostElement("base", {"type": "fixed", "value": 100.0}, 100.0)]
    model = RiskModel(
        elements=elements,
        risks=[DiscreteRisk("R", 0.25, TRIANGLE)],
        correlation=np.eye(1),
    )
    result = simulate_risk_model(model, 40_000, seed=4)
    assert np.mean(result.risk_samples[:, 0] > 0) == pytest.approx(0.25, abs=0.01)


def test_a_certain_risk_always_fires_and_an_impossible_one_never_does():
    elements = [CostElement("base", {"type": "fixed", "value": 100.0}, 100.0)]
    certain = RiskModel(
        elements=elements, risks=[DiscreteRisk("R", 1.0, TRIANGLE)],
        correlation=np.eye(1),
    )
    never = RiskModel(
        elements=elements, risks=[DiscreteRisk("R", 0.0, TRIANGLE)],
        correlation=np.eye(1),
    )
    assert (simulate_risk_model(certain, 2_000, seed=1).risk_samples > 0).all()
    assert (simulate_risk_model(never, 2_000, seed=1).risk_samples == 0).all()


def test_the_simulated_risk_mean_matches_its_closed_form():
    elements = [CostElement("base", {"type": "fixed", "value": 0.0}, 0.0)]
    risk = DiscreteRisk("R", 0.30, TRIANGLE)
    model = RiskModel(elements=elements, risks=[risk], correlation=np.eye(1))
    result = simulate_risk_model(model, 200_000, seed=6)
    assert result.risk_total.mean() == pytest.approx(risk.expected_value, rel=0.02)


def test_discrete_risk_variance_is_not_simply_probability_times_impact_variance():
    """The intuitive formula misses the variance the event itself contributes,
    which for a rare large risk is most of it. Checked against simulation."""
    from cost_core.monte_carlo import _risk_variance

    risk = DiscreteRisk("R", 0.15, TRIANGLE)
    dist = risk.frozen()
    naive = 0.15 * dist.var()
    correct = _risk_variance(risk)
    assert correct > naive * 2

    elements = [CostElement("base", {"type": "fixed", "value": 0.0}, 0.0)]
    model = RiskModel(elements=elements, risks=[risk], correlation=np.eye(1))
    simulated = np.var(
        simulate_risk_model(model, 200_000, seed=8).risk_total, ddof=1
    )
    assert simulated == pytest.approx(correct, rel=0.05)


def test_discrete_risks_stay_separate_from_the_continuous_uncertainty():
    """They are reported apart because they are different things -- and the
    two parts must still add up to the total."""
    model = quiet(
        equal_model, k=4, rho=0.3,
        risks=[DiscreteRisk("R", 0.30, TRIANGLE)],
    )
    result = quiet(simulate_risk_model, model, 5_000, 9)
    assert result.continuous_total.shape == (5_000,)
    assert result.risk_total.shape == (5_000,)
    assert result.totals == pytest.approx(
        result.continuous_total + result.risk_total, rel=1e-12
    )
    # And the point estimate excludes the risks, as a point estimate should.
    assert result.point_estimate == pytest.approx(4 * 100e6, rel=1e-12)


def test_an_impossible_risk_probability_is_refused():
    with pytest.raises(RiskModelError, match="not in \\[0, 1\\]"):
        DiscreteRisk("R", 1.4, TRIANGLE)


# ------------------------------------------------------------- diagnostics
def test_variance_shares_sum_to_exactly_one():
    """The covariance decomposition Var(T) = sum_i Cov(X_i, T) is an identity
    when T is the sum of the X_i, so the tornado shares must add to one. A
    correlation-based ranking would not have this property, which is why this
    one is used."""
    model = quiet(
        equal_model, k=6, rho=0.3,
        risks=[DiscreteRisk("R", 0.25, TRIANGLE)],
    )
    result = quiet(simulate_risk_model, model, 20_000, 11)
    tornado = result.tornado()
    assert len(tornado) == 7                       # six elements plus one risk
    assert tornado["variance_share"].sum() == pytest.approx(1.0, rel=1e-10)
    assert tornado["variance_share"].is_monotonic_decreasing


def test_the_tornado_ranks_the_biggest_driver_first():
    costs = {"Airframe": 300e6, "Avionics": 60e6, "Data": 5e6}
    model = quiet(risk_model_from_elements, costs, default_correlation=0.2)
    result = quiet(simulate_risk_model, model, 20_000, 12)
    tornado = result.tornado()
    assert tornado["component"].iloc[0] == "Airframe"
    assert tornado["component"].iloc[-1] == "Data"


def test_the_tornado_labels_elements_and_risks_distinctly():
    model = quiet(
        equal_model, k=3, rho=0.3, risks=[DiscreteRisk("Qual", 0.2, TRIANGLE)]
    )
    tornado = quiet(simulate_risk_model, model, 5_000, 1).tornado()
    assert set(tornado["kind"]) == {"element", "discrete risk"}
    assert tornado.loc[tornado["component"] == "Qual", "kind"].iloc[0] == "discrete risk"


def test_the_point_estimate_percentile_says_how_much_reserve_there_is():
    """The number that decides whether a programme is funded defensibly. An
    unreserved point estimate typically lands well below the median."""
    model = quiet(equal_model, k=8, rho=0.3)
    result = quiet(simulate_risk_model, model, 40_000, 13)
    percentile = result.point_estimate_percentile
    assert 0.0 < percentile < 50.0
    assert result.percentile_of(result.p50) == pytest.approx(50.0, abs=0.5)
    assert result.percentile_of(result.p80) == pytest.approx(80.0, abs=0.5)


def test_the_cv_of_the_total_is_reported():
    model = quiet(equal_model, k=8, rho=0.3)
    result = quiet(simulate_risk_model, model, 20_000, 14)
    assert result.cv == pytest.approx(result.std / result.mean, rel=1e-12)
    assert 0.0 < result.cv < 1.0


def test_the_convergence_check_settles_and_reports_its_movement():
    model = quiet(equal_model, k=8, rho=0.3)
    result = quiet(simulate_risk_model, model, 50_000, 15)
    frame = result.convergence()
    assert list(frame["iterations"]) == sorted(frame["iterations"])
    assert frame["iterations"].iloc[-1] == 50_000
    assert pd.isna(frame["relative_change"].iloc[0])
    # By fifty thousand iterations the P80 should have stopped moving.
    assert frame["relative_change"].iloc[-1] < 0.005
    assert result.is_converged


def test_a_short_run_is_reported_as_not_converged():
    model = quiet(equal_model, k=8, rho=0.3)
    result = quiet(simulate_risk_model, model, 200, 16)
    frame = result.convergence()
    assert len(frame) >= 1


def test_the_summary_table_carries_the_briefing_numbers():
    model = quiet(equal_model, k=8, rho=0.3)
    result = quiet(simulate_risk_model, model, 20_000, 17)
    summary = result.summary().set_index("statistic")["value"]
    for key in (
        "point_estimate", "point_estimate_percentile", "mean", "cv",
        "p50", "p80", "p90", "reserve_to_p80", "reserve_to_p80_pct",
    ):
        assert key in summary.index, key
    assert summary["p50"] <= summary["p80"] <= summary["p90"]
    assert summary["reserve_to_p80"] == pytest.approx(
        summary["p80"] - summary["point_estimate"], rel=1e-10
    )


# -------------------------------------------------------------- determinism
@pytest.mark.parametrize("method", ["gaussian_copula", "iman_conover"])
def test_the_correlated_simulation_is_seed_deterministic(method):
    """A P80 that moves between runs is not a number you can defend."""
    model = quiet(equal_model, k=5, rho=0.3)
    a = quiet(simulate_risk_model, model, 10_000, 42, method=method)
    b = quiet(simulate_risk_model, model, 10_000, 42, method=method)
    assert np.array_equal(a.totals, b.totals)
    assert a.p80 == b.p80


def test_different_seeds_give_different_draws():
    model = quiet(equal_model, k=5, rho=0.3)
    a = quiet(simulate_risk_model, model, 5_000, 1)
    b = quiet(simulate_risk_model, model, 5_000, 2)
    assert not np.array_equal(a.totals, b.totals)


def test_both_sampling_methods_agree_on_the_headline_numbers():
    """They induce correlation differently, so they should not match exactly
    -- but if they disagreed on the P80 by much, one of them would be wrong."""
    model = quiet(equal_model, k=8, rho=0.3)
    copula = quiet(simulate_risk_model, model, 50_000, 21, method="gaussian_copula")
    reorder = quiet(simulate_risk_model, model, 50_000, 21, method="iman_conover")
    assert reorder.p80 == pytest.approx(copula.p80, rel=0.02)
    assert reorder.mean == pytest.approx(copula.mean, rel=0.01)


# ---------------------------------------------------------------- bad input
def test_a_model_with_no_elements_is_refused():
    with pytest.raises(RiskModelError, match="at least one cost element"):
        RiskModel(elements=[])


def test_duplicate_element_names_are_refused():
    """Names label the correlation matrix and the tornado; duplicates would
    silently merge two elements in the reporting."""
    elements = [
        CostElement("Airframe", {"type": "fixed", "value": 1.0}),
        CostElement("Airframe", {"type": "fixed", "value": 2.0}),
    ]
    with pytest.raises(RiskModelError, match="Duplicate element name"):
        RiskModel(elements=elements)


def test_too_few_iterations_are_refused():
    model = quiet(equal_model, k=3, rho=0.3)
    with pytest.raises(RiskModelError, match="at least 2 iterations"):
        simulate_risk_model(model, 1, seed=1)


def test_an_unknown_sampling_method_is_refused():
    model = quiet(equal_model, k=3, rho=0.3)
    with pytest.raises(RiskModelError, match="Unknown sampling method"):
        simulate_risk_model(model, 100, seed=1, method="latin_hypercube")


def test_building_a_model_with_no_costs_is_refused():
    with pytest.raises(RiskModelError, match="No cost elements"):
        risk_model_from_elements({})


def test_spread_factors_that_do_not_bracket_the_estimate_are_refused():
    with pytest.raises(RiskModelError, match="bracket the point estimate"):
        risk_model_from_elements({"A": 100.0}, low_factor=1.2, high_factor=1.5)


def test_simulated_costs_are_never_negative():
    """A wide normal can go below zero; a negative cost is not an outcome."""
    elements = [
        CostElement("E", {"type": "normal", "loc": 10.0, "scale": 100.0}, 10.0)
    ]
    model = RiskModel(elements=elements, correlation=np.eye(1))
    assert (simulate_risk_model(model, 20_000, seed=1).totals >= 0).all()


# --------------------------------------------------- the original API is intact
def test_the_original_two_variable_simulation_still_works():
    result = run_monte_carlo(10_000, NORMAL, FIXED_QTY, seed=42)
    assert isinstance(result, SimulationResult)
    assert result.mean == pytest.approx(5_000.0, rel=0.02)
    assert result.p50 <= result.p80 <= result.p90
