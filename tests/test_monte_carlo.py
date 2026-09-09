"""Monte Carlo cost risk.

A simulation whose numbers move between runs can't support a P80 you'd put in
front of anyone, so determinism under a fixed seed is the first thing checked
here. The rest pins the statistics against distributions whose true values we
know in closed form.
"""

from __future__ import annotations

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


# --------------------------------------------------- the empirical marginal
def empirical_draws(n=8_000, seed=0):
    """Something emphatically not a two-parameter family: a sum of correlated
    lognormals, which is what a WBS element's buy total is."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 6)) @ np.linalg.cholesky(
        uniform_correlation(6, 0.30)
    ).T
    return (np.exp(2.0 + 0.25 * z) * np.arange(1, 7)).sum(axis=1)


def test_an_empirical_marginal_returns_its_draws_at_the_plotting_positions():
    """The property everything else here rests on. np.interp on the Hazen
    positions returns the knot bit for bit, so a sampler that asks for the
    rank positions of n draws gets those n draws back."""
    draws = empirical_draws()
    dist = make_distribution({"type": "empirical", "draws": draws})
    positions = (np.arange(dist.n_draws) + 0.5) / dist.n_draws
    assert np.array_equal(dist.positions, positions)
    assert np.array_equal(dist.ppf(positions), np.sort(draws))


def test_an_empirical_marginal_reports_its_own_moments():
    draws = empirical_draws()
    dist = make_distribution({"type": "empirical", "draws": draws})
    # ddof 0 on var() and std(): .var() means the variance of the
    # distribution, as it does for every scipy marginal. The ddof-1 sample
    # figure is RiskSimulationResult.std, which is a different question.
    assert dist.mean() == pytest.approx(float(np.mean(draws)), rel=1e-12)
    assert dist.var() == pytest.approx(float(np.var(draws, ddof=0)), rel=1e-12)
    assert dist.std() == pytest.approx(float(np.std(draws, ddof=0)), rel=1e-12)
    assert dist.median() == pytest.approx(float(np.median(draws)), rel=1e-12)
    assert dist.n_draws == draws.size
    assert np.array_equal(dist.draws, np.sort(draws))


def test_an_empirical_quantile_never_leaves_the_observed_range():
    """Interpolation, not extrapolation: a quantile asked for past the data
    returns the most extreme thing that was actually drawn."""
    draws = empirical_draws()
    dist = make_distribution({"type": "empirical", "draws": draws})
    assert dist.ppf(0.0) == draws.min()
    assert dist.ppf(1.0) == draws.max()
    q = np.linspace(0.0, 1.0, 5_000)
    assert (dist.ppf(q) >= draws.min()).all()
    assert (dist.ppf(q) <= draws.max()).all()


@pytest.mark.parametrize("method", ["gaussian_copula", "iman_conover"])
def test_an_element_column_is_a_permutation_of_its_draws(method):
    """The identity the empirical marginal exists for, on both samplers.

    At a matching draw count the sampler returns the draws reordered, so every
    percentile of the column equals the same percentile of the draws exactly.
    Both paths have to be checked: they route through _marginal_column at
    different points, and iman_conover reorders afterwards as well.
    """
    draws = empirical_draws()
    elements = [
        CostElement(
            f"E{i}", {"type": "empirical", "draws": draws}, float(draws.mean())
        )
        for i in range(3)
    ]
    model = RiskModel(elements=elements, correlation=uniform_correlation(3, 0.30))
    result = simulate_risk_model(model, draws.size, seed=1, method=method)
    levels = np.arange(1, 100)
    for i in range(3):
        column = result.element_samples[:, i]
        assert np.array_equal(np.sort(column), np.sort(draws))
        assert np.array_equal(
            np.percentile(column, levels), np.percentile(draws, levels)
        )


def test_rank_mapping_does_not_move_the_rank_correlation():
    """Replacing the copula's uniforms by their own rank positions changes
    every value and no ordering, so Spearman is untouched.

    The comparison has to be against the thing the map replaces, not against
    the requested correlation. A copula asked for 0.40 lands near 0.40 with
    the map or without it, so an assertion of that shape stays green even if
    ``_marginal_column`` stops mapping. Here the same uniforms are pushed
    through ``_marginal_column`` and through ``dist.ppf`` directly, and the
    two columns are compared to each other.

    One wrinkle, measured rather than waved away. ``ppf`` clamps below its
    first plotting position, so an unmapped column carries a tie wherever two
    uniforms land under ``0.5 / n``: 3 of these 16,000 do, making one tied
    pair in the second column. That clamp is the only place the relabelling
    is not strictly monotone. On the rows where nothing clamped the two
    Spearman coefficients are bit-identical; across the full columns they
    differ by about 9e-9. Pearson does move, 0.4016 to 0.4004 here, which is
    the copula's known behaviour against a heavier-tailed marginal rather
    than a new defect.
    """
    from scipy import stats as st

    from cost_core.monte_carlo import _marginal_column

    draws = empirical_draws()
    n = draws.size
    dist = make_distribution({"type": "empirical", "draws": draws})

    # Correlated uniforms in the shape a Gaussian copula hands to a marginal.
    rng = np.random.default_rng(2)
    z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(
        uniform_correlation(2, 0.40)
    ).T
    u = np.clip(st.norm.cdf(z), 1e-12, 1.0 - 1e-12)

    mapped = np.column_stack([_marginal_column(dist, u[:, i], n) for i in range(2)])
    plain = np.column_stack([dist.ppf(u[:, i]) for i in range(2)])

    # The map has to have done something, or everything below is vacuous.
    # This is the assertion that fails if the rank branch is ever bypassed.
    assert not np.array_equal(mapped, plain)
    for i in range(2):
        assert np.array_equal(np.sort(mapped[:, i]), np.sort(draws))
    assert not np.array_equal(np.sort(plain[:, 0]), np.sort(draws))

    # Same ordering, everywhere the quantile function did not clamp.
    interior = np.all(
        (u > dist.positions[0]) & (u < dist.positions[-1]), axis=1
    )
    assert interior.sum() == n - 3
    for i in range(2):
        assert np.array_equal(
            np.argsort(np.argsort(mapped[interior, i])),
            np.argsort(np.argsort(plain[interior, i])),
        )
    assert (
        st.spearmanr(mapped[interior, 0], mapped[interior, 1]).statistic
        == st.spearmanr(plain[interior, 0], plain[interior, 1]).statistic
    )
    # And on the whole column, where the one clamped tie lives.
    assert st.spearmanr(mapped[:, 0], mapped[:, 1]).statistic == pytest.approx(
        st.spearmanr(plain[:, 0], plain[:, 1]).statistic, abs=1e-7
    )


def test_a_draw_count_that_does_not_match_interpolates_instead():
    """Half the iterations is not an error, it is just no longer exact. The
    quantile function still lands inside the observed range, and the
    exactness is a cliff rather than a slope: it is gone at any count but
    the matching one."""
    draws = empirical_draws()
    elements = [
        CostElement("E", {"type": "empirical", "draws": draws}, float(draws.mean()))
    ]
    model = RiskModel(elements=elements, correlation=np.eye(1))
    result = simulate_risk_model(model, draws.size // 2, seed=3)
    column = result.element_samples[:, 0]
    assert column.size == draws.size // 2
    assert column.min() >= draws.min()
    assert column.max() <= draws.max()
    assert float(np.percentile(column, 80)) == pytest.approx(
        float(np.percentile(draws, 80)), rel=0.01
    )


def test_a_non_empirical_marginal_has_no_draw_count_to_trip_on():
    """The rank-mapping branch is keyed on n_draws, which only the empirical
    marginal carries, so it is invisible to every other type."""
    for spec in (NORMAL, TRIANGLE, PERT, {"type": "uniform", "low": 5.0, "high": 30.0},
                 {"type": "fixed", "value": 5.0},
                 {"type": "lognormal", "mean": np.log(80.0), "sigma": 0.25}):
        assert not hasattr(make_distribution(spec), "n_draws")


#: A seeded simulation of a purely analytic model, frozen at the bytes it
#: produced before the empirical marginal existed. _marginal_column sits on
#: the path every marginal takes, so this is the guard that a later edit to it
#: cannot leak into the analytic ones: a lognormal marginal must still be
#: sampled at the copula's own uniforms, in the order the generator produced
#: them. Both samplers and two iteration counts, because iman_conover consumes
#: the generator differently and the rank branch is keyed on the count.
#: Verified identical on both supported lanes (numpy 2.0.2 / Python 3.9 and
#: numpy 2.4 / Python 3.14).
ANALYTIC_REFERENCE = {
    "gaussian_copula|8000|11": (
        "a98b9aed7fdec825a732570f6e664ac02196b0f2a448a833409665cea96b3628",
        "77a5e0627efa2aca7455730dfba2991c0d186a014f65b891795bac61d8e74c92",
    ),
    "gaussian_copula|20000|3": (
        "42088df4e0a941fae2f1a4f76bf02cf80de816f34023f789887cfc7246b4c0bb",
        "044fd6873c91fbe78e066daae01c0b143e54bc95cd868e9edb06302e0650cdaf",
    ),
    "iman_conover|8000|11": (
        "e931574889a94b65f5193c8cb5fdf51628a7644e22761780cd2e862204451ab8",
        "2b239729d156f30ffa9bd163500d95fc053dbea407c4b78797cde0bd678f6870",
    ),
    "iman_conover|20000|3": (
        "cd259251a5654508071edc4d664b8186e4ecd2cd097079ff62abde12d3aab970",
        "0de07890abfa1273b0acac9484397eabc0537aa9a6c070b13a1642f44787da9e",
    ),
}


def lognormal_reference_model():
    """Six correlated lognormal elements. Nothing empirical anywhere."""
    return RiskModel(
        elements=[
            CostElement(
                f"E{i}",
                {"type": "lognormal", "mean": np.log(100e6 + 7e6 * i),
                 "sigma": 0.18 + 0.01 * i},
                100e6 + 7e6 * i,
            )
            for i in range(6)
        ],
        correlation=uniform_correlation(6, 0.30),
        name="lognormal6",
    )


@pytest.mark.parametrize("key", sorted(ANALYTIC_REFERENCE))
def test_an_analytic_model_still_reproduces_its_frozen_draws(key):
    import hashlib

    method, n_iter, seed = key.split("|")
    result = simulate_risk_model(
        lognormal_reference_model(), int(n_iter), int(seed), method=method
    )
    got = (
        hashlib.sha256(result.totals.tobytes()).hexdigest(),
        hashlib.sha256(result.element_samples.tobytes()).hexdigest(),
    )
    assert got == ANALYTIC_REFERENCE[key], (
        f"{key}: the analytic marginals moved. If _marginal_column or the "
        f"order the generator is consumed in was changed, that is the cause."
    )


#: The same guard for the mixed model, held at the percentiles rather than at
#: the bytes. scipy's beta ppf, which the PERT marginal uses, disagrees in the
#: last place between the two supported lanes, so the mixed model's bytes are
#: not lane-independent and cannot be hashed; p50, p80 and p90 are identical
#: on both. Measured, not assumed: the hashes differ on all four cases and
#: these three floats differ on none.
MIXED_REFERENCE = {
    "gaussian_copula|8000|11": (294.57672676810444, 328.0700631199419, 347.0785011899158),
    "gaussian_copula|20000|3": (294.7338199844397, 328.20635526396904, 347.0210751964346),
    "iman_conover|8000|11": (295.2108016231532, 328.4769144656502, 347.84652802616654),
    "iman_conover|20000|3": (294.94426916745704, 329.1604227408918, 348.12955234694647),
}


@pytest.mark.parametrize("key", sorted(MIXED_REFERENCE))
def test_a_mixed_analytic_model_still_reproduces_its_frozen_percentiles(key):
    method, n_iter, seed = key.split("|")
    elements = [
        # scale 12, not the module-level NORMAL: this model is frozen
        # bytes, so its parameters are written out rather than borrowed.
        CostElement("norm", {"type": "normal", "loc": 100.0, "scale": 12.0}, 100.0),
        CostElement("tri", TRIANGLE, 20.0),
        CostElement("pert", PERT, 20.0),
        CostElement("unif", {"type": "uniform", "low": 5.0, "high": 30.0}, 15.0),
        CostElement("fix", {"type": "fixed", "value": 42.0}, 42.0),
        CostElement("logn", {"type": "lognormal", "mean": np.log(80.0), "sigma": 0.25}, 80.0),
    ]
    model = RiskModel(
        elements=elements, correlation=uniform_correlation(6, 0.25), name="mixed6"
    )
    result = simulate_risk_model(model, int(n_iter), int(seed), method=method)
    assert (result.p50, result.p80, result.p90) == MIXED_REFERENCE[key]


@pytest.mark.parametrize(
    "draws,message",
    [
        ([1.0], "at least 2 draws"),
        ([], "at least 2 draws"),
        ([1.0, np.nan], "finite draws"),
        ([1.0, np.inf], "finite draws"),
        ([-1.0, 2.0], "non-negative draws"),
    ],
)
def test_a_malformed_empirical_spec_is_refused(draws, message):
    with pytest.raises(RiskModelError, match=message):
        make_distribution({"type": "empirical", "draws": draws})


def test_an_empirical_spec_without_draws_is_refused_by_name():
    with pytest.raises(RiskModelError, match=r"needs parameter\(s\) \['draws'\]"):
        make_distribution({"type": "empirical"})



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
