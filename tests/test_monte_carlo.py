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
