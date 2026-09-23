# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Intervals and the simulated buy total, on the lot cost engine's own run.

These come from the desktop tool's risk bridge tests. The enrichment layer
takes the objects run_lot_cost_model already produced, so these care mostly
about one thing: the risk numbers must describe the same lots, the same
selected model and the same point estimate as the projections sheet. If they
ever drift apart, the tool is reporting a distribution around a number it is
not showing anyone.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import cost_core.lotmodel as M
from cost_core.lotmodel.enrich import (projection_intervals,
                                       selected_model_name, simulate_buy)


def risk(ctx, proj, summary, *, level=0.80, n_iter=4000, seed=11,
         lot_correlation=0.30, simulate=True):
    """Intervals and, optionally, a simulated buy, straight from enrich.

    The same calls the desktop tool's bridge makes, with the same defaults
    its tests used, gathered into one object so each test reads as it did.
    """
    model = selected_model_name(summary)
    intervals = projection_intervals(ctx, proj, model, level=level)
    res = SimpleNamespace(
        model=model,
        intervals=intervals,
        total_point=float(intervals["Lot Cost ($)"].sum()),
        total_lower=float(intervals["Lot Cost Lower"].sum()),
        total_upper=float(intervals["Lot Cost Upper"].sum()),
    )
    if simulate:
        buy = simulate_buy(
            ctx, proj, model, n_iter=n_iter, seed=seed,
            lot_correlation=lot_correlation,
        )
        res.point_percentile = float(buy.point_estimate_percentile)
        res.p50 = float(buy.p50)
        res.p80 = float(buy.p80)
        res.p90 = float(buy.p90)
        res.sim_cv = float(buy.cv)
    return res


@pytest.fixture(scope="module")
def fitted(analogy_df, estimate_df):
    proj, ctx = M.run_lot_cost_model(analogy_df, estimate_df)
    summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
    return ctx, proj, summary


@pytest.fixture(scope="module")
def result(fitted):
    ctx, proj, summary = fitted
    return risk(ctx, proj, summary, n_iter=20000)


class TestAgreementWithTheModel:
    """The whole point of enriching rather than refitting."""

    def test_uses_the_model_the_tool_selected(self, fitted, result):
        _, _, summary = fitted
        selected = summary.loc[summary["Item"] == "SELECTED"]
        picked = [c for c in ("LC", "Rate", "LC+Rate")
                  if selected.iloc[0][c] == "YES"]
        assert result.model == picked[0]

    def test_point_estimate_matches_the_projections_sheet(self, fitted, result):
        _, proj, _ = fitted
        col = f"{result.model} Lot Cost After Complexity ($)"
        assert result.total_point == pytest.approx(
            proj[col].sum(), rel=1e-6
        )

    def test_one_interval_row_per_forecast_lot(self, fitted, result):
        _, proj, _ = fitted
        assert len(result.intervals) == len(proj)

    def test_simulation_centres_on_the_point_estimate(self, result):
        assert 35 < result.point_percentile < 65


class TestIntervals:
    def test_interval_brackets_the_point_estimate(self, result):
        iv = result.intervals
        assert (iv["Unit Cost Lower"] <= iv["Unit Cost ($K)"]).all()
        assert (iv["Unit Cost ($K)"] <= iv["Unit Cost Upper"]).all()

    def test_total_is_the_sum_of_the_lots(self, result):
        assert result.total_point == pytest.approx(
            result.intervals["Lot Cost ($)"].sum(), rel=1e-9
        )
        assert result.total_lower < result.total_point < result.total_upper

    def test_the_band_tracks_the_selected_model(self, result):
        # The example selects LC+Rate, whose unit cost turns back up on the
        # small final lot, so this deliberately does not assert a falling
        # curve. What must hold is that the band follows the point estimate.
        iv = result.intervals
        point = iv["Unit Cost ($K)"].to_numpy()
        lower = iv["Unit Cost Lower"].to_numpy()
        upper = iv["Unit Cost Upper"].to_numpy()
        assert np.all(np.diff(np.sign(np.diff(point))) >= -1)  # one turn
        np.testing.assert_array_less(lower, point)
        np.testing.assert_array_less(point, upper)

    def test_a_wider_level_gives_a_wider_interval(self, fitted):
        ctx, proj, summary = fitted
        narrow = risk(ctx, proj, summary, level=0.50, simulate=False)
        wide = risk(ctx, proj, summary, level=0.95, simulate=False)
        assert (wide.total_upper - wide.total_lower) > (
            narrow.total_upper - narrow.total_lower
        )


class TestSimulation:
    def test_percentiles_are_ordered(self, result):
        assert result.p50 < result.p80 < result.p90

    def test_same_seed_reproduces_the_answer(self, fitted):
        ctx, proj, summary = fitted
        a = risk(ctx, proj, summary)
        b = risk(ctx, proj, summary)
        assert a.p80 == b.p80

    def test_a_different_seed_moves_the_answer(self, fitted):
        ctx, proj, summary = fitted
        a = risk(ctx, proj, summary, seed=1)
        b = risk(ctx, proj, summary, seed=2)
        assert a.p80 != b.p80

    def test_more_correlation_widens_the_buy(self, fitted):
        # Consecutive lots moving together stops the shocks cancelling.
        ctx, proj, summary = fitted
        low = risk(ctx, proj, summary, lot_correlation=0.0)
        high = risk(ctx, proj, summary, lot_correlation=0.9)
        assert high.sim_cv > low.sim_cv


class TestComplexityFactor:
    def test_is_already_in_the_intervals(self, analogy_df, estimate_df):
        # The projections carry complexity, so the intervals inherit it and
        # doubling the factor doubles the whole distribution.
        doubled = estimate_df.copy()
        doubled["Complexity"] = estimate_df["Complexity"] * 2

        def total(est):
            proj, ctx = M.run_lot_cost_model(analogy_df, est)
            summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
            return risk(ctx, proj, summary, simulate=False).total_point

        assert total(doubled) == pytest.approx(2 * total(estimate_df), rel=1e-6)


class TestBasis:
    def test_continuation_is_cheaper_than_pricing_from_unit_one(
        self, analogy_df, estimate_df
    ):
        def total(prior):
            proj, ctx = M.run_lot_cost_model(
                analogy_df, estimate_df, {"FcstPriorUnits": prior}
            )
            summary = M.generate_analyst_summary(ctx, {"Program": "TEST"})
            return risk(ctx, proj, summary, simulate=False).total_point

        assert total(110) < total(0)
