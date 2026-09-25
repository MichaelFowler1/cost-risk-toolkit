# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Schedule risk and joint cost and schedule confidence.

The deterministic schedule is checked against a textbook network whose float
is known by hand. The simulation is checked against things that are true by
construction rather than against recorded output: with no uncertainty every
draw is the point estimate; every simulated cost is rebuilt exactly from its
parts; a certain risk on a critical activity moves the finish by its delay,
and one on an activity with enough float does not; two uncertain paths that
merge finish later on average than either alone (merge bias); joint
confidence never exceeds either marginal; and every point on the frontier
reaches the confidence it was drawn for.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from cost_core.schedule import (Activity, Project, Risk, ScheduleError,
                                critical_path, point_estimate, simulate)


def tri(lo, hi):
    return {"type": "triangular", "left": lo, "mode": 1.0, "right": hi}


def textbook(**kw):
    # A(3) -> B(4) -> D(5);  A -> C(2) -> D.  Critical A-B-D, finish 12, C has float 2.
    acts = [
        Activity("A", 3, kw.get("ua"), burn_rate=1.0),
        Activity("B", 4, kw.get("ub"), ["A"], fixed_cost=10, burn_rate=2.0),
        Activity("C", 2, kw.get("uc"), ["A"], fixed_cost=5),
        Activity("D", 5, kw.get("ud"), ["B", "C"], burn_rate=0.5),
    ]
    return Project(acts, kw.get("risks", ()), standing_army=kw.get("army", 0.0))


# ------------------------------------------------------------------ network ---
def test_critical_path_matches_the_hand_calculation():
    cpm = critical_path(textbook()).set_index("activity")
    assert list(cpm.early_start) == [0, 3, 3, 7]
    assert list(cpm.early_finish) == [3, 7, 5, 12]
    assert cpm.loc["C", "total_float"] == pytest.approx(2)
    assert list(cpm.index[cpm.critical]) == ["A", "B", "D"]
    assert point_estimate(textbook(army=1.0)) == (12.0, pytest.approx(3 + 10 + 8 + 5 + 2.5 + 12))


def test_lags_and_leads_move_the_schedule():
    p = Project([Activity("A", 3), Activity("B", 4, predecessors=[("A", 2)]),
                 Activity("C", 5, predecessors=[("A", -1)])])
    cpm = critical_path(p).set_index("activity")
    assert cpm.loc["B", "early_start"] == 5 and cpm.loc["C", "early_start"] == 2
    assert cpm.early_finish.max() == 9


def test_bad_networks_are_refused():
    with pytest.raises(ScheduleError, match="cycle"):
        Project([Activity("A", 1, predecessors=["B"]), Activity("B", 1, predecessors=["A"])])
    with pytest.raises(ScheduleError, match="not an activity"):
        Project([Activity("A", 1, predecessors=["Z"])])
    with pytest.raises(ScheduleError, match="itself"):
        Project([Activity("A", 1, predecessors=["A"])])
    with pytest.raises(ScheduleError, match="Duplicate"):
        Project([Activity("A", 1), Activity("A", 2)])
    with pytest.raises(ScheduleError, match="unknown"):
        Project([Activity("A", 1)], [Risk("r", 0.5, ["B"], 1)])
    with pytest.raises(ScheduleError, match="duration"):
        Activity("A", -1)
    with pytest.raises(ScheduleError, match="probability"):
        Risk("r", 1.5)


# --------------------------------------------------------------- simulation ---
def test_without_uncertainty_every_draw_is_the_point_estimate():
    r = simulate(textbook(army=1.0), n_iter=500)
    assert np.all(r.finish == 12) and np.allclose(r.cost, r.point_cost)
    assert r.point_jcl == 1.0
    crit = r.criticality().set_index("activity").criticality
    assert crit[["A", "B", "D"]].tolist() == [1.0, 1.0, 1.0] and crit["C"] == 0.0


def test_every_cost_rebuilds_from_its_parts():
    p = textbook(ua=tri(0.8, 1.5), ub=tri(0.9, 2.0), uc=tri(0.5, 3.0), ud=tri(0.9, 1.3), army=2.0,
                 risks=[Risk("late part", 0.4, ["C"], {"type": "uniform", "low": 1, "high": 4}, 7)])
    r = simulate(p, n_iter=5000, seed=3)
    rates = np.array([1.0, 2.0, 0.0, 0.5])
    fixed = 10 + 5
    rebuilt = fixed + r.durations @ rates + 2.0 * r.finish + 7 * r.risk_hits["late part"]
    assert np.allclose(r.cost, rebuilt)
    # and the finish is the network's longest path through the drawn durations
    d = pd.DataFrame(r.durations, columns=r.activity_ids)
    longest = d.A + np.maximum(d.B, d.C) + d.D
    assert np.allclose(r.finish, longest)


def test_a_risk_moves_the_finish_only_when_it_beats_the_float():
    on_path = simulate(textbook(risks=[Risk("r", 1.0, ["B"], 3)]), n_iter=100)
    assert np.all(on_path.finish == 15)
    absorbed = simulate(textbook(risks=[Risk("r", 1.0, ["C"], 1)]), n_iter=100)
    assert np.all(absorbed.finish == 12)
    overrun = simulate(textbook(risks=[Risk("r", 1.0, ["C"], 3)]), n_iter=100)
    assert np.all(overrun.finish == 13)
    assert overrun.criticality().set_index("activity").criticality["C"] == 1.0


def test_merging_paths_finish_later_than_the_critical_path_says():
    # Two identical uncertain paths into one finish. Each alone averages its
    # own mean; the later of the two averages more (merge bias).
    u = {"type": "normal", "loc": 1.0, "scale": 0.15}
    one = simulate(Project([Activity("X", 10, u)], duration_correlation=0.0), n_iter=40_000, seed=1)
    two = simulate(Project([Activity("X", 10, u), Activity("Y", 10, u)], duration_correlation=0.0),
                   n_iter=40_000, seed=1)
    assert one.finish.mean() == pytest.approx(10, abs=0.05)
    assert two.finish.mean() > 10.6  # E[max of two N(10, 1.5)] = 10 + 1.5/sqrt(pi) = 10.85


def test_correlated_durations_widen_the_schedule():
    chain = [Activity(f"a{i}", 5, tri(0.8, 1.6), [f"a{i-1}"] if i else []) for i in range(6)]
    spread = {}
    for rho in (0.0, 0.8):
        r = simulate(Project(chain, duration_correlation=rho), n_iter=20_000, seed=2)
        spread[rho] = np.percentile(r.finish, 90) - np.percentile(r.finish, 10)
    assert spread[0.8] > 1.5 * spread[0.0]


def test_seeded_runs_repeat():
    p = textbook(ua=tri(0.8, 1.5), ub=tri(0.9, 2.0))
    a, b = simulate(p, n_iter=1000, seed=9), simulate(p, n_iter=1000, seed=9)
    assert np.array_equal(a.finish, b.finish) and np.array_equal(a.cost, b.cost)


# --------------------------------------------------------------------- JCL ---
def jcl_project():
    return textbook(ua=tri(0.8, 1.5), ub=tri(0.9, 2.0), uc=tri(0.5, 3.0), ud=tri(0.9, 1.4),
                    army=1.5, risks=[Risk("late part", 0.3, ["B"], 2, 4)])


def test_joint_confidence_never_beats_either_marginal():
    r = simulate(jcl_project(), n_iter=20_000, seed=4)
    c70, t70 = np.quantile(r.cost, 0.7), np.quantile(r.finish, 0.7)
    j = r.joint(c70, t70)
    assert j <= min(np.mean(r.cost <= c70), np.mean(r.finish <= t70))
    assert j < 0.7  # the two P70s together fall short of 70% jointly
    assert r.point_jcl < 0.2


def test_every_frontier_point_reaches_its_confidence():
    r = simulate(jcl_project(), n_iter=20_000, seed=4)
    f = r.frontier(0.7, points=25)
    assert len(f) > 5
    assert (f.joint >= 0.7 - 1e-12).all()
    assert np.all(np.diff(f.cost) <= 1e-9)  # more time never needs more money
    # and at the latest date the budget needed is the plain cost percentile
    k = math.ceil(0.7 * r.n_iter)
    assert f.cost.iloc[-1] == pytest.approx(np.sort(r.cost)[k - 1])


def test_cost_at_and_finish_at_are_tight_and_say_when_impossible():
    r = simulate(jcl_project(), n_iter=10_000, seed=5)
    t = float(np.quantile(r.finish, 0.85))
    c = r.cost_at(0.7, t)
    assert r.joint(c, t) >= 0.7 and r.joint(c - 1e-6, t) < 0.7
    assert r.finish_at(0.7, c) <= t
    assert math.isnan(r.cost_at(0.7, float(np.quantile(r.finish, 0.5))))
    with pytest.raises(ScheduleError):
        r.frontier(1.2)


def test_the_standing_army_charges_for_every_month():
    base = simulate(jcl_project(), n_iter=3000, seed=6)
    p = jcl_project()
    heavier = simulate(Project(p.activities, p.risks, standing_army=p.standing_army + 1.0),
                       n_iter=3000, seed=6)
    assert np.allclose(heavier.cost - base.cost, base.finish)


# ------------------------------------------------------------ spec + CLI ---
EXAMPLE = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "jcl_example.json"


def test_example_spec_loads():
    from cost_core.schedule.spec import load_project

    p, settings = load_project(EXAMPLE)
    assert settings["units"] == "$M" and settings["confidence"] == 0.7
    assert [a.id for a in p.activities][0] == "design"
    fsw = next(a for a in p.activities if a.id == "fsw")
    assert fsw.links() == [("design", -4.0)]
    assert critical_path(p).early_finish.max() == 43


def test_cli_writes_every_table_and_the_chart(tmp_path, monkeypatch, capsys):
    from cost_core import cli

    monkeypatch.setattr("sys.argv", ["ce-core", "jcl", "--spec", str(EXAMPLE),
                                     "--out", str(tmp_path)])
    cli.main()
    for name in ("summary.csv", "critical_path.csv", "criticality.csv", "frontier.csv",
                 "draws.csv", "assumptions.json"):
        assert (tmp_path / name).is_file(), name
    draws = pd.read_csv(tmp_path / "draws.csv")
    assert len(draws) == 20_000
    assert json.loads((tmp_path / "assumptions.json").read_text())["confidence"] == 0.7
    assert "Where the schedule risk is" in capsys.readouterr().out
