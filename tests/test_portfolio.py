# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Portfolio optimisation.

The solver is checked against brute force: on small random portfolios every
combination of options is enumerated, and the integer program has to find the
same best value with every constraint honoured. The rest pins the behaviour a
briefing leans on: mandatory programs, dependencies and exclusive alternatives
respected, an infeasible budget explained by year, a year with slack worth
nothing at the margin, and budget risk that grows with correlated overruns.
"""
import itertools
import random

import numpy as np
import pytest

pulp = pytest.importorskip("pulp")

from cost_core.aoa import Alternative, CostLine, annual, evaluate, spread  # noqa: E402
from cost_core.ingest import InflationTable  # noqa: E402
from cost_core.portfolio import (Candidate, Option, Portfolio, PortfolioError,  # noqa: E402
                                 budget_risk, candidates_from_aoa, frontier,
                                 marginal_value, solve, spend_by_year)

YEARS = [2027, 2028, 2029]


def opt(name, costs, value):
    return Option(name, dict(zip(YEARS, costs)), value)


def brute_force(p: Portfolio):
    """Best value over every assignment of an option (or none) to each candidate."""
    best = -1.0
    choices = [[None] + list(c.options) for c in p.candidates]
    names = [c.name for c in p.candidates]
    for combo in itertools.product(*choices):
        funded = {n: o is not None for n, o in zip(names, combo)}
        if any(c.mandatory and not funded[c.name] for c in p.candidates):
            continue
        if any(funded[c.name] and not funded[r] for c in p.candidates for r in c.requires):
            continue
        if any(sum(funded[n] for n in g) > 1 for g in p.exclusive):
            continue
        spend = {y: sum(o.cost_by_year.get(y, 0.0) for o in combo if o) for y in p.budget}
        if any(spend[y] > p.budget[y] + 1e-9 for y in p.budget):
            continue
        best = max(best, sum(o.value for o in combo if o))
    return best


def random_portfolio(rng):
    cands = []
    for i in range(rng.randint(3, 6)):
        opts = [opt(f"o{j}", [rng.randint(0, 40) for _ in YEARS], rng.randint(1, 30))
                for j in range(rng.randint(1, 3))]
        cands.append(Candidate(f"c{i}", opts))
    names = [c.name for c in cands]
    if rng.random() < 0.5:
        a, b = rng.sample(names, 2)
        cands = [Candidate(c.name, c.options, requires=(b,)) if c.name == a else c for c in cands]
    exclusive = [rng.sample(names, 2)] if rng.random() < 0.5 else []
    budget = {y: rng.randint(30, 90) for y in YEARS}
    return Portfolio(cands, budget, exclusive)


@pytest.mark.parametrize("seed", range(40))
def test_solver_matches_brute_force(seed):
    rng = random.Random(seed)
    p = random_portfolio(rng)
    r = solve(p)
    assert r.value == pytest.approx(brute_force(p))
    assert (r.spend.spend <= r.spend.budget + 1e-9).all()
    funded = set(r.funded)
    for c in p.candidates:
        for req in c.requires:
            assert c.name not in funded or req in funded
    for g in p.exclusive:
        assert len(funded & set(g)) <= 1


def small():
    return Portfolio(
        candidates=[
            Candidate("Sustain fleet", [opt("Full", [30, 30, 30], 5)], mandatory=True),
            Candidate("Radar upgrade", [opt("Full", [20, 20, 0], 12), opt("Deferred", [0, 20, 20], 9)]),
            Candidate("New missile", [opt("Full", [25, 25, 25], 20)]),
            Candidate("Missile integration", [opt("Full", [5, 5, 5], 6)], requires=("New missile",)),
        ],
        budget={2027: 60, 2028: 80, 2029: 80},
    )


def test_constraints_shape_the_answer():
    r = solve(small())
    s = r.selected.set_index("candidate")
    assert s.loc["Sustain fleet", "option"] == "Full"
    # The full radar and the missile cannot both start in 2027; deferring the
    # radar lets everything in, filling every year exactly.
    assert r.choice == {"Sustain fleet": "Full", "Radar upgrade": "Deferred",
                        "New missile": "Full", "Missile integration": "Full"}
    assert r.value == pytest.approx(5 + 9 + 20 + 6)
    spend = spend_by_year(small(), r.choice).set_index("year").spend
    assert list(spend) == [60, 80, 80]


def test_infeasible_budget_names_the_year():
    p = small()
    with pytest.raises(PortfolioError, match="2027: mandatory programs need at least 30"):
        solve(p, budget_override={2027: 20, 2028: 80, 2029: 80})


def test_marginal_value_is_zero_where_money_is_idle_and_names_what_changes():
    # 5 less in 2027: missile and integration no longer both fit there, so
    # integration drops. 2027 is the binding year; money added in 2028 or
    # 2029 buys nothing, money added back in 2027 buys integration.
    p = Portfolio(small().candidates, {2027: 55, 2028: 80, 2029: 80})
    assert solve(p).value == pytest.approx(5 + 9 + 20)
    mv = marginal_value(p, delta=5).set_index("year")
    assert mv.loc[2027, "value_gain"] == pytest.approx(6)
    assert mv.loc[2027, "value_per_unit"] == pytest.approx(6 / 5)
    assert mv.loc[2027, "changes"] == "Missile integration"
    assert mv.loc[2028, "value_gain"] == 0 and mv.loc[2029, "value_gain"] == 0


def test_frontier_never_loses_value_as_budget_grows():
    f = frontier(small(), scales=[0.3, 0.6, 0.8, 1.0, 1.2, 1.5])
    assert np.isnan(f.value.iloc[0])  # mandatory program no longer fits
    vals = f.value.dropna().to_numpy()
    assert np.all(np.diff(vals) >= -1e-9)
    assert f.value.iloc[-1] == pytest.approx(5 + 12 + 20 + 6)


def test_planning_to_a_growth_factor_buys_less():
    base, p80 = solve(small()), solve(small(), cost_factor=1.15)
    assert p80.value <= base.value
    assert (p80.spend.spend <= p80.spend.budget + 1e-9).all()


def test_budget_risk_with_no_growth_is_zero_and_with_certain_growth_is_one():
    p = small()
    choice = solve(p).choice
    none = budget_risk(p, choice, {"type": "fixed", "value": 1.0}, n_iter=200)
    assert (none.p_over_budget == 0).all()
    over = budget_risk(p, choice, {"type": "fixed", "value": 1.2}, n_iter=200)
    assert over.set_index("year").loc[2027, "p_over_budget"] == 1.0  # 55 * 1.2 > 60


def test_correlated_overruns_widen_the_tail():
    p = small()
    choice = solve(p).choice
    growth = {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.5}
    ind = budget_risk(p, choice, growth, n_iter=40_000, seed=1)
    cor = budget_risk(p, choice, growth, n_iter=40_000, seed=1, correlation=0.8)
    assert (cor.p80 > ind.p80).all()
    assert ind.p50.to_numpy() == pytest.approx(cor.p50.to_numpy(), rel=0.02)


def test_bad_portfolios_are_refused():
    with pytest.raises(PortfolioError, match="not a candidate"):
        Portfolio([Candidate("A", [opt("o", [1, 1, 1], 1)], requires=("Z",))], {y: 10 for y in YEARS})
    with pytest.raises(PortfolioError, match="does not cover"):
        Portfolio([Candidate("A", [Option("o", {2035: 1.0}, 1)])], {y: 10 for y in YEARS})
    with pytest.raises(PortfolioError, match="Duplicate"):
        Portfolio([Candidate("A", [opt("o", [1, 1, 1], 1)])] * 2, {y: 10 for y in YEARS})
    with pytest.raises(PortfolioError, match="unknown"):
        Portfolio([Candidate("A", [opt("o", [1, 1, 1], 1)])], {y: 10 for y in YEARS}, [["A", "B"]])


def test_an_aoa_feeds_the_portfolio_as_exclusive_alternatives():
    tri = {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.3}
    alts = [
        Alternative("Upgrade", [CostLine("Dev", "RDT&E", spread(60, 2027, 3), tri),
                                CostLine("O&S", "O&S", annual(10, 2030, 2040))], effectiveness=10),
        Alternative("New", [CostLine("Dev", "RDT&E", spread(120, 2027, 3), tri)], effectiveness=18),
    ]
    aoa = evaluate(alts, base_year=2026, discount_rate=0.02, n_iter=200,
                   inflation=InflationTable.from_rate(0.02, 2026, 2020, 2050))
    cands, groups = candidates_from_aoa(aoa, years=YEARS, prefix="Sensor: ")
    assert [c.name for c in cands] == ["Sensor: Upgrade", "Sensor: New"]
    assert sum(cands[0].options[0].cost_by_year.values()) == pytest.approx(60)  # O&S is outside
    p = Portfolio(cands + [Candidate("Other", [opt("Full", [20, 20, 20], 15)])],
                  {y: 50 for y in YEARS}, groups)
    r = solve(p)
    assert len(set(r.funded) & set(groups[0])) == 1
    # New alone (18) loses to Upgrade plus Other (10 + 15): New and Other
    # together need 60 a year against 50.
    assert r.choice == {"Sensor: Upgrade": "Fund", "Sensor: New": None, "Other": "Full"}


# ------------------------------------------------------------ spec + CLI ---
EXAMPLE = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "portfolio_example.json"


def test_example_spec_loads_and_its_choice_respects_every_rule():
    from cost_core.portfolio.spec import load_portfolio

    p, settings = load_portfolio(EXAMPLE)
    assert settings["units"] == "TY $M" and settings["delta"] == 50
    r = solve(p)
    assert r.choice["Fleet sustainment"] == "Full"
    assert len(set(r.funded) & set(p.exclusive[0])) <= 1
    assert r.choice["Missile integration"] is None or r.choice["Long-range missile"] is not None
    assert (r.spend.spend <= r.spend.budget + 1e-9).all()


def test_cli_writes_every_table(tmp_path, monkeypatch, capsys):
    from cost_core import cli

    monkeypatch.setattr("sys.argv", ["ce-core", "portfolio", "--spec", str(EXAMPLE),
                                     "--out", str(tmp_path)])
    cli.main()
    for name in ("selected.csv", "spend.csv", "marginal_value.csv", "frontier.csv",
                 "budget_risk.csv"):
        assert (tmp_path / name).is_file(), name
    out = capsys.readouterr().out
    assert "Fleet sustainment" in out and "breaks its budget" in out
