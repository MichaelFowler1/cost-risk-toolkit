# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Life-cycle cost for an analysis of alternatives.

The money arithmetic is checked against closed forms, not recorded output: a
constant-rate index makes then-year dollars a power of (1 + rate), and present
value is a power of (1 + discount rate), so both have exact answers. The rest
checks the decisions the module exists to support: that discounting can
reverse a ranking (the reason A-94 asks for present value), that a dominated
alternative is named, and that uncertainty drawn from SAR history is one
observation per program.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.aoa import (AoAError, Alternative, CostLine, annual,
                           describe_growth, evaluate, growth_factor,
                           historical_growth, spread)
from cost_core.aoa.lcc import _scale_spec
from cost_core.ingest import InflationTable
from cost_core.monte_carlo import make_distribution

INFL = InflationTable.from_rate(0.02, base_year=2026, first_year=2020, last_year=2070)


def tri(left, right):
    return {"type": "triangular", "left": left, "mode": 1.0, "right": right}


def one_line(name, years, uncertainty=None, effectiveness=None):
    return Alternative(name, [CostLine("all", "Procurement", years, uncertainty)],
                       effectiveness=effectiveness)


# ----------------------------------------------------------------- phasing ---
@pytest.mark.parametrize("profile", ["uniform", "front", "back", "bell"])
def test_spread_lands_on_the_total(profile):
    s = spread(1234.5, 2027, 7, profile)
    assert sorted(s) == list(range(2027, 2034))
    assert sum(s.values()) == pytest.approx(1234.5, abs=1e-9)
    assert all(v > 0 for v in s.values())


def test_spread_shapes():
    front, back, bell = (spread(100, 2027, 5, p) for p in ("front", "back", "bell"))
    assert front[2027] > front[2031] and back[2027] < back[2031]
    assert bell[2029] > bell[2027] and bell[2027] == pytest.approx(bell[2031])


def test_phasing_errors():
    with pytest.raises(AoAError):
        spread(100, 2027, 0)
    with pytest.raises(AoAError):
        spread(100, 2027, 3, "wedge")
    with pytest.raises(AoAError):
        annual(10, 2030, 2029)


# ------------------------------------------------------------------- money ---
def test_then_year_and_present_value_are_exact():
    a = one_line("A", {2026: 100.0, 2030: 100.0})
    b = one_line("B", {2026: 1.0})
    r = evaluate([a, b], base_year=2026, inflation=INFL, discount_rate=0.03, n_iter=100)
    row = r.lines[r.lines.alternative == "A"].iloc[0]
    assert row.by == pytest.approx(200.0)
    assert row.ty == pytest.approx(100 + 100 * 1.02 ** 4)
    assert row.pv == pytest.approx(100 + 100 / 1.03 ** 4)


def test_pv_year_moves_the_discounting_point():
    a = one_line("A", {2030: 100.0})
    r = evaluate([a, one_line("B", {2026: 1.0})], base_year=2026, inflation=INFL,
                 discount_rate=0.03, pv_year=2028, n_iter=100)
    assert r.lines.iloc[0].pv == pytest.approx(100 / 1.03 ** 2)


def test_discounting_can_reverse_the_ranking():
    # Spend 100 now, or 105 fourteen years out. Constant dollars favour now;
    # at a 2% real rate the deferred 105 is worth about 79.6 today.
    now = one_line("Now", {2026: 100.0})
    later = one_line("Later", {2040: 105.0})
    by = evaluate([now, later], base_year=2026, inflation=INFL, discount_rate=0.02,
                  basis="by", n_iter=200)
    pv = evaluate([now, later], base_year=2026, inflation=INFL, discount_rate=0.02,
                  basis="pv", n_iter=200)
    assert list(by.summary.p_cheapest) == [1.0, 0.0]
    assert list(pv.summary.p_cheapest) == [0.0, 1.0]
    assert pv.summary.pv.iloc[1] == pytest.approx(105 / 1.02 ** 14)


def test_no_uncertainty_means_no_spread():
    r = evaluate([one_line("A", {2027: 50.0}), one_line("B", {2027: 60.0})],
                 base_year=2026, inflation=INFL, discount_rate=0.0, n_iter=500)
    assert np.all(r.draws["A"] == pytest.approx(50.0))
    assert r.summary.p80.iloc[1] == pytest.approx(60.0)


# ------------------------------------------------------------- uncertainty ---
@pytest.mark.parametrize("spec", [
    tri(0.8, 1.6),
    {"type": "pert", "left": 0.9, "mode": 1.0, "right": 1.5},
    {"type": "uniform", "low": 0.9, "high": 1.3},
    {"type": "normal", "loc": 1.0, "scale": 0.1},
    {"type": "lognormal", "mean": 0.05, "sigma": 0.2},
])
def test_scaling_a_factor_scales_its_mean(spec):
    k = 250.0
    assert make_distribution(_scale_spec(spec, k)).mean() == pytest.approx(
        k * make_distribution(spec).mean())


def test_simulated_mean_matches_the_factor_mean():
    r = evaluate([one_line("A", {2026: 100.0}, tri(0.8, 1.6)), one_line("B", {2026: 1.0})],
                 base_year=2026, inflation=INFL, discount_rate=0.0, n_iter=40_000, seed=3)
    assert r.summary["mean"].iloc[0] == pytest.approx(100 * (0.8 + 1.0 + 1.6) / 3, rel=0.01)


def test_seeded_runs_repeat_and_adding_an_alternative_leaves_others_alone():
    a = one_line("A", {2026: 100.0}, tri(0.8, 1.6))
    b = one_line("B", {2026: 110.0}, tri(0.9, 1.2))
    c = one_line("C", {2026: 90.0}, tri(0.7, 2.0))
    kw = dict(base_year=2026, inflation=INFL, discount_rate=0.02, n_iter=2000, seed=5)
    r1, r2 = evaluate([a, b], **kw), evaluate([a, b], **kw)
    r3 = evaluate([a, b, c], **kw)
    assert np.array_equal(r1.draws["A"], r2.draws["A"])
    assert np.array_equal(r1.draws["B"], r3.draws["B"])
    assert r3.summary.p_cheapest.sum() == pytest.approx(1.0)


def test_s_curves_are_monotone():
    r = evaluate([one_line("A", {2026: 100.0}, tri(0.8, 1.6)), one_line("B", {2026: 1.0})],
                 base_year=2026, inflation=INFL, discount_rate=0.02, n_iter=5000)
    curve = r.s_curves()["A"].to_numpy()
    assert np.all(np.diff(curve) >= 0)


# --------------------------------------------------------------- frontier ---
def test_dominated_alternative_is_named_and_off_the_frontier():
    cheap_good = one_line("Cheap and good", {2026: 100.0}, effectiveness=0.8)
    dear_poor = one_line("Dear and poor", {2026: 150.0}, effectiveness=0.6)
    dear_best = one_line("Dear and best", {2026: 200.0}, effectiveness=0.95)
    r = evaluate([cheap_good, dear_poor, dear_best], base_year=2026, inflation=INFL,
                 discount_rate=0.02, n_iter=200)
    s = r.summary.set_index("alternative")
    assert s.loc["Dear and poor", "dominated_by"] == "Cheap and good"
    assert list(s.on_frontier) == [True, False, True]
    assert s.loc["Cheap and good", "cost_per_effectiveness"] == pytest.approx(100 / 0.8)


def test_without_effectiveness_there_is_no_frontier():
    r = evaluate([one_line("A", {2026: 1.0}), one_line("B", {2026: 2.0})],
                 base_year=2026, inflation=INFL, discount_rate=0.02, n_iter=100)
    assert "on_frontier" not in r.summary


# ------------------------------------------------------------------ errors ---
def test_bad_analyses_are_refused():
    a, b = one_line("A", {2026: 1.0}), one_line("B", {2026: 2.0})
    kw = dict(base_year=2026, inflation=INFL, n_iter=100)
    with pytest.raises(AoAError, match="at least two"):
        evaluate([a], discount_rate=0.02, **kw)
    with pytest.raises(AoAError, match="fraction"):
        evaluate([a, b], discount_rate=2.0, **kw)  # 2 meant as 2%
    with pytest.raises(AoAError, match="unique"):
        evaluate([a, one_line("A", {2026: 3.0})], discount_rate=0.02, **kw)
    with pytest.raises(AoAError, match="basis"):
        evaluate([a, b], discount_rate=0.02, basis="nominal", **kw)
    with pytest.raises(AoAError, match="phase"):
        CostLine("x", "Sustainment", {2026: 1.0})
    with pytest.raises(AoAError, match="duplicate"):
        Alternative("A", [CostLine("x", "O&S", {2026: 1.0}), CostLine("x", "O&S", {2027: 1.0})])


# ----------------------------------------------------------- SAR history ---
def _panel(rows):
    cols = ["program", "subprogram", "cycle", "cycle_year", "measure", "comparison",
            "unit_cost_growth_pct", "quantity_change_pct", "checks_ok"]
    return pd.DataFrame(rows, columns=cols)


def test_history_takes_each_programs_latest_report_once():
    rows = []
    for i in range(25):
        rows.append((f"P{i}", None, "Dec 2019", 2019, "APUC", "original", 5.0, 0.0, True))
        rows.append((f"P{i}", None, "Dec 2022", 2022, "APUC", "original", 10.0 + i, 0.0, True))
    rows.append(("P0", None, "Dec 2022", 2022, "APUC", "current", 99.0, 0.0, True))
    rows.append(("P1", None, "Dec 2023", 2023, "APUC", "original", 500.0, 0.0, False))
    g = historical_growth(_panel(rows))
    assert len(g) == 25 and g.growth_pct.min() == 10.0  # latest checked row, original baseline
    spec = growth_factor(g)
    assert spec["type"] == "empirical" and spec["draws"][0] == pytest.approx(1.10)
    d = describe_growth(g)
    assert d["programs"] == 25 and d["median"] == pytest.approx(22.0)


def test_history_filters_quantity_changes_and_refuses_thin_samples():
    rows = [(f"P{i}", None, "Dec 2022", 2022, "APUC", "original", 10.0, 60.0 if i % 2 else 0.0, True)
            for i in range(30)]
    assert len(historical_growth(_panel(rows), max_quantity_change_pct=20, min_programs=10)) == 15
    with pytest.raises(AoAError, match="Only 15"):
        historical_growth(_panel(rows), max_quantity_change_pct=20)


def test_history_feeds_a_cost_line():
    rows = [(f"P{i}", None, "Dec 2022", 2022, "APUC", "original", float(i), 0.0, True)
            for i in range(40)]
    spec = growth_factor(historical_growth(_panel(rows)))
    line = one_line("Hist", spread(100.0, 2027, 4), spec)
    r = evaluate([line, one_line("B", {2026: 1.0})], base_year=2026, inflation=INFL,
                 discount_rate=0.0, basis="by", n_iter=20_000, seed=2)
    assert r.summary["mean"].iloc[0] == pytest.approx(100 * (1 + 19.5 / 100), rel=0.01)


# ------------------------------------------------------------ spec + CLI ---
EXAMPLE = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "aoa_example.json"


def test_example_spec_loads_and_runs():
    from cost_core.aoa.spec import load_spec, run_spec

    kw = load_spec(EXAMPLE)
    assert [a.name for a in kw["alternatives"]] == [
        "Upgrade in place", "New development", "Buy commercial"]
    assert kw["discount_rate"] == 0.02 and kw["units"] == "BY2026 $M"
    dev = kw["alternatives"][1].lines[0]
    assert sum(dev.by_year.values()) == pytest.approx(2400)
    r = run_spec(EXAMPLE)
    s = r.summary.set_index("alternative")
    assert s.loc["Buy commercial", "dominated_by"] == "Upgrade in place"
    assert r.assumptions["units"] == "BY2026 $M"


def test_spec_errors_name_the_problem(tmp_path):
    import json
    from cost_core.aoa.spec import load_spec

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"base_year": 2026, "alternatives": []}))
    with pytest.raises(AoAError, match="discount_rate"):
        load_spec(bad)
    line = {"name": "x", "phase": "O&S", "annual": {"amount": 1, "first": 2027, "last": 2028},
            "by_year": {"2027": 1}}
    bad.write_text(json.dumps({"base_year": 2026, "discount_rate": 0.02, "inflation_rate": 0.02,
                               "alternatives": [{"name": "A", "lines": [line]}]}))
    with pytest.raises(AoAError, match="exactly one"):
        load_spec(bad)


def test_cli_writes_the_comparison(tmp_path, monkeypatch, capsys):
    from cost_core import cli

    monkeypatch.setattr("sys.argv", ["ce-core", "aoa", "--spec", str(EXAMPLE),
                                     "--out", str(tmp_path)])
    cli.main()
    for name in ("summary.csv", "lines.csv", "s_curves.csv", "assumptions.json"):
        assert (tmp_path / name).is_file()
    assert "Upgrade in place" in capsys.readouterr().out
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert summary.p_cheapest.sum() == pytest.approx(1.0)
