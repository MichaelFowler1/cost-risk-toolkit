# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Earned value metrics, earned schedule and the forecast at completion.

The metrics are checked against a small program worked by hand. The forecast
is checked against things true by construction (constant performance gives
exactly the CPI and SPI(t) formulas; a program on plan finishes on plan and
on budget) and against synthetic programs whose true outcome is known, where
the P50 and P80 have to hold as often as they say they do.
"""
import numpy as np
import pandas as pd
import pytest

from cost_core.evm import EvmData, EvmError, earned_schedule, forecast


def frame(pv, ev, ac, eac=None, **extra):
    n = len(pv)
    pad = lambda x: list(x) + [np.nan] * (n - len(x))  # noqa: E731
    d = {"period": list(range(1, n + 1)), "bcws": pv, "bcwp": pad(ev), "acwp": pad(ac)}
    if eac is not None:
        d["eac"] = pad(eac)
    d.update(extra)
    return pd.DataFrame(d)


# Periodic: plan 10, 20, 30, 20, 20 (BAC 100). Earned 8, 15, 22; spent 10, 18, 27.
HAND = frame([10, 20, 30, 20, 20], [8, 15, 22], [10, 18, 27], eac=[None, None, 110])


def test_metrics_match_the_hand_calculation():
    m = EvmData.from_frame(HAND).metrics().iloc[-1]
    # Cumulative at period 3: PV 60, EV 45, AC 55.
    assert (m.bcws, m.bcwp, m.acwp) == (60, 45, 55)
    assert m.cv == -10 and m.sv == -15
    assert m.cpi == pytest.approx(45 / 55) and m.spi == pytest.approx(0.75)
    assert m.cpi_period == pytest.approx(22 / 27) and m.spi_period == pytest.approx(22 / 30)
    assert m.tcpi_bac == pytest.approx(55 / 45)
    # ES: PV passes 30 at t=2 and 60 at t=3, so 45 is halfway: 2.5.
    assert m.es == pytest.approx(2.5) and m.spi_t == pytest.approx(2.5 / 3)
    assert m.sv_t == pytest.approx(-0.5)
    assert m.ieac_cpi == pytest.approx(55 + 55 / (45 / 55))
    assert m.ieac_cpi_spi == pytest.approx(55 + 55 / (45 / 55 * 0.75))
    assert m.ieac_linear == 110
    assert m.ieac_t == pytest.approx(5 / (2.5 / 3))
    assert m.tspi == pytest.approx((5 - 2.5) / (5 - 3))
    assert m.tcpi_eac == pytest.approx(55 / (110 - 55))


def test_earned_schedule_edges():
    pv = [10, 30, 60, 100]
    assert earned_schedule(pv, 0) == 0 and earned_schedule(pv, 5) == 0.5
    assert earned_schedule(pv, 100) == 4 and earned_schedule(pv, 120) == 4
    # A flat stretch of the baseline: the later end, as Lipke counts it.
    assert earned_schedule([10, 10, 20], 10) == 2
    assert list(earned_schedule(pv, np.array([30, 80]))) == [2, 3.5]


def test_cumulative_and_periodic_input_agree():
    cum = frame(list(np.cumsum([10, 20, 30, 20, 20])), [8, 23, 45], [10, 28, 55])
    a = EvmData.from_frame(HAND).metrics()
    b = EvmData.from_frame(cum, cumulative=True).metrics()
    cols = ["bcws", "bcwp", "acwp", "cpi", "spi_t"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_accounts_sum_to_the_program():
    a = frame([5, 10, 15, 10, 10], [4, 7, 11], [5, 9, 13], wbs=["1.1"] * 5)
    b = frame([5, 10, 15, 10, 10], [4, 8, 11], [5, 9, 14], wbs=["1.2"] * 5)
    data = EvmData.from_frame(pd.concat([a, b]))
    assert set(data.accounts) == {"1.1", "1.2"}
    m = data.metrics().iloc[-1]
    assert (m.bcws, m.bcwp, m.acwp) == (60, 45, 55) and data.bac == 100
    assert data.accounts["1.2"].metrics().iloc[-1].acwp == 28


def test_accounts_that_start_and_end_at_different_times():
    # A runs periods 1-3 with its baseline over by 2; B starts in period 2.
    # Both are statused to period 3; B has no row for period 3 (nothing that
    # month, as omitted zeros).
    a = pd.DataFrame({"period": [1, 2, 3], "wbs": "A", "bcws": [10, 10, 0],
                      "bcwp": [8, 9, 3], "acwp": [9, 10, 3]})
    b = pd.DataFrame({"period": [2, 4, 5], "wbs": "B", "bcws": [5, 5, 5],
                      "bcwp": [4, None, None], "acwp": [6, None, None]})
    data = EvmData.from_frame(pd.concat([a, b]))
    assert data.status == 3 and data.periods == [1, 2, 3, 4, 5]
    assert list(data.bcws) == [10, 25, 25, 30, 35]
    assert list(data.bcwp) == [8, 21, 24] and list(data.acwp) == [9, 25, 28]
    assert list(data.accounts["B"].bcwp) == [0, 4, 4]


def test_periods_sort_as_dates_whatever_their_format():
    df = frame([10, 20, 30], [8, 15], [9, 17])
    df["period"] = ["1/31/2024", "2/29/2024", "10/31/2024"]
    shuffled = df.iloc[[2, 0, 1]]
    data = EvmData.from_frame(shuffled)
    assert data.periods == ["1/31/2024", "2/29/2024", "10/31/2024"]
    assert list(data.bcwp) == [8, 23]


def test_a_program_eac_needs_every_account_to_give_one():
    a = frame([5, 10, 15], [4, 7], [5, 9], eac=[None, 31], wbs=["a"] * 3)
    b = frame([5, 10, 15], [4, 8], [5, 9], wbs=["b"] * 3)
    data = EvmData.from_frame(pd.concat([a, b]))
    assert data.eac is None
    assert data.accounts["a"].eac[-1] == 31
    both = EvmData.from_frame(pd.concat([a, b.assign(eac=[None, 32, None])]))
    assert both.eac[-1] == 63


def test_bad_data_is_refused():
    with pytest.raises(EvmError, match="blank before the status"):
        EvmData.from_frame(frame([10, 20, 30], [8, np.nan, 5], [10, 10, 10]))
    with pytest.raises(EvmError, match="Missing column"):
        EvmData.from_frame(pd.DataFrame({"period": [1], "bcws": [1]}))
    with pytest.raises(EvmError, match="falls"):
        EvmData.from_frame(frame([10, 5, 30], [8], [10]), cumulative=True)


def test_flags_catch_an_optimistic_eac():
    data = EvmData.from_frame(frame([10, 20, 30, 20, 20], [8, 15, 22], [10, 18, 27],
                                    eac=[None, None, 100]))
    f = data.flags().set_index("flag")["raised"]
    # EAC 100 needs a CPI of 1.22 from here against 0.818 so far, is below
    # every independent EAC, and implies a final CPI of 1.0 at 45% complete.
    assert f["TCPI to the EAC exceeds the CPI by more than 0.10"]
    assert f["Contractor EAC below every independent EAC"]
    assert f["EAC implies the CPI recovering by more than 0.10 after 20% complete"]
    assert f["TCPI to BAC above 1.10"]
    # EAC 110 implies a final CPI of 0.909, under 0.10 above 0.818: not raised.
    g = EvmData.from_frame(HAND).flags().set_index("flag")["raised"]
    assert g["TCPI to the EAC exceeds the CPI by more than 0.10"]
    assert not g["EAC implies the CPI recovering by more than 0.10 after 20% complete"]
    honest = EvmData.from_frame(frame([10, 20, 30, 20, 20], [8, 15, 22], [10, 18, 27],
                                      eac=[None, None, 125]))
    h = honest.flags().set_index("flag")["raised"]
    assert not h["TCPI to the EAC exceeds the CPI by more than 0.10"]
    assert not h["Contractor EAC below every independent EAC"]


def test_spi_recovers_while_spi_t_does_not():
    # Plan: 10 a period for 10 periods, then the baseline carries on at zero.
    # The program earns 7 a period. At period 13 it is 91% complete: SPI is
    # 91/100 = 0.91, but it has earned 9.1 periods of schedule in 13, so
    # SPI(t) is 0.70 and the finish is forecast at 10 / 0.70 = 14.3 periods.
    pv = np.cumsum([10.0] * 10 + [0.0] * 5)
    ev = np.cumsum([7.0] * 13)
    data = EvmData(list(range(1, 16)), pv, ev, ev / 0.95, bac=100)
    m = data.metrics().iloc[-1]
    assert data.planned_duration == 10
    assert m.spi == pytest.approx(0.91) and m.es == pytest.approx(9.1)
    assert m.spi_t == pytest.approx(9.1 / 13) and m.ieac_t == pytest.approx(10 * 13 / 9.1)
    assert data.flags().set_index("flag").loc["SPI has recovered but SPI(t) has not", "raised"]


# ------------------------------------------------------------- forecast ---
def steady(periods=36, status=12, speed=0.8, cpi=0.9, bac=1000.0):
    """A program that performs identically every period: every period earns
    ``speed`` of a period of schedule at efficiency ``cpi``."""
    w = np.sin(np.linspace(0.1, np.pi - 0.1, periods))
    pv = np.cumsum(w / w.sum() * bac)
    grid, x = np.concatenate([[0.0], pv]), np.arange(periods + 1)
    ev = np.interp(np.arange(1, status + 1) * speed, x, grid)
    return EvmData(list(range(1, periods + 1)), pv, ev, ev / cpi, bac=bac)


def test_constant_performance_reproduces_the_formulas_exactly():
    data = steady()
    f = forecast(data, n_iter=500, seed=0)
    m = data.metrics().iloc[-1]
    assert np.allclose(f.eac, m.ieac_cpi) and np.allclose(f.finish, m.ieac_t)
    assert f.finish[0] == pytest.approx(36 / 0.8)


def test_a_program_on_plan_finishes_on_plan_and_on_budget():
    f = forecast(steady(speed=1.0, cpi=1.0), n_iter=200, seed=0)
    assert np.allclose(f.eac, 1000) and np.allclose(f.finish, 36)
    assert f.joint(1000 + 1e-6, 36 + 1e-9) == 1.0


def test_forecast_is_reproducible_and_joint_never_exceeds_either_marginal():
    data = EvmData.from_frame(_synthetic(7)[0])
    a, b = forecast(data, n_iter=3000, seed=4), forecast(data, n_iter=3000, seed=4)
    assert np.array_equal(a.eac, b.eac) and np.array_equal(a.finish, b.finish)
    c, t = np.quantile(a.eac, 0.7), np.quantile(a.finish, 0.7)
    assert a.joint(c, t) <= min(a.confidence_of_cost(c), float(np.mean(a.finish <= t)))
    assert (a.eac >= data.acwp[-1]).all() and (a.finish >= data.status).all()
    s = a.summary().set_index("measure")["value"]
    assert "confidence of IEAC (CPI)" in s.index


def test_too_short_a_record_is_refused():
    with pytest.raises(EvmError, match="usable periods"):
        forecast(EvmData.from_frame(HAND))


def _synthetic(seed, periods=36, status=15, cpi=0.9, speed=0.85, sd=0.15, bac=1000.0):
    """A program whose every period's pace and efficiency are drawn at
    random around fixed rates, run to completion: the history up to
    ``status``, and the true final cost and finish."""
    rng = np.random.default_rng(seed)
    w = np.sin(np.linspace(0.05, np.pi - 0.05, periods))
    pv = np.cumsum(w / w.sum() * bac)
    grid, x = np.concatenate([[0.0], pv]), np.arange(periods + 1)
    es = cost = t = 0.0
    ev, ac = [], []
    while es < periods - 1e-9:
        step = max(speed * (1 + sd * rng.standard_normal()), 0.05)
        new = min(es + step, periods)
        cost += (np.interp(new, x, grid) - np.interp(es, x, grid)) / max(
            cpi * (1 + sd * rng.standard_normal()), 0.3)
        t += (periods - es) / step if new >= periods else 1.0
        es = new
        ev.append(np.interp(new, x, grid))
        ac.append(cost)
    df = frame(list(np.diff(grid)), list(np.diff(np.concatenate([[0.0], ev[:status]]))),
               list(np.diff(np.concatenate([[0.0], ac[:status]]))))
    return df, cost, t


def test_the_range_holds_as_often_as_it_says():
    # 120 programs, each forecast from 15 periods and then run to the end.
    # The P50 and the P80 have to contain the truth about as often as their
    # names say; before the Bayesian bootstrap the P80 held the finish 49%
    # of the time.
    hits = {"eac50": [], "eac80": [], "fin80": []}
    for s in range(120):
        df, cost, finish = _synthetic(500 + s)
        f = forecast(EvmData.from_frame(df), n_iter=2000, seed=s)
        hits["eac50"].append(cost <= np.quantile(f.eac, 0.5))
        hits["eac80"].append(cost <= np.quantile(f.eac, 0.8))
        hits["fin80"].append(finish <= np.quantile(f.finish, 0.8))
    rate = {k: float(np.mean(v)) for k, v in hits.items()}
    assert 0.38 <= rate["eac50"] <= 0.62, rate
    assert 0.70 <= rate["eac80"] <= 0.90, rate
    assert 0.70 <= rate["fin80"] <= 0.90, rate


def test_the_range_narrows_as_the_program_matures():
    widths = []
    for status in (8, 24):
        wide = []
        for s in range(20):
            df, _, _ = _synthetic(900 + s, status=status)
            f = forecast(EvmData.from_frame(df), n_iter=2000, seed=s)
            wide.append(np.quantile(f.eac, 0.9) - np.quantile(f.eac, 0.1))
        widths.append(np.median(wide))
    assert widths[1] < 0.6 * widths[0]


def test_cli_writes_every_table_and_names_the_warning(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from cost_core import cli

    example = Path(__file__).resolve().parents[1] / "docs" / "evm_example.csv"
    monkeypatch.setattr("sys.argv", ["ce-core", "evm", "--data", str(example), "--iters", "2000",
                                     "--units", "$K", "--out", str(tmp_path)])
    cli.main()
    for name in ("metrics.csv", "flags.csv", "summary.csv", "forecast_percentiles.csv",
                 "forecast_draws.csv", "accounts.csv", "assumptions.json"):
        assert (tmp_path / name).is_file(), name
    accounts = pd.read_csv(tmp_path / "accounts.csv")
    assert len(accounts) == 3 and accounts.bac.sum() == pytest.approx(15500.4)
    assert "Contractor EAC below every independent EAC" in capsys.readouterr().out
    pytest.importorskip("matplotlib")
    assert (tmp_path / "evm.png").stat().st_size > 10_000
