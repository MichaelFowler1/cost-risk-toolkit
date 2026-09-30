# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Wrong answers found by the 2026-09-30 correctness sweep, one test each.

Every number here is invented; the lot data come from a Crawford 85% curve
with a first-unit cost of 1,000 (lot cost = sum over its units of
1000 * u ** log2(0.85)).
"""

import numpy as np
import pandas as pd
import pytest

from cost_core import cli


def lots_csv(tmp_path, units, costs, name="lots.csv", **extra):
    path = tmp_path / name
    pd.DataFrame({"units": units, "cost": costs, **extra}).to_csv(path, index=False)
    return path


def refused(argv, caplog, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(a) for a in argv])
    return caplog.text + capsys.readouterr().err


# ------------------------------------------------------------ lot engine
RAMP = [5, 10, 15, 20, 25, 30]
NOISY_RAMP = [4285.61, 5399.69, 7320.85, 8280.21, 9353.93, 10378.31]


def test_an_lc_rate_fit_that_never_settles_is_not_selected(tmp_path):
    # Lot sizes that rise in step with their midpoints: the plain fixed-point
    # iteration never settled, and the unsettled LC+Rate fit won anyway with
    # a 0.00% learning slope and a first-unit cost of zero.
    from cost_core.lots import LotSeries, analyse_lots

    report = analyse_lots(LotSeries.read(lots_csv(tmp_path, RAMP, NOISY_RAMP),
                                         dollar_year=2025))
    assert report.selected_model == "LC"
    slope = 2 ** report.fit.ctx["b_lc"]
    assert 0.80 < slope < 0.90


def test_a_file_that_crashed_the_solver_now_fits(tmp_path, capsys):
    path = lots_csv(tmp_path, [10, 20, 30, 40, 50],
                    [6766.64, 10806.61, 13566.49, 14327.98, 15396.22])
    cli.main(["fit-lots", "--csv", str(path), "--dollar-year", "2025"])
    out = capsys.readouterr().out
    assert "LC selected" in out and "85." in out


def test_the_comparison_shows_the_rate_t_statistic_and_the_reason(tmp_path):
    from cost_core.lots import LotSeries, analyse_lots

    report = analyse_lots(LotSeries.read(lots_csv(tmp_path, RAMP, NOISY_RAMP),
                                         dollar_year=2025))
    items = list(report.model_comparison()["Item"])
    assert "t (rate coefficient)" in items and "Selection basis" in items


def test_a_lot_with_no_cost_in_the_middle_is_refused_not_renumbered(tmp_path):
    from cost_core.lots import LotInputError, LotSeries

    costs = [4031.09, 5829.97, None, 8422.45, 9481.82, 10445.65]
    with pytest.raises(LotInputError, match="renumber the units of every later lot"):
        LotSeries.read(lots_csv(tmp_path, RAMP, costs), dollar_year=2025)


def test_lots_with_no_cost_after_the_last_costed_lot_are_just_left_out(tmp_path):
    from cost_core.lots import LotSeries

    costs = [4031.09, 5829.97, 7229.60, 8422.45, None, None]
    assert len(LotSeries.read(lots_csv(tmp_path, RAMP, costs), dollar_year=2025).quantities) == 4


@pytest.mark.parametrize("basis,column", [("recurring", "recurring_cost"),
                                          ("total", "total_cost")])
def test_with_two_cost_columns_the_declared_basis_decides(tmp_path, basis, column):
    from cost_core.lots import LotSeries

    recurring = [7116.06, 6260.43, 6348.95, 6581.41, 6859.52, 7153.38]
    total = [r + n for r, n in zip(recurring, [20000, 8000, 3000, 0, 0, 0])]
    path = tmp_path / "both.csv"
    pd.DataFrame({"units": [10, 12, 14, 16, 18, 20], "recurring_cost": recurring,
                  "total_cost": total}).to_csv(path, index=False)
    series = LotSeries.read(path, dollar_year=2025, cost_basis=basis)
    expected = recurring if column == "recurring_cost" else total
    assert np.allclose(series.costs, expected)


def test_forecast_reprices_under_the_reports_own_selection_rule(tmp_path):
    from cost_core.lots import LotSeries, analyse_lots

    path = lots_csv(tmp_path, [10, 25, 15, 30, 20, 25],
                    [7565.38, 11278.99, 6297.79, 11091.64, 6864.72, 8203.75])
    report = analyse_lots(LotSeries.read(path, dollar_year=2025), t_gate=10.0,
                          forecast=[20, 20])
    assert report.selected_model == "LC"
    assert np.allclose(report.forecast([20, 20])["Lot Cost ($)"].to_numpy(dtype=float),
                       report.intervals()["Lot Cost ($)"].to_numpy(dtype=float))


def test_simulate_needs_a_forecast(tmp_path, caplog, capsys):
    path = lots_csv(tmp_path, [20] * 6, [12010.76, 8557.34, 7910.12, 7506.22, 7282.08, 6668.22])
    msg = refused(["fit-lots", "--csv", path, "--dollar-year", 2025, "--simulate", 2000],
                  caplog, capsys)
    assert "needs --forecast" in msg


def test_a_unit_fit_has_no_confidence_interval_for_a_lot():
    from cost_core.learning_curve import FitError, fit_curve

    units = np.arange(1, 21)
    fit = fit_curve(theory="crawford", units=units, costs=1000 * units ** np.log2(0.85))
    with pytest.raises(FitError, match="no confidence interval"):
        fit.forecast_lots([[21, 30]], kind="confidence")
    assert len(fit.forecast_lots([[21, 30]], kind="prediction")) == 1


def test_program_lots_must_be_in_fiscal_year_order():
    from cost_core.program.rollup import Program, ProgramError, fitted

    analogy = pd.DataFrame({"Lot": [1, 2, 3, 4], "Fiscal Year": [2020, 2021, 2022, 2023],
                            "Lot Quantity": [10, 12, 14, 16],
                            "AUC ($K)": [700.0, 620.0, 600.0, 580.0]})
    with pytest.raises(ProgramError, match="fiscal-year order"):
        Program("P", [2028, 2027, 2029], [fitted("Airframe", analogy, [10, 20, 30])]).validate()
