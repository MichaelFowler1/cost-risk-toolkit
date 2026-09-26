# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Broken and awkward inputs to the older commands: each one here once gave a
traceback, a NaN on screen or a silently wrong answer."""

import json

import pandas as pd
import pytest

from cost_core import cli, plain
from cost_core.examples import example_path


def evm_example():
    return pd.read_csv(example_path("evm"))


def refused(argv, caplog, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(a) for a in argv])
    return caplog.text + capsys.readouterr().err


# ------------------------------------------------------------------- EVM
def test_rows_without_an_account_name_are_refused_not_dropped(tmp_path, caplog, capsys):
    f = evm_example()
    f.loc[[3, 4], "wbs"] = None
    path = tmp_path / "evm.csv"
    f.to_csv(path, index=False)
    msg = refused(["evm", "--data", path, "--out", tmp_path / "o"], caplog, capsys)
    assert "2 row(s) have no control account name (rows 5, 6)" in msg


def test_earned_value_beyond_the_budget_is_a_warning_sign():
    from cost_core.evm import EvmData

    f = evm_example()
    status = f.dropna(subset=["acwp"])["period"].max()
    f.loc[(f["period"] == status) & (f["wbs"] == f["wbs"].iloc[0]), "bcwp"] = f["bcws"].sum() * 3
    flags = EvmData.from_frame(f).flags().set_index("flag")
    assert flags.loc["Earned value exceeds the budget at completion", "raised"]


def test_the_tcpi_warning_names_the_cpi_needed():
    from cost_core.evm import EvmData

    flags = EvmData.from_frame(evm_example()).flags().set_index("flag")
    detail = flags.loc["TCPI to the EAC exceeds the CPI by more than 0.10", "detail"]
    assert "has to run at a CPI of" in detail and "more efficiently" not in detail


def test_no_work_earned_yet_reads_as_words_not_nan(tmp_path, capsys):
    f = evm_example()
    f["bcwp"] = 0.0
    path = tmp_path / "evm.csv"
    f.to_csv(path, index=False)
    cli.main(["evm", "--data", str(path), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "No work has been earned yet against" in out
    assert "There's no forecast of the final cost yet" in out
    assert "NaN" not in out and "inf" not in out


def test_the_status_summary_appears_when_there_is_too_little_history(tmp_path, capsys):
    f = evm_example()
    f = f[f["period"] <= sorted(f["period"].unique())[1]]
    path = tmp_path / "evm.csv"
    f.to_csv(path, index=False)
    cli.main(["evm", "--data", str(path), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert "What this means:" in out and "Each dollar spent so far" in out


@pytest.mark.parametrize("command", [["evm", "--data", "x.csv"], ["full-run", "--out", "x"]])
def test_too_few_iterations_are_refused(command, caplog, capsys):
    msg = refused(command + ["--iters", 10], caplog, capsys)
    assert "use 1,000 or more" in msg


# -------------------------------------------------------------- fit-lots
def lots(tmp_path):
    path = tmp_path / "lots.csv"
    pd.DataFrame({"lot": [f"Lot {i}" for i in range(1, 7)],
                  "units": [22, 18, 25, 30, 30, 36],
                  "cost": [96.8e6, 70.2e6, 90e6, 100.5e6, 96e6, 111.6e6]}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_complexity_factor_must_be_positive(tmp_path, value, caplog, capsys):
    msg = refused(["fit-lots", "--csv", lots(tmp_path), "--dollar-year", 2026,
                   "--complexity", value, "--forecast", "30"], caplog, capsys)
    assert "must be greater than 0" in msg


@pytest.mark.parametrize("flag", ["--forecast", "--price-lots"])
def test_lot_lists_that_are_not_numbers_say_how_to_write_them(tmp_path, flag, caplog, capsys):
    msg = refused(["fit-lots", "--csv", lots(tmp_path), "--dollar-year", 2026, flag, "abc"],
                  caplog, capsys)
    assert f'{flag} takes lot sizes as whole numbers' in msg and "invalid literal" not in msg


def test_an_lc_fit_prints_no_nan_for_the_rate_term(tmp_path, capsys):
    cli.main(["fit-lots", "--csv", str(lots(tmp_path)), "--dollar-year", "2026"])
    assert "NaN" not in capsys.readouterr().out


# ------------------------------------------------------------------ specs
def spec(tmp_path, kind, **changes):
    s = json.loads(example_path(kind).read_text(encoding="utf-8"))
    s.update(changes)
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(s), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", [1.2, 0])
def test_a_jcl_confidence_outside_0_and_1_is_refused(tmp_path, value, caplog, capsys):
    msg = refused(["jcl", "--spec", spec(tmp_path, "jcl", confidence=value, n_iter=2000),
                   "--out", tmp_path / "o"], caplog, capsys)
    assert "confidence is a fraction between 0 and 1" in msg


def test_a_jcl_confidence_flag_outside_0_and_1_is_refused(tmp_path, caplog, capsys):
    msg = refused(["jcl", "--spec", spec(tmp_path, "jcl", n_iter=2000), "--confidence", 5,
                   "--out", tmp_path / "o"], caplog, capsys)
    assert "--confidence is a fraction between 0 and 1" in msg


def test_a_jcl_spec_naming_a_missing_schedule_says_where_it_looked(tmp_path, caplog, capsys):
    msg = refused(["jcl", "--spec", spec(tmp_path, "jcl", mspdi="gone.xml"),
                   "--out", tmp_path / "o"], caplog, capsys)
    assert "names the schedule 'gone.xml', which isn't there" in msg


@pytest.mark.parametrize("changes,message", [
    ({"growth": {"type": "triangular", "left": 1.5, "mode": 1.0, "right": 0.9}},
     "the growth range"),
    ({"delta": -50}, "delta, the extra money to test in one year, must be more than 0"),
    ({"growth_correlation": 2}, "growth_correlation must be between -1 and 1"),
])
def test_bad_portfolio_settings_are_refused_up_front(tmp_path, changes, message):
    from cost_core.portfolio.optimize import PortfolioError
    from cost_core.portfolio.spec import load_portfolio

    with pytest.raises(PortfolioError, match=message):
        load_portfolio(spec(tmp_path, "portfolio", **changes))


@pytest.mark.parametrize("changes,message", [({"inflation_rate": -0.5}, "outside -10% to 50%"),
                                             ({"pv_year": 1900}, "more than a century")])
def test_implausible_aoa_settings_are_refused(tmp_path, changes, message):
    from cost_core.aoa import AoAError
    from cost_core.aoa.spec import load_spec

    with pytest.raises(AoAError, match=message):
        load_spec(spec(tmp_path, "aoa", **changes))


def test_the_aoa_demo_prints_no_nan(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["demo", "aoa"])
    assert "NaN" not in capsys.readouterr().out


def test_the_portfolio_demo_prints_no_nan(tmp_path, monkeypatch, capsys):
    pytest.importorskip("pulp")
    monkeypatch.chdir(tmp_path)
    cli.main(["demo", "portfolio"])
    assert "NaN" not in capsys.readouterr().out
