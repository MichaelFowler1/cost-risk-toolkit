# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Reading the IPMDAR Contract Performance Dataset.

Real CPDs are contractor data and not public, so the datasets here are
written by the tests from ``docs/evm_example.csv`` using the table and field
names of the IPMDAR CPD Data Exchange Instructions (12 March 2020). The
check is that reading the CPD gives exactly the numbers the same program
gives from the CSV, whichever way it is packaged and delivered.
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cost_core.evm import EvmData, EvmError, forecast
from cost_core.evm.ipmdar import load_dataset, read_ipmdar

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "evm_example.csv"
STATUS = 14


def _example():
    df = pd.read_csv(EXAMPLE)
    periods = sorted(df.period.unique())
    names = list(dict.fromkeys(df.wbs))
    ids = {n: f"CA{i + 1}" for i, n in enumerate(names)}
    return df, periods, names, ids


def cpd_tables(status=STATUS, time_phased=True, by_work_package=False, pmb_shift=0.0):
    """The example program as a CPD reported at ``status``."""
    df, periods, names, ids = _example()
    df = df.assign(ca=df.wbs.map(ids), pid=df.period.map({p: i + 1 for i, p in enumerate(periods)}))
    ends = pd.to_datetime([p + "-01" for p in periods]) + pd.offsets.MonthEnd(0)
    cal = [{"ID": i + 1, "StartDate": (e - pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d"),
            "EndDate": e.strftime("%Y-%m-%d"), "WorkingHours": 160} for i, e in enumerate(ends)]
    past, fut = df[df.pid <= status], df[df.pid > status]

    def to_date(col):
        if time_phased:
            rows = [{"ControlAccountID": r.ca, "ReportingPeriodID": int(r.pid),
                     "Value_Dollars": float(getattr(r, col))} for r in past.itertuples()]
        else:
            cum = past.groupby("ca")[col].sum()
            rows = [{"ControlAccountID": ca, "Value_Dollars": float(v)} for ca, v in cum.items()]
        rows = [r for r in rows if r["Value_Dollars"] != 0]  # the DEI omits zero records
        if by_work_package:
            # CA1's work split 60/40 across two work packages.
            out = []
            for r in rows:
                if r["ControlAccountID"] == "CA1":
                    for wp, share in (("WP1a", 0.6), ("WP1b", 0.4)):
                        q = {k: v for k, v in r.items() if k != "ControlAccountID"}
                        out.append({**q, "WorkPackageID": wp, "Value_Dollars": r["Value_Dollars"] * share})
                else:
                    out.append(r)
            rows = out
        return rows

    contractor_eac = past[past.pid == status].groupby("ca").eac.sum()
    acwp = past.groupby("ca").acwp.sum()
    etc_by_ca = contractor_eac - acwp
    fut_bcws = fut.groupby("ca").bcws.sum()
    etc = [{"ControlAccountID": r.ca, "ReportingPeriodID": int(r.pid),
            "Value_Dollars": float(r.bcws / fut_bcws[r.ca] * etc_by_ca[r.ca])}
           for r in fut.itertuples()]
    pmb = {"SummaryElementID": "PMB",
           "BCWS_CumulativeToDate_Dollars": float(past.bcws.sum()),
           "BCWP_CumulativeToDate_Dollars": float(past.bcwp.sum()) + pmb_shift,
           "ACWP_CumulativeToDate_Dollars": float(past.acwp.sum()),
           "BAC_Dollars": float(df.bcws.sum()), "EAC_Dollars": float(contractor_eac.sum())}
    return {
        "DatasetMetadata": [{"SecurityMarking": "UNCLASSIFIED", "ReportingPeriodID": status,
                             "ProgramName": "Example program", "ContractNumber": "X-0000"}],
        "DatasetConfiguration": [{"ToDate_TimePhased": time_phased,
                                  "BCWP_ToDate_ByWorkPackage": by_work_package,
                                  "ACWP_ToDate_ByWorkPackage": by_work_package,
                                  "BCWS_ToDate_ByWorkPackage": by_work_package}],
        "ReportingCalendar": cal,
        "ControlAccounts": [{"ID": ids[n], "Name": n, "IsSummaryLevelPlanningPackage": False}
                            for n in names],
        "WorkPackages": [{"ID": "WP1a", "ControlAccountID": "CA1", "Name": "a"},
                         {"ID": "WP1b", "ControlAccountID": "CA1", "Name": "b"}],
        "BCWS_ToDate": to_date("bcws"), "BCWP_ToDate": to_date("bcwp"),
        "ACWP_ToDate": to_date("acwp"),
        "BCWS_ToComplete": [{"ControlAccountID": r.ca, "ReportingPeriodID": int(r.pid),
                             "Value_Dollars": float(r.bcws)} for r in fut.itertuples()],
        "EST_ToComplete": etc,
        "SummaryPerformance": [pmb],
    }


def write_folder(tables, root):
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")
    return root


@pytest.fixture
def csv_data():
    return EvmData.read(EXAMPLE)


def _same(a: EvmData, b: EvmData):
    assert a.status == b.status and a.bac == pytest.approx(b.bac)
    assert np.allclose(a.bcws, b.bcws) and np.allclose(a.bcwp, b.bcwp)
    assert np.allclose(a.acwp, b.acwp)


def test_a_time_phased_dataset_reads_as_the_csv_does(tmp_path, csv_data):
    data = read_ipmdar(write_folder(cpd_tables(), tmp_path / "cpd"))
    _same(data, csv_data)
    assert data.name == "Example program"
    assert sorted(a.name for a in data.accounts.values()) == sorted(csv_data.accounts)
    assert data.eac[-1] == pytest.approx(16300.0) and np.isnan(data.eac[:-1]).all()
    assert any("folder of" in n for n in data.notes)
    assert any("reconcile" in n and "0.5%" in n for n in data.notes)
    m, c = data.metrics().iloc[-1], csv_data.metrics().iloc[-1]
    assert m.cpi == pytest.approx(c.cpi) and m.spi_t == pytest.approx(c.spi_t)


def test_every_packaging_reads_the_same(tmp_path, csv_data):
    tables = cpd_tables()
    folder = write_folder(tables, tmp_path / "f")
    one = tmp_path / "cpd.json"
    one.write_text(json.dumps(tables), encoding="utf-8")
    z = tmp_path / "cpd.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, rows in tables.items():
            zf.writestr(f"CPD/{name}.json", json.dumps(rows))
    for path in (folder, one, z):
        _same(read_ipmdar(path), csv_data)
    assert "ZIP" in load_dataset(z).notes[0] and "keyed by table" in load_dataset(one).notes[0]


def test_work_packages_roll_up_into_their_accounts(tmp_path, csv_data):
    data = read_ipmdar(write_folder(cpd_tables(by_work_package=True), tmp_path / "wp"))
    _same(data, csv_data)


def test_monthly_cumulative_deliveries_rebuild_the_history(tmp_path, csv_data):
    paths = [write_folder(cpd_tables(status=s, time_phased=False), tmp_path / f"m{s}")
             for s in (14, 9, 10, 11, 12, 13)]  # any order
    data = read_ipmdar(paths)
    # From the first delivery (period 9) on, every cumulative value is exact;
    # before it only period 9's totals are known, spread evenly.
    assert data.status == csv_data.status and data.bac == pytest.approx(csv_data.bac)
    for col in ("bcws", "bcwp", "acwp"):
        assert np.allclose(getattr(data, col)[8:], getattr(csv_data, col)[8:]), col
    assert np.allclose(data.bcws[:9], csv_data.bcws[8] * np.arange(1, 10) / 9)
    assert any("spread evenly" in n for n in data.notes)
    m, c = data.metrics().iloc[-1], csv_data.metrics().iloc[-1]
    assert m.cpi == pytest.approx(c.cpi)
    # Periods 10 to 14 are known one by one; 1 to 9 only in total.
    assert data.first_observed == 10
    assert np.allclose(np.diff(data.bcwp)[9:], np.diff(csv_data.bcwp)[9:])
    f = forecast(data, n_iter=500, seed=0)
    assert any("known only in total" in n for n in f.notes)
    one = read_ipmdar(paths[0])
    assert any("single status period" in n for n in one.notes)


def test_a_missing_month_is_named(tmp_path):
    paths = [write_folder(cpd_tables(status=s, time_phased=False), tmp_path / f"m{s}")
             for s in (10, 11, 13, 14)]
    assert any("[12]" in n for n in read_ipmdar(paths).notes)


def test_the_time_phased_flag_as_text(tmp_path):
    tables = cpd_tables(status=14, time_phased=False)
    tables["DatasetConfiguration"][0]["ToDate_TimePhased"] = "false"
    assert not load_dataset(write_folder(tables, tmp_path / "s")).time_phased
    tables["DatasetConfiguration"][0]["ToDate_TimePhased"] = "true"
    assert load_dataset(write_folder(tables, tmp_path / "t")).time_phased


def test_an_account_with_no_records_is_left_out_and_named(tmp_path, csv_data):
    tables = cpd_tables()
    tables["ControlAccounts"].append({"ID": "CA99", "Name": "Closed account",
                                      "IsSummaryLevelPlanningPackage": False})
    data = read_ipmdar(write_folder(tables, tmp_path / "c"))
    _same(data, csv_data)
    assert "CA99" not in data.accounts
    assert any("Closed account" in n for n in data.notes)


def test_detail_that_does_not_match_the_pmb_is_reported(tmp_path):
    data = read_ipmdar(write_folder(cpd_tables(pmb_shift=400.0), tmp_path / "x"))
    assert any("BCWP" in n and "against the PMB" in n for n in data.notes)


def test_what_is_not_a_cpd_is_refused(tmp_path):
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "Other.json").write_text("[]", encoding="utf-8")
    with pytest.raises(EvmError, match="Contract Performance Dataset"):
        read_ipmdar(tmp_path / "junk")
    tables = cpd_tables()
    tables["BCWP_ToDate"].append({"WorkPackageID": "WP9", "ReportingPeriodID": 1,
                                  "Value_Dollars": 1.0})
    with pytest.raises(EvmError, match="WP9"):
        read_ipmdar(write_folder(tables, tmp_path / "bad"))
    paths = [write_folder(cpd_tables(status=12, time_phased=False), tmp_path / f"d{i}")
             for i in range(2)]
    with pytest.raises(EvmError, match="same period"):
        read_ipmdar(paths)


def test_cli_reads_a_cpd(tmp_path, monkeypatch, capsys):
    from cost_core import cli

    folder = write_folder(cpd_tables(), tmp_path / "cpd")
    monkeypatch.setattr("sys.argv", ["ce-core", "evm", "--ipmdar", str(folder), "--iters", "500",
                                     "--out", str(tmp_path / "o")])
    cli.main()
    assert "Contractor EAC below every independent EAC" in capsys.readouterr().out
    notes = json.loads((tmp_path / "o" / "assumptions.json").read_text())["import_notes"]
    assert any("reconcile" in n for n in notes)
