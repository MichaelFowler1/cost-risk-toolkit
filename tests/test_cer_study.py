# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""ce-core cer: a CER study read from Excel, checked against hand arithmetic.

Every number here is invented.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cost_core import cli
from cost_core.cer.study import CerError, analyse, read_workbook, write_workbook


def workbook(path, data, estimate=None, settings=None):
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame(data).to_excel(xl, sheet_name="Data", index=False)
        if estimate is not None:
            pd.DataFrame(estimate).to_excel(xl, sheet_name="Estimate", index=False)
        if settings is not None:
            pd.DataFrame(settings, columns=["Setting", "Value"]).to_excel(
                xl, sheet_name="Settings", index=False)
    return path


def line_data(n=10, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(10, 100, n)
    return {"Program": [f"P{i}" for i in range(n)], "Cost": 5 + 2 * x + rng.normal(0, 8, n),
            "Weight": x}


def test_the_template_reads_back_with_its_settings(tmp_path):
    inp = read_workbook(write_workbook(tmp_path / "cer.xlsx"))
    assert inp.response == "Cost" and inp.predictors == ("Weight (lb)", "Power (kW)")
    assert (inp.form.value, inp.method, inp.level) == ("log_log", "mupe", 0.8)
    assert inp.units == "millions" and inp.dollar_year == "BY2026"
    assert len(inp.used) == 12 and list(inp.estimate["name"]) == ["Next radar", "Large radar"]


def test_log_log_ols_matches_least_squares_on_the_logs(tmp_path):
    inp = read_workbook(write_workbook(tmp_path / "cer.xlsx"))
    study = analyse(inp, method="ols")
    d = inp.used
    design = np.column_stack([np.ones(len(d)), np.log(d["Weight (lb)"]), np.log(d["Power (kW)"])])
    beta, *_ = np.linalg.lstsq(design, np.log(d["Cost"]), rcond=None)
    got = study.coefficients()["estimate"].to_numpy()
    assert got == pytest.approx(beta, rel=1e-9)


def test_the_prediction_interval_is_the_textbook_one(tmp_path):
    data = line_data()
    p = workbook(tmp_path / "line.xlsx", data, estimate={"Program": ["New"], "Weight": [64.0]},
                 settings=[("Form", "linear"), ("Method", "OLS"), ("Confidence", 0.9)])
    est = analyse(read_workbook(p)).estimates.iloc[0]
    x, y = np.asarray(data["Weight"]), np.asarray(data["Cost"])
    n = len(x)
    b1, b0 = np.polyfit(x, y, 1)
    s = np.sqrt(np.sum((y - (b0 + b1 * x)) ** 2) / (n - 2))
    half = stats.t.ppf(0.95, n - 2) * s * np.sqrt(1 + 1 / n + (64 - x.mean()) ** 2
                                                   / np.sum((x - x.mean()) ** 2))
    assert est["estimate"] == pytest.approx(b0 + b1 * 64, rel=1e-9)
    assert (est["lower"], est["upper"]) == pytest.approx((b0 + b1 * 64 - half,
                                                          b0 + b1 * 64 + half), rel=1e-9)


def test_rows_left_out_are_named_with_the_reason(tmp_path):
    data = line_data()
    data["Use"] = ["yes"] * 8 + ["no", 1.0]
    data["Note"] = [""] * 8 + ["data suspect", ""]
    data["Weight"] = list(data["Weight"][:-1]) + [None]
    inp = read_workbook(workbook(tmp_path / "d.xlsx", data))
    assert len(inp.used) == 8
    assert list(inp.excluded["why"]) == ["data suspect", "no Weight given"]
    assert "2 of 10 rows are left out" in inp.notes[0]


def test_a_year_column_is_not_taken_as_a_driver(tmp_path):
    data = {**line_data(), "FY": list(range(2010, 2020))}
    inp = read_workbook(workbook(tmp_path / "d.xlsx", data))
    assert inp.predictors == ("Weight",)
    assert any("'FY' is not used as a driver" in n for n in inp.notes)
    named = read_workbook(workbook(tmp_path / "e.xlsx", data,
                                   settings=[("Drivers", "Weight; FY"), ("Form", "linear")]))
    assert named.predictors == ("Weight", "FY")


@pytest.mark.parametrize("change, message", [
    (lambda d: d.__setitem__("Cost", ["lots"] + list(d["Cost"][1:])), "Data sheet, row 2"),
    (lambda d: d.__setitem__("Weight", [-1.0] + list(d["Weight"][1:])),
     "log-log CER needs every value above zero"),
    (lambda d: d.__setitem__("Use", ["maybe"] + ["yes"] * 9), "Use is yes or no"),
])
def test_bad_data_is_refused_with_its_row(tmp_path, change, message):
    data = line_data()
    change(data)
    with pytest.raises(CerError, match=message):
        read_workbook(workbook(tmp_path / "d.xlsx", data))


def test_too_few_programs_and_a_missing_estimate_driver_are_refused(tmp_path):
    small = {"Program": ["A", "B"], "Cost": [10.0, 20.0], "Weight": [1.0, 2.0]}
    with pytest.raises(CerError, match="2 programs can't fit 2 parameters"):
        analyse(read_workbook(workbook(tmp_path / "s.xlsx", small)))
    with pytest.raises(CerError, match="Estimate sheet has no column for 'Weight'"):
        read_workbook(workbook(tmp_path / "e.xlsx", line_data(),
                               estimate={"Program": ["New"], "Mass": [5.0]}))


def test_extrapolation_and_a_weak_driver_are_said(tmp_path):
    from cost_core import plain

    study = analyse(read_workbook(write_workbook(tmp_path / "cer.xlsx")))
    assert list(study.estimates["outside_data"]) == ["", "yes"]
    rng = np.random.default_rng(8)
    data = line_data(seed=8)
    data["Colour"] = rng.uniform(1, 9, 10)
    weak = analyse(read_workbook(workbook(tmp_path / "w.xlsx", data,
                                          settings=[("Form", "linear")])))
    assert "Colour's effect can't be told apart from zero" in " ".join(plain.cer(weak, ""))


def test_a_csv_of_the_data_alone_fits(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame(line_data()).to_csv(p, index=False)
    study = analyse(read_workbook(p, drivers=["Weight"]), form="linear")
    assert study.cer.predictors == ("Weight",) and study.estimates.empty


def test_the_command_writes_every_output(tmp_path, capsys):
    from openpyxl import load_workbook

    p = write_workbook(tmp_path / "cer.xlsx")
    cli.main(["cer", "--data", str(p), "--out", str(tmp_path / "o"), "--marking", "CUI"])
    out = capsys.readouterr().out
    assert "Next radar" in out and "outside the data" in out
    for f in ("report.xlsx", "brief.pptx", "methods.csv", "coefficients.csv", "estimates.csv",
              "diagnostics.csv", "data.csv", "assumptions.json", "cer_fit.png"):
        assert (tmp_path / "o" / f).exists(), f
    wb = load_workbook(tmp_path / "o" / "report.xlsx")
    assert {"Estimates", "Methods compared", "Coefficients", "Diagnostics", "Data",
            "Log-space bias", "Assumptions"} <= set(wb.sheetnames)
    assert wb["Summary"].oddHeader.center.text == "CUI"


def test_flags_override_the_workbook(tmp_path, capsys):
    import json

    p = write_workbook(tmp_path / "cer.xlsx")
    cli.main(["cer", "--data", str(p), "--out", str(tmp_path / "o"), "--method", "ols",
              "--confidence", "0.9", "--drivers", "Weight (lb)"])
    capsys.readouterr()
    record = json.loads((tmp_path / "o" / "assumptions.json").read_text())
    assert record["method"] == "OLS" and record["interval"] == "90% prediction interval"
    assert record["drivers"] == "Weight (lb)"


def test_demo_template_and_open_know_the_command(tmp_path, capsys, caplog):
    cli.main(["demo", "cer", "--out", str(tmp_path / "demo")])
    assert (tmp_path / "demo" / "report.xlsx").exists()
    cli.main(["template", "cer", "--out", str(tmp_path / "mine.xlsx")])
    assert "ce-core cer --data" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["template", "cer", "--out", str(tmp_path / "mine.csv")])
    from cost_core import opener

    assert opener.plan(tmp_path / "mine.xlsx").argv[0] == "cer"


def test_the_chart_draws_one_driver_with_its_band(tmp_path):
    pytest.importorskip("matplotlib")
    from cost_core.reporting.charts import plot_cer_diagnostics

    study = analyse(read_workbook(workbook(tmp_path / "l.xlsx", line_data(),
                                           settings=[("Form", "linear")])))
    assert plot_cer_diagnostics(study.cer, tmp_path / "c.png", level=0.9,
                                units="$K").exists()


def test_a_text_cell_stops_the_command_with_its_row(tmp_path, capsys, caplog):
    data = line_data()
    data["Cost"] = ["lots"] + list(data["Cost"][1:])
    p = workbook(tmp_path / "d.xlsx", data)
    with pytest.raises(SystemExit):
        cli.main(["cer", "--data", str(p), "--out", str(tmp_path / "o")])
    assert "Data sheet, row 2: the Cost 'lots' is not a number" in (
        caplog.text + capsys.readouterr().err)
