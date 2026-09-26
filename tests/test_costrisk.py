# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Cost risk from an estimate in Excel: the reader, the analysis, the outputs."""

import numpy as np
import pandas as pd
import pytest

from cost_core import cli, costrisk, plain
from cost_core.costrisk import (
    EXAMPLE_ELEMENTS,
    EXAMPLE_PAIRS,
    EXAMPLE_RISKS,
    CostRiskError,
    analyse,
    read_workbook,
    write_workbook,
)
from cost_core.examples import example_path


def workbook(tmp_path, elements=EXAMPLE_ELEMENTS, risks=EXAMPLE_RISKS, pairs=EXAMPLE_PAIRS,
             settings=costrisk.EXAMPLE_SETTINGS, name="est.xlsx"):
    return write_workbook(tmp_path / name, elements, risks, pairs, settings)


@pytest.fixture(scope="module")
def result():
    return analyse(read_workbook(example_path("cost-risk")), n_iter=5000)


def test_the_shipped_example_is_the_one_the_code_writes(tmp_path):
    # The .xlsx in cost_core/examples is generated; this keeps it from drifting.
    shipped = read_workbook(example_path("cost-risk"))
    fresh = read_workbook(workbook(tmp_path))
    pd.testing.assert_frame_equal(shipped.elements, fresh.elements)
    pd.testing.assert_frame_equal(shipped.risks, fresh.risks)
    np.testing.assert_array_equal(shipped.model.correlation, fresh.model.correlation)
    assert (shipped.units, shipped.n_iter, shipped.seed) == (fresh.units, fresh.n_iter, fresh.seed)


def test_the_workbook_reads_into_the_model_it_describes():
    inputs = read_workbook(example_path("cost-risk"))
    m = inputs.model
    assert m.element_names == list(EXAMPLE_ELEMENTS["Element"])
    assert m.point_estimate == pytest.approx(EXAMPLE_ELEMENTS["Point Estimate"].sum())
    soft = m.elements[3]
    assert soft.distribution == {"type": "triangular", "left": 31.0, "mode": 36.0, "right": 90.0}
    assert m.elements[4].distribution["type"] == "pert"
    assert m.elements[7].distribution == {"type": "fixed", "value": 4.0}
    assert [r.name for r in m.risks] == list(EXAMPLE_RISKS["Risk"])
    assert m.risks[0].probability == 0.25 and m.risks[0].affects == "3.0 Antenna subsystem"
    # A listed pair takes its own value, every other pair the default.
    assert m.correlation[3, 5] == m.correlation[5, 3] == 0.6
    assert m.correlation[0, 7] == 0.3
    assert (inputs.units, inputs.n_iter, inputs.seed) == ("millions", 20000, 1)
    assert any("8.0 Initial spares" in n for n in inputs.notes)


def test_the_answer_holds_together(result):
    conf = result.confidence_table()
    assert conf["cost"].is_monotonic_increasing
    assert conf["reserve"].iloc[-1] == pytest.approx(conf["cost"].iloc[-1] - 156.0)
    drivers = result.drivers()
    assert drivers["variance_share"].sum() == pytest.approx(1.0)
    assert drivers.iloc[0]["component"] == "4.0 Software"
    spares = result.element_table().set_index("element").loc["8.0 Initial spares"]
    assert spares["p50"] == spares["p80"] == 4.0
    # Correlation between elements that overrun together widens the total.
    assert result.impact.correlated.p80 > result.impact.independent.p80
    risks = result.risk_table()
    assert risks["expected_cost"].iloc[0] == pytest.approx(0.25 * (4 + 6 + 10) / 3)


def test_the_same_seed_gives_the_same_answer():
    inputs = read_workbook(example_path("cost-risk"))
    a, b = analyse(inputs, n_iter=2000, seed=7), analyse(inputs, n_iter=2000, seed=7)
    np.testing.assert_array_equal(a.sim.totals, b.sim.totals)


def test_plain_words_name_the_numbers_and_the_driver(result):
    lines = plain.cost_risk(result, "$M")
    text = " ".join(lines)
    assert f"${result.sim.p80:,.0f}M" in text and "'4.0 Software'" in text
    assert "independent" in text and "8.0 Initial spares" in text
    assert all(line.endswith(".") for line in lines)


def test_a_csv_of_elements_alone_is_enough(tmp_path):
    path = tmp_path / "est.csv"
    EXAMPLE_ELEMENTS.to_csv(path, index=False)
    inputs = read_workbook(path)
    assert len(inputs.model.elements) == 8 and not inputs.model.risks
    assert inputs.model.correlation[0, 1] == costrisk.DEFAULT_CORRELATION == 0.3


def test_headings_are_matched_loosely(tmp_path):
    frame = pd.DataFrame({"WBS Element": ["a", "b"], "Min": [1, 2], "Mode": [2, 3],
                          "Max": [4, 5]})
    path = tmp_path / "loose.xlsx"
    frame.to_excel(path, sheet_name="Sheet1", index=False)
    inputs = read_workbook(path)
    # No point estimate column: the most likely stands in for it.
    assert inputs.model.point_estimate == 5.0


@pytest.mark.parametrize("change,message", [
    (lambda e: e.drop(columns="Element"), "no column for 'Element'"),
    (lambda e: e.drop(columns=["Point Estimate", "Most Likely"]), "Point Estimate"),
    (lambda e: e.assign(Low=[50.0] + list(e["Low"][1:])), "Low <= Most Likely <= High"),
    (lambda e: e.assign(High=[None] + list(e["High"][1:])), "High is blank"),
    (lambda e: e.assign(Distribution=["beta"] + list(e["Distribution"][1:])), "'beta'"),
    (lambda e: pd.concat([e, e.iloc[[0]]]), "more than once"),
    (lambda e: e.assign(Low=["about 11"] + list(e["Low"][1:])), "'about 11' is not a number"),
])
def test_a_bad_elements_sheet_says_what_is_wrong(tmp_path, change, message):
    with pytest.raises(CostRiskError, match=message):
        read_workbook(workbook(tmp_path, elements=change(EXAMPLE_ELEMENTS.copy())))


@pytest.mark.parametrize("probability,ok", [(0.3, True), ("30%", True), (30, False),
                                            (-0.1, False)])
def test_probabilities_are_fractions_or_percentages(tmp_path, probability, ok):
    risks = EXAMPLE_RISKS.copy()
    risks["Probability"] = risks["Probability"].astype(object)
    risks.loc[0, "Probability"] = probability
    path = workbook(tmp_path, risks=risks)
    if ok:
        assert read_workbook(path).model.risks[0].probability == pytest.approx(0.3)
    else:
        with pytest.raises(CostRiskError, match="not between 0 and 1"):
            read_workbook(path)


def test_a_risk_on_an_element_that_is_not_there_is_refused(tmp_path):
    risks = EXAMPLE_RISKS.assign(Element=["9.0 Nothing"] + list(EXAMPLE_RISKS["Element"][1:]))
    with pytest.raises(CostRiskError, match="'9.0 Nothing' is not on the Elements sheet"):
        read_workbook(workbook(tmp_path, risks=risks))


def test_correlation_pairs_must_name_real_elements(tmp_path):
    pairs = pd.DataFrame([("4.0 Software", "4.0 software", 0.5)], columns=EXAMPLE_PAIRS.columns)
    with pytest.raises(CostRiskError, match="names have to match exactly"):
        read_workbook(workbook(tmp_path, pairs=pairs))


def test_correlations_that_cannot_all_hold_are_repaired_and_said(tmp_path):
    names = list(EXAMPLE_ELEMENTS["Element"][:3])
    pairs = pd.DataFrame([(names[0], names[1], 0.9), (names[1], names[2], 0.9),
                          (names[0], names[2], -0.9)], columns=EXAMPLE_PAIRS.columns)
    inputs = read_workbook(workbook(tmp_path, pairs=pairs))
    assert np.linalg.eigvalsh(inputs.model.correlation).min() > -1e-9
    assert any("can't all hold at once" in n for n in inputs.notes)


def test_the_command_writes_the_report_and_takes_overrides(tmp_path, capsys):
    import openpyxl

    out = tmp_path / "o"
    cli.main(["cost-risk", "--data", str(example_path("cost-risk")), "--out", str(out),
              "--iters", "3000", "--seed", "5", "--units", "thousands"])
    printed = capsys.readouterr().out
    assert "What this means:" in printed and "costs in $K" in printed
    wb = openpyxl.load_workbook(out / "report.xlsx")
    for sheet in ("Summary", "Confidence", "Drivers", "Elements", "Risks", "Correlation",
                  "Correlation effect", "Convergence", "Assumptions"):
        assert sheet in wb.sheetnames, sheet
    assert pd.read_csv(out / "summary.csv").set_index("statistic").loc["iterations", "value"] \
        == 3000
    assert (out / "confidence.csv").is_file() and (out / "drivers.csv").is_file()


def test_the_briefing_is_written_when_pptx_is_installed(tmp_path, result):
    pytest.importorskip("pptx")
    from cost_core.reporting.brief import cost_risk_brief

    assert cost_risk_brief(result, tmp_path).is_file()


def test_a_broken_workbook_stops_with_the_reason(tmp_path, caplog):
    path = workbook(tmp_path, elements=EXAMPLE_ELEMENTS.drop(columns="Element"))
    with pytest.raises(SystemExit):
        cli.main(["cost-risk", "--data", str(path), "--out", str(tmp_path / "o")])
    assert "Cost risk failed" in caplog.text and "'Element'" in caplog.text


# ------------------------------------------------ fixed in 2.4.2, one each
def test_a_blank_row_does_not_shift_the_row_an_error_names(tmp_path):
    e = pd.concat([EXAMPLE_ELEMENTS.iloc[:3], pd.DataFrame([[None] * 6],
                   columns=EXAMPLE_ELEMENTS.columns), EXAMPLE_ELEMENTS.iloc[3:]]
                  ).reset_index(drop=True)
    e.loc[6, "Low"] = 99.0   # 6.0 Integration and test, on Excel row 8
    with pytest.raises(CostRiskError, match="row 8 "):
        read_workbook(workbook(tmp_path, elements=e))


def test_a_misspelt_setting_is_refused_not_ignored(tmp_path):
    settings = pd.DataFrame([("Units", "millions"), ("Default Corelation", 0.9)],
                            columns=["Setting", "Value"])
    with pytest.raises(CostRiskError, match="row 3: 'Default Corelation' is not a setting"):
        read_workbook(workbook(tmp_path, settings=settings))


@pytest.mark.parametrize("value,message", [("lots", "row 2: the Iterations 'lots' is not a number"),
                                           (-5, "row 2: Iterations is a whole number"),
                                           (2.5, "row 2: Iterations is a whole number")])
def test_a_bad_setting_value_names_its_row(tmp_path, value, message):
    settings = pd.DataFrame([("Iterations", value)], columns=["Setting", "Value"])
    with pytest.raises(CostRiskError, match=message):
        read_workbook(workbook(tmp_path, settings=settings))


def test_an_estimate_with_no_uncertainty_says_so_instead_of_crashing(tmp_path):
    bare = EXAMPLE_ELEMENTS[["Element", "Point Estimate"]]
    inputs = read_workbook(workbook(tmp_path, elements=bare, risks=EXAMPLE_RISKS.iloc[0:0],
                                    pairs=EXAMPLE_PAIRS.iloc[0:0]))
    with pytest.raises(CostRiskError, match="no uncertainty to simulate.*156"):
        analyse(inputs, n_iter=2000)


def test_a_padded_estimate_reads_sensibly(tmp_path):
    padded = EXAMPLE_ELEMENTS.assign(**{"Point Estimate": EXAMPLE_ELEMENTS["High"].fillna(4.0)})
    lines = plain.cost_risk(analyse(read_workbook(workbook(tmp_path, elements=padded)),
                                    n_iter=3000), "$M")
    text = " ".join(lines)
    assert "above every simulated cost" in text and "below the point estimate" in text
    assert "nan" not in text and "$-" not in text and "100% of" not in text


def test_a_point_estimate_outside_its_range_is_flagged(tmp_path):
    e = EXAMPLE_ELEMENTS.copy()
    e.loc[0, "Point Estimate"] = 12000.0
    notes = read_workbook(workbook(tmp_path, elements=e)).notes
    assert any("12,000, outside its own range of 11 to 18" in n for n in notes)


def test_a_negative_seed_is_refused_in_words(tmp_path):
    with pytest.raises(CostRiskError, match="whole number, 0 or more"):
        analyse(read_workbook(example_path("cost-risk")), n_iter=2000, seed=-1)


def test_a_damaged_workbook_says_so(tmp_path):
    bad = tmp_path / "broken.xlsx"
    bad.write_bytes(b"PK\x03\x04not really a workbook")
    with pytest.raises(CostRiskError, match="damaged, still downloading"):
        read_workbook(bad)


def test_most_likely_left_blank_shows_the_point_estimate_used(tmp_path):
    e = EXAMPLE_ELEMENTS.copy()
    e["Most Likely"] = e["Most Likely"].astype(object)
    e.loc[0, "Most Likely"] = None
    row = read_workbook(workbook(tmp_path, elements=e)).elements.iloc[0]
    assert row["most_likely"] == row["point_estimate"] == 12.0


def test_a_correlation_pair_with_a_blank_name_says_what_is_missing(tmp_path):
    p = EXAMPLE_PAIRS.copy()
    p["Element B"] = p["Element B"].astype(object)
    p.loc[0, "Element B"] = None
    with pytest.raises(CostRiskError, match="row 2: name both elements"):
        read_workbook(workbook(tmp_path, pairs=p))
