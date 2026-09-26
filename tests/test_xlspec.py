# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Excel workbooks for the JCL, AoA and portfolio specs."""

import json
import math

import pandas as pd
import pytest

from cost_core import cli, xlspec
from cost_core.examples import example_path
from cost_core.xlspec import SpecError, read_spec, write_workbook


def example(kind):
    return json.loads(example_path(kind).read_text(encoding="utf-8"))


def frames(kind, spec=None):
    """The sheets write_workbook would write, as frames a test can change."""
    spec = spec or example(kind)
    out = xlspec._FRAMES[kind](spec)
    settings = xlspec._portfolio_settings(spec) if kind == "portfolio" else spec
    out["Settings"] = xlspec._settings_frame(kind, settings)
    return out


def book(tmp_path, sheets, name="spec.xlsx"):
    path = tmp_path / name
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for sheet, frame in sheets.items():
            frame.to_excel(xl, sheet_name=sheet, index=False)
    return path


def same(a, b, where="$"):
    if isinstance(a, dict):
        assert set(a) == set(b), (where, set(a) ^ set(b))
        for k in a:
            same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isclose(float(a), float(b), rel_tol=1e-12), (where, a, b)
    else:
        assert a == b, (where, a, b)


@pytest.mark.parametrize("kind", ["jcl", "aoa", "portfolio"])
def test_every_example_survives_the_round_trip(kind, tmp_path):
    spec = example(kind)
    back = read_spec(write_workbook(spec, kind, tmp_path / f"{kind}.xlsx"))
    same(spec, back)


def test_the_kind_of_workbook_is_told_by_its_sheets(tmp_path):
    for kind in ("jcl", "aoa", "portfolio"):
        assert xlspec.kind_of(frames(kind)) == kind
    with pytest.raises(SpecError, match="no Activities"):
        read_spec(book(tmp_path, {"Sheet1": pd.DataFrame({"a": [1]})}))


def test_a_workbook_and_its_json_give_the_same_jcl(tmp_path):
    from cost_core.schedule.spec import load_project
    from cost_core.schedule.jcl import simulate

    xl = write_workbook(example("jcl"), "jcl", tmp_path / "jcl.xlsx")
    a, sa = load_project(example_path("jcl"))
    b, sb = load_project(xl)
    assert sa == sb
    ra, rb = simulate(a, n_iter=2000, seed=1), simulate(b, n_iter=2000, seed=1)
    assert (ra.cost == rb.cost).all() and (ra.finish == rb.finish).all()


def test_json_still_works_and_bad_json_says_so(tmp_path):
    assert read_spec(example_path("aoa")) == example("aoa")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SpecError, match="not valid JSON"):
        read_spec(bad)


@pytest.mark.parametrize("text,link", [
    ("design", "design"), ("design -4", ["design", -4.0]), ("design-4", ["design", -4.0]),
    ("bus SS +2", ["bus", 2.0, "SS"]), ("bus ff", ["bus", 0.0, "FF"]),
    ("i&t", "i&t"),
])
def test_predecessors_are_written_the_way_people_write_them(text, link):
    assert xlspec._link(text, ["design", "bus", "i&t"], "here") == link


def test_link_text_reads_back_as_the_same_link():
    ids = ["a", "b"]
    for p in ("a", ["a", -4], ["b", 2, "SS"], {"id": "a", "lag": 1, "type": "FF"}):
        text = xlspec._link_text(p)
        back = xlspec._link(text, ids, "here")
        expect = xlspec._link(xlspec._link_text(back), ids, "here")
        assert back == expect


def test_jcl_ranges_are_in_months_not_factors(tmp_path):
    f = frames("jcl")
    acts = f["Activities"]
    assert acts.loc[0, "Duration"] == 12 and acts.loc[0, "Duration High"] == pytest.approx(16.8)
    spec = read_spec(book(tmp_path, f))
    assert spec["activities"][0]["duration_uncertainty"]["right"] == 1.4


# -------------------------------------------------------- one per message
def _jcl(change):
    f = frames("jcl")
    change(f)
    return f


def _set(sheet, col, row, value):
    def change(f):
        f[sheet][col] = f[sheet][col].astype(object)
        f[sheet].loc[row, col] = value
    return change


JCL_ERRORS = [
    (lambda f: f.update(Activities=f["Activities"].iloc[0:0]), "Activities sheet has no rows"),
    (lambda f: f.update(Activities=f["Activities"].drop(columns="ID")), "no 'ID' column"),
    (_set("Activities", "ID", 1, "design"), "appear more than once"),
    (lambda f: f.update(Activities=f["Activities"].drop(
        columns=["Duration", "Duration Most Likely"])), "no Duration"),
    (_set("Activities", "Predecessors", 1, "nothing"), "'nothing' is not an activity ID"),
    (_set("Activities", "Duration Low", 0, 20), "Low <= Most Likely <= High"),
    (_set("Activities", "Duration High", 0, None), "both a Low and a High"),
    (_set("Activities", "Duration Distribution", 0, "beta"), "'beta' is not triangular"),
    (_set("Activities", "Duration Low", 0, "soon"), "'soon' is not a number"),
    (_set("Risks", "Probability", 0, 3), "between 0 and 1"),
    (_set("Risks", "Risk", 0, None), "the risk has no name"),
    (_set("Risks", "Activities", 0, "nowhere"), "'nowhere' is not an activity ID"),
    (lambda f: f.update(Settings=pd.concat([f["Settings"], pd.DataFrame(
        [("Colour", "blue")], columns=["Setting", "Value"])])), "'Colour' is not a setting"),
]


@pytest.mark.parametrize("change,message", JCL_ERRORS)
def test_a_bad_jcl_workbook_says_what_is_wrong(tmp_path, change, message):
    with pytest.raises(SpecError, match=message):
        read_spec(book(tmp_path, _jcl(change)))


def test_a_uniform_range_needs_both_ends(tmp_path):
    f = frames("jcl")
    for change in (_set("Activities", "Duration Distribution", 0, "uniform"),
                   _set("Activities", "Duration High", 0, None)):
        change(f)
    with pytest.raises(SpecError, match="uniform range needs both"):
        read_spec(book(tmp_path, f))


def test_a_percentage_probability_is_read(tmp_path):
    spec = read_spec(book(tmp_path, _jcl(_set("Risks", "Probability", 0, "30%"))))
    assert spec["risks"][0]["probability"] == pytest.approx(0.3)


def _aoa(change):
    f = frames("aoa")
    change(f)
    return f


AOA_ERRORS = [
    (lambda f: f.update(Lines=f["Lines"].iloc[0:0]), "Lines sheet has no rows"),
    (lambda f: f.update(Lines=f["Lines"].drop(columns="Phase")), "no 'Phase' column"),
    (_set("Lines", "Alternative", 0, "Teleport"), "'Teleport' is not on the Alternatives"),
    (_set("Lines", "Line", 0, None), "give the Alternative and the Line"),
    (_set("Lines", "Start Year", 0, None), "needs a Start Year"),
    (_set("Lines", "Last Year", 2, None), "needs a First Year and a Last Year"),
    (lambda f: f.update(Lines=f["Lines"].assign(Total=None, **{"Annual Amount": None})),
     "no cost"),
    (lambda f: f.update(Phased=f["Phased"].drop(columns=[2061, 2062])),
     "one column per year"),
    (lambda f: f.update(Phased=pd.concat([f["Phased"], f["Phased"].assign(Line="Scrap")])),
     "Scrap that is not on the Lines sheet"),
    (_set("Alternatives", "Alternative", 0, None), "no name"),
    (lambda f: f.update(Lines=f["Lines"][f["Lines"]["Alternative"] != "Buy commercial"]),
     "'Buy commercial' has no cost lines"),
]


@pytest.mark.parametrize("change,message", AOA_ERRORS)
def test_a_bad_aoa_workbook_says_what_is_wrong(tmp_path, change, message):
    with pytest.raises(SpecError, match=message):
        read_spec(book(tmp_path, _aoa(change)))


def test_aoa_lines_without_an_alternatives_sheet_make_their_own(tmp_path):
    f = frames("aoa")
    del f["Alternatives"]
    spec = read_spec(book(tmp_path, f))
    assert [a["name"] for a in spec["alternatives"]] == \
        [a["name"] for a in example("aoa")["alternatives"]]


def _port(change):
    f = frames("portfolio")
    change(f)
    return f


PORTFOLIO_ERRORS = [
    (lambda f: f.update(Budget=f["Budget"].iloc[0:0]), "Budget sheet has no rows"),
    (_set("Budget", "Budget", 0, None), "give both the Year and the Budget"),
    (lambda f: f.update(Candidates=f["Candidates"].iloc[0:0]), "Candidates sheet has no rows"),
    (lambda f: f.update(Candidates=f["Candidates"][["Candidate", "Option", "Value"]]),
     "one cost column per year"),
    (_set("Candidates", "Candidate", 0, None), "no candidate name"),
    (_set("Candidates", "Option", 0, None), "the Option has no name"),
    (_set("Candidates", "Value", 0, None), "no Value"),
    (_set("Candidates", "Mandatory", 0, "maybe"), "Mandatory is yes or no"),
    (_set("Candidates", "Requires", 0, "Moon base"), "requires 'Moon base'"),
    (_set("Exclusive", "Candidates", 0, "Sensor: new development; Moon base"),
     "'Moon base' is not a candidate"),
    (lambda f: f.update(Settings=pd.concat([f["Settings"], pd.DataFrame(
        [("Frontier Scales", "big; small")], columns=["Setting", "Value"])])),
     "Frontier Scales is a list"),
]


@pytest.mark.parametrize("change,message", PORTFOLIO_ERRORS)
def test_a_bad_portfolio_workbook_says_what_is_wrong(tmp_path, change, message):
    with pytest.raises(SpecError, match=message):
        read_spec(book(tmp_path, _port(change)))


def test_a_distribution_a_workbook_cannot_hold_is_refused():
    spec = example("aoa")
    spec["alternatives"][0]["lines"][0]["uncertainty"] = {"type": "lognormal", "mean": 0,
                                                          "sigma": 0.2}
    with pytest.raises(SpecError, match="keep this spec as JSON"):
        xlspec._aoa_frames(spec)


def test_a_json_template_is_still_available(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["template", "jcl", "--out", "mine.json"])
    assert json.loads((tmp_path / "mine.json").read_text()) == example("jcl")
    cli.main(["template", "jcl"])
    assert (tmp_path / "my_jcl.xlsx").is_file()
    capsys.readouterr()


def test_a_bad_workbook_stops_the_command_with_the_reason(tmp_path, caplog):
    path = book(tmp_path, _aoa(_set("Lines", "Start Year", 0, None)))
    with pytest.raises(SystemExit):
        cli.main(["aoa", "--spec", str(path), "--out", str(tmp_path / "o")])
    assert "Lines sheet, row 2" in caplog.text and "Start Year" in caplog.text
