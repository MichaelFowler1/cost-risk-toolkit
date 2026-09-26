# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""What someone meets the first time they use the package.

Someone who has just run ``pip install cost-core`` and types ``ce-core``
should get a guide, not an error; ``ce-core demo`` should work with no files
at all; ``ce-core template`` should hand them a file that runs unchanged;
and a mistake should say how to fix it. Each of those is checked here from
an empty folder, the way a new user would meet it.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cost_core import cli, plain
from cost_core.examples import EXAMPLES, example_path


def run(argv, capsys):
    cli.main(argv)
    return capsys.readouterr().out


def test_every_example_is_installed_with_the_package():
    for topic in EXAMPLES:
        assert example_path(topic).is_file(), topic
    with pytest.raises(KeyError, match="there are"):
        example_path("nope")


def test_ce_core_alone_is_a_guide_not_an_error(capsys):
    out = run([], capsys)
    assert "What do you want to do?" in out and "ce-core demo evm" in out
    for command in ("evm", "cost-risk", "schedule-check", "jcl", "aoa", "portfolio", "fit-lots", "template"):
        assert command in out


def test_python_dash_m_runs_the_same_command():
    # For machines where pip's scripts folder is not on the PATH.
    done = subprocess.run([sys.executable, "-m", "cost_core"], capture_output=True, text=True)
    assert done.returncode == 0 and "What do you want to do?" in done.stdout


@pytest.mark.parametrize("topic", ["evm", "cost-risk", "schedule", "jcl", "aoa", "portfolio"])
def test_every_demo_runs_from_an_empty_folder(topic, tmp_path, monkeypatch, capsys):
    if topic == "portfolio":
        pytest.importorskip("pulp")
    monkeypatch.chdir(tmp_path)
    out = run(["demo", topic], capsys)
    assert "What this means:" in out and "Now with your own data:" in out
    assert (tmp_path / "ce-core-demo" / topic).is_dir()
    assert any((tmp_path / "ce-core-demo" / topic).iterdir())


@pytest.mark.parametrize("topic,command", [
    ("evm", ["evm", "--data", "my_evm.xlsx", "--iters", "1000", "--out", "o"]),
    ("cost-risk", ["cost-risk", "--data", "my_estimate.xlsx", "--iters", "2000", "--out", "o"]),
    ("jcl", ["jcl", "--spec", "my_jcl.xlsx", "--out", "o"]),
    ("aoa", ["aoa", "--spec", "my_aoa.xlsx", "--out", "o"]),
    ("portfolio", ["portfolio", "--spec", "my_portfolio.xlsx", "--out", "o"]),
    ("lots", ["fit-lots", "--csv", "my_lots.csv", "--dollar-year", "2026"]),
])
def test_every_template_runs_unchanged(topic, command, tmp_path, monkeypatch, capsys):
    pytest.importorskip("pulp") if topic == "portfolio" else None
    monkeypatch.chdir(tmp_path)
    out = run(["template", topic], capsys)
    assert "Wrote" in out and "ce-core" in out
    run(command, capsys)


def test_the_evm_template_is_an_excel_workbook_with_instructions(tmp_path, monkeypatch, capsys):
    import openpyxl

    monkeypatch.chdir(tmp_path)
    run(["template", "evm"], capsys)
    wb = openpyxl.load_workbook("my_evm.xlsx")
    assert wb.sheetnames == ["Data", "Instructions"]
    assert [c.value for c in wb["Data"][1]][:5] == [
        "Period", "Control Account", "Planned Value (BCWS)", "Earned Value (BCWP)",
        "Actual Cost (ACWP)"]


def test_a_template_is_never_overwritten_by_accident(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run(["template", "jcl"], capsys)
    with pytest.raises(SystemExit):
        run(["template", "jcl"], capsys)
    run(["template", "jcl", "--force"], capsys)
    run(["template", "jcl", "--out", "other/second.json"], capsys)
    assert (tmp_path / "other" / "second.json").is_file()


def test_the_schedule_template_says_how_to_save_one_from_project(capsys):
    out = run(["template", "schedule"], capsys)
    assert "Save As" in out and "XML" in out


@pytest.mark.parametrize("argv,topic", [
    (["evm", "--data", "missing.xlsx"], "evm"),
    (["cost-risk", "--data", "missing.xlsx"], "cost-risk"),
    (["jcl", "--spec", "missing.json"], "jcl"),
    (["aoa", "--spec", "missing.json"], "aoa"),
    (["portfolio", "--spec", "missing.json"], "portfolio"),
    (["schedule-check", "--mspdi", "missing.xml"], "schedule"),
])
def test_a_missing_file_says_how_to_get_one(argv, topic, tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(argv)
    message = caplog.text
    assert "Can't find" in message and f"ce-core demo {topic}" in message
    assert "ce-core template" in message


def test_no_internet_for_the_sar_panel_points_to_the_offline_route(monkeypatch, caplog):
    import cost_core.public.catalog as catalog
    from cost_core.public import FetchError

    def offline(*a, **k):
        raise FetchError("Wayback CDX listing failed (no response)")

    monkeypatch.setattr(catalog, "wayback_listing", offline)
    with pytest.raises(SystemExit):
        cli.main(["sar-panel", "--list-cycles"])
    assert "Couldn't reach the Internet Archive" in caplog.text
    assert "ce-core sar-panel --dir" in caplog.text


def test_plain_words_read_as_sentences():
    assert plain._chance(0.0).startswith("almost no chance")
    assert plain._chance(0.004) == "less than a 1% chance"
    assert plain._chance(0.45) == "a 45% chance"
    assert plain._chance(0.8) == "an 80% chance" and plain._chance(0.11) == "an 11% chance"
    assert plain._chance(1.0).startswith("near certainty")
    assert plain._money(17016.4, "$K") == "$17,016K" and plain._money(229, "$M") == "$229M"
    assert plain._money(5, "") == "5" and plain._money(5, "EUR") == "5 EUR"
    # Words that need no quoting at any prompt stand for the $ labels.
    assert plain.units_label("thousands") == "$K" and plain.units_label("Millions") == "$M"
    assert plain.units_label("EUR") == "EUR"


def test_evm_in_plain_words_matches_the_numbers():
    from cost_core.evm import EvmData, forecast

    data = EvmData.read(example_path("evm"))
    fc = forecast(data, n_iter=2000, seed=0)
    lines = plain.evm(data, fc, "$K")
    cpi = data.metrics().iloc[-1].cpi
    assert f"{cpi * 100:.0f} cents" in lines[0]
    # The overrun is 1/CPI - 1 (at CPI 0.5 the work costs twice its budget).
    assert f"{1 / cpi - 1:.0%} over its budget" in lines[0]
    assert f"${np.quantile(fc.eac, 0.5):,.0f}K" in lines[2]
    assert any("optimistic" in line for line in lines)


def test_the_portfolio_demo_without_a_solver_says_what_to_install(tmp_path, monkeypatch, caplog):
    import cost_core.portfolio.optimize as optimize

    def no_pulp():
        raise ImportError("portfolio optimisation needs PuLP")

    monkeypatch.setattr(optimize, "_pulp", no_pulp)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["demo", "portfolio"])
    assert 'pip install "cost-core[optimize]"' in caplog.text


def test_columns_named_by_the_caller_and_ambiguous_ones():
    from cost_core.evm import EvmData, EvmError

    base = pd.DataFrame({"Month": [1, 2, 1, 2], "PV": [5, 5, 5, 5], "EV": [4, 4, 5, 5],
                         "AC": [5, 5, 5, 5], "Control Account": ["a", "a", "b", "b"]})
    # The account column named explicitly is still found.
    data = EvmData.from_frame(base, account="Control Account")
    assert set(data.accounts) == {"a", "b"}
    # Two columns that could both be the period: neither is guessed.
    two = base.assign(Date=[1, 2, 1, 2]).drop(columns="Control Account")
    with pytest.raises(EvmError, match="reporting period"):
        EvmData.from_frame(two)
    # A custom period name that is missing is reported, not a crash.
    with pytest.raises(EvmError, match="'When'"):
        EvmData.from_frame(base, period="When")


def test_the_overrun_in_plain_words_is_one_over_cpi():
    from types import SimpleNamespace

    class Stub:
        name, bac, planned_duration = "p", 100.0, 10

        def metrics(self):
            return pd.DataFrame([{"cpi": 0.5, "sv_t": 0.0}])

        def flags(self):
            return pd.DataFrame({"flag": [], "raised": [], "detail": []})

    fc = SimpleNamespace(eac=np.array([200.0, 210.0]), finish=np.array([10.0, 10.0]),
                         contractor_eac=None, confidence_of_cost=lambda c: 0.0)
    assert "cost 100% over its budget" in plain.evm(Stub(), fc)[0]


def test_a_schedule_that_is_not_xml_says_how_to_save_one(tmp_path, caplog):
    mpp = tmp_path / "plan.mpp"
    mpp.write_bytes(b"\xd0\xcf\x11\xe0 not xml")
    with pytest.raises(SystemExit):
        cli.main(["schedule-check", "--mspdi", str(mpp)])
    assert "not an XML file" in caplog.text and "Save As > XML" in caplog.text


@pytest.mark.parametrize("n,text", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                                    (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
                                    (50, "50th"), (101, "101st"), (112, "112th")])
def test_ordinals_read_the_way_people_say_them(n, text):
    from cost_core.plain import ordinal

    assert ordinal(n) == text
