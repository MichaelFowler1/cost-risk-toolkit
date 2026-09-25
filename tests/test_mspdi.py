# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Reading Microsoft Project XML, and the DCMA 14-point check.

The schedule example, ``cost_core/examples/example_ims.xml``, was written by
hand for these tests (and ships as the example ``ce-core demo`` runs): ten
tasks under two summaries, with a finished task and one in progress, every
link type, a lag, an elapsed lag, a percentage lag and a lead, a hard
constraint, a baseline and a status date. Every expected number below was
worked out by hand from the file, not read back from the code, and the
comments show the working. Days are working days of 8 hours; a month is 20.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from cost_core.schedule import ScheduleError, critical_path, simulate
from cost_core.schedule.dcma import dcma_14_point
from cost_core.examples import example_path
from cost_core.schedule.mspdi import iso_duration_minutes, read_mspdi

IMS = example_path("schedule")


@pytest.fixture(scope="module")
def ims():
    return read_mspdi(IMS)


def test_iso_durations():
    assert iso_duration_minutes("PT8H0M0S") == 480
    assert iso_duration_minutes("PT0H0M30S") == 0.5
    assert iso_duration_minutes("-PT16H0M0S") == -960
    assert iso_duration_minutes("P1DT2H") == 1560
    assert iso_duration_minutes(None) is None
    with pytest.raises(ScheduleError):
        iso_duration_minutes("8 hours")


def test_tasks_read_in_working_days_and_currency(ims):
    t = ims.tasks.set_index("uid")
    assert ims.name == "Small IMS" and len(t) == 10 and 0 not in t.index
    assert list(ims.detail["uid"]) == [2, 3, 5, 6, 7, 8, 9, 10]
    assert t.loc[3, "duration_days"] == 20 and t.loc[3, "remaining_days"] == 10
    assert t.loc[2, "remaining_days"] == 0  # 100% complete
    assert t.loc[5, "cost"] == 6000 and t.loc[5, "fixed_cost"] == 1000  # hundredths
    assert t.loc[10, "total_float_days"] == -1 and t.loc[9, "total_float_days"] == 50
    assert t.loc[8, "constraint"] == "MSO" and t.loc[5, "baseline_duration_days"] == 50
    assert t.loc[5, "parent_uid"] == 4 and pd.isna(t.loc[7, "parent_uid"])
    # The -65535 placeholder on task 9 is not a resource.
    assert t.loc[9, "resources"] == 0 and t.loc[5, "resources"] == 1
    assert ims.status_date.isoformat() == "2026-03-02T17:00:00"
    assert ims.baseline_finish.isoformat() == "2026-05-22T17:00:00"


def test_links_keep_their_type_and_lag_in_working_days(ims):
    links = {(r.pred_uid, r.succ_uid): r for r in ims.links.itertuples()}
    assert links[5, 6].type == "SS" and links[5, 6].lag_days == 5       # 24000 tenths of a minute
    assert links[6, 7].type == "FF" and links[6, 7].lag_days == 12.5    # 50% of Software's 25 days
    assert links[7, 8].lag_days == pytest.approx(2 * 5 / 7)             # 2 elapsed days
    assert links[7, 10].lag_days == -2                                  # a lead
    assert links[1, 4].type == "FS"                                     # summary to summary


def test_summary_links_move_onto_the_detail_tasks(ims):
    d = ims.detail_links()
    pairs = set(zip(d.pred_uid, d.succ_uid))
    # Design (Requirements, Preliminary design) -> Build (Fabricate, Software).
    assert {(2, 5), (2, 6), (3, 5), (3, 6)} <= pairs
    assert not pairs & {(1, 4)} and len(d) == 10
    row = d[(d.pred_uid == 3) & (d.succ_uid == 6)].iloc[0]
    assert (row.pred_name, row.succ_name) == ("Preliminary design", "Software")
    assert any("summary" in n for n in ims.notes) and any("constraint" in n for n in ims.notes)


def test_the_network_schedules_as_worked_by_hand(ims):
    # Remaining work from the status date. Requirements is done (0 days).
    # Preliminary design 0-10. Fabricate after it, 10-40. Software after
    # Preliminary design (10) and 5 days after Fabricate starts (15): 15-40.
    # Integrate after Fabricate (40), and finishing 12.5 days after Software
    # does (52.5, so starting 42.5): 42.5-52.5. Ship 2 elapsed days (1.43
    # working) later: 53.93. Test prep 2 days before Integrate ends: 50.5-55.5.
    cpm = critical_path(ims.to_project()).set_index("activity")
    days = cpm[["early_start", "early_finish"]] * 20
    assert days.loc["T6"].tolist() == pytest.approx([15, 40])
    assert days.loc["T7"].tolist() == pytest.approx([42.5, 52.5])
    assert days.loc["T8", "early_start"] == pytest.approx(52.5 + 10 / 7)
    assert days.loc["T10"].tolist() == pytest.approx([50.5, 55.5])
    assert set(cpm.index[cpm.critical]) == {"T2", "T3", "T5", "T6", "T7", "T10"}


def test_costs_split_into_fixed_and_burn(ims):
    p = ims.to_project(remaining=False)
    fab = next(a for a in p.activities if a.id == "T5")
    # $6,000 in all, $1,000 of it fixed; the other $5,000 over 1.5 months.
    assert fab.fixed_cost == 1000 and fab.burn_rate == pytest.approx(5000 / 1.5)
    done = next(a for a in ims.to_project().activities if a.id == "T2")
    assert done.duration == 0 and done.duration_uncertainty is None


def test_uncertainty_default_and_per_task_overrides(ims):
    tri = {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.4}
    wide = {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 2.5}
    p = ims.to_project(tri, per_task={"Software": wide, 9: None})
    spec = {a.id: a.duration_uncertainty for a in p.activities}
    assert spec["T5"] == tri and spec["T6"] == wide and spec["T9"] is None
    assert spec["T8"] is None  # a milestone has no duration to vary
    r = simulate(p, n_iter=2000, seed=3)
    assert (r.finish >= 55.5 / 20 * 0.9 - 1e-9).all() and r.finish.mean() > 55.5 / 20


def test_dcma_14_point_as_worked_by_hand(ims):
    r = dcma_14_point(ims)
    t = r.table.set_index("check")
    # Incomplete detail tasks: 3, 5, 6, 7, 8, 9, 10. Ship has no successor and
    # Orphan no links; Test prep ends the project and is not counted.
    assert (t.loc[1, "count"], t.loc[1, "base"]) == (2, 7)
    assert r.tasks[1] == ["Ship", "Orphan"]
    # Ten links into incomplete tasks: one lead, three lags, two not FS.
    assert t.loc[2, "value"] == pytest.approx(0.1) and t.loc[3, "value"] == pytest.approx(0.3)
    assert t.loc[4, "value"] == pytest.approx(0.8) and not t.loc[4, "passed"]
    assert r.tasks[5] == ["Ship"] and r.tasks[6] == ["Orphan"] and r.tasks[7] == ["Test prep"]
    assert r.tasks[8] == ["Fabricate"]        # 50 baseline days
    assert r.tasks[9] == ["Software"]         # forecast start before the status date
    assert r.tasks[10] == ["Orphan", "Test prep"]
    # Requirements finished late, Preliminary design is not finished: both
    # missed; one of the two due is done, so BEI is 0.5.
    assert r.tasks[11] == ["Requirements", "Preliminary design"]
    assert t.loc[14, "value"] == pytest.approx(0.5)
    assert t.loc[12, "passed"] and "Preliminary design" in t.loc[12, "note"]
    # 88 calendar days from status to finish, 7 past the baseline finish,
    # both at 5/7 working days per calendar day.
    assert t.loc[13, "value"] == pytest.approx((88 - 7) / 88)
    assert (r.passed, r.failed) == (1, 13)


def test_dcma_passes_a_clean_schedule(tmp_path):
    xml = ['<Project xmlns="http://schemas.microsoft.com/project"><Title>Clean</Title>',
           "<StartDate>2026-01-05T08:00:00</StartDate><FinishDate>2026-01-30T17:00:00</FinishDate>",
           "<StatusDate>2026-01-02T17:00:00</StatusDate><Tasks>",
           "<Task><UID>0</UID><OutlineLevel>0</OutlineLevel><Summary>1</Summary>"
           "<Baseline><Number>0</Number><Finish>2026-01-30T17:00:00</Finish></Baseline></Task>"]
    starts = ["2026-01-05", "2026-01-12", "2026-01-19"]
    for uid, start in enumerate(starts, 1):
        link = (f"<PredecessorLink><PredecessorUID>{uid - 1}</PredecessorUID><Type>1</Type>"
                "</PredecessorLink>") if uid > 1 else ""
        xml.append(f"<Task><UID>{uid}</UID><Name>Step {uid}</Name><OutlineLevel>1</OutlineLevel>"
                   f"<Duration>PT40H0M0S</Duration><Start>{start}T08:00:00</Start>"
                   f"<Cost>100</Cost>{link}</Task>")
    xml.append("</Tasks></Project>")
    path = tmp_path / "clean.xml"
    path.write_text("".join(xml), encoding="utf-8")
    r = dcma_14_point(read_mspdi(path))
    t = r.table.set_index("check")
    assert r.failed == 0, r.table
    assert t.loc[11, "passed"] is None and t.loc[14, "passed"] is None  # nothing due yet


def test_a_file_that_is_not_mspdi_is_refused(tmp_path):
    path = tmp_path / "x.xml"
    path.write_text("<Workbook/>", encoding="utf-8")
    with pytest.raises(ScheduleError, match="not a Microsoft Project XML"):
        read_mspdi(path)


def test_a_file_without_a_namespace_reads_the_same(tmp_path, ims):
    bare = IMS.read_text(encoding="utf-8").replace(
        ' xmlns="http://schemas.microsoft.com/project"', "")
    path = tmp_path / "bare.xml"
    path.write_text(bare, encoding="utf-8")
    other = read_mspdi(path)
    pd.testing.assert_frame_equal(other.tasks, ims.tasks)
    pd.testing.assert_frame_equal(other.links, ims.links)


def test_jcl_spec_can_point_at_a_schedule_and_the_cli_runs_it(tmp_path, monkeypatch, capsys):
    from cost_core import cli
    from cost_core.schedule.spec import load_project

    spec = {"mspdi": str(IMS), "n_iter": 3000, "seed": 2, "standing_army": 100,
            "duration_uncertainty": {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.3},
            "per_task": {"10": None},
            "risks": [{"name": "Late test article", "probability": 0.4,
                       "activities": ["Integrate"], "delay": 0.5}]}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    project, settings = load_project(path)
    assert project.risks[0].activities == ["T7"] and project.standing_army == 100
    assert next(a for a in project.activities if a.id == "T10").duration_uncertainty is None
    assert any("summary" in n for n in settings["notes"])

    monkeypatch.setattr("sys.argv", ["ce-core", "jcl", "--spec", str(path), "--out", str(tmp_path / "j")])
    cli.main()
    notes = json.loads((tmp_path / "j" / "assumptions.json").read_text())["schedule_import_notes"]
    assert any("Calendars" in n for n in notes)

    monkeypatch.setattr("sys.argv", ["ce-core", "schedule-check", "--mspdi", str(IMS),
                                     "--out", str(tmp_path / "d")])
    cli.main()
    out = capsys.readouterr().out
    assert "1 passed, 13 failed" in out
    tasks = pd.read_csv(tmp_path / "d" / "dcma_tasks.csv")
    assert set(tasks[tasks.check == 1].task) == {"Ship", "Orphan"}


def test_an_unknown_task_in_a_spec_is_named(tmp_path):
    from cost_core.schedule.spec import load_project

    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"mspdi": str(IMS), "risks": [
        {"name": "r", "probability": 0.1, "activities": ["No such task"]}]}), encoding="utf-8")
    with pytest.raises(ScheduleError, match="No such task"):
        load_project(path)


@pytest.mark.parametrize("remaining_cost", ["<RemainingCost>10000000</RemainingCost>", ""])
def test_money_already_spent_is_sunk_not_spread_again(tmp_path, remaining_cost):
    # A 10-month task (200 days), 90% complete, $1M in all, $100K of it
    # left, whether the file says so or it follows from percent complete.
    # The $900K spent is fixed and certain; the $100K left burns over the
    # remaining month. Before, the whole $1M burned over that one month.
    xml = ('<Project xmlns="http://schemas.microsoft.com/project"><Tasks>'
           '<Task><UID>1</UID><Name>Build</Name><OutlineLevel>1</OutlineLevel>'
           '<Duration>PT1600H0M0S</Duration><RemainingDuration>PT160H0M0S</RemainingDuration>'
           f'<PercentComplete>90</PercentComplete><Cost>100000000</Cost>{remaining_cost}'
           '</Task></Tasks></Project>')
    path = tmp_path / "p.xml"
    path.write_text(xml, encoding="utf-8")
    sched = read_mspdi(path)
    (act,) = sched.to_project().activities
    assert act.duration == pytest.approx(1.0)
    assert act.fixed_cost == pytest.approx(900_000) and act.burn_rate == pytest.approx(100_000)
    (full,) = sched.to_project(remaining=False).activities
    assert full.burn_rate == pytest.approx(100_000) and full.duration == pytest.approx(10)


def test_critical_path_test_follows_start_to_start_links():
    # A drives B through a start-to-start link, and B drives the finish.
    # Lengthening A would not move the finish; starting it later does.
    from cost_core.schedule import Activity, Project
    from cost_core.schedule.dcma import _critical_path_test

    p = Project([Activity("A", 2), Activity("B", 10, None, [("A", 1, "SS")])])
    bad, note = _critical_path_test(p, {})
    assert bad == [] and "moved the finish 30 months" in note


def test_a_file_with_no_tasks_says_so(tmp_path):
    # Calendar- or resource-only files, of which MPXJ's samples hold dozens,
    # used to fail with KeyError: 'percent_complete'.
    path = tmp_path / "empty.xml"
    path.write_text('<Project xmlns="http://schemas.microsoft.com/project"><Title>Cal</Title>'
                    "<Tasks/></Project>", encoding="utf-8")
    sched = read_mspdi(path)
    assert len(sched.tasks) == 0 and len(sched.detail) == 0
    with pytest.raises(ScheduleError, match="no tasks to assess"):
        dcma_14_point(sched)
