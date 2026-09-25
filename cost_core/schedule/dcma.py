# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
dcma.py - The DCMA 14-point schedule assessment.

A JCL is only as good as the schedule under it: dangling tasks, hard
constraints and leads let a network finish on a date that no logic supports,
and a simulation of that network is confident about nothing. The Defense
Contract Management Agency's 14-point assessment is the standard first look
at whether a schedule's logic can be trusted, and is what a DoD program's
schedule is held to. GAO's Schedule Assessment Guide (GAO-16-89G) covers the
same ground.

Each check runs on the incomplete detail tasks (not summaries, not finished
work), as the assessment specifies, and reports the count, the base, the
share and the threshold, with the tasks that failed so they can be fixed.
The thresholds are DCMA's: 5% for most, none at all for leads, negative float
and invalid dates, 90% finish-to-start, and 0.95 for the two indices.

Two choices the assessment leaves open, made explicitly here:

* **Missing logic** does not count the one task that starts the project
  and the one that ends it (the task with the earliest start and no
  predecessor, and the one with the latest finish and no successor), since
  every network has to begin and end somewhere.
* **The critical path test** (check 12) is run on the network as
  :func:`cost_core.schedule.critical_path` computes it: a long delay is added
  to the earliest critical task and the project finish has to move by
  exactly that much, so the delay has to travel the whole path.
  A break in the logic (a constraint or a dangling task holding the finish)
  shows up as the finish not moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from cost_core.schedule.jcl import Activity, Project, critical_path
from cost_core.schedule.mspdi import MspdiSchedule

#: The hard constraints: ones that fix a date regardless of the logic.
HARD_CONSTRAINTS = ("MSO", "MFO", "SNLT", "FNLT")

#: Float or duration over this many working days is "high" (two months).
HIGH_DAYS = 44.0


@dataclass
class DcmaResult:
    """The assessment.

    Attributes:
        table: One row per check: ``check``, ``name``, ``count``, ``base``,
            ``value``, ``threshold``, ``passed`` (None where the file does
            not hold what the check needs) and ``note``.
        tasks: For each check number, the names of the tasks that failed it.
    """

    table: pd.DataFrame
    tasks: Dict[int, List[str]] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return int((self.table["passed"] == True).sum())  # noqa: E712 - None is neither

    @property
    def failed(self) -> int:
        return int((self.table["passed"] == False).sum())  # noqa: E712


def _share(count: int, base: int) -> Optional[float]:
    return count / base if base else None


def dcma_14_point(schedule: MspdiSchedule) -> DcmaResult:
    """Run the 14 checks on a schedule read by :func:`read_mspdi`."""
    t = schedule.tasks
    detail = t[~t["summary"]]
    incomplete = detail[detail["percent_complete"] < 100]
    inc_uids = set(incomplete["uid"])
    names = dict(zip(t["uid"], t["name"]))
    links = schedule.detail_links()
    status = schedule.status_date
    rows: List[dict] = []
    failed: Dict[int, List[str]] = {}

    def add(no, name, bad, base, threshold, compare="max", note="", value=None):
        count = len(bad) if bad is not None else None
        v = value if value is not None else (_share(count, base) if count is not None else None)
        if v is None:
            ok = None
        elif compare == "max":
            ok = v <= threshold + 1e-12
        else:
            ok = v >= threshold - 1e-12
        rows.append({"check": no, "name": name, "count": count, "base": base, "value": v,
                     "threshold": threshold, "passed": ok, "note": note})
        failed[no] = [names.get(u, str(u)) for u in (bad or [])]

    # 1. Logic: incomplete tasks with no predecessor or no successor.
    has_pred = set(links["succ_uid"])
    has_succ = set(links["pred_uid"])
    no_pred = [u for u in incomplete["uid"] if u not in has_pred]
    no_succ = [u for u in incomplete["uid"] if u not in has_succ]
    exempt = set()
    starts = detail[~detail["uid"].isin(has_pred)].sort_values("start")
    if len(starts):
        exempt.add(int(starts["uid"].iloc[0]))
    ends = detail[~detail["uid"].isin(has_succ)].sort_values("finish")
    if len(ends):
        exempt.add(int(ends["uid"].iloc[-1]))
    missing = sorted((set(no_pred) | set(no_succ)) - exempt)
    add(1, "Logic (missing predecessor or successor)", missing, len(incomplete), 0.05,
        note="the project's first and last task are not counted")

    # 2-4. Leads, lags and relationship types, on links into incomplete tasks.
    live = links[links["succ_uid"].isin(inc_uids)]
    lead = live[live["lag_days"] < 0]
    lag = live[live["lag_days"] > 0]
    add(2, "Leads (negative lag)", sorted(set(lead["succ_uid"])), len(live), 0.0,
        value=_share(len(lead), len(live)))
    add(3, "Lags", sorted(set(lag["succ_uid"])), len(live), 0.05,
        value=_share(len(lag), len(live)))
    non_fs = live[live["type"] != "FS"]
    fs_share = _share(len(live) - len(non_fs), len(live))
    add(4, "Relationship types (finish-to-start share)", sorted(set(non_fs["succ_uid"])),
        len(live), 0.90, compare="min", value=fs_share)
    rows[-1]["count"] = len(live) - len(non_fs)

    # 5. Hard constraints.
    hard = incomplete[incomplete["constraint"].isin(HARD_CONSTRAINTS)]
    add(5, "Hard constraints", list(hard["uid"]), len(incomplete), 0.05,
        note="MSO, MFO, SNLT and FNLT")

    # 6-7. Float.
    tf = incomplete["total_float_days"]
    add(6, f"High float (over {HIGH_DAYS:g} working days)",
        list(incomplete.loc[tf > HIGH_DAYS, "uid"]), len(incomplete), 0.05)
    add(7, "Negative float", list(incomplete.loc[tf < -1e-9, "uid"]), len(incomplete), 0.0)

    # 8. High duration, on the baseline duration where there is one.
    work = incomplete[~incomplete["milestone"]]
    dur = work["baseline_duration_days"].fillna(work["duration_days"])
    add(8, f"High duration (over {HIGH_DAYS:g} working days)",
        list(work.loc[dur > HIGH_DAYS, "uid"]), len(work), 0.05,
        note="baseline duration where set")

    # 9. Invalid dates: forecasts before the status date, actuals after it.
    if status is not None:
        bad = []
        for r in detail.itertuples(index=False):
            if pd.notna(r.actual_start) and r.actual_start > status:
                bad.append(r.uid)
            elif pd.notna(r.actual_finish) and r.actual_finish > status:
                bad.append(r.uid)
            elif r.percent_complete < 100 and (
                    (pd.isna(r.actual_start) and pd.notna(r.start) and r.start < status)
                    or (pd.notna(r.finish) and r.finish < status)):
                bad.append(r.uid)
        add(9, "Invalid dates", bad, len(detail), 0.0,
            note="forecast before, or actual after, the status date")
    else:
        add(9, "Invalid dates", None, len(detail), 0.0, note="no status date in the file")

    # 10. Resources: incomplete work with no resource and no cost.
    no_res = work[(work["resources"] == 0) & (work["cost"] <= 0)]
    add(10, "Resources (no resource or cost)", list(no_res["uid"]), len(work), 0.0)

    # 11. Missed tasks: due by the status date on the baseline, and not
    # finished on it.
    has_bl = detail[detail["baseline_finish"].notna()]
    if status is not None and len(has_bl):
        due = has_bl[has_bl["baseline_finish"] <= status]
        missed = due[due["actual_finish"].isna() | (due["actual_finish"] > due["baseline_finish"])]
        add(11, "Missed tasks", list(missed["uid"]), len(due), 0.05)
        # 14. Baseline execution index: finished over due.
        done = int(detail["actual_finish"].notna().sum())
        bei = done / len(due) if len(due) else None
        add(14, "Baseline execution index (BEI)", [], len(due), 0.95, compare="min", value=bei,
            note=f"{done} tasks finished of {len(due)} baselined to finish by the status date")
        rows[-1]["count"] = done
        failed[14] = list(missed["name"]) if bei is not None and bei < 0.95 else []
    else:
        why = "no status date" if status is None else "no baseline"
        add(11, "Missed tasks", None, 0, 0.05, note=why)
        add(14, "Baseline execution index (BEI)", None, 0, 0.95, compare="min", note=why)

    # 12. Critical path test.
    project = schedule.to_project(remaining=True)
    bad12, note12 = _critical_path_test(project, names)
    add(12, "Critical path test", bad12, None, 0.0, note=note12,
        value=None if bad12 is None else float(len(bad12)))
    rows[-1]["count"] = None if bad12 is None else len(bad12)

    # 13. Critical path length index: (length + project float) / length.
    cpli, note13 = _cpli(schedule, project)
    add(13, "Critical path length index (CPLI)", [], None, 0.95, compare="min",
        value=cpli, note=note13)
    rows[-1]["count"] = None

    table = pd.DataFrame(rows).sort_values("check").reset_index(drop=True)
    return DcmaResult(table=table, tasks=failed)


def _critical_path_test(project: Project, names):
    cpm = critical_path(project)
    crit = cpm[cpm["critical"] & (cpm["duration"] > 0)]
    if not len(crit):
        return None, "no critical task with a duration"
    # The earliest: a delay there has to travel the whole path to the finish.
    probe = crit.sort_values("early_start", kind="stable").iloc[0]["activity"]
    finish = float(cpm["early_finish"].max())
    delay = 30.0  # months: far longer than any float, so it has to show
    acts = [Activity(a.id, a.duration + (delay if a.id == probe else 0.0), None,
                     a.predecessors) for a in project.activities]
    moved = float(critical_path(Project(acts))["early_finish"].max()) - finish
    ok = abs(moved - delay) < 1e-6
    label = names.get(int(probe[1:]), probe)
    return ([] if ok else [int(probe[1:])],
            f"{delay:g} months added to {label!r} moved the finish {moved:g} months")


def _cpli(schedule: MspdiSchedule, project: Project):
    if schedule.baseline_finish is None or schedule.project_finish is None:
        return None, "no baseline finish for the project"
    start = schedule.status_date or schedule.project_start
    if start is None:
        return None, "no status date or project start"
    days_per_calendar_day = (schedule.minutes_per_week / schedule.minutes_per_day) / 7.0
    length = (schedule.project_finish - start).total_seconds() / 86400 * days_per_calendar_day
    slack = ((schedule.baseline_finish - schedule.project_finish).total_seconds() / 86400
             * days_per_calendar_day)
    if length <= 0:
        return None, "the project finishes before the status date"
    return (length + slack) / length, (
        f"critical path {length:.1f} working days, project float {slack:+.1f} against the baseline finish")
