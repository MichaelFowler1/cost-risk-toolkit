# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
mspdi.py - Read a Microsoft Project schedule saved as XML.

A program's integrated master schedule usually lives in Microsoft Project,
and "Save As > XML" writes it as MSPDI, Microsoft's published XML schema for
project data. Primavera P6, Deltek Open Plan and most other tools export
MSPDI too. This reads one into two tables, the tasks and the links, and
builds a :class:`cost_core.schedule.Project` from it, so a JCL or a DCMA
check can run on the schedule people actually manage rather than on a copy
retyped into JSON. Only the standard library's XML parser is used.

**Units.** MSPDI stores durations as ISO 8601 working time ("PT16H0M0S" is
two eight-hour days), lags and float in tenths of a minute, and money in
hundredths of the currency unit. Everything here comes out in working days
(tables) and months (the JCL network), using the file's own
``MinutesPerDay`` and ``DaysPerMonth`` (480 and 20 unless the file says
otherwise). A duration or lag entered as elapsed time ("3 edays") counts
every calendar day, so it is scaled by the file's working days per calendar
day (5/7 in a standard week). A lag given as a percentage is that share of
the predecessor's duration, as Project reads it.

**What the network keeps and what it doesn't.** Summary tasks are outline
headings, not work. A link to or from one is moved onto every detail task
under it, which is how Project itself schedules it. Calendars, resource
levelling and date constraints are not modelled: the JCL network is pure
logic, one working-time calendar, and every such simplification is listed in
:attr:`MspdiSchedule.notes`. A task in progress keeps only its remaining
duration and a finished one none, so a simulation runs from the status date
onwards.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from cost_core.schedule.jcl import Activity, Project, ScheduleError

#: MSPDI ``Type`` of a ``PredecessorLink``.
LINK_TYPE_CODES = {0: "FF", 1: "FS", 2: "SF", 3: "SS"}

#: MSPDI ``ConstraintType``.
CONSTRAINT_TYPES = {0: "ASAP", 1: "ALAP", 2: "MSO", 3: "MFO", 4: "SNET", 5: "SNLT",
                    6: "FNET", 7: "FNLT"}

#: Duration and lag formats counted in elapsed (calendar) time.
_ELAPSED_FORMATS = {4, 6, 8, 10, 12, 20, 36, 38, 40, 42, 44, 52}
_PERCENT_FORMATS = {19, 20, 51, 52}

_ISO_DURATION = re.compile(
    r"^(-)?P(?:(\d+(?:\.\d+)?)Y)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)D)?"
    r"(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?$")

#: Columns of :attr:`MspdiSchedule.tasks`.
TASK_COLUMNS = (
    "uid", "id", "name", "wbs", "outline_level", "parent_uid", "summary", "milestone",
    "duration_days", "remaining_days", "percent_complete", "start", "finish",
    "actual_start", "actual_finish", "baseline_start", "baseline_finish",
    "baseline_duration_days", "total_float_days", "free_float_days", "critical",
    "constraint", "constraint_date", "deadline", "cost", "fixed_cost", "remaining_cost",
    "resources",
    "elapsed_duration",
)

#: Columns of :attr:`MspdiSchedule.links`.
LINK_COLUMNS = ("pred_uid", "succ_uid", "pred_name", "succ_name", "type", "lag_days",
                "lag_format")


def iso_duration_minutes(text: Optional[str]) -> Optional[float]:
    """Minutes in an ISO 8601 duration as MSPDI writes it.

    >>> iso_duration_minutes("PT16H0M0S")
    960.0
    >>> iso_duration_minutes("PT0H30M0S")
    30.0

    Days, months and years are counted as 24 hours, 30 days and 365 days,
    their ISO meanings; MSPDI writes working time in hours, so they are rare.
    """
    if text is None or not text.strip():
        return None
    m = _ISO_DURATION.match(text.strip())
    if not m:
        raise ScheduleError(f"Not an ISO 8601 duration: {text!r}.")
    sign, y, mo, d, h, mi, s = m.groups()
    minutes = (float(y or 0) * 365 * 1440 + float(mo or 0) * 30 * 1440 + float(d or 0) * 1440
               + float(h or 0) * 60 + float(mi or 0) + float(s or 0) / 60)
    return -minutes if sign else minutes


@dataclass
class MspdiSchedule:
    """A schedule read from MSPDI.

    Attributes:
        name: The project's title, or the file name.
        tasks: One row per task (summaries included, the project summary
            task 0 left out); columns :data:`TASK_COLUMNS`, durations and
            float in working days, money in currency units.
        links: One row per predecessor link as the file has it;
            columns :data:`LINK_COLUMNS`.
        minutes_per_day, days_per_month, minutes_per_week: The file's own
            working-time settings, used for every conversion.
        status_date: The file's status date, if set.
        project_start, project_finish: The project's dates as scheduled.
        baseline_finish: The project summary task's baseline finish, if set.
        notes: What was simplified on the way in.
    """

    name: str
    tasks: pd.DataFrame
    links: pd.DataFrame
    minutes_per_day: float = 480.0
    days_per_month: float = 20.0
    minutes_per_week: float = 2400.0
    status_date: Optional[datetime] = None
    project_start: Optional[datetime] = None
    project_finish: Optional[datetime] = None
    baseline_finish: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)

    @property
    def detail(self) -> pd.DataFrame:
        """The tasks that are work: every task that is not a summary."""
        return self.tasks[~self.tasks["summary"]]

    def detail_links(self) -> pd.DataFrame:
        """The links between detail tasks, with every summary-task link
        moved onto the detail tasks under that summary."""
        leaves = self._leaves()
        names = dict(zip(self.tasks["uid"], self.tasks["name"]))
        rows = []
        for r in self.links.itertuples(index=False):
            for p in leaves.get(r.pred_uid, [r.pred_uid]):
                for s in leaves.get(r.succ_uid, [r.succ_uid]):
                    if p != s:
                        rows.append({**r._asdict(), "pred_uid": p, "succ_uid": s,
                                     "pred_name": names[p], "succ_name": names[s]})
        out = pd.DataFrame(rows, columns=list(LINK_COLUMNS))
        return out.drop_duplicates(subset=["pred_uid", "succ_uid", "type", "lag_days"])

    def _leaves(self) -> Dict[int, List[int]]:
        """Each summary task's detail tasks, however deep."""
        children: Dict[int, List[int]] = {}
        for t in self.tasks.itertuples(index=False):
            if pd.notna(t.parent_uid):
                children.setdefault(int(t.parent_uid), []).append(int(t.uid))
        summary = set(self.tasks.loc[self.tasks["summary"], "uid"].astype(int))

        def leaves(uid):
            out = []
            for c in children.get(uid, []):
                out.extend(leaves(c) if c in summary else [c])
            return out

        return {uid: leaves(uid) for uid in summary}

    def to_project(
        self,
        duration_uncertainty: Optional[Mapping[str, Any]] = None,
        *,
        per_task: Optional[Mapping[Union[int, str], Mapping[str, Any]]] = None,
        standing_army: float = 0.0,
        duration_correlation: float = 0.5,
        cost_correlation: float = 0.3,
        remaining: bool = True,
    ) -> Project:
        """The schedule as a JCL :class:`Project`, in months.

        Args:
            duration_uncertainty: Factor spec (see
                :func:`cost_core.monte_carlo.make_distribution`) applied to
                every detail task with a duration; None leaves them certain.
            per_task: Specs for particular tasks, keyed by UID or name,
                overriding the default. ``None`` as a value makes a task
                certain.
            standing_army: Cost per month for as long as the project runs.
            duration_correlation, cost_correlation: As on :class:`Project`.
            remaining: Use each task's remaining duration (the default, for
                a schedule with progress on it) rather than its full one.

        Each task's cost is split the way the JCL model wants it: the
        ``FixedCost`` field is time-independent, and the rest of its
        ``Cost`` (the resources) is spread over its duration as a burn rate.
        On a finished task, or one with no duration, all of it is fixed.

        With ``remaining``, what a task has already spent (its ``Cost``
        less its ``RemainingCost``, or its percent complete of the cost
        where the file gives no remaining cost) is certain, sunk and fixed,
        and only the remaining cost varies with the remaining duration. So
        the simulated total is still the whole program's, with nothing
        counted twice. The fixed cost is taken to accrue in proportion to
        progress, the Project default.
        """
        per_task = dict(per_task or {})
        month = self.days_per_month
        links = self.detail_links()
        preds: Dict[int, list] = {}
        for r in links.itertuples(index=False):
            preds.setdefault(int(r.succ_uid), []).append(
                (_act_id(r.pred_uid), r.lag_days / month, r.type))
        acts = []
        for t in self.detail.itertuples(index=False):
            days = t.remaining_days if remaining and pd.notna(t.remaining_days) else t.duration_days
            days = float(days or 0.0)
            spec = duration_uncertainty
            for key in (int(t.uid), t.name):
                if key in per_task:
                    spec = per_task[key]
            if days <= 0:
                spec = None
            cost, fixed = float(t.cost or 0.0), float(t.fixed_cost or 0.0)
            sunk = 0.0
            if remaining:
                done = min(max(float(t.percent_complete or 0.0) / 100.0, 0.0), 1.0)
                left = (float(t.remaining_cost) if pd.notna(t.remaining_cost)
                        else cost * (1.0 - done))
                left = min(max(left, 0.0), cost)
                sunk = cost - left
                fixed = fixed * (1.0 - done)
                cost = left
            variable = max(cost - fixed, 0.0)
            months = days / month
            if months > 0:
                burn = variable / months
            else:
                burn, fixed = 0.0, fixed + variable
            fixed += sunk
            acts.append(Activity(_act_id(t.uid), months, dict(spec) if spec else None,
                                 preds.get(int(t.uid), []), fixed_cost=fixed,
                                 burn_rate=burn, name=t.name or ""))
        return Project(acts, standing_army=standing_army,
                       duration_correlation=duration_correlation,
                       cost_correlation=cost_correlation, name=self.name)


def _act_id(uid) -> str:
    return f"T{int(uid)}"


def _get(el, tag) -> Optional[str]:
    v = el.findtext(tag) if el is not None else None
    return v.strip() if v is not None and v.strip() != "" else None


def _date(v: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(v) if v else None


def read_mspdi(path) -> MspdiSchedule:
    """Read a Microsoft Project XML (MSPDI) file.

    Raises:
        ScheduleError: The file is not MSPDI, or a link has a type code
            MSPDI does not define.
    """
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        hint = (" Microsoft Project's own .mpp files can't be read directly: open it in "
                "Project and use File > Save As > XML Format." if path.suffix.lower() == ".mpp"
                else " Save the schedule from Microsoft Project with File > Save As > XML Format.")
        raise ScheduleError(f"{path.name} is not an XML file ({exc}).{hint}") from None
    # Project writes the http://schemas.microsoft.com/project namespace and
    # some exporters write none; drop it so both read the same.
    for el in root.iter():
        el.tag = el.tag.rsplit("}", 1)[-1]
    if root.tag != "Project":
        raise ScheduleError(f"{path.name}: not a Microsoft Project XML file (root <{root.tag}>).")
    notes: List[str] = []
    g = _get

    mpd = float(g(root, "MinutesPerDay") or 480)
    mpw = float(g(root, "MinutesPerWeek") or 2400)
    dpm = float(g(root, "DaysPerMonth") or 20)
    working_per_calendar_day = (mpw / mpd) / 7.0

    def days(minutes: Optional[float], fmt: Optional[int]) -> Optional[float]:
        if minutes is None:
            return None
        if fmt in _ELAPSED_FORMATS:
            return minutes / 1440.0 * working_per_calendar_day
        return minutes / mpd

    # Resources assigned, per task: a real resource, not the placeholder -65535.
    resources: Dict[int, int] = {}
    for a in root.iter("Assignment"):
        tuid, ruid = g(a, "TaskUID"), g(a, "ResourceUID")
        if tuid is not None and ruid is not None and int(ruid) >= 0:
            resources[int(tuid)] = resources.get(int(tuid), 0) + 1

    tasks, raw_links = [], []
    parents: List[int] = []  # outline stack: uid at each level
    baseline_finish = None
    for t in root.iter("Task"):
        uid = int(g(t, "UID"))
        if g(t, "IsNull") == "1":
            continue
        level = int(g(t, "OutlineLevel") or 1)
        baselines = {int(b.findtext("Number") or 0): b for b in t.findall("Baseline")}
        b0 = baselines.get(0)
        if uid == 0:
            baseline_finish = _date(g(b0, "Finish"))
            continue
        if g(t, "ExternalTask") == "1":
            notes.append(f"Task {uid} ({g(t, 'Name')}) is external to this file and was left out.")
            continue
        del parents[max(level - 1, 0):]
        parent = parents[-1] if parents else None
        parents.append(uid)
        fmt = int(g(t, "DurationFormat") or 7)
        dur = days(iso_duration_minutes(g(t, "Duration")), fmt)
        rem = days(iso_duration_minutes(g(t, "RemainingDuration")), fmt)
        pct = float(g(t, "PercentComplete") or 0)
        if pct >= 100:
            rem = 0.0
        slack = g(t, "TotalSlack")
        free = g(t, "FreeSlack")
        ctype = int(g(t, "ConstraintType") or 0)
        tasks.append({
            "uid": uid, "id": int(g(t, "ID") or uid), "name": g(t, "Name") or "",
            "wbs": g(t, "WBS"), "outline_level": level, "parent_uid": parent,
            "summary": g(t, "Summary") == "1", "milestone": g(t, "Milestone") == "1",
            "duration_days": dur, "remaining_days": rem, "percent_complete": pct,
            "start": _date(g(t, "Start")), "finish": _date(g(t, "Finish")),
            "actual_start": _date(g(t, "ActualStart")), "actual_finish": _date(g(t, "ActualFinish")),
            "baseline_start": _date(g(b0, "Start")) if b0 is not None else None,
            "baseline_finish": _date(g(b0, "Finish")),
            "baseline_duration_days": (days(iso_duration_minutes(g(b0, "Duration")), fmt)
                                       if b0 is not None else None),
            "total_float_days": float(slack) / 10 / mpd if slack is not None else None,
            "free_float_days": float(free) / 10 / mpd if free is not None else None,
            "critical": g(t, "Critical") == "1",
            "constraint": CONSTRAINT_TYPES.get(ctype, str(ctype)),
            "constraint_date": _date(g(t, "ConstraintDate")),
            "deadline": _date(g(t, "Deadline")),
            "cost": float(g(t, "Cost") or 0) / 100,
            "fixed_cost": float(g(t, "FixedCost") or 0) / 100,
            "remaining_cost": (float(g(t, "RemainingCost")) / 100
                               if g(t, "RemainingCost") is not None else None),
            "resources": resources.get(uid, 0),
            "elapsed_duration": fmt in _ELAPSED_FORMATS,
        })
        for link in t.findall("PredecessorLink"):
            raw_links.append((int(g(link, "PredecessorUID")), uid,
                              int(g(link, "Type") if g(link, "Type") is not None else 1),
                              float(g(link, "LinkLag") or 0), int(g(link, "LagFormat") or 7),
                              g(link, "CrossProject") == "1"))

    task_df = pd.DataFrame(tasks, columns=list(TASK_COLUMNS))
    task_df["parent_uid"] = task_df["parent_uid"].astype("Int64")
    for col in ("duration_days", "remaining_days", "baseline_duration_days",
                "total_float_days", "free_float_days", "percent_complete", "cost",
                "fixed_cost", "remaining_cost"):
        task_df[col] = pd.to_numeric(task_df[col], errors="coerce").astype(float)
    # Typed even when there are no tasks (a calendar- or resource-only file),
    # so a filter on them is a filter and not a column selection.
    for col in ("summary", "milestone", "critical", "elapsed_duration"):
        task_df[col] = task_df[col].astype(bool)
    for col in ("uid", "id", "outline_level", "resources"):
        task_df[col] = task_df[col].astype(int)
    names = dict(zip(task_df["uid"], task_df["name"]))
    dur_by_uid = dict(zip(task_df["uid"], task_df["duration_days"]))
    links = []
    for pred, succ, code, lag, lag_fmt, cross in raw_links:
        if cross or pred not in names:
            notes.append(f"A link into task {succ} from task {pred}, which is not in this "
                         "file, was left out.")
            continue
        if code not in LINK_TYPE_CODES:
            raise ScheduleError(f"Task {succ}: unknown link type code {code}.")
        if lag_fmt in _PERCENT_FORMATS:
            # Stored as the percentage itself: 50 is half the predecessor.
            lag_days = lag / 100.0 * float(dur_by_uid.get(pred) or 0.0)
        else:
            lag_days = days(lag / 10.0, lag_fmt)
        links.append({"pred_uid": pred, "succ_uid": succ, "pred_name": names[pred],
                      "succ_name": names[succ], "type": LINK_TYPE_CODES[code],
                      "lag_days": lag_days, "lag_format": lag_fmt})
    link_df = pd.DataFrame(links, columns=list(LINK_COLUMNS))

    n_constrained = int((~task_df["summary"] & ~task_df["constraint"].isin(["ASAP"])).sum())
    if n_constrained:
        notes.append(f"{n_constrained} task(s) carry a date constraint, which the JCL "
                     "network does not model; the DCMA check reports them.")
    if task_df["elapsed_duration"].any():
        notes.append("Elapsed durations were converted to working days at "
                     f"{working_per_calendar_day:.3f} working days per calendar day.")
    n_summary_links = int(link_df["pred_uid"].isin(task_df.loc[task_df["summary"], "uid"]).sum()
                          + link_df["succ_uid"].isin(task_df.loc[task_df["summary"], "uid"]).sum())
    if n_summary_links:
        notes.append(f"{n_summary_links} link end(s) on summary tasks were moved onto the "
                     "detail tasks under them.")
    notes.append("Calendars and resource levelling are not modelled: every task works "
                 f"{mpd / 60:g} hours a day, {dpm:g} days a month.")
    return MspdiSchedule(
        name=g(root, "Title") or g(root, "Name") or path.stem,
        tasks=task_df, links=link_df, minutes_per_day=mpd, days_per_month=dpm,
        minutes_per_week=mpw, status_date=_date(g(root, "StatusDate")),
        project_start=_date(g(root, "StartDate")), project_finish=_date(g(root, "FinishDate")),
        baseline_finish=baseline_finish, notes=notes)
