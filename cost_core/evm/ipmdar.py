# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
ipmdar.py - Read the IPMDAR Contract Performance Dataset.

The Integrated Program Management Data and Analysis Report (DI-MGMT-81861)
replaced the paper CPR formats with data: the Contract Performance Dataset
(CPD), a set of related JSON tables. Table and field names here follow the
*IPMDAR Contract Performance Dataset Version 1.0 Data Exchange Instructions*
(OUSD, 12 March 2020). The tables used:

* ``DatasetMetadata``: ``ReportingPeriodID`` (the status period) and the
  contract and program names;
* ``DatasetConfiguration``: ``ToDate_TimePhased`` (whether to-date values
  come per period or only cumulative to date) and whether each metric is
  reported by work package or control account;
* ``ReportingCalendar``: ``ID`` (1, 2, 3, ... in order), ``StartDate``,
  ``EndDate``;
* ``ControlAccounts`` and ``WorkPackages``: IDs, names, and each work
  package's ``ControlAccountID``;
* ``BCWS_ToDate``, ``BCWP_ToDate``, ``ACWP_ToDate``: to-date values by
  control account or work package, per period when time-phased;
* ``BCWS_ToComplete``, ``EST_ToComplete``: the rest of the baseline and the
  contractor's estimate to complete, per future period;
* ``SummaryPerformance``: the PMB totals (BAC, EAC, cumulative BCWS, BCWP
  and ACWP) the detail is reconciled against.

Everything is read at control account level, in burdened dollars
(``Value_Dollars``); work packages are summed into their accounts. The DEI
omits records that carry only zeros, so an account with no to-date record
is an account with nothing to date, not a missing one.

**Two kinds of delivery.** With ``ToDate_TimePhased`` true, one dataset
holds the whole history and is enough for the forecast. With it false, a
dataset has only cumulative-to-date values, one point; pass several
monthly deliveries to :func:`read_ipmdar` and the history is rebuilt from
them, one status period per dataset.

**Packaging.** The DEI defines the tables; how they are packaged is in the
companion File Format Specification, which was not to hand when this was
written. So both packagings in use are read: a folder or ZIP archive
holding one JSON file per table (``BCWP_ToDate.json`` and so on), each an
array of records, and a single JSON file whose top-level keys are the table
names. :attr:`IpmdarDataset.notes` says which was found.

**Reconciliation.** The DEI requires the detail, plus summary-level
overhead, COM and G&A where those are not flagged non-add, to sum to the
PMB less undistributed budget. The reader sums the control accounts and
compares the result with ``SummaryPerformance``'s PMB, and says by how much
they differ, rather than trusting either silently.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from cost_core.evm.metrics import EvmData, EvmError

#: The CPD tables this reader uses.
TABLES = ("DatasetMetadata", "DatasetConfiguration", "ReportingCalendar", "ControlAccounts",
          "WorkPackages", "BCWS_ToDate", "BCWP_ToDate", "ACWP_ToDate", "BCWS_ToComplete",
          "EST_ToComplete", "SummaryPerformance", "ContractData")

_METRICS = ("BCWS_ToDate", "BCWP_ToDate", "ACWP_ToDate", "BCWS_ToComplete", "EST_ToComplete")


@dataclass
class IpmdarDataset:
    """One Contract Performance Dataset, as tables.

    Attributes:
        tables: Each table read, as a DataFrame; absent tables are empty.
        status_period: ``DatasetMetadata.ReportingPeriodID``.
        time_phased: ``DatasetConfiguration.ToDate_TimePhased``.
        source: Where it was read from.
        notes: How it was packaged, and anything the reader had to decide.
    """

    tables: Dict[str, pd.DataFrame]
    status_period: int
    time_phased: bool
    source: str = ""
    notes: List[str] = field(default_factory=list)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    @property
    def metadata(self) -> dict:
        md = self.table("DatasetMetadata")
        return md.iloc[0].to_dict() if len(md) else {}

    @property
    def calendar(self) -> pd.DataFrame:
        cal = self.table("ReportingCalendar").copy()
        if not len(cal):
            raise EvmError(f"{self.source}: no ReportingCalendar.")
        cal["ID"] = cal["ID"].astype(int)
        return cal.sort_values("ID").reset_index(drop=True)

    def by_account(self, metric: str) -> pd.DataFrame:
        """A metric's burdened dollars by control account (and period, where
        the table carries one), with work packages summed into accounts."""
        t = self.table(metric)
        cols = ["ControlAccountID", "ReportingPeriodID", "Value_Dollars"]
        if not len(t):
            return pd.DataFrame(columns=cols)
        t = t.copy()
        if "WorkPackageID" in t.columns and t["WorkPackageID"].notna().any():
            wp = self.table("WorkPackages")
            if not len(wp):
                raise EvmError(f"{self.source}: {metric} is by work package but there is "
                               "no WorkPackages table to map them to accounts.")
            to_ca = dict(zip(wp["ID"].astype(str), wp["ControlAccountID"].astype(str)))
            wp_ids = t["WorkPackageID"].astype(str)
            unknown = sorted(set(wp_ids[t["WorkPackageID"].notna()]) - set(to_ca))
            if unknown:
                raise EvmError(f"{self.source}: {metric} names work package(s) {unknown[:5]} "
                               "that WorkPackages does not list.")
            mapped = wp_ids.map(to_ca)
            has_ca = t["ControlAccountID"].notna() if "ControlAccountID" in t.columns \
                else pd.Series(False, index=t.index)
            t["ControlAccountID"] = np.where(has_ca, t.get("ControlAccountID"), mapped)
        if "ReportingPeriodID" not in t.columns:
            t["ReportingPeriodID"] = np.nan
        t["ControlAccountID"] = t["ControlAccountID"].astype(str)
        t["Value_Dollars"] = pd.to_numeric(t.get("Value_Dollars"), errors="coerce").fillna(0.0)
        keys = ["ControlAccountID", "ReportingPeriodID"]
        return (t.groupby(keys, dropna=False)["Value_Dollars"].sum().reset_index()[cols])

    def pmb(self) -> dict:
        """The PMB row of ``SummaryPerformance``, if reported."""
        sp = self.table("SummaryPerformance")
        if not len(sp) or "SummaryElementID" not in sp.columns:
            return {}
        row = sp[sp["SummaryElementID"].astype(str) == "PMB"]
        return row.iloc[0].to_dict() if len(row) else {}


# ------------------------------------------------------------------- read ---
def _records(obj, name) -> pd.DataFrame:
    if isinstance(obj, dict):
        if name in obj and isinstance(obj[name], list):
            obj = obj[name]
        elif len(obj) == 1 and isinstance(next(iter(obj.values())), list):
            obj = next(iter(obj.values()))
        else:
            obj = [obj]  # a one-record table written as an object
    if not isinstance(obj, list):
        raise EvmError(f"Table {name}: expected a list of records.")
    return pd.DataFrame(obj)


def load_dataset(path: Union[str, Path]) -> IpmdarDataset:
    """Read one CPD from a folder, a ZIP archive or a single JSON file."""
    path = Path(path)
    raw: Dict[str, object] = {}
    if path.is_dir():
        files = {p.stem: p for p in path.glob("*.json")}
        for name, p in files.items():
            raw[name] = json.loads(p.read_text(encoding="utf-8-sig"))
        packaging = f"a folder of {len(files)} JSON table files"
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".json")]
            for n in names:
                with z.open(n) as fh:
                    raw[Path(n).stem] = json.load(io.TextIOWrapper(fh, encoding="utf-8-sig"))
        packaging = f"a ZIP archive of {len(names)} JSON table files"
    elif path.suffix.lower() == ".json":
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(doc, dict):
            raise EvmError(f"{path.name}: expected a JSON object keyed by table name.")
        raw = doc
        packaging = "one JSON file keyed by table name"
    else:
        raise EvmError(f"{path}: not a folder, ZIP archive or JSON file.")
    tables = {name: _records(raw[name], name) for name in TABLES if name in raw}
    if "ReportingCalendar" not in tables or "BCWP_ToDate" not in tables:
        found = sorted(raw)[:12]
        raise EvmError(f"{path.name}: no ReportingCalendar or BCWP_ToDate table; found "
                       f"{found}. Is this an IPMDAR Contract Performance Dataset?")
    md = tables.get("DatasetMetadata", pd.DataFrame())
    if not len(md) or "ReportingPeriodID" not in md.columns:
        raise EvmError(f"{path.name}: DatasetMetadata has no ReportingPeriodID.")
    cfg = tables.get("DatasetConfiguration", pd.DataFrame())
    phased = bool(cfg["ToDate_TimePhased"].iloc[0]) if len(cfg) and \
        "ToDate_TimePhased" in cfg.columns else "ReportingPeriodID" in tables["BCWP_ToDate"]
    return IpmdarDataset(tables=tables, status_period=int(md["ReportingPeriodID"].iloc[0]),
                         time_phased=_truthy(phased), source=path.name,
                         notes=[f"{path.name}: read as {packaging}."])


def _truthy(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes")


def read_ipmdar(paths: Union[str, Path, Iterable[Union[str, Path]]], *,
                bac: Optional[float] = None) -> EvmData:
    """Read one or more monthly CPDs into :class:`EvmData`, by control account.

    Args:
        paths: One dataset with time-phased to-date values, or several
            monthly datasets (any order) whose history is rebuilt from their
            cumulative values. With several, the latest supplies the
            baseline to complete and the estimate to complete.
        bac: Budget at completion; the control accounts' baseline total
            when omitted.

    The returned data's ``notes`` record the packaging, the reconciliation
    against the PMB and anything assumed.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    sets = sorted((load_dataset(p) for p in paths), key=lambda d: d.status_period)
    if len({d.status_period for d in sets}) < len(sets):
        raise EvmError("Two datasets report the same period.")
    latest = sets[-1]
    notes = [n for d in sets for n in d.notes]
    cal = latest.calendar
    periods = cal["ID"].tolist()
    status = latest.status_period
    if status not in periods:
        raise EvmError(f"Status period {status} is not in the ReportingCalendar.")
    k = periods.index(status) + 1

    accounts = _account_names(latest)
    # Baseline to complete and the estimate to complete, from the latest.
    btc = latest.by_account("BCWS_ToComplete")
    etc = latest.by_account("EST_ToComplete")

    if latest.time_phased:
        if len(sets) > 1:
            notes.append("The latest dataset is time-phased and holds the whole history; "
                          "the earlier ones were not needed.")
        todate = {m: latest.by_account(m) for m in ("BCWS_ToDate", "BCWP_ToDate", "ACWP_ToDate")}
        per_period = True
    else:
        if len(sets) == 1:
            notes.append("To-date values are cumulative only, so this one dataset gives "
                         "a single status period; pass the earlier monthly datasets too "
                         "for the history the forecast needs.")
        todate = {m: _stack_cumulative(sets, m) for m in ("BCWS_ToDate", "BCWP_ToDate",
                                                           "ACWP_ToDate")}
        per_period = False
    ids = sorted(set(accounts) | {a for t in list(todate.values()) + [btc, etc]
                                  for a in t["ControlAccountID"]})
    rows = []
    for ca in ids:
        for i, pid in enumerate(periods):
            rows.append({"period": cal.loc[i, "EndDate"], "pid": pid, "wbs": ca})
    frame = pd.DataFrame(rows)
    for col, metric in (("bcws", "BCWS_ToDate"), ("bcwp", "BCWP_ToDate"), ("acwp", "ACWP_ToDate")):
        frame[col] = _series(frame, todate[metric])
    # The baseline after the status period comes from BCWS_ToComplete.
    future = frame["pid"].map(lambda p: periods.index(p) + 1 > k)
    btc_map = _lookup(btc)
    frame.loc[future, "bcws"] = [btc_map.get((w, p), 0.0)
                                 for w, p in zip(frame.loc[future, "wbs"], frame.loc[future, "pid"])]
    frame.loc[future, ["bcwp", "acwp"]] = np.nan
    if not per_period:
        # Cumulative values into per-period ones, account by account.
        for col in ("bcws", "bcwp", "acwp"):
            past = ~future
            frame.loc[past, col] = frame[past].groupby("wbs")[col].diff().fillna(frame.loc[past, col])
    frame["period"] = frame["period"].astype(str).str[:10]
    data = EvmData.from_frame(frame.drop(columns="pid"), bac=bac, name=_program_name(latest))
    for ca, acct in data.accounts.items():
        acct.name = accounts.get(ca, ca)

    # The contractor's EAC at the status period: to date plus to complete.
    etc_total = float(etc["Value_Dollars"].sum()) if len(etc) else np.nan
    pmb = latest.pmb()
    eac = np.full(data.status, np.nan)
    if np.isfinite(etc_total) and len(etc):
        eac[-1] = float(data.acwp[-1]) + etc_total
    elif "EAC_Dollars" in pmb:
        eac[-1] = float(pmb["EAC_Dollars"])
    data.eac = eac if np.isfinite(eac[-1]) else None
    notes.extend(_reconcile(data, pmb))
    if not per_period and len(sets) > 1:
        # The first delivery's cumulative values lump every period before it;
        # performance is known period by period only after it.
        data.first_observed = periods.index(sets[0].status_period) + 2
        notes.append(f"History starts at period {sets[0].status_period}, the first delivery; "
                     "its cumulative totals are spread evenly over the periods before it, "
                     "so per-period figures there are not the contractor's.")
        gaps = sorted(set(range(sets[0].status_period, status + 1))
                      - {d.status_period for d in sets})
        if gaps:
            notes.append(f"No delivery for period(s) {gaps}; their performance is counted in "
                         "the next delivery's period.")
    data.notes = notes
    return data


def _account_names(ds: IpmdarDataset) -> Dict[str, str]:
    ca = ds.table("ControlAccounts")
    if not len(ca):
        return {}
    return dict(zip(ca["ID"].astype(str), ca.get("Name", ca["ID"]).astype(str)))


def _program_name(ds: IpmdarDataset) -> str:
    md = ds.metadata
    return str(md.get("ProgramName") or md.get("ContractName") or ds.source)


def _lookup(t: pd.DataFrame) -> dict:
    return {(str(a), int(p)): float(v) for a, p, v in
            zip(t["ControlAccountID"], t["ReportingPeriodID"], t["Value_Dollars"]) if pd.notna(p)}


def _series(frame: pd.DataFrame, t: pd.DataFrame) -> List[float]:
    table = _lookup(t)
    return [table.get((w, p), 0.0) for w, p in zip(frame["wbs"], frame["pid"])]


def _stack_cumulative(sets: List[IpmdarDataset], metric: str) -> pd.DataFrame:
    """Each dataset's cumulative-to-date values, stamped with its period.

    Periods between two deliveries are filled with the earlier cumulative
    value, so the history has one row per calendar period; the gap's
    performance lands in the period of the later delivery. Periods before
    the first delivery share its totals evenly.
    """
    parts = []
    for d in sets:
        t = d.by_account(metric)
        t = t.groupby("ControlAccountID", as_index=False)["Value_Dollars"].sum()
        t["ReportingPeriodID"] = d.status_period
        parts.append(t)
    known = pd.concat(parts, ignore_index=True)
    cal = sets[-1].calendar["ID"].tolist()
    status = sets[-1].status_period
    first = sets[0].status_period
    rows = []
    for ca, g in known.groupby("ControlAccountID"):
        by_p = dict(zip(g["ReportingPeriodID"], g["Value_Dollars"]))
        last = 0.0
        for p in cal:
            if p > status:
                break
            if p in by_p:
                last = by_p[p]
            # Before the first delivery only the total is known, so it is
            # spread evenly over the periods up to it (period IDs run 1, 2,
            # 3, ... by the DEI). The forecast does not draw on them.
            before = by_p.get(first, 0.0) * p / first
            rows.append({"ControlAccountID": ca, "ReportingPeriodID": p,
                         "Value_Dollars": last if p >= first else before})
    return pd.DataFrame(rows, columns=["ControlAccountID", "ReportingPeriodID", "Value_Dollars"])


def _reconcile(data: EvmData, pmb: dict) -> List[str]:
    """Compare the control accounts' sum with the PMB summary."""
    if not pmb:
        return ["No SummaryPerformance PMB row to reconcile the control accounts against."]
    out = []
    ours = {"BCWS": float(data.bcws[data.status - 1]), "BCWP": float(data.bcwp[-1]),
            "ACWP": float(data.acwp[-1])}
    for key, value in ours.items():
        theirs = pmb.get(f"{key}_CumulativeToDate_Dollars")
        if theirs is None or pd.isna(theirs):
            continue
        gap = value - float(theirs)
        if abs(gap) > 0.005 * max(abs(float(theirs)), 1.0):
            out.append(f"Control accounts sum to {key} {value:,.0f} against the PMB's "
                       f"{float(theirs):,.0f} ({gap:+,.0f}); summary-level OH, COM or G&A "
                       "not flagged non-add would account for a shortfall.")
    if not out:
        out.append("Control accounts reconcile with the PMB's cumulative BCWS, BCWP and ACWP "
                   "within 0.5%.")
    return out
