# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
phasing.py - An estimate spread into fiscal years, in then-year dollars.

What ``ce-core phase`` runs. The step between an estimate and a budget: each
line of the estimate, a total in base-year dollars, is spread over the years
it will be spent in, with a profile that matches how that kind of money goes
out, and then inflated to then-year dollars with the index that applies to
it. The result is the spend by fiscal year and by appropriation that goes
into a budget position.

The workbook has a Phasing sheet (one row per estimate line), an optional
Settings sheet (the base year, the units, the default index) and an optional
Index sheet (the inflation indices, in the layout ``ce-core inflate`` reads).
``ce-core template phase`` writes one with an invented example.

**Profiles.** ``uniform``; ``front`` and ``back``, weighted linearly toward
the start or the end; ``bell``, a symmetric hump; ``rayleigh``, the classic
shape of development spending, rising fast and tailing off, with its peak
where the Peak column puts it (as a fraction of the duration, 0.38 by
default: where a Rayleigh curve that has spent 97% by the end peaks); and a
list of percentages, one per year (``10;30;40;20``), for an office's own
standard phasing. The first four are the same shapes the AoA spreads with.

**Indices.** Different appropriations inflate differently, so each line can
name its own index from the Index sheet (or a table given with
``--index``). With no index at all, the phasing is in base-year dollars only
and says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cost_core.costrisk import (CostRiskError, _blank, _norm, excel_rows, headingless,
                                open_workbook)
from cost_core.costrisk import _number as _costrisk_number


class PhaseError(ValueError):
    """A phasing workbook that can't be used; the message says where and why."""


def _number(value, sheet: str, row: int, what: str) -> Optional[float]:
    try:
        return _costrisk_number(value, sheet, row, what)
    except CostRiskError as e:
        raise PhaseError(str(e)) from None


#: Where a Rayleigh curve peaks when the Peak column is blank, as a fraction
#: of the duration: a Rayleigh that has spent 97% of its total by the end of
#: the duration peaks at 1 / sqrt(-2 ln 0.03) = 0.378 of the way through.
RAYLEIGH_PEAK = 0.378

PROFILES = ("uniform", "front", "back", "bell", "rayleigh")

LINE_COLUMNS = {
    "line": ("line", "element", "wbs_element", "wbs", "name", "item", "description"),
    "appropriation": ("appropriation", "approp", "color_of_money", "colour_of_money",
                      "phase", "category"),
    "amount": ("amount", "base_year_amount", "total", "cost", "estimate", "by_amount",
               "point_estimate"),
    "start": ("start", "start_year", "start_fy", "first_year", "first_fy", "begin"),
    "years": ("years", "duration", "number_of_years", "n_years"),
    "end": ("end", "end_year", "end_fy", "last_year", "last_fy", "finish"),
    "profile": ("profile", "shape", "spread", "phasing"),
    "peak": ("peak", "rayleigh_peak", "peak_at"),
    "index": ("index", "index_name", "inflation_index"),
}

SETTING_NAMES = {"base_year": "base_year", "dollar_year": "base_year", "by": "base_year",
                 "units": "units", "index": "index", "index_name": "index",
                 "inflation_rate": "inflation_rate", "rate": "inflation_rate",
                 "description": None, "notes": None}


# ------------------------------------------------------------------ profiles
def profile_weights(profile: str, years: int, peak: Optional[float] = None) -> np.ndarray:
    """The share of the total spent in each of ``years`` years, summing to 1."""
    if years < 1:
        raise PhaseError(f"A line needs at least one year to spread over; got {years}.")
    k = np.arange(1, years + 1, dtype=float)
    p = _norm(profile) or "uniform"
    if p == "rayleigh":
        peak = RAYLEIGH_PEAK if peak is None else float(peak)
        if not 0.0 < peak < 1.0:
            raise PhaseError(f"A Rayleigh peak is a fraction of the duration between 0 and "
                             f"1 (0.38 is usual); got {peak:g}.")
        # The cumulative Rayleigh, scaled so its mode sits at ``peak`` of the
        # duration, read at each year's end and renormalised so the truncated
        # tail is spent within the duration.
        cum = 1.0 - np.exp(-((np.r_[0.0, k] / years) ** 2) / (2.0 * peak ** 2))
        w = np.diff(cum)
    else:
        w = {"uniform": np.ones(years), "front": k[::-1], "back": k,
             "bell": np.sin(np.pi * (k - 0.5) / years)}.get(p)
        if w is None:
            raise PhaseError(f"Unknown profile {profile!r}: use uniform, front, back, bell "
                             "or rayleigh, or a percentage for each year (10;30;40;20).")
    return w / w.sum()


def _percentages(text: str) -> Optional[List[float]]:
    """A profile written as one number per year, or None if it isn't one."""
    parts = [s for s in re.split(r"[;,\s]+", str(text).replace("%", "").strip()) if s]
    try:
        values = [float(s) for s in parts]
    except ValueError:
        return None
    return values or None


def _fy(value):
    """FY2027 or 2027 as 2027; anything else as given, for _number to judge."""
    if isinstance(value, str):
        text = value.strip().upper()
        if text.startswith("FY"):
            return text[2:].strip()
    return value


def spread(total: float, start: int, weights: np.ndarray) -> Dict[int, float]:
    """``total`` over consecutive fiscal years from ``start``, landing on the
    total exactly."""
    amounts = total * np.asarray(weights, dtype=float)
    amounts[-1] = total - amounts[:-1].sum()
    return {start + i: float(a) for i, a in enumerate(amounts)}


# ------------------------------------------------------------------- reading
@dataclass
class PhaseLine:
    line: str
    appropriation: str
    amount: float
    start: int
    years: int
    profile: str
    weights: np.ndarray = field(repr=False)
    index: Optional[str] = None
    row: int = 0


@dataclass
class PhaseInput:
    name: str
    lines: List[PhaseLine]
    base_year: int
    units: str = ""
    indices: Dict[str, Dict[int, float]] = field(default_factory=dict)
    index_source: str = ""
    default_index: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    stated: frozenset = frozenset()


def _columns(frame: pd.DataFrame, sheet: str, required: Sequence[str]) -> Dict[str, str]:
    have = {_norm(c): c for c in frame.columns}
    found = {}
    for key, aliases in LINE_COLUMNS.items():
        for alias in aliases:
            if alias in have:
                found[key] = have[alias]
                break
    missing = [k for k in required if k not in found]
    if missing:
        raise PhaseError(f"The {sheet} sheet has no column for "
                         f"{', '.join(repr(k.title()) for k in missing)}. Its headings are "
                         f"{', '.join(repr(str(c)) for c in frame.columns)}; ce-core "
                         "template phase writes one with the headings it reads.")
    return found


def _index_table(frame: pd.DataFrame, where: str) -> Dict[str, Dict[int, float]]:
    have = {_norm(c): c for c in frame.columns}
    cols = [have.get(c) for c in ("index_name", "fiscal_year", "index_value")]
    if None in cols:
        raise PhaseError(f"{where} needs index_name, fiscal_year and index_value columns, "
                         "the layout ce-core template inflate writes.")
    out: Dict[str, Dict[int, float]] = {}
    frame = frame.dropna(how="all")
    for i, rec in zip(excel_rows(frame), frame.to_dict("records")):
        name = str(rec[cols[0]]).strip()
        year = _number(rec[cols[1]], where, i, "fiscal_year")
        value = _number(rec[cols[2]], where, i, "index_value")
        if year is None or value is None or not name or name == "nan":
            raise PhaseError(f"{where}, row {i}: give the index name, the fiscal year and "
                             "the index value.")
        if value <= 0:
            raise PhaseError(f"{where}, row {i}: an index value of {value:g} can't be a "
                             "price index.")
        if int(year) in out.setdefault(name, {}):
            raise PhaseError(f"{where}, row {i}: {name!r} already has FY{int(year)}.")
        out[name][int(year)] = value
    return out


def _read_settings(frame) -> Tuple[Dict, frozenset]:
    out: Dict = {"base_year": None, "units": "", "index": None, "inflation_rate": None}
    stated = set()
    if frame is None:
        return out, frozenset()
    frame = headingless(frame, SETTING_NAMES).dropna(how="all")
    for row, rec in zip(excel_rows(frame), frame.itertuples(index=False)):
        if len(rec) < 2 or _blank(rec[0]):
            continue
        label = _norm(rec[0])
        if label not in SETTING_NAMES:
            raise PhaseError(f"Settings sheet, row {row}: {str(rec[0]).strip()!r} is not a "
                             "setting. The settings are Base Year, Units, Index and "
                             "Inflation Rate.")
        key, value = SETTING_NAMES[label], rec[1]
        if key is None or _blank(value):
            continue
        stated.add(key)
        if key in ("units", "index"):
            out[key] = str(value).strip()
        elif key == "base_year":
            text = str(value).strip().upper().replace("BY", "").replace("FY", "")
            v = _number(text, "Settings", row, "Base Year")
            out[key] = int(v)
        else:
            text = str(value).strip()
            v = _number(text[:-1] if text.endswith("%") else value, "Settings", row,
                        "Inflation Rate")
            out[key] = v / 100.0 if text.endswith("%") or abs(v) >= 1 else v
    return out, frozenset(stated)


def read_workbook(path, index_path=None, index_name: Optional[str] = None,
                  base_year: Optional[int] = None) -> PhaseInput:
    """Read a phasing workbook; ``index_path`` is an index table (CSV or
    workbook) used in place of the Index sheet."""
    path = Path(path)
    raw = open_workbook(path, PhaseError) if path.suffix.lower() != ".csv" \
        else {"Phasing": pd.read_csv(path)}
    sheets = {_norm(k): v for k, v in raw.items()}
    titles = {_norm(k): str(k) for k in raw}
    for alias in ("phasing", "lines", "estimate", "spread"):
        if alias in sheets:
            if alias != "phasing":
                sheets["phasing"] = sheets.pop(alias)
                titles["phasing"] = titles.pop(alias)
            break
    else:
        first = next(iter(raw), None)
        if first is None:
            raise PhaseError(f"{path.name} has no sheets.")
        sheets["phasing"] = raw[first]
        titles["phasing"] = titles.pop(_norm(first))
    notes: List[str] = []
    unread = [t for k, t in titles.items()
              if k not in ("phasing", "settings", "index", "instructions")]
    if unread:
        notes.append(f"The sheet{'s' if len(unread) > 1 else ''} "
                     f"{', '.join(map(repr, unread))} {'were' if len(unread) > 1 else 'was'} "
                     "not read: the sheets read are Phasing, Settings and Index.")
    settings, stated = _read_settings(sheets.get("settings"))
    if base_year is not None:
        settings["base_year"], stated = int(base_year), stated | {"base_year"}
    if index_name:
        settings["index"] = index_name
    if settings["base_year"] is None:
        raise PhaseError("Say which year's dollars the amounts are in: Base Year on the "
                         "Settings sheet (2026, say), or --base-year.")
    by = settings["base_year"]

    indices: Dict[str, Dict[int, float]] = {}
    source = ""
    if index_path is not None:
        ip = Path(index_path)
        if not ip.is_file():
            raise PhaseError(f"There's no index table at {ip}.")
        table = pd.read_excel(ip) if ip.suffix.lower() in (".xlsx", ".xlsm") \
            else pd.read_csv(ip)
        indices, source = _index_table(table, ip.name), f"{ip.name}"
    elif "index" in sheets and not sheets["index"].dropna(how="all").empty:
        indices, source = _index_table(sheets["index"], "Index sheet"), \
            f"the Index sheet of {path.name}"
    elif settings["inflation_rate"] is not None:
        rate = settings["inflation_rate"]
        if not -0.1 <= rate <= 0.5:
            raise PhaseError(f"An inflation rate of {rate:g} is outside -10% to 50% a year; "
                             "it's a fraction, 0.021 for 2.1%.")
        name = f"constant {rate:.2%} a year"
        indices = {name: {y: (1.0 + rate) ** (y - by) for y in range(by - 60, by + 81)}}
        source = f"a constant {rate:.2%} a year, from the Settings sheet"
        settings["index"] = settings["index"] or name

    frame = sheets["phasing"].dropna(how="all")
    if frame.empty:
        raise PhaseError("The Phasing sheet has no rows: one row per estimate line, with "
                         "its amount, start year, years and profile.")
    cols = _columns(frame, "Phasing", ["line", "amount", "start"])
    lines = []
    for i, rec in zip(excel_rows(frame), frame.to_dict("records")):
        get = lambda k: rec.get(cols[k]) if k in cols else None  # noqa: E731
        name = str(get("line")).strip() if not _blank(get("line")) else f"row {i}"
        where = f"Phasing sheet, row {i} ({name})"
        amount = _number(get("amount"), "Phasing", i, "amount")
        start = _number(_fy(get("start")), "Phasing", i, "start year")
        if amount is None or start is None:
            raise PhaseError(f"{where}: give the amount and the start year.")
        if start != int(start):
            raise PhaseError(f"{where}: the start year {start:g} isn't a whole year.")
        years = _number(get("years"), "Phasing", i, "years")
        end = _number(_fy(get("end")), "Phasing", i, "end year")
        if end is not None and end < start:
            raise PhaseError(f"{where}: it ends in FY{int(end)}, before it starts in "
                             f"FY{int(start)}.")
        text = "" if _blank(get("profile")) else str(get("profile")).strip()
        if text and _percentages(text) and len(_percentages(text)) == 1 and years not in (None, 1):
            raise PhaseError(f"{where}: one percentage for {int(years)} years; give one per "
                             "year, or a named profile.")
        shares = _percentages(text) if text else None
        if years is None and end is not None:
            years = end - start + 1
        if years is None and shares:
            years = len(shares)
        if years is None:
            raise PhaseError(f"{where}: give the number of years (or the end year).")
        if years != int(years) or years < 1:
            raise PhaseError(f"{where}: {years:g} isn't a whole number of years.")
        if end is not None and int(start) + int(years) - 1 != int(end):
            raise PhaseError(f"{where}: {int(years)} years from FY{int(start)} ends in "
                             f"FY{int(start) + int(years) - 1}, not FY{int(end)}.")
        years = int(years)
        if shares:
            if len(shares) != years:
                raise PhaseError(f"{where}: {len(shares)} percentages for {years} years; "
                                 "give one per year.")
            if any(s < 0 for s in shares):
                raise PhaseError(f"{where}: the percentages can't be negative.")
            if sum(shares) <= 0:
                raise PhaseError(f"{where}: the percentages are all zero, so nothing is "
                                 "spent in any year.")
            total = sum(shares)
            if abs(total - 100.0) > 0.5 and abs(total - 1.0) > 0.005:
                notes.append(f"{name}'s percentages add up to {total:g}, not 100; they were "
                             "scaled to the line's amount.")
            weights, profile = np.asarray(shares) / total, "percentages " + text
        else:
            peak = _number(get("peak"), "Phasing", i, "peak")
            if peak is not None and peak > 1:
                peak /= 100.0
            try:
                weights = profile_weights(text or "uniform", years, peak)
            except PhaseError as e:
                raise PhaseError(f"{where}: {e}") from None
            profile = _norm(text or "uniform")
            if profile == "rayleigh":
                profile += f" (peak {RAYLEIGH_PEAK if peak is None else peak:.2f})"
        approp = "" if _blank(get("appropriation")) else str(get("appropriation")).strip()
        idx = None if _blank(get("index")) else str(get("index")).strip()
        lines.append(PhaseLine(name, approp or "(not given)", float(amount), int(start), years,
                               profile, weights, idx, i))

    default = settings["index"]
    if indices:
        if default is None:
            if len(indices) > 1 and any(ln.index is None for ln in lines):
                raise PhaseError(f"The index table holds {len(indices)} indices "
                                 f"({', '.join(sorted(indices))}); name one under Index on "
                                 "the Settings sheet, or on each line.")
            default = next(iter(indices))
        for ln in lines:
            name = ln.index or default
            if name not in indices:
                raise PhaseError(f"Phasing sheet, row {ln.row} ({ln.line}): there's no index "
                                 f"{name!r}; the table has {', '.join(sorted(indices))}.")
            ln.index = name
        if default not in indices:
            raise PhaseError(f"There's no index {default!r}; the table has "
                             f"{', '.join(sorted(indices))}.")
    else:
        notes.append("No inflation index was given, so the phasing is in base-year dollars "
                     "only. Add an Index sheet (or --index) for then-year dollars.")
    return PhaseInput(name=path.stem, lines=lines, base_year=by, units=settings["units"],
                      indices=indices, index_source=source, default_index=default,
                      notes=notes, stated=stated)


# ---------------------------------------------------------------- phasing
@dataclass
class PhaseResult:
    inputs: PhaseInput
    long: pd.DataFrame
    units: str = ""

    @property
    def has_then_year(self) -> bool:
        return bool(self.inputs.indices)

    def wide(self, basis: str = "ty") -> pd.DataFrame:
        """One row per line, one column per fiscal year, and the total."""
        col = "then_year" if basis == "ty" and self.has_then_year else "base_year"
        table = self.long.pivot_table(index=["line", "appropriation"], columns="fiscal_year",
                                      values=col, aggfunc="sum", fill_value=0.0, sort=False)
        table.columns = [f"FY{c}" for c in table.columns]
        table["Total"] = table.sum(axis=1)
        table = table.reset_index()
        total = table.select_dtypes("number").sum()
        return pd.concat([table, pd.DataFrame([{"line": "Total", "appropriation": "",
                                                **total.to_dict()}])], ignore_index=True)

    def by_appropriation(self) -> pd.DataFrame:
        """Spend per appropriation and fiscal year, base-year and then-year."""
        cols = ["base_year"] + (["then_year"] if self.has_then_year else [])
        return (self.long.groupby(["appropriation", "fiscal_year"], sort=False)[cols].sum()
                .reset_index())

    def totals(self) -> Dict[str, float]:
        by = float(self.long["base_year"].sum())
        ty = float(self.long["then_year"].sum()) if self.has_then_year else float("nan")
        return {"base_year": by, "then_year": ty}

    def peak(self) -> Tuple[int, float]:
        col = "then_year" if self.has_then_year else "base_year"
        per = self.long.groupby("fiscal_year")[col].sum()
        return int(per.idxmax()), float(per.max())

    @property
    def assumptions(self) -> Dict:
        i = self.inputs
        return {"data": i.name, "base year": f"BY{i.base_year}", "lines": len(i.lines),
                "fiscal years": f"FY{self.long['fiscal_year'].min()} to "
                                f"FY{self.long['fiscal_year'].max()}",
                "units": self.units or "as entered",
                "inflation index": ", ".join(sorted({ln.index for ln in i.lines if ln.index}))
                                   or "none (base-year dollars only)",
                "index source": i.index_source or "none",
                "profiles": ", ".join(sorted({ln.profile.split(" (")[0] for ln in i.lines}))}


def phase(inputs: PhaseInput, units: Optional[str] = None) -> PhaseResult:
    """Spread every line and inflate it."""
    rows = []
    for ln in inputs.lines:
        for fy, amount in spread(ln.amount, ln.start, ln.weights).items():
            factor = 1.0
            if ln.index:
                idx = inputs.indices[ln.index]
                for y in (fy, inputs.base_year):
                    if y not in idx:
                        raise PhaseError(f"The index {ln.index!r} covers FY{min(idx)} to "
                                         f"FY{max(idx)} and {ln.line} needs FY{y}. Extend "
                                         "the table rather than guessing past its end.")
                factor = idx[fy] / idx[inputs.base_year]
            rows.append({"line": ln.line, "appropriation": ln.appropriation,
                         "fiscal_year": fy, "base_year": amount, "factor": factor,
                         "then_year": amount * factor if ln.index else np.nan,
                         "index": ln.index or "", "profile": ln.profile})
    long = pd.DataFrame(rows)
    return PhaseResult(inputs=inputs, long=long,
                       units=inputs.units if units is None else units)


# ------------------------------------------------------------- the template
INSTRUCTIONS = [
    ("What this is", "An estimate's lines and how each is spent over time, for a budget "
     "in then-year dollars: ce-core phase --data <this file>. Every number in the example "
     "is invented, and so is its 2% index."),
    ("Phasing sheet", "One row per estimate line. Amount is its total in base-year dollars "
     "(the Base Year on the Settings sheet). Start is the first fiscal year it's spent in, "
     "Years how many (or give End instead). Appropriation groups the lines in the "
     "results (RDT&E, Procurement, O&M)."),
    ("Profile", "How the money goes out: uniform (the default); front or back (weighted "
     "toward the start or the end); bell (a symmetric hump); rayleigh, the usual shape of "
     "development spending, with Peak setting where it peaks as a fraction of the duration "
     "(0.38 if blank); or one percentage per year, such as 10;30;40;20."),
    ("Index column", "Optional. Which inflation index on the Index sheet the line uses; "
     "RDT&E, procurement and O&M money each have their own. Blank lines use the Index on "
     "the Settings sheet."),
    ("Index sheet", "The inflation indices: index_name, fiscal_year, index_value, from your "
     "agency's published tables. The example's index is an INVENTED 2% a year: replace "
     "it. Or give Inflation Rate on the Settings sheet for a constant rate, or leave both "
     "out for base-year dollars only."),
]

EXAMPLE_LINES = pd.DataFrame(
    [("Prototype development", "RDT&E", 120.0, 2027, 5, "rayleigh", None, ""),
     ("Test and evaluation", "RDT&E", 30.0, 2029, 3, "bell", None, ""),
     ("Production lots", "Procurement", 260.0, 2031, 5, "10;20;25;25;20", None, ""),
     ("Initial spares", "Procurement", 18.0, 2032, 3, "front", None, ""),
     ("Sustainment, first years", "O&M", 45.0, 2032, 5, "uniform", None, "")],
    columns=["Line", "Appropriation", "Amount", "Start", "Years", "Profile", "Peak",
             "Index"])
EXAMPLE_SETTINGS = pd.DataFrame(
    [("Base Year", 2026), ("Units", "millions"),
     ("Index", "ILLUSTRATIVE 2% - replace with the published index")],
    columns=["Setting", "Value"])


def write_workbook(out, lines: pd.DataFrame = EXAMPLE_LINES,
                   settings: pd.DataFrame = EXAMPLE_SETTINGS,
                   index: Optional[pd.DataFrame] = None) -> Path:
    """Write a workbook laid out the way :func:`read_workbook` reads it."""
    from openpyxl.styles import Alignment, Font

    from cost_core.inflate import illustrative_index

    out = Path(out)
    index = illustrative_index(2020, 2050) if index is None else index
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        lines.to_excel(xl, sheet_name="Phasing", index=False)
        settings.to_excel(xl, sheet_name="Settings", index=False)
        index.to_excel(xl, sheet_name="Index", index=False)
        pd.DataFrame(INSTRUCTIONS, columns=["Sheet or topic", "What to put there"]) \
            .to_excel(xl, sheet_name="Instructions", index=False)
        widths = {"Phasing": (28, 15, 10, 8, 8, 18, 8, 30), "Settings": (14, 52),
                  "Index": (52, 12, 12), "Instructions": (22, 100)}
        for name, cols in widths.items():
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for j, w in enumerate(cols):
                ws.column_dimensions[chr(ord("A") + j)].width = w
        for row in xl.sheets["Instructions"].iter_rows(min_row=2):
            row[0].alignment = Alignment(vertical="top")
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
    return out
