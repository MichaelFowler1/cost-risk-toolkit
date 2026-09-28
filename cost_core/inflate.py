# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
inflate.py - Base-year and then-year dollars, from an index you supply.

Every estimate is stated in some year's dollars, and the budget is asked for
in the dollars of the year the money is spent (then-year). Converting between
the two is arithmetic, but the index behind it has to come from the right
place: the agency's published tables (the NASA New Start Inflation Index,
the OSD inflation guidance, a program office's own). cost-core never carries
those numbers itself. You give it the table; it does the arithmetic, shows it,
and keeps the index it used beside every converted amount.

The table is a CSV (or a sheet) with ``index_name``, ``fiscal_year`` and
``index_value`` columns: raw indices, so any year converts to any other by a
ratio. A weighted index (one per appropriation, already spread over its
outlay years) goes in the same shape; it is still a number per fiscal year.

    ce-core inflate --index my_index.csv --data phased.csv --from by2024 --to ty
    ce-core inflate --index my_index.csv --amount 100 --year 2027 --from by2024 --to ty
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from cost_core.ingest.inflation import INDEX_COLUMNS, InflationError, InflationTable


class InflateError(ValueError):
    """A conversion that can't be done as asked; the message says why."""


@dataclass(frozen=True)
class Basis:
    """``ty`` (each amount in its own year's dollars) or ``by2026``."""

    kind: str
    year: Optional[int] = None

    def __str__(self) -> str:
        return "then-year" if self.kind == "ty" else f"BY{self.year}"


def parse_basis(text: str) -> Basis:
    t = str(text).strip().lower().replace(" ", "").replace("fy", "")
    if t in ("ty", "then-year", "thenyear", "then"):
        return Basis("ty")
    m = re.fullmatch(r"(?:by|base-?year)?(\d{4})", t)
    if m:
        return Basis("by", int(m.group(1)))
    raise InflateError(f"{text!r} isn't a dollar basis: use ty for then-year dollars or "
                       "by2026 (any year) for base-year dollars.")


def read_index(path, name: Optional[str] = None) -> Tuple[Dict[int, float], str]:
    """One index from a table, and its name."""
    path = Path(path)
    if not path.is_file():
        raise InflateError(f"There's no index table at {path}. It's a CSV with index_name, "
                           "fiscal_year and index_value columns, from your agency's "
                           "published inflation tables (ce-core template inflate writes the "
                           "layout).")
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        frame = pd.read_excel(path)
        tmp = path.with_suffix(".csv.tmp")
        frame.to_csv(tmp, index=False)
        try:
            table = InflationTable.load(tmp, source=f"loaded from {path}")
        finally:
            tmp.unlink()
    else:
        try:
            table = InflationTable.load(path)
        except InflationError as e:
            raise InflateError(str(e)) from None
    names = sorted(table.indices)
    if name is None:
        if len(names) > 1:
            raise InflateError(f"{path.name} holds {len(names)} indices "
                               f"({', '.join(names)}); choose one with --index-name.")
        name = names[0]
    if name not in table.indices:
        raise InflateError(f"{path.name} has no index {name!r}; it has {', '.join(names)}.")
    return table.indices[name], name


def fiscal_year(values, start_month: int = 10) -> np.ndarray:
    """Fiscal years from years or dates. With the U.S. government's year
    (October start), 15 October 2025 is FY2026."""
    s = pd.Series(values)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and ((num >= 1900) & (num <= 2300)).all():
        return num.astype(int).to_numpy()
    s = s.astype(str).str.strip().str.replace(r"^FY\s*", "", regex=True, case=False)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and ((num >= 1900) & (num <= 2300)).all():
        return num.astype(int).to_numpy()
    dates = pd.to_datetime(s, errors="coerce")
    if dates.isna().any():
        bad = list(s[dates.isna()][:3])
        raise InflateError(f"Can't read {bad} as fiscal years or dates.")
    shift = (dates.dt.month >= start_month).astype(int) if start_month != 1 else 0
    return (dates.dt.year + shift).to_numpy().astype(int)


def factors(index: Dict[int, float], years, src: Basis, dst: Basis) -> np.ndarray:
    """What to multiply each amount by. ``years`` is the fiscal year each
    amount is spent in (for then-year dollars, the year it's stated in)."""
    def value(y):
        y = int(y)
        if y not in index:
            raise InflateError(f"The index covers FY{min(index)} to FY{max(index)} and was "
                               f"asked for FY{y}. Extend the table rather than guessing past "
                               "its end.")
        return index[y]

    years = np.asarray(years, dtype=int)
    frm = np.array([value(y) for y in years]) if src.kind == "ty" else \
        np.full(len(years), value(src.year))
    to = np.array([value(y) for y in years]) if dst.kind == "ty" else \
        np.full(len(years), value(dst.year))
    return to / frm


AMOUNT_ALIASES = ("amount", "cost", "value", "dollars", "total", "estimate", "budget")
YEAR_ALIASES = ("fiscal_year", "fy", "year", "date", "period", "spend_year")


def _pick(frame: pd.DataFrame, given: Optional[str], aliases, what: str) -> str:
    if given:
        if given not in frame.columns:
            raise InflateError(f"No column {given!r}; the columns are "
                               f"{', '.join(map(str, frame.columns))}.")
        return given
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in frame.columns}
    for alias in aliases:
        if alias in norm:
            return norm[alias]
    raise InflateError(f"Can't tell which column holds the {what}; name it with "
                       f"--{'amount' if what == 'amounts' else 'year'}-col. The columns are "
                       f"{', '.join(map(str, frame.columns))}.")


def convert(frame: pd.DataFrame, index: Dict[int, float], src: Basis, dst: Basis,
            amount_col: Optional[str] = None, year_col: Optional[str] = None,
            start_month: int = 10, index_name: str = "index") -> pd.DataFrame:
    """The table with the converted amount, the factor and the index values
    beside every row; nothing in it is overwritten."""
    a = _pick(frame, amount_col, AMOUNT_ALIASES, "amounts")
    y = _pick(frame, year_col, YEAR_ALIASES, "fiscal years")
    amounts = pd.to_numeric(frame[a], errors="coerce")
    if amounts.isna().any():
        rows = ", ".join(str(int(i) + 2) for i in frame.index[amounts.isna()][:5])
        raise InflateError(f"The {a} column has blanks or text on row(s) {rows}.")
    fy = fiscal_year(frame[y], start_month)
    f = factors(index, fy, src, dst)
    out = frame.copy()
    out["fiscal_year"] = fy
    out[f"factor_{str(src)}_to_{str(dst)}".replace("-", "_")] = f
    out[f"{a}_{str(dst)}".replace("-", "_")] = amounts.to_numpy() * f
    out["index_used"] = index_name
    return out


# ------------------------------------------------------------ the template
def illustrative_index(first: int = 2020, last: int = 2045, base: int = 2026,
                       rate: float = 0.02) -> pd.DataFrame:
    """An INVENTED constant-rate index in the table layout, for the template
    and the demo only. Published indices come from the agency's own tables."""
    return pd.DataFrame({"index_name": "ILLUSTRATIVE 2% - replace with the published index",
                         "fiscal_year": range(first, last + 1),
                         "index_value": [round((1 + rate) ** (y - base), 6)
                                         for y in range(first, last + 1)]},
                        columns=list(INDEX_COLUMNS))


def example_phasing() -> pd.DataFrame:
    """An invented BY2026 estimate phased by fiscal year."""
    return pd.DataFrame({"element": ["Development"] * 3 + ["Production"] * 4,
                         "fiscal_year": [2027, 2028, 2029, 2029, 2030, 2031, 2032],
                         "amount": [40.0, 55.0, 25.0, 30.0, 60.0, 60.0, 45.0]})
