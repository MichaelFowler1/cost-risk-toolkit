# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
mathx.py - The lot geometry and the input parsing behind the lot cost model.

Ported unchanged from the original spreadsheet-replacement script, because the
numbers it produces are the reference this package is held to.

**Lot midpoint.** A lot's cost is priced at its algebraic midpoint -- the unit
whose cost equals the lot average. Under a power curve that midpoint depends on
the slope, which is the parameter being fitted, so the two have to be solved
together. :func:`lmp_func` is one evaluation of the midpoint at a given slope;
the fixed-point loop that chases the two together lives in
:mod:`cost_core.lotmodel.models`.

**Unit tracking.** :func:`track_units` turns a run of lot quantities into the
first and last unit number of each lot, carrying any units built before the
series starts.

**Input parsing.** :func:`find_col` and :func:`to_num` read the analyst's
spreadsheet: column names that differ only in case or spacing, and money
written with dollar signs and thousands separators.

The estimating maths itself is no longer here. The regression that used to sit
in this module has moved to :mod:`cost_core.lotmodel.models`, which declares
the three lot models for :mod:`cost_core.fitting` so that every fit in the
package goes through one estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_core.lotmodel.config import SETTINGS


def find_col(df_columns: list, candidates: list) -> str | None:
    """Case-insensitive search for the first matching column name."""
    col_map = {c.strip().lower(): c for c in df_columns}
    for cand in candidates:
        if cand.strip().lower() in col_map:
            return col_map[cand.strip().lower()]
    return None


def to_num(val):
    """Clean string currency/commas and convert to numeric float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned) if cleaned != "" else np.nan
        except ValueError:
            return np.nan
    return np.nan


def lmp_func(
    s: float, e: float, q: float, b: float | None
) -> float | None:
    """Calculate Lot Midpoint (LMP) given Start, End, Qty, and b-slope."""
    if b is None or pd.isna(b):
        return np.nan
    if q <= 1:
        return float(s)
    if abs(b) < 1e-12:
        return (s + e) / 2.0
    if abs(b + 1.0) < 1e-6:
        lo = max(s - 0.5, 1e-6)
        return q / (np.log(e + 0.5) - np.log(lo))

    p = b + 1.0
    lo = max(s - 0.5, 1e-6)
    v = ((e + 0.5) ** p - lo**p) / (p * q)
    if v <= 0:
        return np.nan
    return v ** (1.0 / b)


def track_units(quantities: np.ndarray, prior: int):
    cums = np.cumsum(quantities) + prior
    starts = cums - quantities + 1
    return [{"S": s, "E": e} for s, e in zip(starts, cums)]


