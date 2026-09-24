# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
growth.py - Uncertainty from what real programs actually did.

The spread put on a cost line in an AoA is usually a judgement: "could come in
10% under or 50% over". The SAR panel from :mod:`cost_core.public` offers
something better for the lines it covers: how far unit cost has actually moved
from the original baseline on every major program DoD reports on. This turns
that history into a factor distribution a :class:`~cost_core.aoa.CostLine` can
carry, so the uncertainty on a new estimate rests on the record of past ones.

**One number per program.** A program appears in the panel once per report,
and the same program's growth in consecutive years is the same fact seen
twice, not fifteen independent observations. So each program contributes its
latest report only, which is also its most complete record of growth.

**Growth against the original baseline, in constant dollars.** The SAR states
both sides of that comparison in the same base year, so inflation is already
out of it. What is left includes quantity effects: a cut in quantity raises
unit cost with nothing going wrong in the estimate. Filter on
``quantity_change_pct`` when that matters, as it does for a procurement line
whose quantity is fixed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from cost_core.aoa.lcc import AoAError


def historical_growth(
    unit_cost: pd.DataFrame,
    measure: str = "APUC",
    max_quantity_change_pct: Optional[float] = None,
    min_programs: int = 20,
) -> pd.DataFrame:
    """Each program's latest unit cost growth against its original baseline.

    Args:
        unit_cost: ``SarPanel.unit_cost`` from :func:`cost_core.public.build_sar_panel`.
        measure: ``"APUC"`` (procurement) or ``"PAUC"`` (whole acquisition).
        max_quantity_change_pct: Keep only programs whose quantity moved by at
            most this much either way, to separate estimating growth from
            quantity changes. None keeps all.
        min_programs: Refuse to return fewer programs than this; a
            distribution fitted to a handful of programs is a guess with a
            histogram attached.

    Returns:
        One row per program and subprogram: ``program, subprogram, cycle,
        growth_pct, quantity_change_pct``. Only rows whose arithmetic checks
        held are used.
    """
    need = {"comparison", "measure", "cycle_year", "program", "unit_cost_growth_pct",
            "quantity_change_pct", "checks_ok"}
    missing = need - set(unit_cost.columns)
    if missing:
        raise AoAError(f"unit_cost is missing column(s) {sorted(missing)}; pass "
                       f"SarPanel.unit_cost.")
    uc = unit_cost[(unit_cost["comparison"] == "original") & (unit_cost["measure"] == measure)
                   & unit_cost["checks_ok"].astype(bool)
                   & unit_cost["unit_cost_growth_pct"].notna()].copy()
    if max_quantity_change_pct is not None:
        uc = uc[uc["quantity_change_pct"].abs() <= max_quantity_change_pct]
    uc["subprogram"] = uc.get("subprogram", pd.Series(index=uc.index, dtype=object)).fillna("")
    order = uc.sort_values(["cycle_year", "cycle"])
    latest = order.groupby(["program", "subprogram"], sort=True).tail(1)
    out = latest[["program", "subprogram", "cycle", "unit_cost_growth_pct",
                  "quantity_change_pct"]].rename(columns={"unit_cost_growth_pct": "growth_pct"})
    if len(out) < min_programs:
        raise AoAError(f"Only {len(out)} programs meet the filter; need {min_programs} "
                       f"(lower min_programs to override).")
    return out.reset_index(drop=True)


def growth_factor(growth: pd.DataFrame) -> Dict[str, Any]:
    """A factor distribution for a CostLine from :func:`historical_growth`.

    Returns an ``empirical`` spec whose draws are ``1 + growth_pct / 100``, one
    per program, so a line carrying it is resampled from the actual history
    rather than from a shape fitted to it.
    """
    factors = 1.0 + growth["growth_pct"].to_numpy(dtype=float) / 100.0
    if (factors <= 0).any():
        raise AoAError("A growth of -100% or less cannot be a cost factor.")
    return {"type": "empirical", "draws": factors}


def describe_growth(growth: pd.DataFrame) -> pd.Series:
    """Programs, mean, median and P20/P80 of the growth, in percent."""
    g = growth["growth_pct"].to_numpy(dtype=float)
    return pd.Series({"programs": len(g), "mean": g.mean(), "median": np.median(g),
                      "p20": np.percentile(g, 20), "p80": np.percentile(g, 80),
                      "share_over_baseline": float((g > 0).mean())})
