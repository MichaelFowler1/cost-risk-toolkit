# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
economic.py - Each alternative against the status quo: does it pay for itself?

An AoA's life-cycle costs say which alternative costs least. An economic
analysis asks the question a decision maker funding a change asks: if we
spend more now, do we get it back, how soon, and at what return? So each
alternative is set against the status quo year by year, in base-year dollars
discounted at the AoA's real rate, the way OMB Circular A-94 and DoD's
economic analysis guidance (DoDI 7041.03) set it out:

``investment``      the extra one-time cost over the status quo: RDT&E,
                    procurement and MILCON, in present value.
``savings``         the operating cost it avoids: O&S and disposal, the
                    status quo's less the alternative's, in present value.
``net savings``     savings less investment: the status quo's life-cycle
                    present value less the alternative's.
``SIR``             savings over investment. Above 1, the savings repay the
                    investment at the discount rate.
``payback``         the first fiscal year by which the discounted savings have
                    repaid the discounted investment.
``break-even``      the same in constant dollars, undiscounted.
``IRR``             the real rate at which the stream of yearly differences
                    has a present value of zero: the return on the extra
                    investment, to set beside the discount rate. Reported
                    only when the stream changes sign once, which is when it
                    is unique.
``uniform annual``  each alternative's present value spread as a level
                    amount over its own years, which is how alternatives with
                    different service lives are compared fairly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize

#: The phases counted as investment; everything else is operating cost.
INVESTMENT_PHASES = ("RDT&E", "Procurement", "MILCON")


@dataclass
class EconomicAnalysis:
    """Every alternative measured against the status quo."""

    status_quo: str
    summary: pd.DataFrame
    by_year: pd.DataFrame
    discount_rate: float
    pv_year: int
    notes: List[str] = field(default_factory=list)


def _yearly(alt) -> pd.DataFrame:
    """An alternative's base-year cost by fiscal year, split into investment
    and operating."""
    rows: Dict[int, List[float]] = {}
    for line in alt.lines:
        kind = 0 if line.phase in INVESTMENT_PHASES else 1
        for y, v in line.by_year.items():
            rows.setdefault(int(y), [0.0, 0.0])[kind] += float(v)
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["investment", "operating"])
    return frame.sort_index()


def _first_year_repaid(years: np.ndarray, flows: np.ndarray) -> Optional[int]:
    """The first year by which the running total of ``flows`` (savings less
    investment) has come back to zero or above after going below it.

    None when it never does; the first year when it never went below zero
    (the alternative costs less from the start)."""
    running = np.cumsum(flows)
    if (running >= -1e-9).all():
        return int(years[0])
    last_negative = int(np.max(np.nonzero(running < -1e-9)[0]))
    if last_negative == len(running) - 1:
        return None
    return int(years[last_negative + 1])


def _irr(years: np.ndarray, flows: np.ndarray) -> Optional[float]:
    """The real IRR of yearly ``flows``, if there is exactly one."""
    nz = flows[np.abs(flows) > 1e-9]
    if len(nz) < 2 or (np.diff(np.sign(nz)) != 0).sum() != 1:
        return None
    t = years - years[0]

    def npv(r: float) -> float:
        return float(np.sum(flows / (1.0 + r) ** t))

    grid = np.r_[np.linspace(-0.95, 1.0, 400), np.linspace(1.0, 20.0, 200)[1:]]
    values = [npv(r) for r in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return float(optimize.brentq(npv, a, b, xtol=1e-12))
    return None


def economic_analysis(result, status_quo: str) -> EconomicAnalysis:
    """Measure every alternative in an :class:`~cost_core.aoa.AoAResult`
    against ``status_quo``, one of its alternatives."""
    from cost_core.aoa.lcc import AoAError

    alts = {a.name: a for a in result.alternatives}
    if status_quo not in alts:
        raise AoAError(f"The status quo {status_quo!r} isn't one of the alternatives "
                       f"({', '.join(alts)}).")
    a = result.assumptions
    rate, pv_year = float(a["discount_rate"]), int(a.get("pv_year") or a["base_year"])
    tables = {name: _yearly(alt) for name, alt in alts.items()}
    years = sorted(set().union(*(t.index for t in tables.values())))
    disc = pd.Series({y: 1.0 / (1.0 + rate) ** (y - pv_year) for y in years})
    sq = tables[status_quo].reindex(years, fill_value=0.0)
    notes: List[str] = []

    def span(name):
        idx = tables[name].index
        return int(idx.min()), int(idx.max())

    rows, by_year = [], []
    for name, table in tables.items():
        t = table.reindex(years, fill_value=0.0)
        pv_total = float(((t["investment"] + t["operating"]) * disc).sum())
        first, last = span(name)
        own = [y for y in years if first <= y <= last]
        uniform = pv_total / float(disc[own].sum())
        row = {"alternative": name, "status_quo": "yes" if name == status_quo else "",
               "first_year": first, "last_year": last, "pv_life_cycle": pv_total,
               "uniform_annual_cost": uniform}
        if name != status_quo:
            invest = float(((t["investment"] - sq["investment"]) * disc).sum())
            savings = float(((sq["operating"] - t["operating"]) * disc).sum())
            diff = (sq["investment"] + sq["operating"]) - (t["investment"] + t["operating"])
            yrs = np.asarray(years)
            row.update({
                "investment_pv": invest, "savings_pv": savings,
                "net_savings_pv": savings - invest,
                "sir": savings / invest if invest > 1e-9 else np.nan,
                "payback_year": _first_year_repaid(yrs, (diff * disc).to_numpy()),
                "break_even_year": _first_year_repaid(yrs, diff.to_numpy()),
                "irr": _irr(yrs, diff.to_numpy()),
            })
            draws = getattr(result, "draws", {})
            if name in draws and status_quo in draws:
                row["chance_cheaper"] = float(np.mean(draws[name] < draws[status_quo]))
            for y in years:
                by_year.append({"alternative": name, "fiscal_year": y,
                                "status_quo_cost": float(sq.loc[y].sum()),
                                "alternative_cost": float(t.loc[y].sum()),
                                "saving": float(diff[y]), "discount_factor": float(disc[y]),
                                "saving_pv": float(diff[y] * disc[y])})
            if span(name)[1] != span(status_quo)[1]:
                notes.append(f"{name!r} runs to FY{span(name)[1]} and the status quo to "
                             f"FY{span(status_quo)[1]}: years one of them has no cost in "
                             "count as zero, which flatters the shorter one. Compare the "
                             "uniform annual costs, or give them the same service life.")
        rows.append(row)
    summary = pd.DataFrame(rows)
    for col in ("payback_year", "break_even_year"):
        if col in summary:
            summary[col] = pd.array([None if v is None or pd.isna(v) else int(v)
                                     for v in summary[col]], dtype="Int64")
    by_year = pd.DataFrame(by_year)
    if len(by_year):
        by_year["cumulative_saving_pv"] = by_year.groupby("alternative")["saving_pv"].cumsum()
    if "chance_cheaper" in summary:
        notes.append(f"The chance each alternative costs less than the status quo compares "
                     f"their {a.get('basis', 'pv')} draws, simulated independently; costs "
                     "they share would move together, so the real chance is further from "
                     "50% than shown.")
    return EconomicAnalysis(status_quo=status_quo, summary=summary, by_year=by_year,
                            discount_rate=rate, pv_year=pv_year, notes=notes)
