# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
checks.py - The monthly questions: is the data sound, and which accounts need
a variance report?

Before a CPI means anything the data under it has to hold together, and the
first hour of every month's EVM review goes on finding the rows that don't:
earned value that went backwards, cost charged to work that earned nothing,
accounts that have earned their whole budget and are still being charged.
:func:`data_checks` lists them per control account, with the period and the
numbers, as questions to ask rather than verdicts; a negative period value is
sometimes a legitimate correction.

:func:`variance_breaches` lists the accounts whose cumulative cost or schedule
variance breaks the thresholds (in percent, dollars, or both) and so, under
most contracts' reporting rules, owe a variance analysis report. The
thresholds are the contract's to set; the defaults (10%) are only a start.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

CHECK_COLUMNS = ["account", "period", "check", "detail"]


def _accounts(data) -> Dict[str, object]:
    return dict(data.accounts) if data.accounts else {data.name: data}


def data_checks(data) -> pd.DataFrame:
    """Rows that need a question asked, across every account up to status."""
    rows = []
    for name, acct in _accounts(data).items():
        m = acct.metrics()
        if m.empty:
            continue
        periods = m["period"].astype(str).tolist()
        ev, ac = m["bcwp"].to_numpy(float), m["acwp"].to_numpy(float)
        d_ev = np.diff(np.concatenate([[0.0], ev]))
        d_ac = np.diff(np.concatenate([[0.0], ac]))
        bac = float(acct.bac)
        tol = 1e-9 * max(bac, 1.0)
        for k, p in enumerate(periods):
            if d_ev[k] < -tol:
                rows.append((name, p, "Earned value went down",
                             f"BCWP fell by {-d_ev[k]:,.2f} this period (cumulative "
                             f"{ev[k - 1]:,.2f} to {ev[k]:,.2f}). A correction? Ask for the "
                             "reason; otherwise work was un-earned."))
            if d_ac[k] < -tol:
                rows.append((name, p, "Actual cost went down",
                             f"ACWP fell by {-d_ac[k]:,.2f} this period. A credit or a "
                             "transfer? Ask where the cost went."))
        last = len(periods) - 1
        p = periods[last]
        if ev[last] > bac + tol:
            rows.append((name, p, "Earned value beyond the account's budget",
                         f"Cumulative BCWP {ev[last]:,.2f} against a budget of {bac:,.2f}: more "
                         "work claimed done than the account holds."))
        complete = ev[last] >= bac - tol
        if complete and d_ac[last] > tol:
            rows.append((name, p, "Complete but still charging",
                         f"The account has earned its whole budget ({bac:,.2f}) yet took "
                         f"{d_ac[last]:,.2f} of cost this period. Should it be closed?"))
        elif not complete and d_ac[last] > tol and abs(d_ev[last]) <= tol:
            rows.append((name, p, "Cost with no earned value",
                         f"{d_ac[last]:,.2f} charged this period and nothing earned. Level of "
                         "effort, or work that isn't being credited?"))
        if d_ev[last] > tol and abs(d_ac[last]) <= tol:
            rows.append((name, p, "Earned value with no cost",
                         f"{d_ev[last]:,.2f} earned this period with no cost recorded. Is the "
                         "cost late, or charged to another account?"))
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def thresholds(data) -> dict:
    """The variance thresholds set on this data (by ``ce-core evm``), or the
    defaults."""
    return dict(getattr(data, "variance_thresholds", None) or {})


def variance_breaches(data, cv_pct: float = 10.0, sv_pct: float = 10.0,
                      cv_dollars: Optional[float] = None,
                      sv_dollars: Optional[float] = None) -> pd.DataFrame:
    """Accounts whose cumulative CV or SV breaks a threshold, worst first.

    A variance breaks the percent threshold when it's more than ``cv_pct``
    (``sv_pct``) percent of the BCWP (BCWS), and the dollar threshold when
    it's more than that many dollars either way. With both given, both must
    be broken, as contracts usually write it; with one, that one decides.
    """
    rows = []
    for name, acct in _accounts(data).items():
        m = acct.metrics()
        if m.empty:
            continue
        r = m.iloc[-1]
        cv_p = 100.0 * r.cv / r.bcwp if r.bcwp else np.nan
        sv_p = 100.0 * r.sv / r.bcws if r.bcws else np.nan
        reasons = []
        for label, value, pct, pct_lim, dol_lim in (("cost", r.cv, cv_p, cv_pct, cv_dollars),
                                                    ("schedule", r.sv, sv_p, sv_pct, sv_dollars)):
            over_pct = pct_lim is not None and np.isfinite(pct) and abs(pct) > pct_lim
            over_dol = dol_lim is not None and abs(value) > dol_lim
            if pct_lim is not None and dol_lim is not None:
                broken = over_pct and over_dol
            else:
                broken = over_pct or over_dol
            if broken:
                reasons.append(f"{label} variance {value:+,.2f} ({pct:+.1f}%)")
        if reasons:
            rows.append({"account": name, "cv": r.cv, "cv_pct": cv_p, "sv": r.sv,
                         "sv_pct": sv_p, "cpi": r.cpi, "spi": r.spi,
                         "why": "; ".join(reasons)})
    frame = pd.DataFrame(rows, columns=["account", "cv", "cv_pct", "sv", "sv_pct", "cpi",
                                        "spi", "why"])
    if len(frame):
        frame = frame.assign(_worst=frame[["cv_pct", "sv_pct"]].abs().max(axis=1)) \
            .sort_values("_worst", ascending=False).drop(columns="_worst")
    return frame.reset_index(drop=True)
