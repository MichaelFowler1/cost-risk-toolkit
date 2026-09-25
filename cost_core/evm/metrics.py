# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
metrics.py - Earned value metrics, earned schedule and the independent EACs.

The data is the three numbers every EVM report is built on, by reporting
period: the budgeted cost of work scheduled (BCWS, planned value), the
budgeted cost of work performed (BCWP, earned value) and the actual cost of
work performed (ACWP). The baseline runs to the end of the program; earned
value and actuals stop at the status period. Everything else is arithmetic
on those, and the arithmetic is standard (the EIA-748 guidelines and the DoD
EVM system interpretation guide). Two parts of it are not as well known as
they should be.

**Earned schedule.** SPI (EV over PV) measures schedule in dollars, and
every program finishes with SPI equal to 1.0, since in the end all the work
is earned, however late. So a late program's SPI drifts back towards 1.0
over its last third and stops saying anything. Earned schedule (Lipke,
2003) measures it in time instead: ES is the time at which the baseline
planned to have earned what has been earned so far, and SPI(t) is ES over
the actual time. SPI(t) keeps saying the program is late until it finishes,
and PD / SPI(t) is an estimate of when it will.

**The warning signs.** An estimate at completion that asks the remaining
work to be done far more efficiently than the work so far is the most common
optimism in EVM. The to-complete performance index (TCPI) says how
efficient: the remaining budget over the remaining cost. The research on
DoD contracts (Christensen and Heise, 1993; Christensen and Payne, 1992) is
that the cumulative CPI rarely improves by more than 0.10 once a contract is
20% complete, so :meth:`EvmData.flags` flags a contractor EAC whose TCPI
exceeds the CPI by more than 0.10, one below every independent EAC, and a
late program whose SPI has recovered while its SPI(t) has not.

Time is in reporting periods: the end of period ``k`` (counting from 1) is
time ``k``, and the program starts at time 0. Money is in whatever units the
data uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


class EvmError(ValueError):
    """Raised on EVM data that cannot be analysed."""


#: Columns of :meth:`EvmData.metrics`.
METRIC_COLUMNS = (
    "period", "time", "bcws", "bcwp", "acwp", "cv", "sv", "cpi", "spi", "cpi_period",
    "spi_period", "pct_complete", "pct_spent", "tcpi_bac", "es", "sv_t", "spi_t", "tspi",
    "ieac_cpi", "ieac_cpi_spi", "ieac_cpi_spi_t", "ieac_cpi_3", "ieac_weighted",
    "ieac_linear", "ieac_t", "eac", "tcpi_eac",
)


def _div(a, b):
    """a / b, NaN where b is zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)


def earned_schedule(pv_cum: Sequence[float], ev: Union[float, np.ndarray]):
    """The earned schedule of ``ev`` against a cumulative baseline.

    ``pv_cum[k]`` is the cumulative planned value at the end of period
    ``k + 1``. ES is the time at which the baseline reaches ``ev``: the
    number of whole periods whose cumulative PV is at most ``ev``, plus the
    fraction of the next period's planned value that ``ev`` covers (Lipke's
    definition). Earned value at or above the whole baseline gives the
    baseline's length.

    >>> earned_schedule([10, 30, 60, 100], 45)
    2.5
    """
    pv = np.concatenate([[0.0], np.asarray(pv_cum, dtype=float)])
    ev = np.asarray(ev, dtype=float)
    c = np.searchsorted(pv, ev, side="right") - 1  # largest C with pv[C] <= ev
    c = np.clip(c, 0, len(pv) - 1)
    last = c >= len(pv) - 1
    nxt = np.minimum(c + 1, len(pv) - 1)
    step = pv[nxt] - pv[c]
    frac = np.where(last | (step <= 0), 0.0, (ev - pv[c]) / np.where(step > 0, step, 1.0))
    out = np.where(ev <= 0, 0.0, c + frac)
    return float(out) if out.ndim == 0 else out


@dataclass
class EvmData:
    """Cumulative BCWS, BCWP and ACWP for one program or control account.

    Attributes:
        periods: Label of every baseline period, e.g. month-end dates.
        bcws: Cumulative planned value at the end of each period, through
            the end of the baseline.
        bcwp, acwp: Cumulative earned value and actual cost at the end of
            each period up to the status period.
        bac: Budget at completion.
        eac: The contractor's estimate at completion in each status period
            (its latest revised estimate), where reported.
        name: Label for reports.
        accounts: The same, per control account or WBS element, when the
            data was given at that level.
    """

    periods: List
    bcws: np.ndarray
    bcwp: np.ndarray
    acwp: np.ndarray
    bac: float
    eac: Optional[np.ndarray] = None
    name: str = "program"
    accounts: Dict[str, "EvmData"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bcws = np.asarray(self.bcws, dtype=float)
        self.bcwp = np.asarray(self.bcwp, dtype=float)
        self.acwp = np.asarray(self.acwp, dtype=float)
        if len(self.bcwp) != len(self.acwp):
            raise EvmError("BCWP and ACWP must cover the same periods.")
        if not 1 <= len(self.bcwp) <= len(self.bcws):
            raise EvmError("Need at least one status period, and no more than the baseline has.")
        if len(self.periods) != len(self.bcws):
            raise EvmError("One period label per baseline period.")
        if self.bac <= 0:
            raise EvmError(f"BAC must be positive; got {self.bac}.")
        if np.any(np.diff(self.bcws) < -1e-9 * max(1.0, self.bac)):
            raise EvmError("Cumulative BCWS falls between periods; was periodic data "
                           "given as cumulative?")
        if self.eac is not None:
            self.eac = np.asarray(self.eac, dtype=float)

    # ------------------------------------------------------------- building
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        cumulative: bool = False,
        bac: Optional[float] = None,
        name: str = "program",
        period: str = "period",
        account: Optional[str] = "wbs",
    ) -> "EvmData":
        """Build from a table with one row per period (and per account).

        Columns: ``period``, ``bcws``, ``bcwp``, ``acwp``, and optionally
        ``eac`` (the contractor's estimate) and ``wbs`` (control account or
        WBS element). Rows after the status period carry the baseline only,
        with ``bcwp`` and ``acwp`` blank. Values are per period unless
        ``cumulative`` is set.

        Args:
            bac: Budget at completion; the baseline's total when omitted.
            account: The column naming the control account, if any. With it,
                the program is the sum of the accounts and each account is
                kept in :attr:`accounts`.
        """
        need = {period, "bcws", "bcwp", "acwp"}
        missing = need - set(df.columns)
        if missing:
            raise EvmError(f"Missing column(s) {sorted(missing)}.")
        if account and account in df.columns:
            accounts = {}
            for key, part in df.groupby(account, sort=False):
                accounts[str(key)] = cls.from_frame(part, cumulative=cumulative, name=str(key),
                                                   period=period, account=None)
            statuses = {k: a.status for k, a in accounts.items()}
            if len(set(statuses.values())) > 1:
                raise EvmError(f"Accounts end at different status periods: {statuses}.")
            pieces = [a._frame() for a in accounts.values()]
            total = pd.concat(pieces).groupby("period", sort=False).sum(min_count=1)
            total = total.reindex(sorted(total.index))
            out = cls._from_cumulative(total.index.tolist(), total, bac, name)
            out.accounts = accounts
            return out
        d = df.sort_values(period)
        if d[period].duplicated().any():
            raise EvmError("A period appears twice; give one row per period "
                           "(or name the account column).")
        cols = ["bcws", "bcwp", "acwp"] + (["eac"] if "eac" in d.columns else [])
        vals = d[cols].apply(pd.to_numeric, errors="coerce")
        if not cumulative:
            for c in ("bcws", "bcwp", "acwp"):
                known = vals[c].notna()
                vals.loc[known, c] = vals.loc[known, c].cumsum()
        return cls._from_cumulative(d[period].tolist(), vals, bac, name)

    @classmethod
    def _from_cumulative(cls, periods, vals: pd.DataFrame, bac, name) -> "EvmData":
        if vals["bcws"].isna().any():
            raise EvmError("BCWS is blank in some period; the baseline has to be complete.")
        status = vals["bcwp"].notna() & vals["acwp"].notna()
        if not status.any():
            raise EvmError("No period has both BCWP and ACWP.")
        n = int(np.flatnonzero(status.to_numpy())[-1]) + 1
        if not status.iloc[:n].all():
            raise EvmError("BCWP or ACWP is blank before the status period.")
        eac = None
        if "eac" in vals.columns and vals["eac"].iloc[:n].notna().any():
            eac = vals["eac"].iloc[:n].to_numpy(dtype=float)
        bcws = vals["bcws"].to_numpy(dtype=float)
        return cls(periods=list(periods), bcws=bcws, bcwp=vals["bcwp"].iloc[:n].to_numpy(float),
                   acwp=vals["acwp"].iloc[:n].to_numpy(float),
                   bac=float(bac) if bac is not None else float(bcws[-1]), eac=eac, name=name)

    @classmethod
    def read(cls, path, **kwargs) -> "EvmData":
        """Read a CSV or Excel file laid out as for :meth:`from_frame`."""
        path = Path(path)
        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
        return cls.from_frame(df, **kwargs)

    def _frame(self) -> pd.DataFrame:
        n = len(self.bcws)
        pad = lambda a: np.concatenate([a, np.full(n - len(a), np.nan)])  # noqa: E731
        f = pd.DataFrame({"period": self.periods, "bcws": self.bcws, "bcwp": pad(self.bcwp),
                          "acwp": pad(self.acwp)})
        if self.eac is not None:
            f["eac"] = pad(self.eac)
        return f.set_index("period")

    # ----------------------------------------------------------- properties
    @property
    def status(self) -> int:
        """The status period's number: periods elapsed, the actual time."""
        return len(self.bcwp)

    @property
    def planned_duration(self) -> int:
        """Periods the baseline takes to earn the whole budget: the first
        period whose cumulative BCWS reaches it. Periods after that (a
        baseline carried on at zero past its end) do not count."""
        target = min(self.bac, float(self.bcws[-1]))
        tol = 1e-9 * max(1.0, abs(target))
        return int(np.flatnonzero(self.bcws >= target - tol)[0]) + 1

    def earned_schedule(self) -> np.ndarray:
        """Earned schedule in each status period, against the baseline up to
        its planned duration, so it never runs past it."""
        return earned_schedule(self.bcws[:self.planned_duration], self.bcwp)

    # -------------------------------------------------------------- metrics
    def metrics(self) -> pd.DataFrame:
        """Every metric in every status period; columns :data:`METRIC_COLUMNS`."""
        n = self.status
        t = np.arange(1, n + 1, dtype=float)
        pv, ev, ac, bac = self.bcws[:n], self.bcwp, self.acwp, self.bac
        cpi, spi = _div(ev, ac), _div(ev, pv)
        d_ev = np.diff(np.concatenate([[0.0], ev]))
        d_ac = np.diff(np.concatenate([[0.0], ac]))
        d_pv = np.diff(np.concatenate([[0.0], pv]))
        es = np.atleast_1d(self.earned_schedule())
        pd_ = self.planned_duration
        spi_t = es / t
        remaining = bac - ev
        k = 3
        ev_k = ev - np.concatenate([np.zeros(k), ev])[:n]
        ac_k = ac - np.concatenate([np.zeros(k), ac])[:n]
        cpi_3 = _div(ev_k, ac_k)
        m = pd.DataFrame({
            "period": self.periods[:n], "time": t, "bcws": pv, "bcwp": ev, "acwp": ac,
            "cv": ev - ac, "sv": ev - pv, "cpi": cpi, "spi": spi,
            "cpi_period": _div(d_ev, d_ac), "spi_period": _div(d_ev, d_pv),
            "pct_complete": ev / bac, "pct_spent": ac / bac,
            "tcpi_bac": _div(remaining, bac - ac),
            "es": es, "sv_t": es - t, "spi_t": spi_t, "tspi": _div(pd_ - es, pd_ - t),
            "ieac_cpi": ac + _div(remaining, cpi),
            "ieac_cpi_spi": ac + _div(remaining, cpi * spi),
            "ieac_cpi_spi_t": ac + _div(remaining, cpi * spi_t),
            "ieac_cpi_3": ac + _div(remaining, cpi_3),
            "ieac_weighted": ac + _div(remaining, 0.8 * cpi + 0.2 * spi),
            "ieac_linear": ac + remaining,
            "ieac_t": _div(pd_, spi_t),
        })
        eac = self.eac if self.eac is not None else np.full(n, np.nan)
        m["eac"] = eac
        m["tcpi_eac"] = _div(remaining, eac - ac)
        return m[list(METRIC_COLUMNS)]

    def summary(self) -> pd.DataFrame:
        """The status period's numbers, one per row, for a briefing."""
        r = self.metrics().iloc[-1]
        ieacs = r[["ieac_cpi", "ieac_cpi_spi", "ieac_cpi_spi_t", "ieac_cpi_3",
                   "ieac_weighted"]].astype(float)
        rows = [
            ("status period", r["period"]),
            ("budget at completion (BAC)", self.bac),
            ("percent complete", r["pct_complete"]),
            ("percent spent", r["pct_spent"]),
            ("cost variance (CV)", r["cv"]),
            ("schedule variance (SV)", r["sv"]),
            ("CPI", r["cpi"]),
            ("SPI", r["spi"]),
            ("SPI(t)", r["spi_t"]),
            ("schedule variance in periods, SV(t)", r["sv_t"]),
            ("planned duration (periods)", self.planned_duration),
            ("independent completion estimate, IEAC(t) (periods)", r["ieac_t"]),
            ("TCPI to BAC", r["tcpi_bac"]),
            ("independent EACs: lowest", float(ieacs.min())),
            ("independent EACs: highest", float(ieacs.max())),
            ("contractor EAC", r["eac"]),
            ("TCPI to the contractor EAC", r["tcpi_eac"]),
        ]
        return pd.DataFrame(rows, columns=["measure", "value"])

    def flags(self) -> pd.DataFrame:
        """The warning signs at the status period, each with its reason.

        Columns ``flag``, ``raised`` and ``detail``. A flag that cannot be
        assessed (no contractor EAC, say) is not raised and says why.
        """
        r = self.metrics().iloc[-1]
        out = []

        def add(flag, raised, detail):
            out.append({"flag": flag, "raised": bool(raised), "detail": detail})

        if np.isfinite(r["eac"]):
            gap = r["tcpi_eac"] - r["cpi"]
            add("TCPI to the EAC exceeds the CPI by more than 0.10", gap > 0.10,
                f"TCPI(EAC) {r['tcpi_eac']:.3f} against CPI {r['cpi']:.3f}: the remaining "
                f"work would have to be done {gap:+.3f} more efficiently than the work so far.")
            low = min(r["ieac_cpi"], r["ieac_cpi_spi"], r["ieac_cpi_3"], r["ieac_weighted"])
            add("Contractor EAC below every independent EAC", r["eac"] < low,
                f"EAC {r['eac']:,.1f} against the lowest independent EAC {low:,.1f}.")
            implied = self.bac / r["eac"]
            add("EAC implies the CPI recovering by more than 0.10 after 20% complete",
                r["pct_complete"] >= 0.2 and implied - r["cpi"] > 0.10,
                f"The EAC implies a final CPI of {implied:.3f}, against {r['cpi']:.3f} at "
                f"{r['pct_complete']:.0%} complete; programs rarely recover 0.10 past 20%.")
        else:
            for f in ("TCPI to the EAC exceeds the CPI by more than 0.10",
                      "Contractor EAC below every independent EAC",
                      "EAC implies the CPI recovering by more than 0.10 after 20% complete"):
                add(f, False, "No contractor EAC in the data.")
        add("SPI has recovered but SPI(t) has not", r["pct_complete"] > 0.66
            and r["spi"] - r["spi_t"] > 0.10,
            f"SPI {r['spi']:.3f}, SPI(t) {r['spi_t']:.3f} at {r['pct_complete']:.0%} complete: "
            "late in a program SPI returns towards 1.0 whatever the schedule.")
        add("TCPI to BAC above 1.10", r["tcpi_bac"] > 1.10,
            f"Finishing within budget needs a CPI of {r['tcpi_bac']:.3f} from here.")
        return pd.DataFrame(out)
