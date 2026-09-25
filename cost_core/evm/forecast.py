# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
forecast.py - Estimate at completion and completion date as distributions.

Every independent EAC formula answers one question: if the rest of the work
is done at some chosen efficiency, what will it cost? The choice of
efficiency is the whole answer, and the formulas disagree by exactly as much
as the program's performance has varied. So the honest output is not one of
them but a range, and the program's own record is the best evidence of how
wide it is.

**The method.** Each reporting period so far is one observation of how the
program actually performs: how much schedule it earned (the change in earned
schedule) and how efficiently it spent (earned value over actual cost). A
simulated future is built by drawing those periods again, as pairs, and
running them against the remaining baseline:

* each drawn period advances earned schedule by what that period earned;
* the work earned is what the baseline planned between the old and the new
  earned schedule, so a program in its expensive middle earns expensive
  work;
* that work costs its earned value divided by the drawn period's
  efficiency.

Repeated until earned schedule reaches the planned duration, that gives one
completion date and one final cost, together. Because a period's speed and
its efficiency are drawn as a pair, a program that is late when it is over
budget stays so in the simulation, and the joint confidence of a budget and
a date comes out the same way as the JCL's.

Two refinements make the range honest rather than merely wide:

* Each simulated future first draws its own weighting of the record (the
  Bayesian bootstrap), because a few noisy periods say only roughly what the
  program's underlying efficiency and pace are. Without it the range covers
  period-to-period noise alone: on synthetic programs whose truth is known,
  the P80 held the actual cost 65% of the time and the actual finish 49%.
* Performance persists, so each future period's deviation from those rates
  carries into the next, starting from where the latest periods left off, by
  the persistence estimated from the record. Persistent periods are also
  worth fewer independent observations, which widens the first draw.

**How well it is calibrated.** On synthetic programs forecast from 10 and 20
periods of history, the P80 held the actual cost 79 to 82% of the time and
the actual finish 79 to 82% when periods were independent, and 72 to 77%
when each period's performance carried 0.6 into the next: slightly narrow
there, since a short record cannot pin down how persistent the program is.
``tests/test_evm.py`` holds the independent case to its coverage.

**What it assumes.** That the future looks like a resampling of the past:
no step change from a replan, a new subcontractor or a change of phase. The
``recent`` option draws only from the latest periods, for a program whose
early performance no longer says much. With constant performance the method
reduces exactly to the CPI and SPI(t) formulas, which the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from cost_core.evm.metrics import EvmData, EvmError


@dataclass
class EvmForecast:
    """Simulated completions of a program from its status period.

    Attributes:
        eac: Final cost, per simulated completion.
        finish: Completion time in periods from the program start.
        status_time: The status period (actual time).
        planned_duration: The baseline's length in periods.
        bac: Budget at completion.
        contractor_eac: The contractor's EAC at the status period, if given.
        independent: The status period's formula EACs, by name.
        notes: Assumptions and anything unusual in the data used.
    """

    eac: np.ndarray
    finish: np.ndarray
    status_time: float
    planned_duration: float
    bac: float
    contractor_eac: Optional[float] = None
    independent: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def n_iter(self) -> int:
        return int(self.eac.size)

    def confidence_of_cost(self, cost: float) -> float:
        """Chance the final cost is at or under ``cost``."""
        return float(np.mean(self.eac <= cost))

    def joint(self, cost: float, finish: float) -> float:
        """Chance of finishing at or under ``cost`` and by ``finish``."""
        return float(np.mean((self.eac <= cost) & (self.finish <= finish)))

    def percentiles(self, levels=(0.1, 0.5, 0.7, 0.8, 0.9)) -> pd.DataFrame:
        return pd.DataFrame({
            "confidence": list(levels),
            "eac": [float(np.quantile(self.eac, q)) for q in levels],
            "finish": [float(np.quantile(self.finish, q)) for q in levels],
        })

    def summary(self) -> pd.DataFrame:
        """Where the range sits, and where each point estimate falls on it."""
        rows = [("simulated completions", self.n_iter),
                ("EAC P50", float(np.quantile(self.eac, 0.5))),
                ("EAC P80", float(np.quantile(self.eac, 0.8))),
                ("completion P50 (periods)", float(np.quantile(self.finish, 0.5))),
                ("completion P80 (periods)", float(np.quantile(self.finish, 0.8))),
                ("planned duration (periods)", self.planned_duration),
                ("confidence of BAC", self.confidence_of_cost(self.bac)),
                ("joint confidence of BAC and the planned duration",
                 self.joint(self.bac, self.planned_duration))]
        if self.contractor_eac is not None and np.isfinite(self.contractor_eac):
            rows.append(("confidence of the contractor EAC",
                         self.confidence_of_cost(self.contractor_eac)))
        for name, value in self.independent.items():
            if np.isfinite(value):
                rows.append((f"confidence of {name}", self.confidence_of_cost(value)))
        return pd.DataFrame(rows, columns=["measure", "value"])


def _observed_periods(data: EvmData, recent: Optional[int]) -> "tuple[np.ndarray, np.ndarray, list]":
    """Per-period (earned schedule gained, efficiency) pairs that can be used."""
    ev = np.concatenate([[0.0], data.bcwp])
    ac = np.concatenate([[0.0], data.acwp])
    es = np.concatenate([[0.0], np.atleast_1d(data.earned_schedule())])
    d_ev, d_ac, d_es = np.diff(ev), np.diff(ac), np.diff(es)
    notes = []
    usable = (d_ev > 0) & (d_ac > 0)
    lumped = np.arange(1, len(d_ev) + 1) < data.first_observed
    if lumped.any():
        notes.append(f"Periods before {data.first_observed} are known only in total and were "
                     "not drawn from.")
    usable &= ~lumped
    dropped = int((~usable).sum())
    if dropped:
        notes.append(f"{dropped} period(s) with no earned value or no actual cost (or a "
                     "correction that took either back) were left out of the draws.")
    idx = np.flatnonzero(usable)
    if recent is not None:
        idx = idx[-int(recent):]
        notes.append(f"Only the latest {len(idx)} usable periods were drawn from.")
    return d_es[idx], d_ev[idx] / d_ac[idx], notes


def _lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    den = float(np.dot(x, x))
    return float(np.dot(x[1:], x[:-1]) / den) if den > 0 and len(x) > 2 else 0.0


def _persistence(d_es: np.ndarray, eff: np.ndarray) -> float:
    """How much one period's performance carries into the next.

    The larger lag-1 autocorrelation of the two series, held between 0 and
    0.9. Periods that persist are worth fewer independent observations
    (``n (1 - r) / (1 + r)`` of them, the usual effective sample size), and
    the draws are widened to match.
    """
    n = len(d_es)
    r = max(_lag1(d_es), _lag1(eff))
    # A lag-1 estimate from a short series is biased low, by about
    # (1 + 4r) / n (Kendall, 1954); a record of 10 periods from a process
    # at 0.6 reads about 0.25 before the correction.
    return float(np.clip(r + (1 + 4 * r) / n, 0.0, 0.9))


def forecast(
    data: EvmData,
    n_iter: int = 20_000,
    seed: Optional[int] = 0,
    *,
    persistence: Optional[float] = None,
    recent: Optional[int] = None,
    min_periods: int = 4,
    max_factor: float = 10.0,
) -> EvmForecast:
    """Simulate the rest of the program from its own performance record.

    Args:
        data: The program, through its status period.
        n_iter: Simulated completions.
        seed: Random seed; the same seed gives the same draws.
        persistence: How much one period's performance carries into the
            next, from 0 to 0.9; estimated from the record when omitted.
            It sets both how long runs of periods last in the draws and how
            much the record's few periods are trusted.
        recent: Draw only from the latest this many usable periods.
        min_periods: Fewest usable periods to draw from; below it the record
            is too short to say how performance varies.
        max_factor: A simulated program still unfinished at this many times
            the planned duration is stopped there and noted, rather than run
            for ever on a record of periods that earned nothing.

    Raises:
        EvmError: Too few usable periods, or the program is already complete.
    """
    if n_iter < 2:
        raise EvmError(f"Need at least 2 iterations; got {n_iter}.")
    d_es, eff, notes = _observed_periods(data, recent)
    if len(d_es) < min_periods:
        raise EvmError(f"Only {len(d_es)} usable periods; a forecast needs at least "
                       f"{min_periods} to see how performance varies.")
    pv = data.bcws
    pd_ = data.planned_duration
    t0 = float(data.status)
    es0 = float(np.atleast_1d(data.earned_schedule())[-1])
    ev0, ac0 = float(data.bcwp[-1]), float(data.acwp[-1])
    bac_pv = min(data.bac, float(pv[-1]))
    if es0 >= pd_ - 1e-12:
        raise EvmError("The program has earned its whole baseline; there is nothing to forecast.")
    if abs(data.bac - float(pv[-1])) > 1e-9 * data.bac:
        notes.append(f"BAC {data.bac:,.2f} differs from the baseline's total "
                     f"{float(pv[-1]):,.2f}; the remaining work is priced on the baseline "
                     "and scaled to BAC.")
    scale = (data.bac - ev0) / (bac_pv - ev0) if bac_pv > ev0 else 1.0

    rng = np.random.default_rng(seed)
    n_obs = len(d_es)
    rho = _persistence(d_es, eff) if persistence is None else float(persistence)
    # Each simulated future first draws how much weight each period of the
    # record deserves (a Bayesian bootstrap), then draws its periods with
    # those weights. The first draw carries the uncertainty in the program's
    # underlying rate, which a few noisy periods pin down only roughly, and
    # less well still when periods persist; the second carries the
    # period-to-period noise.
    weights = rng.dirichlet(np.full(n_obs, (1 - rho) / (1 + rho)), size=n_iter)
    cum_w = np.cumsum(weights, axis=1)
    cum_w[:, -1] = 1.0

    def fresh():
        u = rng.random(n_iter)
        return np.minimum((cum_w < u[:, None]).sum(axis=1), n_obs - 1)

    # Each future's own underlying rates, and its deviation from them, which
    # starts where the latest periods left off and carries into the next
    # period by rho: y = m + rho (y_prev - m) + sqrt(1 - rho^2) (x - m), with
    # x a period drawn from the weighted record. Its spread matches the
    # record's and its runs last as long.
    m_es = weights @ d_es
    m_eff = weights @ eff
    last = max(1, min(3, n_obs))
    y_es = np.full(n_iter, d_es[-last:].mean())
    y_eff = np.full(n_iter, eff[-last:].mean())
    shrink = np.sqrt(1 - rho ** 2)
    floor_es, floor_eff = 0.05 * float(d_es.mean()), 0.05 * float(eff.mean())

    es = np.full(n_iter, es0)
    ev_at = np.full(n_iter, ev0)
    cost = np.full(n_iter, ac0)
    finish = np.full(n_iter, np.nan)
    elapsed = np.zeros(n_iter)
    active = np.ones(n_iter, dtype=bool)
    limit = max_factor * pd_
    n_stuck = 0
    while active.any():
        j = fresh()
        y_es = np.maximum(m_es + rho * (y_es - m_es) + shrink * (d_es[j] - m_es), floor_es)
        y_eff = np.maximum(m_eff + rho * (y_eff - m_eff) + shrink * (eff[j] - m_eff), floor_eff)
        idx = np.flatnonzero(active)
        step = y_es[idx]
        new_es = np.minimum(es[idx] + step, pd_)
        new_ev = np.minimum(np.interp(new_es, np.arange(len(pv) + 1),
                                      np.concatenate([[0.0], pv])), bac_pv)
        earned = np.maximum(new_ev - ev_at[idx], 0.0) * scale
        cost[idx] += earned / y_eff[idx]
        # The last period is only partly needed: the time it takes is the
        # share of its earned schedule that reaching the end used up.
        done = new_es >= pd_ - 1e-12
        frac = np.where(done, (pd_ - es[idx]) / np.where(step > 0, step, 1.0), 1.0)
        elapsed[idx] += np.clip(frac, 0.0, 1.0)
        es[idx], ev_at[idx] = new_es, new_ev
        finish[idx[done]] = t0 + elapsed[idx[done]]
        active[idx[done]] = False
        stuck = active & (t0 + elapsed >= limit)
        if stuck.any():
            finish[stuck] = t0 + elapsed[stuck]
            # Price the unearned remainder at the program's overall efficiency.
            remaining = (bac_pv - ev_at[stuck]) * scale
            cost[stuck] += remaining / (ev0 / ac0)
            active[stuck] = False
            n_stuck += int(stuck.sum())
    if n_stuck:
        notes.append(f"{n_stuck} simulated completion(s) had not finished by {max_factor:g} "
                     "times the planned duration and were stopped there.")
    r = data.metrics().iloc[-1]
    independent = {name: float(r[col]) for name, col in (
        ("IEAC (CPI)", "ieac_cpi"), ("IEAC (CPI x SPI)", "ieac_cpi_spi"),
        ("IEAC (CPI x SPI(t))", "ieac_cpi_spi_t"), ("IEAC (3-period CPI)", "ieac_cpi_3"))}
    notes.insert(0, f"Drawn from {n_obs} periods of the program's own performance, "
                    f"with persistence {rho:.2f} between periods.")
    return EvmForecast(eac=cost, finish=finish, status_time=t0, planned_duration=pd_,
                       bac=data.bac, contractor_eac=float(r["eac"]), independent=independent,
                       notes=notes, seed=seed)
