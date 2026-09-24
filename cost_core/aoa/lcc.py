# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
lcc.py - Life-cycle cost for an analysis of alternatives.

An AoA asks which of several ways of meeting a need is worth its cost over the
whole life of the thing: development, production, decades of operating and
support, and disposal. Three things make that different from pricing one
program, and this module is built around them.

**Three kinds of money, for three questions.** The same estimate is stated
three ways, and each answers something different:

* *Base-year (constant) dollars* remove inflation. They are how estimates are
  built and compared line by line.
* *Then-year dollars* are what will actually be appropriated in each year, the
  base-year amount inflated to that year. Budgets are in these.
* *Present value* discounts each year's constant dollars back to one year at a
  real discount rate, because a dollar spent in 2040 costs less today than a
  dollar spent in 2027. Alternatives with different timing are compared on
  this: OMB Circular A-94 requires it for federal cost-effectiveness analysis
  and publishes the real rate to use each year (its Appendix C). There is no
  default rate here on purpose; pass the current one and it is recorded.

Discounting constant dollars at a *real* rate is the consistent pairing. Then-
year dollars would be discounted at a nominal rate instead; mixing the two
double-counts inflation, which is the classic AoA arithmetic error.

**Uncertainty is part of the answer.** Each cost line carries a distribution
on a multiplicative factor (1.0 is the point estimate), and the lines of an
alternative are correlated through :mod:`cost_core.monte_carlo`, for the same
reason a WBS roll-up is: the lines share a design, a contractor and a
schedule. The result for each alternative is a distribution, so the useful
outputs are P50 and P80 and the probability each alternative is the cheapest,
not only a ranking of point estimates that the uncertainty can overturn.

**Cost is half of it.** An alternative that costs more and does more is not
worse. When effectiveness scores are supplied, the result reports cost per unit
of effectiveness and which alternatives are dominated (another does at least as
well for no more money), which is the frontier a decision maker actually
chooses along.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cost_core.ingest.inflation import DEFAULT_INDEX, InflationTable
from cost_core.monte_carlo import (CostElement, RiskModel, RiskModelError,
                                   simulate_risk_model, uniform_correlation)

#: Phases a cost line can belong to, in life-cycle order.
PHASES = ("RDT&E", "Procurement", "MILCON", "O&S", "Disposal")


class AoAError(ValueError):
    """Raised on an alternative or analysis that cannot be evaluated."""


# ------------------------------------------------------------------ phasing ---
def spread(total: float, start_year: int, years: int, profile: str = "uniform") -> Dict[int, float]:
    """Phase a total over consecutive fiscal years.

    Args:
        total: Amount to spread, in base-year dollars.
        start_year: First fiscal year.
        years: Number of years.
        profile: ``"uniform"``; ``"front"`` or ``"back"`` (linearly
            weighted toward the start or the end); or ``"bell"``, a
            symmetric hump, the usual shape of development spending.

    Returns:
        ``{fiscal_year: amount}``, summing to ``total`` exactly.
    """
    if years < 1:
        raise AoAError(f"Need at least one year to spread over; got {years}.")
    k = np.arange(1, years + 1, dtype=float)
    weights = {
        "uniform": np.ones(years),
        "front": k[::-1],
        "back": k,
        "bell": np.sin(np.pi * (k - 0.5) / years),
    }.get(profile)
    if weights is None:
        raise AoAError(f"Unknown profile {profile!r}; use uniform, front, back or bell.")
    weights = weights / weights.sum()
    amounts = total * weights
    amounts[-1] = total - amounts[:-1].sum()  # land on the total exactly
    return {start_year + i: float(a) for i, a in enumerate(amounts)}


def annual(amount: float, first_year: int, last_year: int) -> Dict[int, float]:
    """The same amount every year, for recurring cost such as O&S."""
    if last_year < first_year:
        raise AoAError(f"last_year {last_year} precedes first_year {first_year}.")
    return {y: float(amount) for y in range(first_year, last_year + 1)}


# ------------------------------------------------------------------- inputs ---
@dataclass(frozen=True)
class CostLine:
    """One line of an alternative's life-cycle cost.

    Attributes:
        name: Label, unique within the alternative.
        phase: One of :data:`PHASES`.
        by_year: Base-year dollars by fiscal year, e.g. from :func:`spread`.
        uncertainty: Spec for :func:`cost_core.monte_carlo.make_distribution`
            describing a multiplicative factor on the whole line, 1.0 being
            the point estimate. ``{"type": "triangular", "left": 0.9,
            "mode": 1.0, "right": 1.5}`` is a line that could come in 10%
            under or 50% over. None means no uncertainty.
    """

    name: str
    phase: str
    by_year: Mapping[int, float]
    uncertainty: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise AoAError(f"{self.name}: phase {self.phase!r} is not one of {PHASES}.")
        if not self.by_year:
            raise AoAError(f"{self.name}: no fiscal years given.")

    @property
    def total(self) -> float:
        return float(sum(self.by_year.values()))


@dataclass(frozen=True)
class Alternative:
    """One way of meeting the need.

    Attributes:
        name: Label.
        lines: Its cost lines.
        effectiveness: A score on whatever scale the AoA uses (a measure of
            effectiveness, or a weighted value score), higher being better.
            Optional; without it only cost is compared.
        correlation: Uniform correlation between this alternative's lines in
            the simulation. The monte_carlo module's reasoning applies:
            independence is the unsafe assumption.
    """

    name: str
    lines: Sequence[CostLine]
    effectiveness: Optional[float] = None
    correlation: float = 0.3

    def __post_init__(self) -> None:
        if not self.lines:
            raise AoAError(f"{self.name}: an alternative needs at least one cost line.")
        names = [ln.name for ln in self.lines]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise AoAError(f"{self.name}: duplicate line name(s) {sorted(dup)}.")


# ----------------------------------------------------------------- money ---
def _money(line: CostLine, base_year: int, inflation: InflationTable, index: str,
           discount_rate: float, pv_year: int) -> "tuple[float, float, float]":
    by = ty = pv = 0.0
    for year, amount in line.by_year.items():
        by += amount
        ty += amount * inflation.factor(base_year, int(year), index)
        pv += amount / (1.0 + discount_rate) ** (int(year) - pv_year)
    return by, ty, pv


def _scale_spec(spec: Optional[Dict[str, Any]], k: float) -> Dict[str, Any]:
    """A factor distribution rescaled to dollars: ``k`` times the factor."""
    if spec is None:
        return {"type": "fixed", "value": k}
    s = dict(spec)
    kind = str(s.get("type", "")).lower()
    if kind in ("triangular", "pert"):
        for key in ("left", "mode", "right"):
            s[key] = float(s[key]) * k
    elif kind == "uniform":
        s["low"], s["high"] = float(s["low"]) * k, float(s["high"]) * k
    elif kind == "normal":
        s["loc"], s["scale"] = float(s["loc"]) * k, float(s["scale"]) * k
    elif kind == "lognormal":
        if k <= 0:
            raise AoAError("A lognormal factor needs a positive amount to scale.")
        s["mean"] = float(s["mean"]) + math.log(k)
    elif kind == "fixed":
        s["value"] = float(s["value"]) * k
    elif kind == "empirical":
        s["draws"] = np.asarray(s["draws"], dtype=float) * k
    else:
        raise AoAError(f"Cannot scale a {kind!r} factor distribution.")
    return s


# ----------------------------------------------------------------- result ---
@dataclass
class AoAResult:
    """Everything :func:`evaluate` works out.

    Attributes:
        lines: Per line: base-year, then-year and present-value totals.
        summary: Per alternative: the three point totals, the simulated mean,
            P50 and P80 on the chosen basis, the probability of being the
            cheapest, and cost-effectiveness where scores were given.
        draws: Per alternative, the simulated life-cycle cost on the chosen
            basis, one value per iteration.
        assumptions: Every setting the answer depends on, for the record.
    """

    lines: pd.DataFrame
    summary: pd.DataFrame
    draws: Dict[str, np.ndarray]
    assumptions: Dict[str, Any] = field(default_factory=dict)

    def s_curves(self, points: Iterable[float] = range(5, 100, 5)) -> pd.DataFrame:
        """Percentiles of each alternative's simulated cost, one column each."""
        qs = list(points)
        return pd.DataFrame({name: np.percentile(d, qs) for name, d in self.draws.items()},
                            index=pd.Index(qs, name="percentile"))


def _frontier(costs: Sequence[float], eff: Sequence[Optional[float]]) -> "list[Optional[str]]":
    """For each alternative, the name of one that dominates it, or None.

    B dominates A when B costs no more and is at least as effective, and is
    strictly better on one of the two.
    """
    out = []
    for i, (ci, ei) in enumerate(zip(costs, eff)):
        by = None
        if ei is not None:
            for j, (cj, ej) in enumerate(zip(costs, eff)):
                if j != i and ej is not None and cj <= ci and ej >= ei and (cj < ci or ej > ei):
                    by = j
                    break
        out.append(by)
    return out


def evaluate(
    alternatives: Sequence[Alternative],
    *,
    base_year: int,
    inflation: InflationTable,
    discount_rate: float,
    pv_year: Optional[int] = None,
    basis: str = "pv",
    index: str = DEFAULT_INDEX,
    n_iter: int = 20_000,
    seed: Optional[int] = 0,
    units: str = "as entered",
) -> AoAResult:
    """Cost the alternatives and compare them under uncertainty.

    Args:
        alternatives: Two or more, with distinct names.
        base_year: The fiscal year the cost lines are stated in.
        inflation: Index table used to reach then-year dollars.
        discount_rate: Real discount rate for present value, as a fraction
            (0.02 is 2%). OMB Circular A-94 Appendix C gives the current one.
        pv_year: Year values are discounted to; defaults to ``base_year``.
        basis: What the simulation and the ranking use: ``"pv"`` (the A-94
            comparison), ``"by"`` or ``"ty"``.
        index: Which index in ``inflation`` to use.
        n_iter: Iterations per alternative.
        seed: Seed for reproducibility; each alternative draws from its own
            stream derived from it, so adding an alternative does not change
            the draws of the others.
        units: What the cost lines are stated in, e.g. ``"$M"``. Nothing is
            converted; it is recorded and printed on the chart, so a table in
            millions is never read as thousands.

    Raises:
        AoAError: On fewer than two alternatives, duplicate names, a negative
            or implausible discount rate, or an unknown basis.
    """
    if len(alternatives) < 2:
        raise AoAError("An AoA compares at least two alternatives.")
    names = [a.name for a in alternatives]
    if len(set(names)) != len(names):
        raise AoAError(f"Alternative names must be unique; got {names}.")
    if not 0 <= discount_rate < 0.2:
        raise AoAError(f"discount_rate {discount_rate} should be a real rate as a fraction, "
                       f"e.g. 0.02 for 2%.")
    if basis not in ("pv", "by", "ty"):
        raise AoAError(f"basis must be 'pv', 'by' or 'ty'; got {basis!r}.")
    pv_year = base_year if pv_year is None else pv_year

    rows = []
    for alt in alternatives:
        for line in alt.lines:
            by, ty, pv = _money(line, base_year, inflation, index, discount_rate, pv_year)
            years = sorted(int(y) for y in line.by_year)
            rows.append({"alternative": alt.name, "line": line.name, "phase": line.phase,
                         "first_year": years[0], "last_year": years[-1],
                         "by": by, "ty": ty, "pv": pv,
                         "uncertain": line.uncertainty is not None})
    lines = pd.DataFrame(rows)

    draws: Dict[str, np.ndarray] = {}
    for i, alt in enumerate(alternatives):
        sub = lines[lines["alternative"] == alt.name]
        elements = [CostElement(name=ln.name,
                                distribution=_scale_spec(ln.uncertainty, float(amount)),
                                point_estimate=float(amount))
                    for ln, amount in zip(alt.lines, sub[basis])]
        stream = None if seed is None else int(seed) * 1000 + i
        try:
            result = simulate_risk_model(
                RiskModel(elements=elements,
                          correlation=uniform_correlation(len(elements), alt.correlation),
                          name=alt.name),
                n_iter=n_iter, seed=stream)
        except RiskModelError as exc:
            raise AoAError(f"{alt.name}: {exc}") from exc
        draws[alt.name] = result.totals

    stack = np.vstack([draws[n] for n in names])
    cheapest = np.bincount(stack.argmin(axis=0), minlength=len(names)) / stack.shape[1]
    totals = lines.groupby("alternative", sort=False)[["by", "ty", "pv"]].sum()
    summary = pd.DataFrame({
        "alternative": names,
        "by": [totals.loc[n, "by"] for n in names],
        "ty": [totals.loc[n, "ty"] for n in names],
        "pv": [totals.loc[n, "pv"] for n in names],
        "mean": [float(draws[n].mean()) for n in names],
        "p50": [float(np.percentile(draws[n], 50)) for n in names],
        "p80": [float(np.percentile(draws[n], 80)) for n in names],
        "p_cheapest": cheapest,
        "effectiveness": [a.effectiveness for a in alternatives],
    })
    eff = list(summary["effectiveness"])
    if any(e is not None for e in eff):
        summary["cost_per_effectiveness"] = [
            (m / e) if e not in (None, 0) else np.nan for m, e in zip(summary["mean"], eff)]
        dom = _frontier(list(summary["mean"]), eff)
        summary["dominated_by"] = [names[j] if j is not None else None for j in dom]
        summary["on_frontier"] = [e is not None and j is None for e, j in zip(eff, dom)]

    return AoAResult(
        lines=lines, summary=summary, draws=draws,
        assumptions={"base_year": base_year, "pv_year": pv_year,
                     "discount_rate": discount_rate, "basis": basis,
                     "inflation_index": index, "inflation_source": inflation.source,
                     "n_iter": n_iter, "seed": seed, "units": units,
                     "correlation": {a.name: a.correlation for a in alternatives},
                     "dominance_on": "simulated mean cost"},
    )
