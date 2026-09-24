# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
optimize.py - Which programs to fund, as an integer program.

A portfolio decision is a capital budgeting problem: a list of programs, each
fundable in one of a few ways (fully, at a reduced rate, deferred) or not at
all, each way costing a known amount in each fiscal year and buying some value,
and a budget in each year that the chosen set has to fit under. Maximise value.
In a spreadsheet that is Excel Solver with a binary constraint per cell; here
it is written out as a mixed-integer program and solved with CBC through PuLP.

The formulation, with one binary ``x[c, o]`` per candidate ``c`` and option ``o``:

* maximise ``sum value[c, o] * x[c, o]``
* each year ``t``: ``sum cost[c, o, t] * x[c, o] <= budget[t]``
* each candidate: ``sum_o x[c, o] <= 1``, and ``= 1`` when it is mandatory
* each exclusive group (alternatives for the same need, straight from an
  AoA): at most one of its candidates funded
* each dependency ("B requires A"): ``funded(B) <= funded(A)``

**The optimum is the start of the answer.** Three things matter as much in a
briefing, and each has its own function:

* :func:`marginal_value` re-solves with a little more money in one year at a
  time. Its answer, value gained per dollar, is what an extra dollar in 2029
  is worth against one in 2031, which is the question every reprogramming
  argument is really about. (Duals of the LP relaxation would be quicker and
  wrong: a knapsack's value jumps, it does not slope.)
* :func:`frontier` solves across a range of budget levels, so the
  conversation can be about where the knee in value per dollar is rather
  than about one number.
* :func:`budget_risk` takes the chosen portfolio and draws cost growth for
  each program, from the SAR history in :mod:`cost_core.aoa.growth` or any
  factor distribution, to say how likely each year is to break its budget.
  An optimum on point estimates fills the budget exactly, so it usually is.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from cost_core.monte_carlo import make_distribution


class PortfolioError(ValueError):
    """Raised on a portfolio that cannot be built or solved."""


def _pulp():
    try:
        import pulp
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "portfolio optimisation needs PuLP: pip install \"cost-core[optimize]\""
        ) from exc
    return pulp


# ------------------------------------------------------------------- inputs ---
@dataclass(frozen=True)
class Option:
    """One way of funding a candidate.

    Attributes:
        name: e.g. ``"Full"``, ``"Reduced rate"``, ``"Defer two years"``.
        cost_by_year: Cost in each fiscal year, in the budget's units.
        value: What funding it this way buys, on the portfolio's value scale.
    """

    name: str
    cost_by_year: Mapping[int, float]
    value: float

    @property
    def total(self) -> float:
        return float(sum(self.cost_by_year.values()))


@dataclass(frozen=True)
class Candidate:
    """A program the portfolio could fund.

    Attributes:
        name: Unique label.
        options: Ways to fund it; at most one is chosen.
        mandatory: Must be funded (by one of its options).
        requires: Names of candidates that must be funded if this one is.
        category: Free label for grouping in the results.
    """

    name: str
    options: Sequence[Option]
    mandatory: bool = False
    requires: Sequence[str] = ()
    category: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.options:
            raise PortfolioError(f"{self.name}: a candidate needs at least one option.")


@dataclass(frozen=True)
class Portfolio:
    """The whole decision.

    Attributes:
        candidates: The programs.
        budget: Money available in each fiscal year. Years a candidate spends
            in but the budget does not name are an error, not free money.
        exclusive: Groups of candidate names of which at most one may be
            funded, e.g. the alternatives of one AoA.
    """

    candidates: Sequence[Candidate]
    budget: Mapping[int, float]
    exclusive: Sequence[Sequence[str]] = ()

    def __post_init__(self) -> None:
        names = [c.name for c in self.candidates]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise PortfolioError(f"Duplicate candidate name(s) {sorted(dup)}.")
        known = set(names)
        for c in self.candidates:
            for r in c.requires:
                if r not in known:
                    raise PortfolioError(f"{c.name} requires {r!r}, which is not a candidate.")
            for o in c.options:
                extra = sorted(set(o.cost_by_year) - set(self.budget))
                if extra:
                    raise PortfolioError(
                        f"{c.name} / {o.name} spends in {extra}, which the budget does not cover.")
        for group in self.exclusive:
            unknown = sorted(set(group) - known)
            if unknown:
                raise PortfolioError(f"Exclusive group names unknown candidate(s) {unknown}.")

    @property
    def years(self) -> List[int]:
        return sorted(int(y) for y in self.budget)


# ------------------------------------------------------------------ result ---
@dataclass
class PortfolioResult:
    """A solved portfolio.

    Attributes:
        status: ``"Optimal"`` or the solver's reason it is not.
        value: Total value of the chosen options.
        selected: One row per candidate: the option chosen (or none), its
            value and total cost, mandatory and category.
        spend: One row per year: budget, spend, and what is left.
        choice: ``{candidate: option name or None}``, for re-use.
    """

    status: str
    value: float
    selected: pd.DataFrame
    spend: pd.DataFrame
    choice: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def funded(self) -> List[str]:
        return [c for c, o in self.choice.items() if o is not None]


def _infeasibility_hint(p: Portfolio, budget: Mapping[int, float]) -> str:
    must = [c for c in p.candidates if c.mandatory]
    lines = []
    for y in p.years:
        cheapest = sum(min(o.cost_by_year.get(y, 0.0) for o in c.options) for c in must)
        if cheapest > budget[y] + 1e-9:
            lines.append(f"{y}: mandatory programs need at least {cheapest:,.2f} "
                         f"against a budget of {budget[y]:,.2f}")
    return ("; ".join(lines) if lines else
            "check dependencies and exclusive groups against the mandatory programs")


def _binary(pulp, prob, name):
    """A binary variable, the PuLP 3.3+ way when available (the direct
    constructor is deprecated there) and the 2.x way on Python 3.9, where
    3.x does not install."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, cat="Binary")
    return pulp.LpVariable(name, cat="Binary")


def _solver(pulp, time_limit):
    """CBC as PuLP ships it. PuLP 3.3 deprecates the bundled binary in favour
    of a separately installed one; use that when it is there, and the bundled
    one quietly otherwise, since it is the one every install has."""
    kwargs = {"msg": False}
    if time_limit is not None:
        kwargs["timeLimit"] = time_limit
    coin = getattr(pulp, "COIN_CMD", None)
    if coin is not None:
        try:
            solver = coin(**kwargs)
            if solver.available():
                return solver
        except Exception:  # noqa: BLE001 - fall back to the bundled binary
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pulp.PULP_CBC_CMD(**kwargs)


def solve(
    portfolio: Portfolio,
    *,
    budget_scale: float = 1.0,
    budget_override: Optional[Mapping[int, float]] = None,
    cost_factor: float = 1.0,
    time_limit: Optional[float] = None,
) -> PortfolioResult:
    """Choose options to maximise value within every year's budget.

    Args:
        portfolio: The decision.
        budget_scale: Multiply every year's budget, e.g. 0.9 for a 10% cut.
        budget_override: Replace the budget outright (after scaling is ignored).
        cost_factor: Multiply every cost, e.g. to plan to a P80 growth factor
            rather than point estimates.
        time_limit: Seconds for the solver; None for no limit.

    Raises:
        PortfolioError: The mandatory programs alone do not fit, or the
            constraints admit no portfolio at all; the message says which year.
    """
    pulp = _pulp()
    budget = ({int(y): float(v) for y, v in budget_override.items()} if budget_override
              else {int(y): float(v) * budget_scale for y, v in portfolio.budget.items()})
    prob = pulp.LpProblem("portfolio", pulp.LpMaximize)
    x = {}
    for i, c in enumerate(portfolio.candidates):
        for j, o in enumerate(c.options):
            x[c.name, o.name] = _binary(pulp, prob, f"x_{i}_{j}")
    prob += pulp.lpSum(o.value * x[c.name, o.name]
                       for c in portfolio.candidates for o in c.options)
    for y in portfolio.years:
        prob += (pulp.lpSum(o.cost_by_year.get(y, 0.0) * cost_factor * x[c.name, o.name]
                            for c in portfolio.candidates for o in c.options)
                 <= budget[y], f"budget_{y}")
    funded = {c.name: pulp.lpSum(x[c.name, o.name] for o in c.options)
              for c in portfolio.candidates}
    for c in portfolio.candidates:
        prob += (funded[c.name] == 1) if c.mandatory else (funded[c.name] <= 1)
        for r in c.requires:
            prob += funded[c.name] <= funded[r]
    for group in portfolio.exclusive:
        prob += pulp.lpSum(funded[n] for n in group) <= 1

    prob.solve(_solver(pulp, time_limit))
    status = pulp.LpStatus[prob.status]
    if status == "Infeasible":
        raise PortfolioError("No portfolio fits: " + _infeasibility_hint(portfolio, budget))
    if status != "Optimal":
        raise PortfolioError(f"Solver stopped with status {status!r}.")

    choice: Dict[str, Optional[str]] = {}
    rows = []
    for c in portfolio.candidates:
        chosen = next((o for o in c.options if (x[c.name, o.name].value() or 0) > 0.5), None)
        choice[c.name] = chosen.name if chosen else None
        rows.append({"candidate": c.name, "category": c.category, "mandatory": c.mandatory,
                     "option": chosen.name if chosen else None,
                     "value": chosen.value if chosen else 0.0,
                     "cost": chosen.total * cost_factor if chosen else 0.0})
    selected = pd.DataFrame(rows)
    spend = spend_by_year(portfolio, choice, cost_factor)
    spend["budget"] = [budget[y] for y in spend["year"]]
    spend["remaining"] = spend["budget"] - spend["spend"]
    return PortfolioResult(status=status, value=float(selected["value"].sum()),
                           selected=selected, spend=spend[["year", "budget", "spend", "remaining"]],
                           choice=choice)


def spend_by_year(portfolio: Portfolio, choice: Mapping[str, Optional[str]],
                  cost_factor: float = 1.0) -> pd.DataFrame:
    """What a given choice of options spends in each year."""
    by_name = {c.name: c for c in portfolio.candidates}
    totals = {y: 0.0 for y in portfolio.years}
    for cname, oname in choice.items():
        if oname is None:
            continue
        option = next(o for o in by_name[cname].options if o.name == oname)
        for y, v in option.cost_by_year.items():
            totals[int(y)] += float(v) * cost_factor
    return pd.DataFrame({"year": list(totals), "spend": list(totals.values())})


# ---------------------------------------------------------------- analyses ---
def marginal_value(portfolio: Portfolio, delta: float, **kwargs) -> pd.DataFrame:
    """Value gained per unit of money added to one year at a time.

    Re-solves once per year with ``delta`` more in that year only. A year whose
    extra money buys nothing has a marginal value of zero, which is the case
    for moving money out of it.
    """
    if delta <= 0:
        raise PortfolioError("delta must be positive.")
    base = solve(portfolio, **kwargs)
    scale = kwargs.pop("budget_scale", 1.0)
    rows = []
    for y in portfolio.years:
        budget = {yy: v * scale for yy, v in portfolio.budget.items()}
        budget[y] += delta
        r = solve(portfolio, budget_override=budget, **kwargs)
        changed = sorted(c for c in r.choice if r.choice[c] != base.choice[c])
        rows.append({"year": y, "added": delta, "value_gain": r.value - base.value,
                     "value_per_unit": (r.value - base.value) / delta,
                     "changes": ", ".join(changed)})
    return pd.DataFrame(rows)


def frontier(portfolio: Portfolio, scales: Iterable[float] = np.linspace(0.6, 1.4, 9),
             **kwargs) -> pd.DataFrame:
    """Best value at each budget level, as a multiple of the stated budget.

    Levels at which even the mandatory programs do not fit are reported with a
    blank value rather than stopping the sweep.
    """
    rows = []
    total = sum(portfolio.budget.values())
    for s in scales:
        try:
            r = solve(portfolio, budget_scale=float(s), **kwargs)
            rows.append({"budget_scale": float(s), "budget_total": total * s, "value": r.value,
                         "funded": len(r.funded), "spend_total": float(r.spend["spend"].sum())})
        except PortfolioError:
            rows.append({"budget_scale": float(s), "budget_total": total * s, "value": np.nan,
                         "funded": np.nan, "spend_total": np.nan})
    return pd.DataFrame(rows)


def budget_risk(
    portfolio: Portfolio,
    choice: Mapping[str, Optional[str]],
    growth: Dict[str, Any],
    *,
    n_iter: int = 20_000,
    seed: Optional[int] = 0,
    correlation: float = 0.0,
) -> pd.DataFrame:
    """How likely each year is to break its budget once costs grow.

    Each funded program draws one cost factor from ``growth`` (a
    :func:`cost_core.monte_carlo.make_distribution` spec, such as
    :func:`cost_core.aoa.growth_factor` of SAR history) and applies it to all
    its years. ``correlation`` links the programs' draws through a Gaussian
    copula: portfolio overruns are rarely independent, since programs share
    an industrial base and a budget climate.

    Returns one row per year: budget, planned spend, the P50 and P80 of
    simulated spend, and the probability spend exceeds the budget.
    """
    funded = [(c, o) for c, o in choice.items() if o is not None]
    if not funded:
        raise PortfolioError("Nothing is funded, so there is no risk to measure.")
    k = len(funded)
    rng = np.random.default_rng(seed)
    if correlation:
        if not -1.0 / max(k - 1, 1) <= correlation <= 1.0:
            raise PortfolioError(f"correlation {correlation} is not achievable across {k} programs.")
        cov = np.full((k, k), correlation)
        np.fill_diagonal(cov, 1.0)
        z = rng.multivariate_normal(np.zeros(k), cov, size=n_iter)
        from scipy import stats
        u = stats.norm.cdf(z)
    else:
        u = rng.random((n_iter, k))
    dist = make_distribution(growth)
    factors = np.column_stack([np.asarray(dist.ppf(u[:, i])) for i in range(k)])
    by_name = {c.name: c for c in portfolio.candidates}
    rows = []
    for y in portfolio.years:
        planned = np.array([next(o for o in by_name[c].options if o.name == on).cost_by_year.get(y, 0.0)
                            for c, on in funded], dtype=float)
        sims = factors @ planned
        rows.append({"year": y, "budget": float(portfolio.budget[y]), "planned": float(planned.sum()),
                     "p50": float(np.percentile(sims, 50)), "p80": float(np.percentile(sims, 80)),
                     "p_over_budget": float((sims > portfolio.budget[y] + 1e-9).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- from an AoA ---
def candidates_from_aoa(result, years: Optional[Iterable[int]] = None,
                        prefix: str = "") -> "tuple[list, list]":
    """Turn an AoA's alternatives into mutually exclusive portfolio candidates.

    Each alternative becomes a candidate with one option costing its
    base-year lines by year, valued at its effectiveness score; the returned
    exclusive group says at most one of them can be funded. So an AoA feeds
    the portfolio directly, and the portfolio decides whether the need is
    worth meeting at all given everything else competing for the money.

    Args:
        result: An :class:`~cost_core.aoa.AoAResult`.
        years: The budget years to keep, typically the five of the FYDP. An
            AoA runs to disposal decades out, and a portfolio budget does
            not; costs outside ``years`` are left out, which is the usual
            treatment and worth saying out loud in the briefing: the
            portfolio then weighs each alternative's near-term cost only.
        prefix: Prepended to candidate names, to keep several AoAs apart.

    Returns:
        ``(candidates, [group])``.
    """
    keep = None if years is None else {int(y) for y in years}
    cands = []
    for a in result.alternatives:
        if a.effectiveness is None:
            raise PortfolioError(f"{a.name} has no effectiveness score to value it by.")
        costs: Dict[int, float] = {}
        for line in a.lines:
            for y, v in line.by_year.items():
                if keep is None or int(y) in keep:
                    costs[int(y)] = costs.get(int(y), 0.0) + float(v)
        cands.append(Candidate(prefix + a.name, [Option("Fund", costs, float(a.effectiveness))]))
    return cands, [[c.name for c in cands]]
