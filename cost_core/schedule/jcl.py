# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
jcl.py - Schedule risk, and joint cost and schedule confidence (JCL).

A cost S-curve answers "how much money buys 70% confidence?" and says nothing
about when. For a project that carries a large team for its whole life, the
two are the same question: every month of slip is a month of salaries, so a
cost estimate that ignores the schedule is confident about the wrong thing.
NASA's answer is the joint confidence level, the probability of finishing at
or under a cost *and* at or before a date, and it budgets major projects at
the 70% JCL (NASA Cost Estimating Handbook v4.0, Appendix J).

**What a JCL model holds.** Following that appendix:

* a network of activities, each with a most-likely duration and an
  uncertainty on it, linked finish-to-start, start-to-start,
  finish-to-finish or start-to-finish, with optional lags;
* costs split by how they behave: *time-independent* (materials, a fixed-price
  subcontract) that do not care how long the work takes, and *time-dependent*
  (a team's burn rate) that grow with every month the activity runs;
* a *standing army*, the project-level staff (management, systems
  engineering, mission assurance) paid for as long as the project lasts;
* discrete risks, each with a probability, a delay to the activities it hits
  and a cost if it happens.

**What it gives back.** One (finish, cost) pair per simulated project. From the
cloud of them: the joint confidence of any budget and date; the curve of
budget and date pairs that reach a chosen joint confidence (the 70% line on
every JCL chart); the joint confidence of the point estimate, which is usually
sobering; and how often each activity sits on the critical path, which is
where management attention should go.

Two things the cloud shows that separate cost and schedule S-curves hide.
Joint confidence is never higher than either marginal one, so a budget at the
cost P70 and a date at the schedule P70 together are well under 70% jointly.
And parallel paths that merge finish later on average than any one of them
does (merge bias), so a deterministic critical path is optimistic by
construction.

Time is in months from project start; money is in whatever units the inputs
use, and nothing is converted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from cost_core.monte_carlo import _gaussian_copula, make_distribution, validate_correlation


class ScheduleError(ValueError):
    """Raised on a network or analysis that cannot be built or evaluated."""


#: The four ways one activity can hold up another, as scheduling tools name
#: them: finish-to-start (the successor starts after the predecessor
#: finishes, the usual one), start-to-start, finish-to-finish and
#: start-to-finish.
LINK_TYPES = ("FS", "SS", "FF", "SF")

Predecessor = Union[str, Tuple[str, float], Tuple[str, float, str], Mapping[str, Any]]


class Link(NamedTuple):
    """One predecessor relationship: ``pred`` holds up the activity by
    ``type`` with ``lag`` months (negative for a lead)."""

    pred: str
    lag: float = 0.0
    type: str = "FS"


def _link(p: Predecessor) -> Link:
    if isinstance(p, str):
        return Link(p)
    if isinstance(p, Mapping):
        pred, lag, kind = p["id"], p.get("lag", 0.0), p.get("type", "FS")
    else:
        parts = tuple(p)
        pred = parts[0]
        lag = parts[1] if len(parts) > 1 else 0.0
        kind = parts[2] if len(parts) > 2 else "FS"
    kind = str(kind).upper()
    if kind not in LINK_TYPES:
        raise ScheduleError(f"Link type {kind!r} from {pred!r}: use one of {', '.join(LINK_TYPES)}.")
    return Link(str(pred), float(lag), kind)


# ------------------------------------------------------------------- inputs ---
@dataclass(frozen=True)
class Activity:
    """One piece of work in the network.

    Attributes:
        id: Unique label, used by predecessors and risks.
        duration: Most-likely duration in months.
        duration_uncertainty: Spec for
            :func:`cost_core.monte_carlo.make_distribution` describing a
            multiplicative factor on the duration (1.0 is the most likely).
            None means the duration is certain.
        predecessors: Activities that hold this one up: an id (finish-to-
            start), ``(id, lag)`` for a lag in months (negative for a lead),
            ``(id, lag, type)`` with ``type`` one of :data:`LINK_TYPES`, or
            a mapping with keys ``id``, ``lag`` and ``type``.
        fixed_cost: Time-independent cost.
        fixed_cost_uncertainty: Factor spec on the fixed cost.
        burn_rate: Time-dependent cost per month while the activity runs.
        name: Longer label for reports.
    """

    id: str
    duration: float
    duration_uncertainty: Optional[Dict[str, Any]] = None
    predecessors: Sequence[Predecessor] = ()
    fixed_cost: float = 0.0
    fixed_cost_uncertainty: Optional[Dict[str, Any]] = None
    burn_rate: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ScheduleError(f"{self.id}: duration must be >= 0; got {self.duration}.")
        if self.burn_rate < 0 or self.fixed_cost < 0:
            raise ScheduleError(f"{self.id}: costs must be >= 0.")
        self.relations()  # raises on an unknown link type

    def relations(self) -> List[Link]:
        """Every predecessor as a :class:`Link`, with its type."""
        return [_link(p) for p in self.predecessors]

    def links(self) -> List[Tuple[str, float]]:
        """``(id, lag)`` for every predecessor, without the link type.

        Kept for code written against 2.2.0, when every link was
        finish-to-start; :meth:`relations` has the type as well.
        """
        return [(r.pred, r.lag) for r in self.relations()]


@dataclass(frozen=True)
class Risk:
    """A discrete risk: it happens or it does not.

    Attributes:
        name: Label.
        probability: Chance it happens, in [0, 1].
        activities: Ids of the activities it delays; the same delay is added
            to each when it happens.
        delay: Months added: a number, or a spec for
            :func:`cost_core.monte_carlo.make_distribution` for the delay itself.
        cost: Cost added when it happens (on top of what the delay costs
            through burn rates and the standing army).
    """

    name: str
    probability: float
    activities: Sequence[str] = ()
    delay: Union[float, Dict[str, Any]] = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ScheduleError(f"{self.name}: probability must be in [0, 1].")


@dataclass(frozen=True)
class Project:
    """The whole network.

    Attributes:
        activities: The work, in any order.
        risks: Discrete risks.
        standing_army: Cost per month for as long as the project runs.
        duration_correlation: Common correlation between the activities'
            duration factors. Schedules slip together for shared reasons (the
            same team, the same test facility, the same late requirement), so
            independence understates the spread; see the monte_carlo module.
        cost_correlation: Common correlation between the fixed-cost factors.
        cross_correlation: Correlation between each duration factor and each
            fixed-cost factor. Zero by default: a long activity is not
            necessarily an expensive one in its time-independent cost.
        name: Label for reports.
    """

    activities: Sequence[Activity]
    risks: Sequence[Risk] = ()
    standing_army: float = 0.0
    duration_correlation: float = 0.5
    cost_correlation: float = 0.3
    cross_correlation: float = 0.0
    name: str = "project"

    def __post_init__(self) -> None:
        ids = [a.id for a in self.activities]
        if not ids:
            raise ScheduleError("A project needs at least one activity.")
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ScheduleError(f"Duplicate activity id(s) {sorted(dup)}.")
        known = set(ids)
        for a in self.activities:
            for p, _ in a.links():
                if p not in known:
                    raise ScheduleError(f"{a.id} follows {p!r}, which is not an activity.")
                if p == a.id:
                    raise ScheduleError(f"{a.id} cannot follow itself.")
        for r in self.risks:
            unknown = sorted(set(r.activities) - known)
            if unknown:
                raise ScheduleError(f"Risk {r.name!r} names unknown activity(ies) {unknown}.")
        self.order()  # raises on a cycle

    def order(self) -> List[str]:
        """Activity ids in an order where every predecessor comes first."""
        preds = {a.id: {p for p, _ in a.links()} for a in self.activities}
        done: List[str] = []
        placed = set()
        remaining = [a.id for a in self.activities]
        while remaining:
            ready = [i for i in remaining if preds[i] <= placed]
            if not ready:
                raise ScheduleError(f"The network has a cycle among {sorted(remaining)}.")
            for i in ready:
                done.append(i)
                placed.add(i)
            remaining = [i for i in remaining if i not in placed]
        return done


# ------------------------------------------------------------ critical path ---
def _earliest_start(link: Link, start_p, finish_p, duration):
    """The earliest start a link allows its successor, given the
    predecessor's start and finish and the successor's own duration.
    Works on floats and on arrays of iterations alike."""
    if link.type == "FS":
        return finish_p + link.lag
    if link.type == "SS":
        return start_p + link.lag
    if link.type == "FF":
        return finish_p + link.lag - duration
    return start_p + link.lag - duration  # SF


def _latest_finish(link: Link, late_start_s, late_finish_s, duration_p):
    """The latest finish a link allows its predecessor, given the
    successor's late dates and the predecessor's own duration."""
    if link.type == "FS":
        return late_start_s - link.lag
    if link.type == "SS":
        return late_start_s - link.lag + duration_p
    if link.type == "FF":
        return late_finish_s - link.lag
    return late_finish_s - link.lag + duration_p  # SF


def critical_path(project: Project) -> pd.DataFrame:
    """The deterministic schedule on most-likely durations (CPM).

    Returns one row per activity in network order: early and late start and
    finish, total float, and whether it is critical (zero float). The
    project's deterministic finish is the largest early finish. No activity
    starts before the project does, whatever its links allow: a finish-to-
    finish link to a short predecessor does not pull a long successor's
    start before month zero.
    """
    acts = {a.id: a for a in project.activities}
    order = project.order()
    es, ef = {}, {}
    for i in order:
        d = acts[i].duration
        es[i] = max([0.0] + [_earliest_start(r, es[r.pred], ef[r.pred], d)
                             for r in acts[i].relations()])
        ef[i] = es[i] + d
    finish = max(ef.values())
    succs: Dict[str, List[Tuple[str, Link]]] = {i: [] for i in order}
    for i in order:
        for r in acts[i].relations():
            succs[r.pred].append((i, r))
    lf, ls = {}, {}
    for i in reversed(order):
        d = acts[i].duration
        lf[i] = min([finish] + [_latest_finish(r, ls[s], lf[s], d) for s, r in succs[i]])
        ls[i] = lf[i] - d
    rows = [{"activity": i, "name": acts[i].name or i, "duration": acts[i].duration,
             "early_start": es[i], "early_finish": ef[i], "late_start": ls[i],
             "late_finish": lf[i], "total_float": ls[i] - es[i],
             "critical": abs(ls[i] - es[i]) < 1e-9} for i in order]
    return pd.DataFrame(rows)


def point_estimate(project: Project) -> Tuple[float, float]:
    """Finish and cost on most-likely durations, point costs and no risks."""
    cpm = critical_path(project)
    finish = float(cpm["early_finish"].max())
    cost = sum(a.fixed_cost + a.burn_rate * a.duration for a in project.activities)
    return finish, float(cost + project.standing_army * finish)


# -------------------------------------------------------------- simulation ---
@dataclass
class JclResult:
    """A simulated project: one finish and one cost per iteration.

    Attributes:
        finish: Months from start, per iteration.
        cost: Total cost, per iteration.
        durations: Activity durations actually used, (n_iter, activities).
        critical: Whether each activity was on the critical path, same shape.
        activity_ids: Column labels for ``durations`` and ``critical``.
        point_finish, point_cost: The deterministic point estimate.
        notes: Assumptions and repairs made, for the record.
    """

    finish: np.ndarray
    cost: np.ndarray
    durations: np.ndarray
    critical: np.ndarray
    activity_ids: List[str]
    point_finish: float
    point_cost: float
    risk_hits: Dict[str, np.ndarray] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def n_iter(self) -> int:
        return int(self.finish.size)

    # ----------------------------------------------------------- queries
    def joint(self, cost: float, finish: float) -> float:
        """Probability of finishing at or under ``cost`` and by ``finish``."""
        return float(np.mean((self.cost <= cost) & (self.finish <= finish)))

    @property
    def point_jcl(self) -> float:
        """Joint confidence of the deterministic point estimate."""
        return self.joint(self.point_cost, self.point_finish)

    def cost_at(self, confidence: float, finish: float) -> float:
        """Least budget reaching ``confidence`` jointly with date ``finish``.

        Returns NaN when no budget can, because too few iterations finish by
        then at all.
        """
        _check_conf(confidence)
        k = math.ceil(confidence * self.n_iter)
        on_time = np.sort(self.cost[self.finish <= finish])
        return float(on_time[k - 1]) if on_time.size >= k else float("nan")

    def finish_at(self, confidence: float, cost: float) -> float:
        """Earliest date reaching ``confidence`` jointly with budget ``cost``."""
        _check_conf(confidence)
        k = math.ceil(confidence * self.n_iter)
        in_budget = np.sort(self.finish[self.cost <= cost])
        return float(in_budget[k - 1]) if in_budget.size >= k else float("nan")

    def frontier(self, confidence: float = 0.7, points: int = 30) -> pd.DataFrame:
        """Budget and date pairs that each reach ``confidence`` jointly.

        The curve drawn on a JCL chart. Every pair on it has joint confidence
        of at least ``confidence``; moving along it trades months for money.
        It starts at the earliest date that can reach the confidence at all
        (the schedule's own ``confidence`` percentile) and runs to the latest
        simulated finish, where the budget needed is the cost percentile.
        """
        _check_conf(confidence)
        lo = float(np.quantile(self.finish, confidence, method="higher"))
        hi = float(self.finish.max())
        dates = np.unique(np.linspace(lo, hi, points))
        rows = [{"finish": t, "cost": self.cost_at(confidence, t)} for t in dates]
        df = pd.DataFrame(rows).dropna()
        df["joint"] = [self.joint(c, t) for c, t in zip(df["cost"], df["finish"])]
        return df.reset_index(drop=True)

    def summary(self, confidence: float = 0.7) -> pd.DataFrame:
        """The numbers a JCL briefing opens with."""
        cost_p = float(np.quantile(self.cost, confidence))
        fin_p = float(np.quantile(self.finish, confidence))
        rows = [
            ("point estimate: finish (months)", self.point_finish),
            ("point estimate: cost", self.point_cost),
            ("point estimate: joint confidence", self.point_jcl),
            (f"schedule P{confidence * 100:.0f} (months)", fin_p),
            (f"cost P{confidence * 100:.0f}", cost_p),
            (f"joint confidence of those two together", self.joint(cost_p, fin_p)),
            (f"budget for P{confidence * 100:.0f} JCL at the schedule P{confidence * 100:.0f} + 3 months",
             self.cost_at(confidence, fin_p + 3)),
            ("mean finish (months)", float(self.finish.mean())),
            ("mean cost", float(self.cost.mean())),
        ]
        return pd.DataFrame(rows, columns=["measure", "value"])

    def criticality(self) -> pd.DataFrame:
        """How often each activity was critical, and how its duration moved
        the finish and the cost (Spearman rank correlation)."""
        rows = []
        for j, a in enumerate(self.activity_ids):
            d = self.durations[:, j]
            varies = np.ptp(d) > 0
            rows.append({
                "activity": a,
                "criticality": float(self.critical[:, j].mean()),
                "duration_mean": float(d.mean()),
                "rank_corr_finish": float(stats.spearmanr(d, self.finish)[0]) if varies else 0.0,
                "rank_corr_cost": float(stats.spearmanr(d, self.cost)[0]) if varies else 0.0,
            })
        return pd.DataFrame(rows).sort_values("criticality", ascending=False).reset_index(drop=True)


def _check_conf(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ScheduleError(f"confidence must be strictly between 0 and 1; got {confidence}.")


def _block_correlation(kd: int, kc: int, rd: float, rc: float, rx: float) -> np.ndarray:
    m = np.zeros((kd + kc, kd + kc))
    m[:kd, :kd] = rd
    m[kd:, kd:] = rc
    m[:kd, kd:] = rx
    m[kd:, :kd] = rx
    np.fill_diagonal(m, 1.0)
    return m


def simulate(project: Project, n_iter: int = 20_000, seed: Optional[int] = 0) -> JclResult:
    """Simulate the project: durations, risks, the network and the cost.

    Duration and fixed-cost factors are drawn together through a Gaussian
    copula with the project's block correlation (repaired to the nearest valid
    matrix, with a note, if the three settings are jointly impossible). Each
    risk occurs independently with its probability; when it does, its delay is
    added to every activity it names and its cost to the total.
    """
    if n_iter < 2:
        raise ScheduleError(f"Need at least 2 iterations; got {n_iter}.")
    rng = np.random.default_rng(seed)
    order = project.order()
    acts = {a.id: a for a in project.activities}
    col = {a: j for j, a in enumerate(order)}
    notes: List[str] = []

    dur_ids = [a for a in order if acts[a].duration_uncertainty is not None]
    cost_ids = [a for a in order if acts[a].fixed_cost_uncertainty is not None
                and acts[a].fixed_cost > 0]
    marginals = ([make_distribution(acts[a].duration_uncertainty) for a in dur_ids]
                 + [make_distribution(acts[a].fixed_cost_uncertainty) for a in cost_ids])
    factors = np.ones((n_iter, 0))
    if marginals:
        corr = _block_correlation(len(dur_ids), len(cost_ids), project.duration_correlation,
                                  project.cost_correlation, project.cross_correlation)
        corr, repair = validate_correlation(corr)
        notes.extend(repair)
        factors = _gaussian_copula(marginals, corr, n_iter, rng)

    durations = np.tile([acts[a].duration for a in order], (n_iter, 1)).astype(float)
    for k, a in enumerate(dur_ids):
        durations[:, col[a]] *= factors[:, k]
    if (durations < 0).any():
        notes.append("Some sampled durations were negative and were floored at zero; "
                     "check the duration uncertainty specs.")
        durations = np.maximum(durations, 0.0)

    fixed = np.tile([acts[a].fixed_cost for a in order], (n_iter, 1)).astype(float)
    for k, a in enumerate(cost_ids):
        fixed[:, col[a]] *= factors[:, len(dur_ids) + k]

    risk_cost = np.zeros(n_iter)
    hits: Dict[str, np.ndarray] = {}
    for r in project.risks:
        occurs = rng.random(n_iter) < r.probability
        hits[r.name] = occurs
        if isinstance(r.delay, dict):
            delay = np.asarray(make_distribution(r.delay).ppf(rng.random(n_iter)), dtype=float)
        else:
            delay = np.full(n_iter, float(r.delay))
        for a in r.activities:
            durations[:, col[a]] += np.where(occurs, delay, 0.0)
        risk_cost += np.where(occurs, r.cost, 0.0)

    start = np.zeros_like(durations)
    fin = np.zeros_like(durations)
    for a in order:
        j = col[a]
        for r in acts[a].relations():
            k = col[r.pred]
            np.maximum(start[:, j], _earliest_start(r, start[:, k], fin[:, k], durations[:, j]),
                       out=start[:, j])
        fin[:, j] = start[:, j] + durations[:, j]
    finish = fin.max(axis=1)

    # Critical in an iteration: ends the project, or drives a successor that
    # is itself critical, meaning the link from it is the one that set the
    # successor's start.
    succs: Dict[str, List[Tuple[str, Link]]] = {a: [] for a in order}
    for a in order:
        for r in acts[a].relations():
            succs[r.pred].append((a, r))
    crit = np.zeros_like(durations, dtype=bool)
    tol = 1e-9 * max(1.0, float(finish.max()))
    for a in reversed(order):
        j = col[a]
        c = np.isclose(fin[:, j], finish, atol=tol, rtol=0)
        for s, r in succs[a]:
            k = col[s]
            drives = np.isclose(start[:, k],
                                _earliest_start(r, start[:, j], fin[:, j], durations[:, k]),
                                atol=tol, rtol=0)
            c |= crit[:, k] & drives
        crit[:, j] = c

    rates = np.array([acts[a].burn_rate for a in order], dtype=float)
    cost = fixed.sum(axis=1) + durations @ rates + project.standing_army * finish + risk_cost
    pf, pc = point_estimate(project)
    return JclResult(finish=finish, cost=cost, durations=durations, critical=crit,
                     activity_ids=list(order), point_finish=pf, point_cost=pc,
                     risk_hits=hits, notes=notes, seed=seed)
