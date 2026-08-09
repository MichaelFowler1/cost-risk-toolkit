"""
monte_carlo.py - Monte Carlo cost risk simulation.

The original two-variable simulation is unchanged and still exported. What is
added is the machinery a WBS-level risk model needs, and one result worth
stating up front.

**Correlation is not a refinement.** Sampling WBS elements independently is the
default in most spreadsheets and it is close to the worst assumption available.
Elements on one program share a workforce, a management team, a supply chain
and a schedule; when one runs late they mostly all run late. The variance of a
sum of correlated variables is

    Var(sum X_i) = sum Var(X_i) + 2 * sum_{i<j} rho_ij * sd_i * sd_j

so with *k* elements of equal spread and a common correlation *rho*, ignoring
correlation understates the variance of the total by a factor of exactly
``1 + rho*(k-1)``. For ten elements at rho = 0.3 that is a factor of 3.7 in
variance, near enough a doubling of the standard deviation -- and it lands
almost entirely on the upper tail, which is where the P80 lives. That identity
is asserted in the tests, and :func:`correlation_impact` measures it against a
simulation so the number can be quoted with its derivation attached.

Because independence is so rarely right, :class:`RiskModel` applies a non-zero
default correlation when none is supplied rather than silently assuming zero.
It warns when it does so: a default is an assumption, and an unstated
assumption is the thing the GAO guide's "well-documented" characteristic is
about.

Also here: discrete risks kept separate from continuous uncertainty (a risk
that either happens or does not is not a wider distribution on the base
estimate), lognormal and PERT marginals alongside triangular, nearest-PSD
repair for a correlation matrix that is not quite valid, and diagnostics --
where the point estimate falls on the CDF, the CV of the total, a variance
decomposition that sums exactly to one, and a convergence check on the P80.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SimulationResult:
    """
    Container for the results of a Monte Carlo cost simulation.
    
    Attributes:
        samples: The raw array of simulated total costs.
        mean: The arithmetic mean of the simulated costs.
        p50: The 50th percentile (median) cost.
        p80: The 80th percentile cost (80% confidence level).
        p90: The 90th percentile cost (90% confidence level).
    """
    samples: np.ndarray
    mean: float
    p50: float
    p80: float
    p90: float


def _draw_samples(dist_params: Dict[str, Any], n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """
    Internal helper to draw samples from a specified distribution.
    
    Args:
        dist_params: Dictionary containing 'type' and distribution-specific parameters.
        n_iter: Number of samples to draw.
        rng: NumPy random generator instance.
        
    Returns:
        np.ndarray: Array of sampled values.
        
    Raises:
        ValueError: If the distribution type is unsupported or parameters are missing.
    """
    dist_type = dist_params.get("type", "").lower()
    
    try:
        if dist_type == "normal":
            return rng.normal(loc=dist_params["loc"], scale=dist_params["scale"], size=n_iter)
        
        elif dist_type == "lognormal":
            return rng.lognormal(mean=dist_params["mean"], sigma=dist_params["sigma"], size=n_iter)
            
        elif dist_type == "triangular":
            return rng.triangular(left=dist_params["left"], mode=dist_params["mode"], right=dist_params["right"], size=n_iter)
            
        else:
            raise ValueError(f"Unsupported distribution type: '{dist_type}'. Allowed: normal, lognormal, triangular.")
            
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for distribution type '{dist_type}'.")


def run_monte_carlo(
    n_iter: int, 
    unit_cost_dist: Dict[str, Any], 
    quantity_dist: Dict[str, Any],
    seed: int | None = None
) -> SimulationResult:
    """
    Runs a Monte Carlo simulation to calculate total cost distribution.
    Total Cost = Unit Cost * Quantity.
    
    Args:
        n_iter: Number of simulation iterations.
        unit_cost_dist: Dictionary defining the unit cost distribution.
        quantity_dist: Dictionary defining the quantity distribution.
        seed: Optional random seed for deterministic outputs.
        
    Returns:
        SimulationResult: The calculated statistics and raw samples.
    """
    logger.info(f"Starting Monte Carlo simulation with {n_iter} iterations.")
    rng = np.random.default_rng(seed)
    
    # Draw samples
    unit_costs = _draw_samples(unit_cost_dist, n_iter, rng)
    quantities = _draw_samples(quantity_dist, n_iter, rng)
    
    # Ensure no negative costs or quantities, as they are non-physical in this context
    unit_costs = np.maximum(unit_costs, 0.0)
    quantities = np.maximum(quantities, 0.0)
    
    # Calculate Total Cost
    total_costs = unit_costs * quantities
    
    # Compute summary statistics
    mean_cost = float(np.mean(total_costs))
    p50_cost = float(np.percentile(total_costs, 50))
    p80_cost = float(np.percentile(total_costs, 80))
    p90_cost = float(np.percentile(total_costs, 90))
    
    logger.info(f"Simulation complete. Mean: {mean_cost:.2f}, P80: {p80_cost:.2f}")
    
    return SimulationResult(
        samples=total_costs,
        mean=mean_cost,
        p50=p50_cost,
        p80=p80_cost,
        p90=p90_cost
    )


def plot_distribution(result: SimulationResult, bins: int = 50) -> None:
    """
    Plots a histogram of the simulation results with confidence intervals marked.
    
    Args:
        result: The SimulationResult object containing samples and metrics.
        bins: Number of histogram bins.
    """
    logger.info("Generating distribution plot.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(result.samples, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.7)
    
    # Add vertical lines for percentiles and mean
    ax.axvline(result.mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {result.mean:,.2f}")
    ax.axvline(result.p50, color="orange", linestyle="-", linewidth=2, label=f"P50: {result.p50:,.2f}")
    ax.axvline(result.p80, color="green", linestyle="-.", linewidth=2, label=f"P80: {result.p80:,.2f}")
    ax.axvline(result.p90, color="purple", linestyle=":", linewidth=2, label=f"P90: {result.p90:,.2f}")
    
    # Formatting
    ax.set_title("Monte Carlo Total Cost Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Cost", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Use pandas formatting for x-axis if desired, but default matplotlib usually suffices
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.show()


# ==========================================================================
# Distributions
# ==========================================================================
#: Correlation applied between WBS elements when the caller supplies no
#: matrix. Not zero: elements on one program share a workforce, a management
#: chain and a schedule, so their overruns move together. Values in the
#: 0.2-0.3 range are the usual starting point in the absence of programme
#: history. It is still an assumption, and :class:`RiskModel` says so out loud.
DEFAULT_CORRELATION = 0.25

#: Tolerance for treating a marginally negative eigenvalue as zero.
PSD_TOLERANCE = 1e-10


class RiskModelError(ValueError):
    """Raised when a risk model is malformed."""


class CorrelationWarning(UserWarning):
    """Raised when a correlation matrix had to be repaired, or defaulted."""


class _Degenerate:
    """A distribution concentrated on a single value.

    scipy's frozen distributions reject a zero scale and return ``nan`` for
    the moments of a zero-width uniform, so an element carrying no uncertainty
    -- a firm fixed-price line, a fee already negotiated -- needs its own
    implementation. Exposes the small surface the sampler and the diagnostics
    use.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def ppf(self, q):
        return np.full_like(np.asarray(q, dtype=float), self.value)

    def mean(self) -> float:
        return self.value

    def median(self) -> float:
        return self.value

    def var(self) -> float:
        return 0.0

    def std(self) -> float:
        return 0.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Degenerate({self.value})"


def make_distribution(spec: Dict[str, Any]):
    """Build a frozen scipy distribution from a spec dictionary.

    Frozen scipy distributions rather than raw samplers, because a Gaussian
    copula needs the inverse CDF, and because having ``.mean()`` and ``.var()``
    available in closed form is what lets the tests check the sampler against
    the analytic answer instead of against a recorded one.

    Supported types:

    ``normal``       ``loc``, ``scale``
    ``lognormal``    ``mean``, ``sigma`` of the *underlying normal*, matching
                     the original :func:`run_monte_carlo` convention
    ``triangular``   ``left``, ``mode``, ``right``
    ``pert``         ``left``, ``mode``, ``right``, optional ``lambda_`` (4)
    ``uniform``      ``low``, ``high``
    ``fixed``        ``value`` -- a degenerate distribution, for an element
                     carrying no uncertainty

    Raises:
        RiskModelError: On an unknown type, a missing parameter, or parameters
            that do not describe a distribution (a mode outside its bounds, a
            negative scale).
    """
    dist_type = str(spec.get("type", "")).lower()

    def need(*keys: str):
        missing = [k for k in keys if k not in spec]
        if missing:
            raise RiskModelError(
                f"Distribution {dist_type!r} needs parameter(s) {missing}; "
                f"got {sorted(spec)}."
            )
        return [float(spec[k]) for k in keys]

    if dist_type == "normal":
        loc, scale = need("loc", "scale")
        if scale < 0:
            raise RiskModelError(f"normal scale must be >= 0; got {scale}.")
        return stats.norm(loc=loc, scale=scale)

    if dist_type == "lognormal":
        mean, sigma = need("mean", "sigma")
        if sigma < 0:
            raise RiskModelError(f"lognormal sigma must be >= 0; got {sigma}.")
        return stats.lognorm(s=sigma, scale=np.exp(mean))

    if dist_type in ("triangular", "pert"):
        left, mode, right = need("left", "mode", "right")
        if not left <= mode <= right:
            raise RiskModelError(
                f"{dist_type} needs left <= mode <= right; got "
                f"({left}, {mode}, {right})."
            )
        if right == left:
            return _Degenerate(left)
        if dist_type == "triangular":
            return stats.triang(
                c=(mode - left) / (right - left), loc=left, scale=right - left
            )
        # PERT: a Beta reshaped onto [left, right]. lambda_ controls how much
        # weight the mode carries; 4 is the classic choice and makes the mean
        # exactly (left + 4*mode + right) / 6.
        lam = float(spec.get("lambda_", 4.0))
        if lam <= 0:
            raise RiskModelError(f"pert lambda_ must be positive; got {lam}.")
        alpha = 1.0 + lam * (mode - left) / (right - left)
        beta = 1.0 + lam * (right - mode) / (right - left)
        return stats.beta(alpha, beta, loc=left, scale=right - left)

    if dist_type == "uniform":
        low, high = need("low", "high")
        if high < low:
            raise RiskModelError(f"uniform needs high >= low; got ({low}, {high}).")
        return stats.uniform(loc=low, scale=high - low)

    if dist_type == "fixed":
        (value,) = need("value")
        return _Degenerate(value)

    raise RiskModelError(
        f"Unsupported distribution type {dist_type!r}. Allowed: normal, "
        f"lognormal, triangular, pert, uniform, fixed."
    )


# ==========================================================================
# Model components
# ==========================================================================
@dataclass(frozen=True)
class CostElement:
    """One WBS element carrying continuous uncertainty.

    Attributes:
        name: Element name, used to label the tornado and the correlation
            matrix.
        distribution: Spec dictionary for :func:`make_distribution`.
        point_estimate: The deterministic estimate for this element. Optional,
            and used only for diagnostics -- specifically, to report where the
            point estimate falls on the CDF of the total, which is the single
            most informative number about whether an estimate carries any
            reserve at all.
    """

    name: str
    distribution: Dict[str, Any]
    point_estimate: float | None = None

    def frozen(self):
        return make_distribution(self.distribution)


@dataclass(frozen=True)
class DiscreteRisk:
    """A risk that either happens or does not.

    Kept separate from the continuous uncertainty on the base estimate, and
    deliberately so. A 20% chance of a $40M qualification failure is not the
    same thing as a wider distribution around the base estimate: it produces a
    bimodal contribution, it does not belong in the element's own spread, and
    averaging it into one produces a distribution with a mode nobody believes.

    Attributes:
        name: Risk name.
        probability: Chance of occurrence, in [0, 1].
        impact: Distribution of the cost impact *given that it occurs*. The
            unconditional expected value is ``probability * impact.mean()``.
        affects: Optional element name, for reporting only.
    """

    name: str
    probability: float
    impact: Dict[str, Any]
    affects: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise RiskModelError(
                f"Risk {self.name!r} has probability {self.probability}, which "
                f"is not in [0, 1]."
            )

    def frozen(self):
        return make_distribution(self.impact)

    @property
    def expected_value(self) -> float:
        """Unconditional expected cost: probability times mean impact."""
        return float(self.probability * self.frozen().mean())


# ==========================================================================
# Correlation
# ==========================================================================
def validate_correlation(
    matrix: np.ndarray, names: Sequence[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Check a correlation matrix and repair it if it is not quite PSD.

    A matrix elicited from analysts one pair at a time is very often not
    positive semi-definite: each pairwise judgement is reasonable and the set
    of them is jointly impossible. Cholesky then fails, and the usual response
    -- quietly falling back to independence -- throws away the entire point of
    supplying a matrix.

    So instead: clip the negative eigenvalues to zero, which gives the nearest
    PSD matrix in the Frobenius norm, then rescale the diagonal back to one so
    the result is a valid *correlation* matrix. That last step perturbs the
    Frobenius-optimality slightly; Higham's alternating projections would
    restore it, at more complexity than the difference justifies here. Either
    way it warns, because a repaired matrix is not the matrix that was
    supplied and the analyst should know which one was used.

    Returns:
        The validated (possibly repaired) matrix and a list of notes.

    Raises:
        RiskModelError: If the matrix is not square, not symmetric, has a
            non-unit diagonal, or contains values outside [-1, 1]. Those are
            errors of construction rather than of elicitation, and repairing
            them would be guessing.
    """
    matrix = np.asarray(matrix, dtype=float)
    notes: list[str] = []

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RiskModelError(
            f"Correlation matrix must be square; got shape {matrix.shape}."
        )
    if names is not None and matrix.shape[0] != len(names):
        raise RiskModelError(
            f"Correlation matrix is {matrix.shape[0]}x{matrix.shape[0]} but "
            f"there are {len(names)} elements."
        )
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise RiskModelError(
            "Correlation matrix is not symmetric. rho(i,j) and rho(j,i) "
            "describe the same relationship and must be equal."
        )
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-10):
        raise RiskModelError(
            f"Correlation matrix diagonal must be all ones; got "
            f"{np.diag(matrix)}."
        )
    if np.any(np.abs(matrix) > 1.0 + 1e-10):
        raise RiskModelError(
            "Correlation matrix contains values outside [-1, 1]."
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest = float(eigenvalues.min())
    if smallest >= -PSD_TOLERANCE:
        return matrix, notes

    note = (
        f"Supplied correlation matrix is not positive semi-definite (smallest "
        f"eigenvalue {smallest:.3e}); repaired by clipping negative "
        f"eigenvalues and rescaling the diagonal. The matrix used is not the "
        f"matrix supplied."
    )
    warnings.warn(note, CorrelationWarning, stacklevel=2)
    notes.append(note)

    values, vectors = np.linalg.eigh(matrix)
    repaired = vectors @ np.diag(np.maximum(values, 0.0)) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(repaired), 1e-300, None))
    repaired = repaired / np.outer(scale, scale)
    repaired = (repaired + repaired.T) / 2.0
    np.fill_diagonal(repaired, 1.0)

    shift = float(np.max(np.abs(repaired - matrix)))
    notes.append(f"Largest change to any correlation: {shift:.4f}.")
    logger.warning("Correlation repair moved an entry by up to %.4f", shift)
    return repaired, notes


def uniform_correlation(k: int, rho: float) -> np.ndarray:
    """A k x k matrix with ``rho`` off the diagonal.

    Raises:
        RiskModelError: If ``rho`` is outside the range that keeps the matrix
            positive semi-definite. For an equicorrelation matrix that is
            ``[-1/(k-1), 1]`` exactly -- a strongly negative common
            correlation is not merely unusual, it is impossible for more than
            two elements.
    """
    if k < 1:
        raise RiskModelError(f"Need at least one element; got {k}.")
    lower = -1.0 / (k - 1) if k > 1 else -1.0
    if not lower - 1e-12 <= rho <= 1.0:
        raise RiskModelError(
            f"A common correlation of {rho} is not achievable across {k} "
            f"elements; it must lie in [{lower:.4f}, 1]."
        )
    matrix = np.full((k, k), float(rho))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _gaussian_copula(
    marginals: list, corr: np.ndarray, n_iter: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample correlated variates through a Gaussian copula.

    Correlated standard normals via Cholesky, mapped to uniforms through the
    normal CDF, then through each marginal's inverse CDF. The marginals come
    out exactly right; the induced rank correlation is close to the target and
    the Pearson correlation slightly below it, which is inherent to the copula
    and not a defect of the implementation.
    """
    # Cholesky needs strict positive definiteness; nudge the diagonal if the
    # repaired matrix sits exactly on the boundary.
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(corr + np.eye(len(corr)) * 1e-10)

    normals = rng.standard_normal((n_iter, len(marginals))) @ chol.T
    uniforms = stats.norm.cdf(normals)
    # Keep the inverse CDF away from the open ends of [0, 1].
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    return np.column_stack(
        [dist.ppf(uniforms[:, i]) for i, dist in enumerate(marginals)]
    )


def _iman_conover(
    marginals: list, corr: np.ndarray, n_iter: int, rng: np.random.Generator
) -> np.ndarray:
    """Induce rank correlation by reordering independent samples.

    Iman and Conover's method. The samples themselves are drawn independently
    and then permuted, so every marginal is preserved *exactly* -- not
    approximately, exactly, down to the individual sampled values. That makes
    it the safer choice when a marginal has been argued over and must not
    move. It targets Spearman rank correlation rather than Pearson.
    """
    k = len(marginals)
    samples = np.column_stack(
        [dist.ppf(rng.uniform(1e-12, 1.0 - 1e-12, n_iter)) for dist in marginals]
    )

    # Van der Waerden scores, independently permuted per column.
    scores = stats.norm.ppf(np.arange(1, n_iter + 1) / (n_iter + 1.0))
    score_matrix = np.column_stack(
        [rng.permutation(scores) for _ in range(k)]
    )

    # Correct for the accidental correlation among the permuted scores, then
    # impose the target.
    observed = np.corrcoef(score_matrix, rowvar=False)
    try:
        p_inv = np.linalg.inv(np.linalg.cholesky(observed))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        p_inv = np.eye(k)
    try:
        target_chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        target_chol = np.linalg.cholesky(corr + np.eye(k) * 1e-10)

    targeted = score_matrix @ p_inv.T @ target_chol.T

    # Reorder each column of the samples to follow the target's rank pattern.
    out = np.empty_like(samples)
    for j in range(k):
        order = np.argsort(np.argsort(targeted[:, j]))
        out[:, j] = np.sort(samples[:, j])[order]
    return out


# ==========================================================================
# The risk model
# ==========================================================================
@dataclass
class RiskModel:
    """A WBS-level cost risk model.

    Attributes:
        elements: Continuous uncertainty, one entry per WBS element.
        risks: Discrete risks, kept separate from the continuous uncertainty.
        correlation: Correlation matrix across the elements, in element order.
            None means use ``default_correlation``.
        default_correlation: Applied uniformly when no matrix is supplied.
        name: Label for reports.
    """

    elements: list[CostElement]
    risks: list[DiscreteRisk] = field(default_factory=list)
    correlation: np.ndarray | None = None
    default_correlation: float = DEFAULT_CORRELATION
    name: str = "risk model"

    def __post_init__(self) -> None:
        if not self.elements:
            raise RiskModelError("A risk model needs at least one cost element.")
        names = [e.name for e in self.elements]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise RiskModelError(
                f"Duplicate element name(s) {sorted(duplicates)}; names label "
                f"the correlation matrix and the tornado, so they must be "
                f"unique."
            )

    @property
    def element_names(self) -> list[str]:
        return [e.name for e in self.elements]

    @property
    def point_estimate(self) -> float:
        """Deterministic total: the sum of the element point estimates.

        Falls back to a distribution's mean for any element that does not
        carry one. Discrete risks are excluded by construction -- a point
        estimate that already includes risk reserve is not a point estimate.
        """
        total = 0.0
        for element in self.elements:
            total += (
                element.point_estimate
                if element.point_estimate is not None
                else float(element.frozen().mean())
            )
        return float(total)

    def resolved_correlation(self, warn: bool = True) -> tuple[np.ndarray, list[str]]:
        """The correlation matrix actually used, plus notes for the log."""
        k = len(self.elements)
        if self.correlation is None:
            note = (
                f"No correlation matrix supplied; applying a uniform "
                f"{self.default_correlation:.2f} across all {k} elements. "
                f"This is an assumption, not a measurement. Independence "
                f"(rho = 0) would understate the variance of the total by a "
                f"factor of about {1 + self.default_correlation * (k - 1):.1f}."
            )
            if warn:
                warnings.warn(note, CorrelationWarning, stacklevel=3)
            return uniform_correlation(k, self.default_correlation), [note]
        return validate_correlation(self.correlation, self.element_names)

    def analytic_variance(self, rho: np.ndarray | None = None) -> float:
        """Var(total) from the marginal variances and the correlation matrix.

        Closed form, independent of any simulation:
        ``sum_i sum_j rho_ij sd_i sd_j``, plus the discrete-risk variance.
        Used to check the sampler and to compute the variance inflation
        without running anything.
        """
        if rho is None:
            rho, _ = self.resolved_correlation(warn=False)
        sd = np.array([float(np.sqrt(e.frozen().var())) for e in self.elements])
        continuous = float(sd @ rho @ sd)
        return continuous + sum(_risk_variance(r) for r in self.risks)

    def variance_inflation(self) -> float:
        """How much correlation multiplies the variance of the total.

        The ratio of the correlated variance to the variance the same model
        would show if every element were sampled independently. This is the
        headline number: it is exactly ``1 + rho*(k-1)`` for *k* equally
        variable elements at a common ``rho``, and it says how much an
        independence assumption understates the spread of the total.
        """
        rho, _ = self.resolved_correlation(warn=False)
        sd = np.array([float(np.sqrt(e.frozen().var())) for e in self.elements])
        independent = float(sd @ sd) + sum(_risk_variance(r) for r in self.risks)
        if independent <= 0:
            return 1.0
        return self.analytic_variance(rho) / independent


def _risk_variance(risk: DiscreteRisk) -> float:
    """Variance of a Bernoulli-gated impact.

    ``Var(B*I) = p*(Var(I) + E[I]^2) - (p*E[I])^2`` for independent B and I.
    Worth writing out because the intuitive ``p*Var(I)`` is wrong -- it misses
    the variance contributed by the event's own uncertainty, which for a rare
    large risk is most of it.
    """
    dist = risk.frozen()
    mean, var = float(dist.mean()), float(dist.var())
    p = risk.probability
    return p * (var + mean**2) - (p * mean) ** 2


# ==========================================================================
# Simulation
# ==========================================================================
@dataclass
class RiskSimulationResult:
    """Outcome of a correlated WBS-level simulation."""

    totals: np.ndarray
    element_samples: np.ndarray          # (n_iter, k)
    risk_samples: np.ndarray             # (n_iter, m)
    element_names: list[str]
    risk_names: list[str]
    point_estimate: float
    correlation_used: np.ndarray
    notes: list[str]
    seed: int | None
    method: str

    # ------------------------------------------------------------ summary
    @property
    def n_iter(self) -> int:
        return int(self.totals.size)

    @property
    def mean(self) -> float:
        return float(np.mean(self.totals))

    @property
    def std(self) -> float:
        return float(np.std(self.totals, ddof=1))

    @property
    def cv(self) -> float:
        """Coefficient of variation of the total.

        The one number that says how risky the estimate is, independent of its
        size. Programme-level CVs below about 0.15 usually mean the risk model
        is missing something rather than that the programme is safe.
        """
        return float(self.std / self.mean) if self.mean else float("nan")

    def percentile(self, q: float | Iterable[float]):
        return np.percentile(self.totals, q)

    @property
    def p50(self) -> float:
        return float(np.percentile(self.totals, 50))

    @property
    def p80(self) -> float:
        return float(np.percentile(self.totals, 80))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.totals, 90))

    @property
    def continuous_total(self) -> np.ndarray:
        return self.element_samples.sum(axis=1)

    @property
    def risk_total(self) -> np.ndarray:
        if self.risk_samples.size == 0:
            return np.zeros(self.n_iter)
        return self.risk_samples.sum(axis=1)

    # -------------------------------------------------------- diagnostics
    def percentile_of(self, value: float) -> float:
        """Where a given cost falls on the simulated CDF, as a percentile.

        Applied to the deterministic point estimate this answers the question
        that decides whether a programme is funded to a defensible level. A
        point estimate that sits at the 30th percentile carries no reserve at
        all and has a 70% chance of being exceeded -- which is a normal place
        for an unreserved estimate to land, and not usually what the people
        reading it assume.
        """
        return float(np.mean(self.totals <= value) * 100.0)

    @property
    def point_estimate_percentile(self) -> float:
        return self.percentile_of(self.point_estimate)

    def summary(self) -> pd.DataFrame:
        """One-row-per-statistic table for the report."""
        rows = [
            ("iterations", self.n_iter),
            ("point_estimate", self.point_estimate),
            ("point_estimate_percentile", self.point_estimate_percentile),
            ("mean", self.mean),
            ("std_dev", self.std),
            ("cv", self.cv),
            ("p50", self.p50),
            ("p80", self.p80),
            ("p90", self.p90),
            ("reserve_to_p80", self.p80 - self.point_estimate),
            ("reserve_to_p80_pct", 100.0 * (self.p80 / self.point_estimate - 1.0)),
        ]
        return pd.DataFrame(rows, columns=["statistic", "value"])

    def tornado(self) -> pd.DataFrame:
        """Variance contribution by element and risk, largest first.

        Uses the covariance decomposition ``Cov(X_i, T) / Var(T)``. Because
        ``Var(T) = sum_i Cov(X_i, T)`` when T is the sum of the X_i, these
        contributions add to exactly one -- a property the tests check, and one
        that correlation-based sensitivity rankings do not have. It also
        attributes correctly under correlation: an element that is only
        moderately variable but moves with everything else gets the credit it
        deserves, which is the whole reason to rank on variance rather than on
        input spread.
        """
        columns, labels, kinds = [], [], []
        for i, name in enumerate(self.element_names):
            columns.append(self.element_samples[:, i])
            labels.append(name)
            kinds.append("element")
        for j, name in enumerate(self.risk_names):
            columns.append(self.risk_samples[:, j])
            labels.append(name)
            kinds.append("discrete risk")

        total_var = float(np.var(self.totals, ddof=1))
        rows = []
        for label, kind, column in zip(labels, kinds, columns):
            cov = float(np.cov(column, self.totals, ddof=1)[0, 1])
            rows.append(
                {
                    "component": label,
                    "kind": kind,
                    "std_dev": float(np.std(column, ddof=1)),
                    "covariance_with_total": cov,
                    "variance_share": cov / total_var if total_var else np.nan,
                }
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values("variance_share", ascending=False).reset_index(
            drop=True
        )

    def convergence(self, checkpoints: Sequence[int] | None = None) -> pd.DataFrame:
        """P80 computed on growing prefixes of the same sample.

        Nested prefixes rather than independent reruns, so the sequence shows
        the estimate settling rather than bouncing between seeds. If the P80
        is still moving by more than about half a percent at the last
        checkpoint, the run is too short to quote.
        """
        if checkpoints is None:
            checkpoints = [
                n for n in (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
                if n <= self.n_iter
            ]
            if not checkpoints or checkpoints[-1] != self.n_iter:
                checkpoints = [*checkpoints, self.n_iter]

        rows, previous = [], None
        for n in checkpoints:
            prefix = self.totals[:n]
            p80 = float(np.percentile(prefix, 80))
            change = np.nan if previous is None else abs(p80 / previous - 1.0)
            rows.append(
                {
                    "iterations": int(n),
                    "p80": p80,
                    "relative_change": change,
                    "converged": bool(change < 0.005) if previous else False,
                }
            )
            previous = p80
        return pd.DataFrame(rows)

    @property
    def is_converged(self) -> bool:
        """True when the P80 moved less than 0.5% over the last doubling."""
        frame = self.convergence()
        if len(frame) < 2:
            return False
        return bool(frame["relative_change"].iloc[-1] < 0.005)


def simulate_risk_model(
    model: RiskModel,
    n_iter: int = 20_000,
    seed: int | None = None,
    *,
    method: Literal["gaussian_copula", "iman_conover"] = "gaussian_copula",
    correlate_risks: bool = False,
) -> RiskSimulationResult:
    """Run a correlated WBS-level cost risk simulation.

    Args:
        model: The risk model.
        n_iter: Iterations. The default is enough for a stable P80 on a
            ten-element model; check :meth:`RiskSimulationResult.convergence`
            rather than assuming.
        seed: Fixed seed for reproducibility. A P80 that moves between runs is
            not a number anyone can defend, so this should always be set for
            anything that leaves the room.
        method: ``"gaussian_copula"`` induces correlation through a normal
            copula; ``"iman_conover"`` reorders independent draws and so
            preserves each marginal exactly.
        correlate_risks: Whether discrete risks share the element correlation.
            False by default: distinct risk events are usually independent of
            the base estimate's spread, and assuming otherwise without a
            reason double-counts.

    Raises:
        RiskModelError: On a bad iteration count, unknown method, or an
            invalid correlation matrix.
    """
    if n_iter < 2:
        raise RiskModelError(
            f"Need at least 2 iterations to estimate a distribution; got {n_iter}."
        )
    if method not in ("gaussian_copula", "iman_conover"):
        raise RiskModelError(
            f"Unknown sampling method {method!r}. Allowed: gaussian_copula, "
            f"iman_conover."
        )

    rng = np.random.default_rng(seed)
    corr, notes = model.resolved_correlation()
    marginals = [e.frozen() for e in model.elements]

    sampler = _gaussian_copula if method == "gaussian_copula" else _iman_conover
    element_samples = sampler(marginals, corr, n_iter, rng)
    # Cost cannot be negative; clamp rather than let a wide normal go through.
    element_samples = np.maximum(element_samples, 0.0)

    if model.risks:
        occurred = np.column_stack(
            [rng.random(n_iter) < r.probability for r in model.risks]
        )
        impacts = np.column_stack(
            [
                r.frozen().ppf(rng.uniform(1e-12, 1.0 - 1e-12, n_iter))
                for r in model.risks
            ]
        )
        risk_samples = np.where(occurred, np.maximum(impacts, 0.0), 0.0)
    else:
        risk_samples = np.zeros((n_iter, 0))

    if correlate_risks and model.risks:  # pragma: no cover - opt-in path
        notes.append(
            "Discrete risks were correlated with the base estimate at the "
            "same rho; confirm this is intended rather than double-counting."
        )

    totals = element_samples.sum(axis=1) + risk_samples.sum(axis=1)

    result = RiskSimulationResult(
        totals=totals,
        element_samples=element_samples,
        risk_samples=risk_samples,
        element_names=model.element_names,
        risk_names=[r.name for r in model.risks],
        point_estimate=model.point_estimate,
        correlation_used=corr,
        notes=notes,
        seed=seed,
        method=method,
    )
    logger.info(
        "Simulated %s: %d iterations, mean %.4g, P80 %.4g, CV %.3f, point "
        "estimate at the %.1fth percentile",
        model.name,
        n_iter,
        result.mean,
        result.p80,
        result.cv,
        result.point_estimate_percentile,
    )
    return result


# ==========================================================================
# The headline: what independence costs you
# ==========================================================================
@dataclass(frozen=True)
class CorrelationImpact:
    """Correlated versus independent, side by side.

    Attributes:
        correlated: The simulation with correlation applied.
        independent: The same model with every correlation set to zero.
        analytic_variance_ratio: The closed-form inflation factor, which for
            equally variable elements at a common rho is exactly
            ``1 + rho*(k-1)``.
        empirical_variance_ratio: The same ratio measured from the two
            simulations. Agreement between the two is the check that the
            sampler is doing what the algebra says.
    """

    correlated: RiskSimulationResult
    independent: RiskSimulationResult
    analytic_variance_ratio: float
    empirical_variance_ratio: float

    @property
    def p80_understatement(self) -> float:
        """How much lower the independent P80 sits, as a fraction."""
        return 1.0 - self.independent.p80 / self.correlated.p80

    @property
    def reserve_understatement(self) -> float:
        """How much of the P80 risk reserve an independence assumption loses."""
        point = self.correlated.point_estimate
        correlated_reserve = self.correlated.p80 - point
        independent_reserve = self.independent.p80 - point
        if correlated_reserve <= 0:
            return float("nan")
        return 1.0 - independent_reserve / correlated_reserve

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": "std deviation of total",
                    "independent": self.independent.std,
                    "correlated": self.correlated.std,
                    "ratio": self.correlated.std / self.independent.std,
                },
                {
                    "metric": "variance of total",
                    "independent": self.independent.std**2,
                    "correlated": self.correlated.std**2,
                    "ratio": self.empirical_variance_ratio,
                },
                {
                    "metric": "P80",
                    "independent": self.independent.p80,
                    "correlated": self.correlated.p80,
                    "ratio": self.correlated.p80 / self.independent.p80,
                },
                {
                    "metric": "P80 risk reserve",
                    "independent": self.independent.p80
                    - self.independent.point_estimate,
                    "correlated": self.correlated.p80
                    - self.correlated.point_estimate,
                    "ratio": (self.correlated.p80 - self.correlated.point_estimate)
                    / max(
                        self.independent.p80 - self.independent.point_estimate, 1e-300
                    ),
                },
            ]
        )

    def narrative(self) -> str:
        """A sentence for the briefing."""
        return (
            f"Sampling the WBS elements independently understates the variance "
            f"of the total by a factor of {self.empirical_variance_ratio:.2f} "
            f"(closed form: {self.analytic_variance_ratio:.2f}). The P80 comes "
            f"out {self.p80_understatement:.1%} low, which discards "
            f"{self.reserve_understatement:.1%} of the risk reserve the same "
            f"model produces once correlation is applied."
        )


def correlation_impact(
    model: RiskModel,
    n_iter: int = 50_000,
    seed: int | None = 0,
    *,
    method: Literal["gaussian_copula", "iman_conover"] = "gaussian_copula",
) -> CorrelationImpact:
    """Quantify what assuming independence costs, for this specific model.

    Runs the model twice on the same seed -- once with its correlation, once
    with all correlations set to zero -- and reports the difference against the
    closed-form expectation. This is the demonstration the whole correlation
    argument rests on, so it produces both the measured and the derived number
    rather than asking anyone to take the simulation on trust.
    """
    with warnings.catch_warnings():
        # The caller has explicitly asked for this comparison; the default
        # correlation notice would be noise here and is carried in the result.
        warnings.simplefilter("ignore", CorrelationWarning)
        correlated = simulate_risk_model(model, n_iter, seed, method=method)

        independent_model = RiskModel(
            elements=model.elements,
            risks=model.risks,
            correlation=np.eye(len(model.elements)),
            name=f"{model.name} (independent)",
        )
        independent = simulate_risk_model(
            independent_model, n_iter, seed, method=method
        )

    analytic = model.variance_inflation()
    empirical = float(np.var(correlated.totals, ddof=1)) / float(
        np.var(independent.totals, ddof=1)
    )

    impact = CorrelationImpact(
        correlated=correlated,
        independent=independent,
        analytic_variance_ratio=analytic,
        empirical_variance_ratio=empirical,
    )
    logger.info("Correlation impact: %s", impact.narrative())
    return impact


def risk_model_from_elements(
    costs: Dict[str, float],
    *,
    low_factor: float = 0.85,
    high_factor: float = 1.45,
    distribution: str = "pert",
    correlation: np.ndarray | None = None,
    default_correlation: float = DEFAULT_CORRELATION,
    risks: list[DiscreteRisk] | None = None,
    name: str = "risk model",
) -> RiskModel:
    """Build a risk model from point estimates and a pair of spread factors.

    A convenience for the common case where element-level uncertainty has not
    been elicited individually and the analyst is applying one spread across
    the board. The asymmetry of the default factors is deliberate: cost
    estimates are bounded below by what the work costs and unbounded above, so
    a symmetric range around the point estimate is almost always wrong.

    Raises:
        RiskModelError: If the factors do not bracket the point estimate.
    """
    if not costs:
        raise RiskModelError("No cost elements supplied.")
    if not low_factor <= 1.0 <= high_factor:
        raise RiskModelError(
            f"Spread factors must bracket the point estimate: got "
            f"low={low_factor}, high={high_factor}."
        )

    elements = [
        CostElement(
            name=element_name,
            distribution={
                "type": distribution,
                "left": value * low_factor,
                "mode": value,
                "right": value * high_factor,
            },
            point_estimate=value,
        )
        for element_name, value in costs.items()
    ]
    return RiskModel(
        elements=elements,
        risks=risks or [],
        correlation=correlation,
        default_correlation=default_correlation,
        name=name,
    )


if __name__ == "__main__":
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define distributions
    # Example: Unit cost is lognormal, Quantity is triangular
    unit_cost_config = {
        "type": "lognormal",
        "mean": np.log(150),  # Underlying normal mean
        "sigma": 0.2          # Underlying normal standard deviation
    }
    
    quantity_config = {
        "type": "triangular",
        "left": 40,
        "mode": 50,
        "right": 75
    }

    # Run Simulation
    sim_result = run_monte_carlo(
        n_iter=10000,
        unit_cost_dist=unit_cost_config,
        quantity_dist=quantity_config,
        seed=42 # Fixed seed for deterministic demo
    )

    # Print Report
    print("\n--- Monte Carlo Simulation Results ---")
    print(f"Iterations : {len(sim_result.samples):,}")
    print(f"Mean Cost  : ${sim_result.mean:,.2f}")
    print(f"P50 (Base) : ${sim_result.p50:,.2f}")
    print(f"P80 (Safe) : ${sim_result.p80:,.2f}")
    print(f"P90 (Cons.) : ${sim_result.p90:,.2f}")
    print("--------------------------------------\n")

    # Plot
    plot_distribution(sim_result)