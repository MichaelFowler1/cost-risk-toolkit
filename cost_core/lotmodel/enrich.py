"""
enrich.py - The statistics the original tool did not carry.

Everything here reads the models the engine already fitted and adds to them.
Nothing in this module changes a single number in the projections table, the
analyst summary or the fit chart data, and ``tests/test_lotmodel.py`` holds a
golden master that fails if it ever does. The estimate stays the estimate; what
this adds is a statement of how much confidence it can carry.

Five things the point estimates alone cannot say:

**Retransformation bias.** The engine fits ``ln(cost)`` by ordinary least
squares and then exponentiates back to dollars. That step is biased: with
log-space errors of variance s-squared, the retransformed value estimates the
*median* and understates the *mean* by a factor of exp(s^2/2). MUPE and ZMPE
refit the same regressors under a proportional-error loss and drive the mean
percentage error to zero, so the size of the bias can be measured on this
dataset rather than argued about in the abstract.

**Prediction intervals.** A projected lot cost with no interval invites the
reader to treat it as exact. The delta method on the fitted covariance gives an
interval for a *new* lot -- carrying the residual scatter, not just the
uncertainty in where the fitted line sits.

**Influence.** Six analogy lots is a normal sample size here, and at that size a
single lot can set the slope on its own without anything in the summary saying
so. Leverage and Cook's distance name it.

**Risk on the buy.** A distribution over the total of the estimate lots, drawn
from the fitted parameter covariance with a t multiplier, giving P50/P80/P90
and where the point estimate falls on the curve.

**Assumptions.** A written record mapped to the four characteristics of a
reliable estimate in the GAO Cost Estimating and Assessment Guide, separating
what was measured from what was assumed.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cost_core.lotmodel.mathx import lmp_func

logger = logging.getLogger(__name__)

#: Correlation applied between the residual shocks of different estimate lots
#: in the risk simulation. Not zero: consecutive lots on one programme share a
#: workforce, a supply base and a schedule, so treating them as independent
#: lets the shocks cancel and understates the spread of the whole buy.
DEFAULT_LOT_CORRELATION = 0.30

#: Below this many costed analogy lots the fit is reported but flagged. The
#: engine already refuses under three; this is the level at which an interval
#: starts to mean something.
COMFORTABLE_LOTS = 5

MODEL_KEYS = {"LC": "mdl_lc", "Rate": "mdl_rt", "LC+Rate": "mdl_lcr"}


class EnrichmentError(ValueError):
    """Raised when the added statistics cannot be computed for a run."""


# --------------------------------------------------------------------------
# design matrices, rebuilt from the fitted models
# --------------------------------------------------------------------------
def _design(ctx: dict, model_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Rebuild the log-space design matrix, response and slope for a model.

    The engine keeps the fitted coefficients but not the matrix it fitted them
    with, so it is reconstructed here from the same inputs. The learning term
    is the log lot midpoint evaluated at the converged ``b`` -- the midpoint
    depends on the slope, which is why the original fit had to iterate.

    Raises:
        EnrichmentError: If the named model was not fitted for this run.
    """
    key = MODEL_KEYS.get(model_name)
    if key is None:
        raise EnrichmentError(
            f"Unknown model {model_name!r}. Expected one of {sorted(MODEL_KEYS)}."
        )
    model = ctx.get(key)
    if model is None:
        raise EnrichmentError(
            f"The {model_name} model was not fitted for this run, so its "
            f"statistics cannot be computed."
        )

    fit_q = np.asarray(ctx["fit_q"], dtype=float)
    fit_c = np.asarray(ctx["fit_c"], dtype=float)
    fit_se = ctx["fit_se"]
    y = np.log(fit_c)

    b = {"LC": ctx.get("b_lc"), "Rate": ctx.get("b_rt"),
         "LC+Rate": ctx.get("b_br")}[model_name]

    columns = [np.ones(len(y))]
    if model_name in ("LC", "LC+Rate"):
        mid = np.array([
            np.log(lmp_func(se["S"], se["E"], q, b))
            for se, q in zip(fit_se, fit_q)
        ])
        columns.append(mid)
    if model_name in ("Rate", "LC+Rate"):
        columns.append(np.log(fit_q))

    return np.column_stack(columns), y, float(b) if b is not None else float("nan")


# --------------------------------------------------------------------------
# unbiased refits
# --------------------------------------------------------------------------
def _fit_mupe(design: np.ndarray, y_log: np.ndarray, max_iter: int = 200,
              tol: float = 1e-13) -> np.ndarray:
    """Minimum-unbiased-percentage-error fit by iteratively reweighted least
    squares, on the same regressors the engine used.

    Weights are ``1/f_prev``; at the fixed point the normal equation for the
    multiplicative scale collapses to ``sum((y - f)/f) = 0``, which is what
    "unbiased percentage error" means.
    """
    observed = np.exp(y_log)
    beta = np.linalg.lstsq(design, y_log, rcond=None)[0]
    for _ in range(max_iter):
        fitted = np.exp(design @ beta)
        w = 1.0 / np.maximum(fitted, 1e-300)
        # Weighted least squares on the level scale, linearised about the
        # current fit: the model is exp(X beta), so d f / d beta = f * X.
        jac = fitted[:, None] * design
        resid = observed - fitted
        step, *_ = np.linalg.lstsq(jac * w[:, None], resid * w, rcond=None)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _fit_zmpe(design: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Zero-percentage-bias minimum-percentage-error fit.

    Minimises the sum of squared percentage errors subject to the mean
    percentage error being zero. The intercept is a pure multiplicative scale,
    so the constraint is solved for it in closed form -- ``exp(b0) = mean(y/g)``
    -- which makes the zero-bias property exact rather than converged-to, and
    leaves only the slopes to optimise.
    """
    from scipy import optimize

    observed = np.exp(y_log)
    rest = design[:, 1:]

    def complete(slopes: np.ndarray) -> np.ndarray:
        g = np.exp(rest @ slopes) if rest.size else np.ones_like(observed)
        return np.concatenate([[np.log(np.mean(observed / g))], slopes])

    def objective(slopes: np.ndarray) -> float:
        f = np.exp(design @ complete(slopes))
        return float(np.sum(((observed - f) / f) ** 2))

    seed = np.linalg.lstsq(design, y_log, rcond=None)[0][1:]
    if seed.size == 0:
        return complete(seed)
    sol = optimize.minimize(objective, seed, method="Nelder-Mead",
                            options={"xatol": 1e-12, "fatol": 1e-14,
                                     "maxiter": 20000})
    return complete(sol.x)


@dataclass(frozen=True)
class MethodComparison:
    """OLS against MUPE and ZMPE on the selected model's own regressors."""

    model: str
    frame: pd.DataFrame
    log_residual_variance: float
    theoretical_factor: float
    smearing_factor: float
    mupe_over_ols: float
    zmpe_over_ols: float

    @property
    def percent_understated(self) -> float:
        """How much the naive retransformation understates the mean, as a
        percentage. This is the number worth quoting."""
        return (self.theoretical_factor - 1.0) * 100.0


def compare_fitting_methods(ctx: dict, model_name: str) -> MethodComparison:
    """Refit the selected model by MUPE and ZMPE and measure the OLS bias.

    Raises:
        EnrichmentError: If the model was not fitted.
    """
    design, y_log, _ = _design(ctx, model_name)
    observed = np.exp(y_log)
    n, p = design.shape

    beta_ols = np.linalg.lstsq(design, y_log, rcond=None)[0]
    beta_mupe = _fit_mupe(design, y_log)
    beta_zmpe = _fit_zmpe(design, y_log)

    rows, fits = [], {}
    for label, beta in (("OLS", beta_ols), ("MUPE", beta_mupe), ("ZMPE", beta_zmpe)):
        fitted = np.exp(design @ beta)
        fits[label] = fitted
        pct = (observed - fitted) / fitted
        dof = max(n - p, 1)
        rows.append({
            "Method": label,
            "T1 ($K)": float(np.exp(beta[0])),
            "b (learning)": float(beta[1]) if p > 1 and model_name != "Rate" else np.nan,
            "c (rate)": float(beta[-1]) if model_name in ("Rate", "LC+Rate") else np.nan,
            "Mean % error": float(np.mean(pct)),
            "MAPE": float(np.mean(np.abs(pct))),
            "SEE (log)": float(np.sqrt(np.sum((y_log - np.log(fitted)) ** 2) / dof)),
        })

    log_resid = y_log - np.log(fits["OLS"])
    s2 = float(np.sum(log_resid ** 2) / max(n - p, 1))

    return MethodComparison(
        model=model_name,
        frame=pd.DataFrame(rows),
        log_residual_variance=s2,
        theoretical_factor=float(np.exp(s2 / 2.0)),
        smearing_factor=float(np.mean(np.exp(log_resid))),
        mupe_over_ols=float(np.mean(fits["MUPE"] / fits["OLS"])),
        zmpe_over_ols=float(np.mean(fits["ZMPE"] / fits["OLS"])),
    )


# --------------------------------------------------------------------------
# influence on the analogy lots
# --------------------------------------------------------------------------
def influence_diagnostics(ctx: dict, model_name: str,
                          labels: list[str] | None = None) -> pd.DataFrame:
    """Leverage, Cook's distance and DFFITS for each analogy lot.

    At six lots a single analogy can set the slope while every summary
    statistic still looks healthy. Leverage says which lot is unusual in the
    predictors; Cook's distance says which one is actually moving the fit.

    The conventional flags are ``2p/n`` for leverage and ``4/n`` for Cook's
    distance. They are flags, not verdicts -- the largest or smallest lot in a
    sample has high leverage by construction, and dropping it would usually be
    indefensible.
    """
    design, y_log, _ = _design(ctx, model_name)
    n, p = design.shape
    beta = np.linalg.lstsq(design, y_log, rcond=None)[0]
    resid = y_log - design @ beta
    dof = max(n - p, 1)
    sigma = float(np.sqrt(np.sum(resid ** 2) / dof))

    gram_inv = np.linalg.pinv(design.T @ design)
    leverage = np.clip(np.einsum("ij,jk,ik->i", design, gram_inv, design), 0.0, 1.0)
    one_minus_h = np.maximum(1.0 - leverage, 1e-12)

    if sigma <= 1e-12 * max(float(np.max(np.abs(y_log))), 1.0):
        standardised = np.zeros_like(resid)
        cooks = np.zeros_like(resid)
        dffits = np.zeros_like(resid)
    else:
        standardised = resid / (sigma * np.sqrt(one_minus_h))
        cooks = (standardised ** 2 / p) * (leverage / one_minus_h)
        s_minus = np.sqrt(np.maximum(
            (dof * sigma ** 2 - resid ** 2 / one_minus_h) / max(dof - 1, 1), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            dffits = np.where(
                s_minus > 0,
                resid / (s_minus * np.sqrt(one_minus_h))
                * np.sqrt(leverage / one_minus_h), 0.0)
        dffits = np.nan_to_num(dffits)

    if labels is None:
        labels = [f"Analogy lot {i + 1}" for i in range(n)]

    lev_flag, cook_flag = 2.0 * p / n, 4.0 / n
    return pd.DataFrame({
        "Lot": labels,
        "Qty": np.asarray(ctx["fit_q"], dtype=float),
        "Actual ($K)": np.asarray(ctx["fit_c"], dtype=float),
        "Fitted ($K)": np.exp(design @ beta),
        "% error": (np.exp(y_log) - np.exp(design @ beta)) / np.exp(design @ beta) * 100.0,
        "Leverage": leverage,
        "Cook's D": cooks,
        "DFFITS": dffits,
        "High leverage": leverage > lev_flag,
        "Influential": cooks > cook_flag,
    })


# --------------------------------------------------------------------------
# prediction intervals on the projected lots
# --------------------------------------------------------------------------
def projection_intervals(ctx: dict, projections: pd.DataFrame, model_name: str,
                         level: float = 0.80) -> pd.DataFrame:
    """Prediction intervals for each estimate lot under the selected model.

    A *prediction* interval, not a confidence interval: the question is what a
    new lot will cost, not where the fitted line sits. The two differ by the
    residual variance, and that term does not shrink with more analogy lots.

    The multiplier is a t with n-p degrees of freedom, because sigma is
    estimated rather than known. On six analogy lots that is materially wider
    than a normal would give.

    Raises:
        EnrichmentError: If ``level`` is outside (0, 1) or there are no
            residual degrees of freedom.
    """
    if not 0.0 < level < 1.0:
        raise EnrichmentError(f"level must be between 0 and 1; got {level}.")

    design, y_log, b = _design(ctx, model_name)
    n, p = design.shape
    dof = n - p
    if dof <= 0:
        raise EnrichmentError(
            f"The {model_name} model has {n} analogy lots and {p} parameters, "
            f"leaving no residual degrees of freedom, so no interval is "
            f"estimable."
        )

    beta = np.linalg.lstsq(design, y_log, rcond=None)[0]
    resid = y_log - design @ beta
    sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
    gram_inv = np.linalg.pinv(design.T @ design)

    prefix = {"LC": "LC", "Rate": "Rate", "LC+Rate": "LC+Rate"}[model_name]
    mid_col = f"{prefix} Lot Midpoint (unit no.)"
    unit_col = f"{prefix} Unit Cost ($K)"
    cost_col = f"{prefix} Lot Cost After Complexity ($)"
    for col in (mid_col, unit_col, cost_col):
        if col not in projections.columns:
            raise EnrichmentError(
                f"Projections table has no column {col!r}; it does not look "
                f"like output from this engine."
            )

    rows = []
    tcrit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, dof))
    for _, r in projections.iterrows():
        x = [1.0]
        if model_name in ("LC", "LC+Rate"):
            x.append(np.log(float(r[mid_col])))
        if model_name in ("Rate", "LC+Rate"):
            x.append(np.log(float(r["Lot Quantity"])))
        x = np.asarray(x, dtype=float)

        var_mean = float(x @ gram_inv @ x) * sigma ** 2
        se_pred = float(np.sqrt(var_mean + sigma ** 2))
        unit = float(r[unit_col])
        lot_cost = float(r[cost_col])
        lo, hi = np.exp(-tcrit * se_pred), np.exp(tcrit * se_pred)
        rows.append({
            "Lot": r["Lot"],
            "Fiscal Year": r["Fiscal Year"],
            "Lot Quantity": r["Lot Quantity"],
            "Unit Cost ($K)": unit,
            "Unit Cost Lower": unit * lo,
            "Unit Cost Upper": unit * hi,
            "Lot Cost ($)": lot_cost,
            "Lot Cost Lower": lot_cost * lo,
            "Lot Cost Upper": lot_cost * hi,
            "SE (log)": se_pred,
            "Level": level,
            "Kind": "prediction",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# risk on the total buy
# --------------------------------------------------------------------------
@dataclass
class BuyRisk:
    """Distribution of the total cost of the estimate lots."""

    totals: np.ndarray
    per_lot: np.ndarray
    point_estimate: float
    model: str
    n_history_lots: int
    dof: int
    seed: int | None
    lot_correlation: float
    clipped: int = 0

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
        return float(self.std / self.mean) if self.mean else float("nan")

    @property
    def p50(self) -> float:
        return float(np.percentile(self.totals, 50))

    @property
    def p80(self) -> float:
        return float(np.percentile(self.totals, 80))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.totals, 90))

    def percentile_of(self, value: float) -> float:
        return float(np.mean(self.totals <= value) * 100.0)

    @property
    def point_estimate_percentile(self) -> float:
        return self.percentile_of(self.point_estimate)

    def summary(self) -> pd.DataFrame:
        reserve = self.p80 - self.point_estimate
        return pd.DataFrame(
            [("Model", self.model),
             ("Iterations", self.n_iter),
             ("Analogy lots", self.n_history_lots),
             ("Degrees of freedom", self.dof),
             ("Point estimate ($)", self.point_estimate),
             ("Point estimate percentile", self.point_estimate_percentile),
             ("Mean ($)", self.mean),
             ("Std deviation ($)", self.std),
             ("CV", self.cv),
             ("P50 ($)", self.p50),
             ("P80 ($)", self.p80),
             ("P90 ($)", self.p90),
             ("Reserve to P80 ($)", reserve),
             ("Reserve to P80 (%)",
              100.0 * reserve / self.point_estimate if self.point_estimate else np.nan)],
            columns=["Statistic", "Value"])

    def narrative(self) -> str:
        reserve = self.p80 - self.point_estimate
        pct = (100.0 * reserve / self.point_estimate) if self.point_estimate else float("nan")
        return (
            f"Total of the estimate lots under the {self.model} model. The "
            f"point estimate of {self.point_estimate:,.0f} sits at the "
            f"{self.point_estimate_percentile:.0f}th percentile of the risk "
            f"distribution; P50 {self.p50:,.0f}, P80 {self.p80:,.0f}, P90 "
            f"{self.p90:,.0f}. Risk reserve to P80 is {reserve:,.0f} "
            f"({pct:.1f}%). CV {self.cv:.1%}. Uncertainty is measured from the "
            f"{self.n_history_lots} analogy lots, drawn on a t distribution "
            f"with {self.dof} degrees of freedom, with lot-to-lot residuals "
            f"correlated at {self.lot_correlation:.2f}."
        )


def simulate_buy(ctx: dict, projections: pd.DataFrame, model_name: str,
                 n_iter: int = 20_000, seed: int | None = 0,
                 lot_correlation: float = DEFAULT_LOT_CORRELATION) -> BuyRisk:
    """Monte Carlo the total cost of the estimate lots.

    Two sources of uncertainty are propagated and kept distinct. Parameter
    uncertainty is drawn once per iteration from the fitted covariance and
    applied to every lot, because there is one curve per iteration -- on a
    six-lot analogy this dominates. Residual scatter is drawn per lot and
    correlated across lots at ``lot_correlation``.

    The scale factor comes from a scaled inverse chi-square, which makes both
    draws t-distributed with the fit's degrees of freedom. Drawing from a
    normal would understate the interval badly at this sample size, and
    understating risk is the error worth avoiding.

    Raises:
        EnrichmentError: On too few iterations or no residual degrees of freedom.
    """
    if n_iter < 2:
        raise EnrichmentError(
            f"Need at least 2 iterations to form a distribution; got {n_iter}.")

    design, y_log, _ = _design(ctx, model_name)
    n, p = design.shape
    dof = n - p
    if dof <= 0:
        raise EnrichmentError(
            f"The {model_name} model leaves no residual degrees of freedom, so "
            f"no risk distribution can be formed.")

    beta = np.linalg.lstsq(design, y_log, rcond=None)[0]
    resid = y_log - design @ beta
    sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
    cov = np.linalg.pinv(design.T @ design) * sigma ** 2

    prefix = model_name
    mid_col = f"{prefix} Lot Midpoint (unit no.)"
    cost_col = f"{prefix} Lot Cost After Complexity ($)"
    unit_col = f"{prefix} Unit Cost ($K)"

    x_rows, lot_costs = [], []
    for _, r in projections.iterrows():
        x = [1.0]
        if model_name in ("LC", "LC+Rate"):
            x.append(np.log(float(r[mid_col])))
        if model_name in ("Rate", "LC+Rate"):
            x.append(np.log(float(r["Lot Quantity"])))
        x_rows.append(x)
        lot_costs.append(float(r[cost_col]))
    x_new = np.asarray(x_rows, dtype=float)
    base = np.asarray(lot_costs, dtype=float)
    n_lots = base.size

    rng = np.random.default_rng(seed)
    scale = np.sqrt(dof / rng.chisquare(dof, size=n_iter))
    cap = float(np.sqrt(dof / stats.chi2.ppf(1.0 - 1e-4, dof)) * 1e4)
    clipped = int(np.sum(scale > cap))
    if clipped:
        scale = np.minimum(scale, cap)
    if dof <= 2:
        warnings.warn(
            f"Only {dof} degree(s) of freedom, so the risk distribution has "
            + ("no finite mean or variance" if dof == 1 else "no finite variance")
            + ". Read the percentiles; the mean, standard deviation and CV do "
            f"not mean anything here. {clipped} of {n_iter} draws were clipped "
            f"to keep the output finite.",
            RuntimeWarning, stacklevel=2)

    centred = rng.multivariate_normal(np.zeros(p), cov, size=n_iter)
    draws = beta + centred * scale[:, None]
    # Ratio to the fitted curve, so complexity factors and scaling carried in
    # the engine's lot cost stay applied.
    log_ratio = (draws @ x_new.T) - (beta @ x_new.T)

    if n_lots > 1 and lot_correlation != 0.0:
        corr = np.full((n_lots, n_lots), float(lot_correlation))
        np.fill_diagonal(corr, 1.0)
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(corr + np.eye(n_lots) * 1e-10)
        shocks = (rng.standard_normal((n_iter, n_lots)) @ chol.T) * sigma
    else:
        shocks = rng.standard_normal((n_iter, n_lots)) * sigma
    shocks = shocks * scale[:, None]

    per_lot = base * np.exp(log_ratio + shocks)
    return BuyRisk(totals=per_lot.sum(axis=1), per_lot=per_lot,
                   point_estimate=float(base.sum()), model=model_name,
                   n_history_lots=n, dof=dof, seed=seed,
                   lot_correlation=lot_correlation, clipped=clipped)


# --------------------------------------------------------------------------
# the whole added layer
# --------------------------------------------------------------------------
@dataclass
class Enrichment:
    """Everything the added statistics produced for one run."""

    selected_model: str
    methods: MethodComparison
    influence: pd.DataFrame
    intervals: pd.DataFrame
    risk: BuyRisk
    warnings_raised: list[str] = field(default_factory=list)

    def sheets(self) -> dict[str, pd.DataFrame]:
        """Extra workbook sheets, keyed by sheet name."""
        return {
            "Fit_Methods": self.methods.frame,
            "Influence": self.influence,
            "Prediction_Intervals": self.intervals,
            "Buy_Risk": self.risk.summary(),
        }


def selected_model_name(summary: pd.DataFrame) -> str:
    """Read which model the analyst summary marked SELECTED.

    Raises:
        EnrichmentError: If no model is marked, which means the engine could
            not fit anything and there is nothing to add statistics to.
    """
    row = summary[summary["Item"] == "SELECTED"]
    if not row.empty:
        for col in ("LC", "Rate", "LC+Rate"):
            if str(row.iloc[0][col]).strip().upper() == "YES":
                return col
    raise EnrichmentError(
        "No model is marked SELECTED in the analyst summary, so the engine "
        "fitted nothing to add statistics to.")


def enrich_run(ctx: dict, projections: pd.DataFrame, summary: pd.DataFrame,
               *, level: float = 0.80, n_iter: int = 20_000,
               seed: int | None = 0,
               lot_correlation: float = DEFAULT_LOT_CORRELATION) -> Enrichment:
    """Compute every added statistic for a completed run.

    Args:
        ctx: The model context returned by ``run_lot_cost_model``.
        projections: The projections table from the same call.
        summary: The analyst summary, used only to read the selected model.
        level: Coverage for the prediction intervals.
        n_iter: Iterations for the buy risk simulation.
        seed: Fixed seed, so a P80 does not move between runs.
        lot_correlation: Residual correlation across estimate lots.
    """
    model = selected_model_name(summary)
    notes: list[str] = []

    n_keep = int(ctx.get("n_keep", 0))
    if n_keep < COMFORTABLE_LOTS:
        notes.append(
            f"{n_keep} costed analogy lots is below the {COMFORTABLE_LOTS} at "
            f"which an interval starts to be informative. Treat the point "
            f"estimate as indicative and read the per-lot errors.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        risk = simulate_buy(ctx, projections, model, n_iter=n_iter, seed=seed,
                            lot_correlation=lot_correlation)
        notes.extend(str(w.message) for w in caught)

    enrichment = Enrichment(
        selected_model=model,
        methods=compare_fitting_methods(ctx, model),
        influence=influence_diagnostics(ctx, model),
        intervals=projection_intervals(ctx, projections, model, level=level),
        risk=risk,
        warnings_raised=notes,
    )

    influential = enrichment.influence.loc[
        enrichment.influence["Influential"], "Lot"].tolist()
    if influential:
        notes.append(
            f"Analogy lot(s) {', '.join(map(str, influential))} exceed the "
            f"conventional Cook's distance flag and are setting this fit. "
            f"Confirm each belongs in the sample before relying on the slope.")

    logger.info(
        "Enriched %s run: bias %.2f%%, P80 %.4g, point estimate at the %.0fth "
        "percentile", model, enrichment.methods.percent_understated,
        risk.p80, risk.point_estimate_percentile)
    return enrichment
