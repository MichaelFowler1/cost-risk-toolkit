"""
fitting.py - Shared estimation engine for learning curves and CERs.

Both a learning curve and a parametric CER are the same statistical object: a
prediction function with a few parameters, fitted to a handful of observations
whose errors are usually *proportional* to the estimate rather than additive.
So the estimator lives here once and is reused by ``cost_core.learning_curve``
and ``cost_core.cer`` rather than being written twice.

Three fitting methods, all minimising a different loss over the same
prediction function:

    OLS   min sum ( r_i )^2                     r = log y - log f   (log link)
                                                r = y - f          (identity link)
    MUPE  min sum ( y_i - f_i )^2 / f_i(prev)^2 solved by iteratively
          reweighted least squares (Book & Lao). At convergence its first-order
          condition for a multiplicative scale parameter is
          sum( (y_i - f_i) / f_i ) = 0, i.e. the *mean percentage error is
          exactly zero* -- the "unbiased percentage error" the name refers to.
    ZMPE  min sum ( (y_i - f_i) / f_i )^2  subject to sum( (y_i - f_i)/f_i ) = 0.
          Same zero-percentage-bias property, imposed as a constraint rather
          than emerging from the normal equations, and a different slope.

Why this matters. The traditional cost-analysis fit is OLS in log space, and
then the analyst exponentiates back to dollars. That retransformation is
biased: if log-space errors are normal with variance s^2, then

    E[ y | x ]  =  f(x) * exp(s^2 / 2)

so the retransformed point estimate is an estimate of the *median*, and it
understates the mean by a factor of exp(s^2/2). On a 30% CV curve that is a
4-5% understatement built into the estimate before any risk analysis starts.
``retransformation_bias()`` measures it three ways and reports all three.

Standard errors and intervals are computed by the delta method from the
Gauss-Newton covariance, which is not an approximation in the log-linear case:
for a log-log fit on unit data it reduces algebraically to the textbook
prediction-interval formula, and ``tests/test_fitting.py`` asserts that
identity to machine precision.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)

Method = Literal["ols", "mupe", "zmpe"]
Link = Literal["log", "identity"]
IntervalKind = Literal["prediction", "confidence"]

METHODS: tuple[str, ...] = ("ols", "mupe", "zmpe")

# Below this many degrees of freedom a fit is reported but flagged. Three is
# the conventional floor in the cost community -- with fewer, the t
# multiplier on any interval is so large that the interval stops being
# informative even though the point estimate still computes cleanly.
MIN_COMFORTABLE_DF = 3


class FitError(ValueError):
    """Raised when a fit cannot be performed or has not converged.

    Deliberately a ValueError: bad input should stop the pipeline, not return
    a plausible-looking number.
    """


# --------------------------------------------------------------------------
# model specification
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """A prediction function plus the metadata the estimator needs.

    Attributes:
        name: Human-readable label carried into the fit result.
        param_names: One name per free parameter, in theta order.
        predict: ``(theta, X) -> yhat``. ``X`` is opaque to the engine; it may
            be a design matrix, a DataFrame of lots, or anything the caller's
            predict function understands.
        link: ``"log"`` if the model's natural error scale is multiplicative
            (costs, hours -- almost everything here), ``"identity"`` if
            additive. This drives what OLS minimises and whether intervals
            are multiplicative or additive.
        log_scale_index: Index of a parameter that enters as ``exp(theta_k)``
            multiplying the whole prediction. When present, ZMPE solves its
            constraint in closed form for that parameter, which makes the
            zero-bias property exact rather than merely converged-to.
        initial: ``(X, y) -> theta0``. Optional; falls back to ones.
        bounds: Optional ``(lower, upper)`` arrays for the optimiser.
        jacobian: Optional ``(theta, X) -> d f / d theta`` with shape
            ``(n_obs, n_params)``, on the *level* scale; the engine divides by
            ``f`` itself when it needs the log scale. Worth supplying whenever
            it is known. Central differences on a log-linear model lose about
            four digits to cancellation -- the derivative is exact in real
            arithmetic but the function values it differences are large
            relative to the step -- which is enough to stop the fit landing
            exactly on the closed-form OLS solution and to blur the reported
            standard errors in the last digits.
    """

    name: str
    param_names: tuple[str, ...]
    predict: Callable[[np.ndarray, Any], np.ndarray]
    link: Link = "log"
    log_scale_index: int | None = None
    initial: Callable[[Any, np.ndarray], np.ndarray] | None = None
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    jacobian: Callable[[np.ndarray, Any], np.ndarray] | None = None

    @property
    def n_params(self) -> int:
        return len(self.param_names)


# --------------------------------------------------------------------------
# fit result
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FitResult:
    """A fitted model, its uncertainty, and everything needed to defend it."""

    spec: ModelSpec
    method: str
    theta: np.ndarray
    cov: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray
    sigma: float
    n_obs: int
    n_params: int
    n_iter: int
    converged: bool
    X: Any = field(repr=False, default=None)

    # ---------------------------------------------------------------- basics
    @property
    def df(self) -> int:
        """Degrees of freedom: observations minus estimated parameters."""
        return self.n_obs - self.n_params

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.spec.param_names, (float(t) for t in self.theta)))

    @property
    def param_se(self) -> dict[str, float]:
        return dict(
            zip(self.spec.param_names, (float(s) for s in np.sqrt(np.diag(self.cov))))
        )

    # ------------------------------------------------------------- residuals
    @property
    def residuals(self) -> np.ndarray:
        """Observed minus fitted, in the units of the data."""
        return self.observed - self.fitted

    @property
    def percent_errors(self) -> np.ndarray:
        """(observed - fitted) / fitted. The scale MUPE and ZMPE work on."""
        return (self.observed - self.fitted) / self.fitted

    @property
    def log_residuals(self) -> np.ndarray:
        return np.log(self.observed) - np.log(self.fitted)

    @property
    def mean_percent_error(self) -> float:
        """Mean of the percentage errors.

        Exactly zero for MUPE and ZMPE by construction. Positive for a naive
        log-log OLS retransformation, which is the retransformation bias.
        """
        return float(np.mean(self.percent_errors))

    # ------------------------------------------------------- quality metrics
    @property
    def standard_error(self) -> float:
        """Standard error of the estimate, in the units of the data.

        Reported in preference to R^2. R^2 on a log-log fit describes how well
        the *logs* line up and is close to 1 for almost any cost data, which
        makes it useless for discriminating between candidate CERs; the
        standard error is in dollars and answers the question the reviewer
        actually asks.
        """
        if self.spec.link == "identity":
            return self.sigma
        # Multiplicative fit: sigma is a proportion, so scale it to the units
        # of the data at the centre of the fitted range.
        return float(self.sigma * np.mean(self.fitted))

    @property
    def cv(self) -> float:
        """Coefficient of variation of the estimate.

        For a multiplicative fit this *is* sigma -- the relative residual
        spread. For an additive fit it is the residual SD over the mean
        response.
        """
        if self.spec.link == "identity":
            return float(self.sigma / np.mean(self.observed))
        return float(self.sigma)

    @property
    def r_squared(self) -> float:
        """Available, but reported alongside SE and CV rather than instead.

        Computed on the fitting scale (log space for a multiplicative model),
        which is where the fit actually happened.
        """
        y = np.log(self.observed) if self.spec.link == "log" else self.observed
        f = np.log(self.fitted) if self.spec.link == "log" else self.fitted
        ss_res = float(np.sum((y - f) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot

    # ----------------------------------------------------------- guardrails
    @property
    def df_is_adequate(self) -> bool:
        return self.df >= MIN_COMFORTABLE_DF

    def summary(self) -> pd.DataFrame:
        """One-row-per-parameter table for reports."""
        se = np.sqrt(np.diag(self.cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(se > 0, self.theta / se, np.nan)
        pval = (
            2.0 * (1.0 - stats.t.cdf(np.abs(tstat), self.df))
            if self.df > 0
            else np.full_like(tstat, np.nan)
        )
        return pd.DataFrame(
            {
                "parameter": list(self.spec.param_names),
                "estimate": self.theta,
                "std_error": se,
                "t_stat": tstat,
                "p_value": pval,
            }
        )


# --------------------------------------------------------------------------
# numerical Jacobian
# --------------------------------------------------------------------------
def _jacobian(
    spec: ModelSpec, theta: np.ndarray, X: Any, on_log_scale: bool
) -> np.ndarray:
    """d f / d theta, or d log(f) / d theta.

    Uses the spec's analytic Jacobian when it has one, otherwise central
    differences (good to roughly 1e-10 relative, which is well below anything
    that matters in dollars but not tight enough to land on a closed form).
    """
    if spec.jacobian is not None:
        jac = np.asarray(spec.jacobian(theta, X), dtype=float)
        if on_log_scale:
            f = np.asarray(spec.predict(theta, X), dtype=float)
            jac = jac / f[:, None]
        return jac

    n_p = len(theta)
    base = np.asarray(spec.predict(theta, X), dtype=float)
    jac = np.empty((base.size, n_p), dtype=float)
    for k in range(n_p):
        h = 1e-6 * max(abs(float(theta[k])), 1.0)
        up, dn = theta.astype(float).copy(), theta.astype(float).copy()
        up[k] += h
        dn[k] -= h
        f_up = np.asarray(spec.predict(up, X), dtype=float)
        f_dn = np.asarray(spec.predict(dn, X), dtype=float)
        if on_log_scale:
            jac[:, k] = (np.log(f_up) - np.log(f_dn)) / (2.0 * h)
        else:
            jac[:, k] = (f_up - f_dn) / (2.0 * h)
    return jac


def _covariance(jac: np.ndarray, sigma: float, df: int) -> np.ndarray:
    """sigma^2 (J'J)^-1, pseudo-inverted so a rank-deficient fit still reports."""
    jtj = jac.T @ jac
    try:
        inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        logger.warning("Jacobian is singular; falling back to pseudo-inverse.")
        inv = np.linalg.pinv(jtj)
    return (sigma**2) * inv


# --------------------------------------------------------------------------
# the three estimators
# --------------------------------------------------------------------------
def _initial_theta(spec: ModelSpec, X: Any, y: np.ndarray) -> np.ndarray:
    if spec.initial is not None:
        return np.asarray(spec.initial(X, y), dtype=float)
    return np.ones(spec.n_params, dtype=float)


def _gauss_newton_polish(
    spec: ModelSpec,
    X: Any,
    theta: np.ndarray,
    resid: Callable[[np.ndarray], np.ndarray],
    on_log_scale: bool,
    max_steps: int = 2,
) -> np.ndarray:
    """Refine a least-squares solution to the exact stationary point.

    ``scipy.optimize.least_squares`` stops on a relative tolerance, which
    leaves the coefficients about 1e-9 off the true optimum. For the
    log-linear and linear models that dominate this library the residual is
    linear in theta, so a single Gauss-Newton step lands exactly on the
    closed-form OLS solution -- which is what lets the tests assert that our
    OLS *is* the textbook OLS rather than merely close to it.

    A step is rejected if it makes the sum of squares meaningfully worse or
    moves theta a long way, so on a genuinely nonlinear model this can do
    nothing but is never harmful. The tolerance on "worse" has to be relative
    rather than exact: the loss is quadratic and flat at the optimum, so the
    last few steps change it by less than floating-point resolution and an
    exact ``loss < best`` test would reject the very steps that matter.
    """
    theta = np.asarray(theta, dtype=float).copy()
    if spec.bounds is not None:  # a bounded fit may sit legitimately on an edge
        return theta

    best = float(np.sum(resid(theta) ** 2))
    for _ in range(max_steps):
        r = resid(theta)
        jac = -_jacobian(spec, theta, X, on_log_scale=on_log_scale)
        try:
            step = np.linalg.lstsq(jac, r, rcond=None)[0]
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            break
        candidate = theta - step
        if not np.all(np.isfinite(candidate)):
            break
        scale = np.maximum(np.abs(theta), 1.0)
        rel_step = float(np.max(np.abs(step) / scale))
        if rel_step > 0.5:  # not a polish; leave the optimiser's answer alone
            break
        loss = float(np.sum(resid(candidate) ** 2))
        if not np.isfinite(loss) or loss > best * (1.0 + 1e-9) + 1e-300:
            break
        theta, best = candidate, min(loss, best)
        if rel_step < 1e-15:
            break
    return theta


def _fit_ols(spec: ModelSpec, X: Any, y: np.ndarray, theta0: np.ndarray):
    """Least squares on the model's natural scale."""
    log_link = spec.link == "log"

    def resid(theta: np.ndarray) -> np.ndarray:
        f = np.asarray(spec.predict(theta, X), dtype=float)
        if log_link:
            if np.any(f <= 0):
                return np.full_like(y, 1e6)
            return np.log(y) - np.log(f)
        return y - f

    sol = optimize.least_squares(
        resid, theta0, bounds=spec.bounds or (-np.inf, np.inf), xtol=1e-14, ftol=1e-14
    )
    theta = _gauss_newton_polish(spec, X, sol.x, resid, on_log_scale=log_link)
    return theta, int(sol.nfev), bool(sol.success)


def _fit_mupe(
    spec: ModelSpec, X: Any, y: np.ndarray, theta0: np.ndarray, max_iter: int, tol: float
):
    """Iteratively reweighted least squares: weights 1/f_prev^2, refit, repeat.

    At the fixed point the weights stop moving, and the normal equation for a
    multiplicative scale parameter collapses to sum((y - f)/f) = 0.
    """
    theta = np.asarray(theta0, dtype=float)
    f_prev = np.asarray(spec.predict(theta, X), dtype=float)
    if np.any(f_prev <= 0):
        f_prev = np.full_like(y, float(np.mean(y)))

    converged, used = False, 0
    for it in range(1, max_iter + 1):
        weights = 1.0 / f_prev

        def resid(th: np.ndarray, w: np.ndarray = weights) -> np.ndarray:
            f = np.asarray(spec.predict(th, X), dtype=float)
            return (y - f) * w

        sol = optimize.least_squares(
            resid, theta, bounds=spec.bounds or (-np.inf, np.inf), xtol=1e-14, ftol=1e-14
        )
        new_theta = sol.x
        f_new = np.asarray(spec.predict(new_theta, X), dtype=float)
        shift = float(np.max(np.abs(f_new - f_prev) / np.maximum(f_prev, 1e-300)))
        theta, f_prev, used = new_theta, f_new, it
        if shift < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"MUPE did not converge in {max_iter} iterations "
            f"(last relative shift {shift:.3e}). Treat the standard errors as "
            f"indicative and inspect the data for outliers.",
            RuntimeWarning,
            stacklevel=3,
        )
    return theta, used, converged


def _fit_zmpe(
    spec: ModelSpec, X: Any, y: np.ndarray, theta0: np.ndarray, tol: float
):
    """Minimise squared percentage error subject to zero mean percentage error.

    When the spec declares a multiplicative scale parameter the constraint is
    solved for that parameter in closed form and substituted, so it holds to
    machine precision instead of to an optimiser tolerance. Given
    ``f = exp(theta_k) * g(rest)``, the constraint sum(y/f) = n has the unique
    solution ``exp(theta_k) = mean(y / g)``.
    """
    k = spec.log_scale_index

    if k is not None:
        free_idx = [i for i in range(spec.n_params) if i != k]

        def _complete(free: np.ndarray) -> np.ndarray:
            """Rebuild full theta with the scale parameter set by the constraint."""
            th = np.empty(spec.n_params, dtype=float)
            th[free_idx] = free
            th[k] = 0.0
            g = np.asarray(spec.predict(th, X), dtype=float)  # scale factor exp(0) = 1
            th[k] = float(np.log(np.mean(y / g)))
            return th

        def objective(free: np.ndarray) -> float:
            f = np.asarray(spec.predict(_complete(free), X), dtype=float)
            return float(np.sum(((y - f) / f) ** 2))

        if len(free_idx) == 0:
            return _complete(np.array([])), 1, True

        sol = optimize.minimize(
            objective,
            np.asarray(theta0, dtype=float)[free_idx],
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20_000},
        )
        return _complete(sol.x), int(sol.nit), bool(sol.success)

    # General case: explicit equality constraint.
    def objective(th: np.ndarray) -> float:
        f = np.asarray(spec.predict(th, X), dtype=float)
        return float(np.sum(((y - f) / f) ** 2))

    def constraint(th: np.ndarray) -> float:
        f = np.asarray(spec.predict(th, X), dtype=float)
        return float(np.sum((y - f) / f))

    sol = optimize.minimize(
        objective,
        np.asarray(theta0, dtype=float),
        method="SLSQP",
        constraints=[{"type": "eq", "fun": constraint}],
        options={"ftol": 1e-14, "maxiter": 5_000},
    )
    if abs(constraint(sol.x)) > tol * max(1.0, len(y)):
        warnings.warn(
            "ZMPE constraint residual is larger than tolerance; the "
            "zero-percentage-bias property is only approximate for this model.",
            RuntimeWarning,
            stacklevel=3,
        )
    return sol.x, int(sol.nit), bool(sol.success)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def fit(
    spec: ModelSpec,
    X: Any,
    y: np.ndarray | pd.Series,
    method: Method = "ols",
    *,
    max_iter: int = 100,
    tol: float = 1e-12,
    allow_low_df: bool = True,
) -> FitResult:
    """Fit ``spec`` to ``y`` by ``method``.

    Args:
        spec: The model.
        X: Predictors, in whatever shape ``spec.predict`` expects.
        y: Observed responses. Must be positive for multiplicative methods.
        method: ``"ols"``, ``"mupe"`` or ``"zmpe"``.
        max_iter: IRLS iteration cap for MUPE.
        tol: Convergence tolerance.
        allow_low_df: If False, raise instead of warn when degrees of freedom
            fall below :data:`MIN_COMFORTABLE_DF`.

    Raises:
        FitError: On an unknown method, non-positive data where the method
            requires positives, or fewer observations than parameters.
    """
    y = np.asarray(y, dtype=float).ravel()
    method = str(method).lower()  # type: ignore[assignment]
    if method not in METHODS:
        raise FitError(f"Unknown fitting method {method!r}. Allowed: {list(METHODS)}.")

    n = y.size
    p = spec.n_params
    if n < p:
        raise FitError(
            f"{n} observations cannot identify {p} parameters "
            f"({spec.name}). Reduce the model or find more data points."
        )
    if n == p:
        raise FitError(
            f"{n} observations and {p} parameters leaves zero degrees of "
            f"freedom, so the fit would interpolate and no standard error or "
            f"interval is estimable. Refusing to report a number that cannot "
            f"be qualified."
        )

    needs_positive = spec.link == "log" or method in ("mupe", "zmpe")
    if needs_positive and np.any(y <= 0):
        raise FitError(
            f"Method {method!r} with link {spec.link!r} works on proportional "
            f"errors, which are undefined at or below zero. Found "
            f"{int(np.sum(y <= 0))} non-positive response value(s)."
        )

    theta0 = _initial_theta(spec, X, y)
    if theta0.size != p:
        raise FitError(
            f"Initial guess has {theta0.size} entries but the model has {p} "
            f"parameters."
        )

    if method == "ols":
        theta, n_iter, converged = _fit_ols(spec, X, y, theta0)
    elif method == "mupe":
        theta, n_iter, converged = _fit_mupe(spec, X, y, theta0, max_iter, tol)
    else:
        # Seed ZMPE from the MUPE solution: same neighbourhood, far fewer
        # Nelder-Mead steps, and it keeps the two comparable.
        try:
            seed, _, _ = _fit_mupe(spec, X, y, theta0, max_iter, tol)
        except Exception:  # pragma: no cover - defensive, MUPE is the easy fit
            seed = theta0
        theta, n_iter, converged = _fit_zmpe(spec, X, y, seed, tol)

    fitted = np.asarray(spec.predict(theta, X), dtype=float)
    df = n - p

    if method == "ols" and spec.link == "identity":
        resid = y - fitted
    elif method == "ols":
        resid = np.log(y) - np.log(fitted)
    else:
        resid = (y - fitted) / fitted
    sigma = float(np.sqrt(np.sum(resid**2) / df))

    on_log_scale = not (method == "ols" and spec.link == "identity")
    jac = _jacobian(spec, theta, X, on_log_scale=on_log_scale)
    cov = _covariance(jac, sigma, df)

    if df < MIN_COMFORTABLE_DF:
        msg = (
            f"{spec.name}: {n} observations and {p} parameters leaves {df} "
            f"degree(s) of freedom, below the {MIN_COMFORTABLE_DF} normally "
            f"required. Intervals will be very wide and the standard error is "
            f"poorly determined."
        )
        if not allow_low_df:
            raise FitError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    logger.info(
        "Fitted %s by %s: sigma=%.4g, df=%d, mean pct error=%.3g",
        spec.name,
        method.upper(),
        sigma,
        df,
        float(np.mean((y - fitted) / fitted)),
    )

    return FitResult(
        spec=spec,
        method=method,
        theta=np.asarray(theta, dtype=float),
        cov=cov,
        observed=y,
        fitted=fitted,
        sigma=sigma,
        n_obs=n,
        n_params=p,
        n_iter=n_iter,
        converged=converged,
        X=X,
    )


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------
def predict_with_interval(
    result: FitResult,
    X_new: Any,
    *,
    level: float = 0.80,
    kind: IntervalKind = "prediction",
) -> pd.DataFrame:
    """Point estimate plus an interval for new predictor values.

    The distinction between the two kinds is not cosmetic and is the single
    most common error in a cost estimate:

    ``"confidence"``
        An interval on the *mean* response at ``X_new`` -- where the fitted
        line is. It shrinks toward zero as the sample grows.
    ``"prediction"``
        An interval on a *single new observation* at ``X_new`` -- where the
        next actual program will land. It carries the residual scatter as
        well as the parameter uncertainty, so it never shrinks below the
        inherent spread of the data no matter how many points you have.

    A cost estimate is a forecast of one new program, so the prediction
    interval is the correct one and is the default here.

    Returns:
        DataFrame with ``fit``, ``lower``, ``upper``, ``se``, ``level``,
        ``kind``.

    Raises:
        FitError: If ``level`` is outside (0, 1) or the fit has no residual df.
    """
    if not 0.0 < level < 1.0:
        raise FitError(f"level must be in (0, 1), got {level}.")
    if result.df <= 0:
        raise FitError("Cannot form an interval with zero residual degrees of freedom.")
    if kind not in ("prediction", "confidence"):
        raise FitError(
            f"kind must be 'prediction' or 'confidence', got {kind!r}."
        )

    multiplicative = not (result.method == "ols" and result.spec.link == "identity")
    f0 = np.asarray(result.spec.predict(result.theta, X_new), dtype=float)
    grad = _jacobian(result.spec, result.theta, X_new, on_log_scale=multiplicative)

    # Parameter uncertainty at the new point: g' Sigma g, row by row.
    var_mean = np.einsum("ij,jk,ik->i", grad, result.cov, grad)
    var_mean = np.maximum(var_mean, 0.0)

    var = var_mean + (result.sigma**2 if kind == "prediction" else 0.0)
    se = np.sqrt(var)
    tcrit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, result.df))

    if multiplicative:
        lower, upper = f0 * np.exp(-tcrit * se), f0 * np.exp(tcrit * se)
    else:
        lower, upper = f0 - tcrit * se, f0 + tcrit * se

    return pd.DataFrame(
        {
            "fit": f0,
            "lower": lower,
            "upper": upper,
            "se": se,
            "level": level,
            "kind": kind,
        }
    )


# --------------------------------------------------------------------------
# retransformation bias
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RetransformationBias:
    """How much a naive log-log OLS retransformation understates the mean.

    Attributes:
        log_residual_variance: s^2, the variance of the log-space residuals.
        theoretical_factor: exp(s^2 / 2). Under lognormal errors this is
            exactly the ratio of the mean to the median, so it is the factor
            the naive retransformed estimate is short by.
        smearing_factor: Duan's nonparametric smearing estimate,
            mean(exp(residual)). Makes no distributional assumption, so when
            it agrees with ``theoretical_factor`` the lognormal assumption is
            doing no work; when they diverge, the log-space errors are not
            normal and the theoretical factor should not be trusted.
        observed_mean_ratio: mean(y / f) in the fitting sample. Note this is
            *algebraically the same quantity* as ``smearing_factor`` for a
            log-link fit -- exp(log y - log f) is y/f -- so the two agreeing
            is an identity, not corroboration. Both are reported because
            reviewers ask for each by name; ``test_fitting.py`` asserts the
            identity so nobody mistakes it for evidence.
        mupe_ratio: mean(f_mupe / f_ols) over the fitting sample -- how much
            higher MUPE places the curve than naive OLS.
        zmpe_ratio: the same against ZMPE.
        percent_understated: (theoretical_factor - 1) * 100, the headline
            number for a slide.
    """

    log_residual_variance: float
    theoretical_factor: float
    smearing_factor: float
    observed_mean_ratio: float
    mupe_ratio: float | None
    zmpe_ratio: float | None

    @property
    def percent_understated(self) -> float:
        return (self.theoretical_factor - 1.0) * 100.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": [
                    "log_residual_variance",
                    "theoretical_factor_exp_s2_over_2",
                    "duan_smearing_factor",
                    "observed_mean_ratio_y_over_f",
                    "mupe_over_ols",
                    "zmpe_over_ols",
                    "percent_understated",
                ],
                "value": [
                    self.log_residual_variance,
                    self.theoretical_factor,
                    self.smearing_factor,
                    self.observed_mean_ratio,
                    self.mupe_ratio,
                    self.zmpe_ratio,
                    self.percent_understated,
                ],
            }
        )


def retransformation_bias(
    ols: FitResult,
    mupe: FitResult | None = None,
    zmpe: FitResult | None = None,
) -> RetransformationBias:
    """Quantify the bias in ``ols``'s retransformed point estimate.

    Args:
        ols: A log-link OLS fit.
        mupe: Optional MUPE fit of the same model and data, for comparison.
        zmpe: Optional ZMPE fit of the same model and data.

    Raises:
        FitError: If ``ols`` is not a log-link OLS fit, since the bias being
            measured is specific to exponentiating a log-space regression.
    """
    if ols.method != "ols" or ols.spec.link != "log":
        raise FitError(
            "Retransformation bias is a property of exponentiating a log-space "
            f"OLS fit; got method={ols.method!r}, link={ols.spec.link!r}."
        )

    lr = ols.log_residuals
    # The unbiased variance estimate, matching the sigma reported by the fit.
    s2 = float(np.sum(lr**2) / ols.df)

    def _ratio(other: FitResult | None) -> float | None:
        if other is None:
            return None
        if other.fitted.shape != ols.fitted.shape:
            raise FitError(
                "Comparison fit was made on a different number of observations."
            )
        return float(np.mean(other.fitted / ols.fitted))

    return RetransformationBias(
        log_residual_variance=s2,
        theoretical_factor=float(np.exp(s2 / 2.0)),
        smearing_factor=float(np.mean(np.exp(lr))),
        observed_mean_ratio=float(np.mean(ols.observed / ols.fitted)),
        mupe_ratio=_ratio(mupe),
        zmpe_ratio=_ratio(zmpe),
    )


def fitting_scale(result: FitResult) -> str:
    """``"log"`` if the fit's residuals live on a proportional scale.

    Every fit here works on one of two scales: proportional (log residuals for
    OLS with a log link, percentage errors for MUPE and ZMPE) or additive
    (level residuals for OLS with an identity link). Diagnostics have to be
    computed on the same scale the fit minimised on, so this is the single
    place that decision is made.
    """
    return "identity" if (result.method == "ols" and result.spec.link == "identity") else "log"


def design_matrix(result: FitResult) -> np.ndarray:
    """The Jacobian on the fitting scale: the design matrix of the fit.

    For a log-log OLS fit this is literally ``[1, log x_1, ..., log x_k]``, so
    the hat matrix built from it is the textbook one and leverage, Cook's
    distance and DFFITS all reduce to their standard formulas.
    """
    return _jacobian(
        result.spec, result.theta, result.X, on_log_scale=fitting_scale(result) == "log"
    )


def fitting_residuals(result: FitResult) -> np.ndarray:
    """Residuals on the scale the fit actually minimised."""
    if fitting_scale(result) == "identity":
        return result.residuals
    if result.method == "ols":
        return result.log_residuals
    return result.percent_errors


def fit_all_methods(
    spec: ModelSpec, X: Any, y: np.ndarray | pd.Series, **kwargs: Any
) -> dict[str, FitResult]:
    """Fit the same model three ways so the caller can report all of them.

    Reporting one number without its alternatives is what makes an estimate
    hard to defend; this makes the comparison the cheap default.
    """
    return {m: fit(spec, X, y, method=m, **kwargs) for m in METHODS}


def compare_methods(fits: dict[str, FitResult]) -> pd.DataFrame:
    """Side-by-side table of the three fits: SE, CV, bias, parameters."""
    rows = []
    for name, f in fits.items():
        row: dict[str, Any] = {
            "method": name.upper(),
            "sigma": f.sigma,
            "std_error": f.standard_error,
            "cv": f.cv,
            "mean_pct_error": f.mean_percent_error,
            "df": f.df,
            "converged": f.converged,
        }
        row.update(f.params)
        rows.append(row)
    return pd.DataFrame(rows)
