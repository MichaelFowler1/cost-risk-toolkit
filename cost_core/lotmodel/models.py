# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
models.py - The three lot models, expressed for :mod:`cost_core.fitting`.

The lot engine used to carry its own least-squares solver. It no longer does:
the three candidate models are declared here as :class:`~cost_core.fitting.ModelSpec`
objects and every coefficient in the projections table comes back through
``cost_core.fitting.fit``. One estimator, one covariance, one place where a
numerical decision is made.

Two things about the arrangement are deliberate and were settled by
measurement rather than by taste.

**The midpoint iteration stays outside the fit.** A lot's cost is priced at its
algebraic midpoint, and under a power curve that midpoint depends on the slope
being estimated. Making the design matrix a function of ``b`` *inside* the
model would let the optimiser move the regressors as it moves the parameters,
and it then converges on a different stationary point, about 1e-3 relative away
in ``b``. So each pass fits a frozen design and the fixed point is chased by
the loop in :func:`solve_lot_model`, exactly as the original Goal Seek did.

**Every spec carries an analytic Jacobian.** ``d f / d theta`` is
``f[:, None] * X`` on the level scale, which is exact. ``fitting`` falls back
to central differences when a spec has none, and those are good to about 1e-10
relative, which is not tight enough here: the covariance of the LC+Rate model
came out 1e-8 away, an order of magnitude past what the goldens allow.

The specs fit the observed cost in levels, not its logarithm. The log link does
the transform, which is what makes ``fitting``'s own positivity check the
engine's positivity check.
"""

from __future__ import annotations

import contextlib
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from cost_core import fitting
from cost_core.fitting import FitResult, ModelSpec
from cost_core.lotmodel.mathx import lmp_func

#: Parameter names per model, in theta order. Index 0 is always the log scale,
#: which is what ``log_scale_index`` tells :mod:`cost_core.fitting`.
PARAM_NAMES = {
    "LC": ("log_t1", "b"),
    "Rate": ("log_t1", "c"),
    "LC+Rate": ("log_t1", "b", "c"),
}


def lot_spec(model: str) -> ModelSpec:
    """The lot model ``model`` as a ModelSpec over a frozen design matrix.

    ``X`` is the ``(n, p)`` log-space design matrix, already evaluated at the
    current ``b``: a column of ones, then ``log(lot midpoint)`` for the models
    that learn, then ``log(lot quantity)`` for the models that carry a rate
    term. The prediction is on the level scale, ``exp(X @ theta)``.

    Raises:
        ValueError: If ``model`` is not one of the three.
    """
    if model not in PARAM_NAMES:
        raise ValueError(
            f"Unknown lot model {model!r}. Expected one of {sorted(PARAM_NAMES)}."
        )

    def predict(theta: np.ndarray, X: Any) -> np.ndarray:
        return np.exp(np.asarray(X, dtype=float) @ np.asarray(theta, dtype=float))

    def jacobian(theta: np.ndarray, X: Any) -> np.ndarray:
        # On the level scale, so d/dtheta exp(X theta) = exp(X theta) * X.
        # fitting divides by f itself when it wants the log scale, and the
        # design matrix it recovers is then X to within an ulp.
        X = np.asarray(X, dtype=float)
        f = np.exp(X @ np.asarray(theta, dtype=float))
        return f[:, None] * X

    def initial(X: Any, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.linalg.lstsq(X, np.log(np.asarray(y, dtype=float)), rcond=None)[0]

    return ModelSpec(
        name=model,
        param_names=PARAM_NAMES[model],
        predict=predict,
        link="log",
        log_scale_index=0,
        initial=initial,
        jacobian=jacobian,
    )


# --------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------
def log_midpoints(fit_se: list, fit_q: np.ndarray, b: float) -> np.ndarray:
    """``log`` of the lot midpoint of each analogy lot at the slope ``b``."""
    return np.array([
        np.log(lmp_func(se["S"], se["E"], q, b))
        for se, q in zip(fit_se, fit_q)
    ])


def design(fit_se: list, fit_q: np.ndarray, b: float, *, use_rate: bool,
           use_lc: bool = True) -> np.ndarray:
    """The log-space design matrix at the slope ``b``.

    ``use_lc`` is False only for the pure Rate model, whose regressors do not
    depend on ``b`` at all and so need no iteration.
    """
    cols = [np.ones(len(fit_q))]
    if use_lc:
        cols.append(log_midpoints(fit_se, fit_q, b))
    if use_rate:
        cols.append(np.log(np.asarray(fit_q, dtype=float)))
    return np.column_stack(cols)


# --------------------------------------------------------------------------
# the singular guard
# --------------------------------------------------------------------------
def is_singular(X: np.ndarray, singular_tol: float) -> bool:
    """True where the design has no unique least-squares solution.

    The test is the determinant of ``X'X`` scaled by the product of its own
    diagonal, on the un-centred matrix, which is the guard the engine has
    always applied. It is kept because it is what decides whether a model is
    reported at all: a design that fails it returns None and the engine drops
    that model rather than reporting whatever a pseudo-inverse would pick.

    The scaled determinant is only a meaningful number down to about 1e-16,
    where double precision runs out. ``SETTINGS["SingularTol"]`` is 1e-12, four
    orders of magnitude clear of that, and it is caller-overridable
    (``run_lot_cost_model`` merges ``config_overrides`` into ``cfg`` with no
    allow-list). Loosened below about 1e-16 the guard stops meaning anything:
    a design that is rank-deficient in double precision passes it, and the two
    solvers then disagree about how to fail. The old normal equations returned
    a dict of nonsense -- one measured case gives R^2 = -707 and a Beta with no
    relation to the data -- and ``fitting.fit`` walks the same design into an
    overflowing Jacobian and raises instead. Neither answer is worth anything;
    the difference is that one says so. Nothing in the corpus reaches it, and
    3661 randomised designs at 1e-12 show no disagreement at all.
    """
    X = np.asarray(X, dtype=float)
    xtx = X.T @ X
    det = np.linalg.det(xtx)
    diag_prod = np.prod(np.abs(np.diag(xtx)))
    if diag_prod <= 0 or (abs(det) / diag_prod) < singular_tol:
        return True
    try:
        np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return True
    return False


# --------------------------------------------------------------------------
# warning policy
# --------------------------------------------------------------------------
_LOW_DF = re.compile(r".*degree\(s\) of freedom, below the \d+ normally required")


def _is_low_df_warning(w: warnings.WarningMessage) -> bool:
    return issubclass(w.category, RuntimeWarning) and bool(_LOW_DF.match(str(w.message)))


@contextlib.contextmanager
def suppress_low_df():
    """Drop ``fitting.fit``'s "below the 3 normally required" warning.

    The engine already refuses fewer than three analogy lots and gates the rate
    models off below four, and it has never warned about the degrees of freedom
    that leaves. Passing that warning through would be a change in behaviour
    dressed up as a refactor, so it is dropped here and nowhere else.

    Every other warning is re-emitted unchanged. A module-level filter would be
    the shorter way to write this and the wrong one: it would also swallow the
    MUPE non-convergence warning, which is the one warning in ``fitting`` an
    analyst has to see.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for w in caught:
        if _is_low_df_warning(w):
            continue
        warnings.warn(w.message, w.category, stacklevel=3)


# --------------------------------------------------------------------------
# one fit
# --------------------------------------------------------------------------
def fit_frozen(model: str, X: np.ndarray, y_level: np.ndarray,
               singular_tol: float) -> FitResult | None:
    """One OLS through ``fitting.fit`` on a frozen design. None if singular.

    ``y_level`` is the observed cost, not its log: the spec's log link does the
    transform.
    """
    X = np.asarray(X, dtype=float)
    if is_singular(X, singular_tol):
        return None
    with suppress_low_df():
        return fitting.fit(lot_spec(model), X, np.asarray(y_level, dtype=float),
                           method="ols")


def ols_dict_from(result: FitResult) -> dict:
    """The fit-space statistics the engine and the summary read, from a FitResult.

    The shape is the one the engine has always consumed, and each entry is
    derived from the FitResult by the formulas the old solver used:

        Beta      the coefficients
        SE        the square roots of the covariance diagonal
        Fitted    the fitted values on the LOG scale, because that is what the
                  summary exponentiates and what the chart data differences
        SSE       sigma^2 * df
        InvDiag   the covariance diagonal divided by sigma^2, i.e. the diagonal
                  of ``(X'X)^-1``, which the summary needs to form the rate
                  t-statistic
        R2 / AdjR2 / F / SSreg / SSresid from the total sum of squares of the
                  log costs
        SEy       sigma
        DF, N, K  sizes
    """
    n = int(result.n_obs)
    k = int(result.n_params)
    df = n - k

    y_log = np.log(np.asarray(result.observed, dtype=float))
    fitted_log = np.log(np.asarray(result.fitted, dtype=float))

    sigma = float(result.sigma)
    s2 = sigma * sigma if df > 0 else np.nan
    sse = s2 * df if df > 0 else float(np.sum((y_log - fitted_log) ** 2))

    ybar = float(np.mean(y_log))
    sst = float(np.sum((y_log - ybar) ** 2))
    ssr = sst - sse

    cov_diag = np.diag(np.asarray(result.cov, dtype=float))
    inv_diag = (cov_diag / s2) if (df > 0 and np.isfinite(s2)) else np.full(k, np.nan)

    r2 = (1.0 - (sse / sst)) if sst > 0 else np.nan
    ar2 = (1.0 - (1.0 - r2) * (n - 1) / df) if (df > 0 and pd.notna(r2)) else np.nan
    fstat = ((ssr / (k - 1)) / (sse / df)) if (df > 0 and k > 1 and sse > 0) else np.nan

    se = []
    for a in range(k):
        v = cov_diag[a] if pd.notna(s2) else np.nan
        se.append(np.sqrt(v) if pd.notna(v) and v > 0 else np.nan)

    return {
        "Beta": np.asarray(result.theta, dtype=float).tolist(),
        "SE": se,
        "Fitted": fitted_log,
        "SSE": float(sse),
        "InvDiag": np.asarray(inv_diag, dtype=float).tolist(),
        "R2": r2,
        "AdjR2": ar2,
        "SEy": np.sqrt(s2) if pd.notna(s2) and s2 > 0 else np.nan,
        "F": fstat,
        "DF": df,
        "SSreg": float(ssr),
        "SSresid": float(sse),
        "N": n,
        "K": k,
    }


def _degenerate_ols_dict(X: np.ndarray, y_log: np.ndarray) -> dict:
    """The same statistics when there are no residual degrees of freedom.

    ``fitting.fit`` refuses a design with ``n <= p``, on the good grounds that
    an interpolating fit cannot be qualified. The engine's older contract is
    that a model is dropped only when its design is singular, and that a fit it
    cannot qualify comes back with NaN in the entries that need a variance. Two
    defensible positions, and swapping one for the other is a behaviour change
    that does not belong inside a refactor, so this branch keeps the older one.

    It is unreachable from ``run_lot_cost_model`` as it stands, which refuses
    fewer than three analogy lots and gates the rate models off below four; the
    smallest designs it can build are (3, 2) and (4, 3). Instrumenting every
    ``fitting.fit`` call the lot engine makes across the whole test suite finds
    8895 of them, the smallest at one degree of freedom and none at ``n <= p``.
    So this branch exists to keep :func:`ols_via_fitting` a drop-in for the
    solver it replaced, not because the engine needs it, and
    tests/test_lotmodel.py calls it directly rather than leaving it unrun.

    :func:`cost_core.lotmodel.enrich.fit_on_design` and
    ``enrich.compare_fitting_methods`` do not apply this branch, and should
    not: they only ever run on a design the engine has already fitted, so
    ``n > p`` holds by construction there, and a guard on a condition that
    cannot arise would be a claim that it can.
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ (X.T @ y_log)
    fitted = X @ beta
    sse = float(np.sum((y_log - fitted) ** 2))
    ybar = float(np.mean(y_log))
    sst = float(np.sum((y_log - ybar) ** 2))
    ssr = sst - sse
    df = n - k
    s2 = (sse / df) if df > 0 else np.nan
    r2 = (1.0 - (sse / sst)) if sst > 0 else np.nan
    ar2 = (1.0 - (1.0 - r2) * (n - 1) / df) if (df > 0 and pd.notna(r2)) else np.nan
    fstat = ((ssr / (k - 1)) / (sse / df)) if (df > 0 and k > 1 and sse > 0) else np.nan
    se = []
    for a in range(k):
        v = s2 * inv[a, a] if pd.notna(s2) else np.nan
        se.append(np.sqrt(v) if pd.notna(v) and v > 0 else np.nan)
    return {
        "Beta": beta.tolist(), "SE": se, "Fitted": fitted, "SSE": sse,
        "InvDiag": np.diag(inv).tolist(), "R2": r2, "AdjR2": ar2,
        "SEy": np.sqrt(s2) if pd.notna(s2) and s2 > 0 else np.nan,
        "F": fstat, "DF": df, "SSreg": ssr, "SSresid": sse, "N": n, "K": k,
    }


def ols_via_fitting(model: str, X: np.ndarray, y_level: np.ndarray,
                    singular_tol: float) -> dict | None:
    """One OLS on a frozen design, reported as the engine's fit-space dict."""
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        if is_singular(X, singular_tol):
            return None
        return _degenerate_ols_dict(X, np.log(np.asarray(y_level, dtype=float)))
    result = fit_frozen(model, X, y_level, singular_tol)
    if result is None:
        return None
    return ols_dict_from(result)


# --------------------------------------------------------------------------
# the iterated solve
# --------------------------------------------------------------------------
def solve_lot_model(fit_q: np.ndarray, fit_c: np.ndarray, fit_se: list,
                    use_rate: bool, cfg: dict) -> dict | None:
    """Fit the LC or LC+Rate model, chasing the midpoint to a fixed point.

    The midpoint depends on ``b`` and ``b`` is estimated from a regression that
    uses the midpoint, so the two are solved together: seed ``b``, rebuild the
    design, refit, repeat until ``b`` stops moving. This is the Goal Seek the
    original workbook did by hand.

    One more fit is taken at the settled ``b`` and it is that fit which is
    reported, with ``Delta`` measuring how far its own coefficient sits from
    the ``b`` its design was built at. ``Converged`` says whether that gap is
    inside the tolerance; a run that exhausts ``MaxIter`` says so.
    """
    fit_q = np.asarray(fit_q, dtype=float)
    fit_c = np.asarray(fit_c, dtype=float)
    model = "LC+Rate" if use_rate else "LC"

    b = cfg["SeedB"]
    delta = 1.0
    iteration = 0

    while iteration < cfg["MaxIter"] and delta > cfg["Tol"]:
        bp = b
        X = design(fit_se, fit_q, bp, use_rate=use_rate)
        fit = ols_via_fitting(model, X, fit_c, cfg["SingularTol"])
        if fit is None:
            return None
        b = fit["Beta"][1]
        delta = abs(b - bp)
        iteration += 1

    X = design(fit_se, fit_q, b, use_rate=use_rate)
    final_fit = ols_via_fitting(model, X, fit_c, cfg["SingularTol"])
    if final_fit is None:
        return None

    resid = abs(final_fit["Beta"][1] - b)
    final_fit["Iter"] = iteration
    final_fit["Delta"] = resid
    final_fit["Converged"] = resid <= cfg["Tol"]
    return final_fit


def solve_rate_model(fit_q: np.ndarray, fit_c: np.ndarray,
                     cfg: dict) -> dict | None:
    """Fit the pure Rate model: one OLS on ``[1, log qty]``.

    No iteration, because nothing in this design depends on the slope.
    """
    fit_q = np.asarray(fit_q, dtype=float)
    X = np.column_stack([np.ones(len(fit_q)), np.log(fit_q)])
    return ols_via_fitting("Rate", X, np.asarray(fit_c, dtype=float),
                           cfg["SingularTol"])
