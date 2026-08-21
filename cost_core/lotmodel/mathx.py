"""
mathx.py - The estimating maths behind the lot cost model.

Ported unchanged from the original spreadsheet-replacement script, because the
numbers it produces are the reference this package is held to. The three pieces
that matter:

**Lot midpoint.** A lot's cost is priced at its algebraic midpoint -- the unit
whose cost equals the lot average. Under a power curve that midpoint depends on
the slope, which is the parameter being fitted, so the two have to be solved
together.

**OLS in log space.** ``ln(unit cost)`` regressed on ``ln(lot midpoint)`` and,
for the rate models, ``ln(lot quantity)``. Matches the Excel/M fit-space
statistics the original tool reported, including the singularity guard.

**The iterative solve.** Because the midpoint depends on ``b`` and ``b`` is
estimated from a regression that uses the midpoint, the fit iterates to a fixed
point. This is the Goal Seek the workbook did by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_core.lotmodel.config import SETTINGS


def find_col(df_columns: list, candidates: list) -> str | None:
    """Case-insensitive search for the first matching column name."""
    col_map = {c.strip().lower(): c for c in df_columns}
    for cand in candidates:
        if cand.strip().lower() in col_map:
            return col_map[cand.strip().lower()]
    return None


def to_num(val):
    """Clean string currency/commas and convert to numeric float."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned) if cleaned != "" else np.nan
        except ValueError:
            return np.nan
    return np.nan


def lmp_func(
    s: float, e: float, q: float, b: float | None
) -> float | None:
    """Calculate Lot Midpoint (LMP) given Start, End, Qty, and b-slope."""
    if b is None or pd.isna(b):
        return np.nan
    if q <= 1:
        return float(s)
    if abs(b) < 1e-12:
        return (s + e) / 2.0
    if abs(b + 1.0) < 1e-6:
        lo = max(s - 0.5, 1e-6)
        return q / (np.log(e + 0.5) - np.log(lo))

    p = b + 1.0
    lo = max(s - 0.5, 1e-6)
    v = ((e + 0.5) ** p - lo**p) / (p * q)
    if v <= 0:
        return np.nan
    return v ** (1.0 / b)


def ols_fit(
    x_cols: list[np.ndarray], y: np.ndarray, singular_tol: float = 1e-12
):
    """Ordinary Least Squares matching Excel/M Fit Space Statistics."""
    n = len(y)
    k = len(x_cols) + 1  # Intercept + predictors
    x = np.column_stack([np.ones(n)] + x_cols)

    xtx = x.T @ x
    xty = x.T @ y

    det = np.linalg.det(xtx)
    diag_prod = np.prod(np.abs(np.diag(xtx)))
    if diag_prod <= 0 or (abs(det) / diag_prod) < singular_tol:
        return None

    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None

    beta = inv @ xty
    fitted = x @ beta
    res = y - fitted
    sse = float(np.sum(res**2))
    ybar = float(np.mean(y))
    sst = float(np.sum((y - ybar) ** 2))
    ssr = sst - sse
    df = n - k

    s2 = (sse / df) if df > 0 else np.nan
    r2 = (1.0 - (sse / sst)) if sst > 0 else np.nan
    ar2 = (
        (1.0 - (1.0 - r2) * (n - 1) / df)
        if (df > 0 and pd.notna(r2))
        else np.nan
    )
    fstat = (
        ((ssr / (k - 1)) / (sse / df))
        if (df > 0 and k > 1 and sse > 0)
        else np.nan
    )

    se = []
    for a in range(k):
        v = s2 * inv[a, a] if pd.notna(s2) else np.nan
        se.append(np.sqrt(v) if pd.notna(v) and v > 0 else np.nan)

    return {
        "Beta": beta.tolist(),
        "SE": se,
        "Fitted": fitted,
        "SSE": sse,
        "InvDiag": np.diag(inv).tolist(),
        "R2": r2,
        "AdjR2": ar2,
        "SEy": np.sqrt(s2) if pd.notna(s2) and s2 > 0 else np.nan,
        "F": fstat,
        "DF": df,
        "SSreg": ssr,
        "SSresid": sse,
        "N": n,
        "K": k,
    }


def solve_model(
    fit_q: np.ndarray,
    fit_c: np.ndarray,
    fit_se: list[dict],
    use_rate: bool,
    cfg: dict,
):
    """Iterative solver for Learning Curve parameter b (Goal Seek equivalent)."""
    ln_y = np.log(fit_c)
    ln_r = np.log(fit_q)

    def mid_at(b_val):
        return np.array(
            [
                np.log(lmp_func(se["S"], se["E"], q, b_val))
                for se, q in zip(fit_se, fit_q)
            ]
        )

    b = cfg["SeedB"]
    delta = 1.0
    iteration = 0

    while iteration < cfg["MaxIter"] and delta > cfg["Tol"]:
        bp = b
        x_pred = [mid_at(bp), ln_r] if use_rate else [mid_at(bp)]
        fit = ols_fit(x_pred, ln_y, cfg["SingularTol"])
        if fit is None:
            return None
        b = fit["Beta"][1]
        delta = abs(b - bp)
        iteration += 1

    x_pred = [mid_at(b), ln_r] if use_rate else [mid_at(b)]
    final_fit = ols_fit(x_pred, ln_y, cfg["SingularTol"])
    if final_fit is None:
        return None

    resid = abs(final_fit["Beta"][1] - b)
    final_fit["Iter"] = iteration
    final_fit["Delta"] = resid
    final_fit["Converged"] = resid <= cfg["Tol"]
    return final_fit


def track_units(quantities: np.ndarray, prior: int):
    cums = np.cumsum(quantities) + prior
    starts = cums - quantities + 1
    return [{"S": s, "E": e} for s, e in zip(starts, cums)]


