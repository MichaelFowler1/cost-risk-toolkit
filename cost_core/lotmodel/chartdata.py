"""
chartdata.py - Actual against fitted, per analogy lot, for the fit charts.

One row per analogy lot with the observed unit cost and what each of the three
models predicts for it, which is what the scatter charts in the workbook plot.
This is the residual view: it shows which analogy lot each model misses, and by
how much.

Ported unchanged from the original script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_core.lotmodel.mathx import lmp_func


def generate_fit_chart_data(
    models_context: dict,
) -> pd.DataFrame:
    """Compute the fit chart evaluation data matching M code Fit Chart Data."""
    ctx = models_context
    cfg = ctx["cfg"]

    fit_q = ctx["fit_q"]
    fit_c = ctx["fit_c"]
    fit_se = ctx["fit_se"]

    t1_lc, b_lc = ctx["t1_lc"], ctx["b_lc"]
    t1_rt, b_rt = ctx["t1_rt"], ctx["b_rt"]
    t1_br, b_br, c_br = (
        ctx["t1_br"],
        ctx["b_br"],
        ctx["c_br"],
    )

    rnd = (
        lambda v, d: (
            round(v, d)
            if (v is not None and pd.notna(v))
            else np.nan
        )
    )

    chart_rows = []
    for k in range(len(fit_q)):
        lot_no = k + 1
        a_qty = int(fit_q[k])
        first_u = int(fit_se[k]["S"])
        last_u = int(fit_se[k]["E"])
        actual = rnd(fit_c[k] * cfg["CostUnitScale"], 2)

        # LC calculations
        lc_mid = rnd(
            lmp_func(first_u, last_u, a_qty, b_lc), 4
        )
        lc_est = (
            rnd(
                t1_lc
                * (
                    lmp_func(first_u, last_u, a_qty, b_lc)
                    ** b_lc
                )
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_lc)
            else np.nan
        )
        lc_res = (
            rnd(actual - lc_est, 2)
            if pd.notna(lc_est)
            else np.nan
        )
        lc_resp = (
            rnd(((actual / lc_est) - 1.0) * 100, 2)
            if (pd.notna(lc_est) and lc_est != 0)
            else np.nan
        )

        # Rate calculations (against Analogy Lot Quantity)
        rt_est = (
            rnd(
                t1_rt
                * (a_qty**b_rt)
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_rt)
            else np.nan
        )
        rt_res = (
            rnd(actual - rt_est, 2)
            if pd.notna(rt_est)
            else np.nan
        )
        rt_resp = (
            rnd(((actual / rt_est) - 1.0) * 100, 2)
            if (pd.notna(rt_est) and rt_est != 0)
            else np.nan
        )

        # LC+Rate calculations (True fitted value with rate term)
        lcr_mid = rnd(
            lmp_func(first_u, last_u, a_qty, b_br), 4
        )
        lcr_est = (
            rnd(
                t1_br
                * (
                    lmp_func(first_u, last_u, a_qty, b_br)
                    ** b_br
                )
                * (a_qty**c_br)
                * cfg["CostUnitScale"],
                2,
            )
            if pd.notna(t1_br)
            else np.nan
        )
        lcr_res = (
            rnd(actual - lcr_est, 2)
            if pd.notna(lcr_est)
            else np.nan
        )
        lcr_resp = (
            rnd(((actual / lcr_est) - 1.0) * 100, 2)
            if (pd.notna(lcr_est) and lcr_est != 0)
            else np.nan
        )

        chart_rows.append(
            {
                "Analogy Lot No.": lot_no,
                "Analogy Lot Quantity": a_qty,
                "First Unit in Lot": first_u,
                "Last Unit in Lot": last_u,
                "Actual AUC ($K)": actual,
                "LC Lot Midpoint": lc_mid,
                "LC Estimated AUC ($K)": lc_est,
                "LC Residual ($K)": lc_res,
                "LC Residual (%)": lc_resp,
                "Rate Estimated AUC ($K)": rt_est,
                "Rate Residual ($K)": rt_res,
                "Rate Residual (%)": rt_resp,
                "LC+Rate Lot Midpoint": lcr_mid,
                "LC+Rate Estimated AUC ($K)": lcr_est,
                "LC+Rate Residual ($K)": lcr_res,
                "LC+Rate Residual (%)": lcr_resp,
            }
        )

    chart_df = pd.DataFrame(chart_rows)
    return chart_df


