"""
engine.py - The core projection engine: fit three models, cost every lot.

Three candidate models are fitted to the analogy lots and each one is used to
price every estimate lot, so the projections table carries all three side by
side rather than only the winner:

``LC``        ln(cost) = ln(T1) + b * ln(lot midpoint)
``Rate``      ln(cost) = ln(T1) + c * ln(lot quantity)
``LC+Rate``   both terms together

Selection happens later, in :mod:`cost_core.lotmodel.summary`, on the
significance of the rate coefficient with an AICc tiebreak. Keeping the fit and
the selection apart means an analyst can see what the models the tool did
*not* pick would have said, which is usually the first question asked.

Ported unchanged from the original script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_core.lotmodel.config import SETTINGS
from cost_core.lotmodel.mathx import (find_col, lmp_func, ols_fit, solve_model,
                                      to_num, track_units)


def run_lot_cost_model(
    analogy_df: pd.DataFrame,
    estimate_df: pd.DataFrame,
    config_overrides: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    cfg = SETTINGS.copy()
    if config_overrides:
        cfg.update(config_overrides)

    if analogy_df.empty:
        raise ValueError(
            f"Analogy table '{cfg['AnalogyTableName']}' contains no rows."
        )
    if estimate_df.empty:
        raise ValueError(
            f"Estimate table '{cfg['EstimateTableName']}' contains no rows."
        )

    # Resolve Columns - Analogy Table
    a_cols = list(analogy_df.columns)
    a_yr = find_col(
        a_cols,
        ["Year", "FY", "Fiscal Year", "Lot FY", "Lot Year", "FY of Lot"],
    )
    a_q = find_col(
        a_cols,
        [
            "Analogy Qtys",
            "Analogy Qty",
            "AnalogyQtys",
            "AnalogyQty",
            "Analogy Quantity",
            "Qty",
            "Qtys",
            "Quantity",
            "Lot Qty",
            "Lot Quantity",
        ],
    )
    a_c = find_col(
        a_cols,
        [
            "AnalogyUnitCost_K",
            "AnalogyUnitCost_Dollars",
            "AnalogyUnitCost",
            "Analogy Unit Cost",
            "AUC",
            "CP24",
            "AUC_K",
            "AUC ($K)",
            "Unit Cost",
            "UnitCost",
        ],
    )
    a_seq = find_col(
        a_cols,
        [
            "AnalogySeq",
            "Analogy Seq",
            "Analogy Lot",
            "Seq",
            "Sequence",
            "Lot",
            "Lot No",
            "Lot No.",
            "LotNo",
        ],
    )

    if not a_q or not a_c:
        raise ValueError(
            f"Analogy table needs a quantity column and a unit-cost column. Seen: {a_cols}"
        )

    df_a = analogy_df.copy()
    df_a["RowNo"] = np.arange(len(df_a))
    df_a["AQty"] = df_a[a_q].apply(to_num)
    df_a["ACost"] = df_a[a_c].apply(to_num)
    df_a["_Seq"] = (
        df_a[a_seq].apply(to_num) if a_seq else np.nan
    )
    df_a["_Yr"] = df_a[a_yr].apply(to_num) if a_yr else np.nan

    unit_keep = df_a[
        df_a["AQty"].notna() & (df_a["AQty"] > 0)
    ].copy()
    n_unit = len(unit_keep)
    seq_ok = (
        n_unit > 0
        and unit_keep["_Seq"].notna().sum() == n_unit
    )
    yr_ok = (
        n_unit > 0
        and unit_keep["_Yr"].notna().sum() == n_unit
    )

    if seq_ok:
        unit_set = unit_keep.sort_values(
            by=["_Seq", "RowNo"], ascending=[True, True]
        ).reset_index(drop=True)
    elif yr_ok:
        unit_set = unit_keep.sort_values(
            by=["_Yr", "RowNo"], ascending=[True, True]
        ).reset_index(drop=True)
    else:
        unit_set = unit_keep.sort_values(
            by=["RowNo"], ascending=[True]
        ).reset_index(drop=True)

    fit_set = unit_set[
        unit_set["ACost"].notna() & (unit_set["ACost"] > 0)
    ].reset_index(drop=True)
    n_keep = len(fit_set)

    # Resolve Columns - Estimate Table
    e_cols = list(estimate_df.columns)
    c_lot = find_col(
        e_cols,
        [
            "Lot",
            "Lot #",
            "LRIP",
            "Lot ID",
            "Lot No",
            "Lot No.",
            "LotNo",
            "LRIP Lot",
        ],
    )
    c_yr = find_col(
        e_cols,
        [
            "Year",
            "FY",
            "Fiscal Year",
            "Lot FY",
            "Lot Year",
            "Delivery FY",
            "Buy FY",
        ],
    )
    c_qty = find_col(
        e_cols,
        [
            "Qty",
            "Quantity",
            "Units",
            "Lot Qty",
            "Lot Quantity",
            "Estimate Qty",
            "Buy Qty",
            "LRIP Qty",
        ],
    )
    c_cf = find_col(
        e_cols,
        [
            "ComplexityFactor",
            "Complexity Factor",
            "Complexity/CP",
            "Complexity",
            "CF",
        ],
    )

    if not c_qty:
        raise ValueError(
            f"Estimate table needs a quantity column. Seen: {e_cols}"
        )

    df_e = estimate_df.copy()
    df_e["RowNo"] = np.arange(len(df_e))
    df_e["Lot"] = (
        df_e[c_lot].fillna("").astype(str)
        if c_lot
        else [f"Lot {i+1}" for i in range(len(df_e))]
    )
    df_e["Year"] = (
        df_e[c_yr].apply(to_num) if c_yr else np.nan
    )
    df_e["Qty"] = df_e[c_qty].apply(to_num)
    df_e["ComplexityFactor"] = (
        df_e[c_cf].apply(to_num) if c_cf else np.nan
    )

    fcst_ord = (
        df_e[df_e["Qty"].notna() & (df_e["Qty"] > 0)]
        .sort_values(
            by=["Year", "RowNo"], ascending=[True, True]
        )
        .reset_index(drop=True)
    )

    cf_vals = []
    last_cf = cfg["DefaultCF"]
    for val in fcst_ord["ComplexityFactor"]:
        if pd.isna(val) or val <= 0:
            cf_vals.append(last_cf)
        else:
            last_cf = float(val)
            cf_vals.append(last_cf)
    fcst_ord["ComplexityFactor"] = cf_vals

    if n_keep < 3:
        raise ValueError(
            f"Learning curve needs at least 3 analogy lots with both quantity and cost. Found: {n_keep}"
        )
    if len(fcst_ord) < 1:
        raise ValueError("No forecast rows found.")

    # Unit Tracking
    unit_q = unit_set["AQty"].to_numpy()
    unit_se = track_units(unit_q, cfg["FitPriorUnits"])
    complete_idx = unit_set[
        unit_set["ACost"].notna() & (unit_set["ACost"] > 0)
    ].index.tolist()

    fit_q = fit_set["AQty"].to_numpy()
    fit_c = fit_set["ACost"].to_numpy()
    fit_se = [unit_se[i] for i in complete_idx]

    fcst_q = fcst_ord["Qty"].to_numpy()
    fcst_se = track_units(fcst_q, cfg["FcstPriorUnits"])

    # Model Fitting
    ln_r = np.log(fit_q)
    ln_y = np.log(fit_c)
    rate_sd = (
        np.std(ln_r, ddof=1) if len(ln_r) > 1 else np.nan
    )

    rate_why = ""
    if n_keep < 4:
        rate_why = "fewer than 4 analogy lots"
    elif pd.isna(rate_sd):
        rate_why = "no spread in ln(lot qty)"
    elif rate_sd < cfg["RateSdFloor"]:
        rate_why = (
            "lot quantities too uniform for a rate term"
        )

    rate_ok = rate_why == ""

    mdl_lc = solve_model(
        fit_q, fit_c, fit_se, use_rate=False, cfg=cfg
    )
    mdl_rt = (
        ols_fit([ln_r], ln_y, cfg["SingularTol"])
        if rate_ok
        else None
    )
    mdl_lcr = (
        solve_model(
            fit_q, fit_c, fit_se, use_rate=True, cfg=cfg
        )
        if rate_ok
        else None
    )

    if mdl_lc is None:
        raise RuntimeError("Learning curve fit failed.")

    gb = (
        lambda m, i: (
            m["Beta"][i]
            if (m and len(m["Beta"]) > i)
            else np.nan
        )
    )
    gs = (
        lambda m, i: (
            m["SE"][i] if (m and len(m["SE"]) > i) else np.nan
        )
    )
    gf = lambda m, f: m.get(f, np.nan) if m else np.nan
    slope = (
        lambda b_val: (
            round((2**b_val) * 100, 2)
            if pd.notna(b_val)
            else np.nan
        )
    )
    t1_of = (
        lambda m: (
            round(
                np.exp(m["Beta"][0]) * cfg["CostUnitScale"], 2
            )
            if (m and "Beta" in m)
            else np.nan
        )
    )

    def stat_msg(m, name):
        if m is None:
            return f"{name} suppressed: {rate_why}"
        if not m.get("Converged", True):
            return f"{name} NOT CONVERGED"
        return f"{name} ok"

    fit_status = f"{stat_msg(mdl_lc, 'LC')}; {stat_msg(mdl_rt, 'Rate')}; {stat_msg(mdl_lcr, 'LC+Rate')}"
    data_source = f"{cfg['AnalogyTableName']} + {cfg['EstimateTableName']}"

    t1_lc, b_lc = np.exp(mdl_lc["Beta"][0]), gb(mdl_lc, 1)
    t1_rt, b_rt = (
        (np.exp(mdl_rt["Beta"][0]), gb(mdl_rt, 1))
        if mdl_rt
        else (np.nan, np.nan)
    )
    t1_br, b_br, c_br = (
        (
            np.exp(mdl_lcr["Beta"][0]),
            gb(mdl_lcr, 1),
            gb(mdl_lcr, 2),
        )
        if mdl_lcr
        else (np.nan, np.nan, np.nan)
    )

    # Projections on Forecast Lots
    res_df = fcst_ord.copy()
    res_df["FirstUnitInLot"] = [se["S"] for se in fcst_se]
    res_df["LastUnitInLot"] = [se["E"] for se in fcst_se]

    res_df["LC_LMP"] = [
        lmp_func(s, e, q, b_lc)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    res_df["LC_UnitCost"] = (
        t1_lc
        * (res_df["LC_LMP"] ** b_lc)
        * cfg["CostUnitScale"]
    )
    res_df["LC_BaseTotal"] = (
        res_df["LC_UnitCost"]
        * res_df["Qty"]
        * cfg["TotalScale"]
    )
    res_df["LC_AdjTotal"] = (
        res_df["LC_BaseTotal"] * res_df["ComplexityFactor"]
    )

    res_df["RT_LMP"] = [
        lmp_func(s, e, q, b_rt)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    if pd.notna(t1_rt):
        if cfg["ToolMatchProjection"]:
            res_df["RT_UnitCost"] = (
                t1_rt
                * (res_df["RT_LMP"] ** b_rt)
                * cfg["CostUnitScale"]
            )
        else:
            res_df["RT_UnitCost"] = (
                t1_rt
                * (res_df["Qty"] ** b_rt)
                * cfg["CostUnitScale"]
            )
        res_df["RT_BaseTotal"] = (
            res_df["RT_UnitCost"]
            * res_df["Qty"]
            * cfg["TotalScale"]
        )
        res_df["RT_AdjTotal"] = (
            res_df["RT_BaseTotal"] * res_df["ComplexityFactor"]
        )
    else:
        res_df["RT_UnitCost"] = res_df["RT_BaseTotal"] = (
            res_df["RT_AdjTotal"]
        ) = np.nan

    res_df["LCR_LMP"] = [
        lmp_func(s, e, q, b_br)
        for s, e, q in zip(
            res_df["FirstUnitInLot"],
            res_df["LastUnitInLot"],
            res_df["Qty"],
        )
    ]
    if pd.notna(t1_br):
        rate_factor = (
            1.0
            if cfg["ToolMatchProjection"]
            else (res_df["Qty"] ** c_br)
        )
        res_df["LCR_UnitCost"] = (
            t1_br
            * (res_df["LCR_LMP"] ** b_br)
            * rate_factor
            * cfg["CostUnitScale"]
        )
        res_df["LCR_BaseTotal"] = (
            res_df["LCR_UnitCost"]
            * res_df["Qty"]
            * cfg["TotalScale"]
        )
        res_df["LCR_AdjTotal"] = (
            res_df["LCR_BaseTotal"]
            * res_df["ComplexityFactor"]
        )
    else:
        res_df["LCR_UnitCost"] = res_df["LCR_BaseTotal"] = (
            res_df["LCR_AdjTotal"]
        ) = np.nan

    # Fit Statistics Columns
    rnd = (
        lambda v, d: (
            round(v, d) if pd.notna(v) else np.nan
        )
    )
    pct = (
        lambda v: (
            round(v * 100, 2) if pd.notna(v) else np.nan
        )
    )

    res_df["LC_T1"] = t1_of(mdl_lc)
    res_df["LC_Icept"] = rnd(gb(mdl_lc, 0), 4)
    res_df["LC_IceptSE"] = rnd(gs(mdl_lc, 0), 4)
    res_df["LC_Learn"] = rnd(gb(mdl_lc, 1), 4)
    res_df["LC_LearnSE"] = rnd(gs(mdl_lc, 1), 4)
    res_df["LC_LearnSlope"] = slope(gb(mdl_lc, 1))
    res_df["LC_R2"] = pct(gf(mdl_lc, "R2"))
    res_df["LC_SEy"] = rnd(gf(mdl_lc, "SEy"), 4)
    res_df["LC_F"] = rnd(gf(mdl_lc, "F"), 2)
    res_df["LC_df"] = gf(mdl_lc, "DF")
    res_df["LC_SSreg"] = rnd(gf(mdl_lc, "SSreg"), 4)
    res_df["LC_SSresid"] = rnd(gf(mdl_lc, "SSresid"), 4)
    res_df["LC_AdjR2"] = pct(gf(mdl_lc, "AdjR2"))

    res_df["RT_T1"] = t1_of(mdl_rt)
    res_df["RT_Icept"] = rnd(gb(mdl_rt, 0), 4)
    res_df["RT_IceptSE"] = rnd(gs(mdl_rt, 0), 4)
    res_df["RT_Rate"] = rnd(gb(mdl_rt, 1), 4)
    res_df["RT_RateSE"] = rnd(gs(mdl_rt, 1), 4)
    res_df["RT_RateSlope"] = slope(gb(mdl_rt, 1))
    res_df["RT_R2"] = pct(gf(mdl_rt, "R2"))
    res_df["RT_SEy"] = rnd(gf(mdl_rt, "SEy"), 4)
    res_df["RT_F"] = rnd(gf(mdl_rt, "F"), 2)
    res_df["RT_df"] = gf(mdl_rt, "DF")
    res_df["RT_SSreg"] = rnd(gf(mdl_rt, "SSreg"), 4)
    res_df["RT_SSresid"] = rnd(gf(mdl_rt, "SSresid"), 4)
    res_df["RT_AdjR2"] = pct(gf(mdl_rt, "AdjR2"))

    res_df["LCR_T1"] = t1_of(mdl_lcr)
    res_df["LCR_Icept"] = rnd(gb(mdl_lcr, 0), 4)
    res_df["LCR_IceptSE"] = rnd(gs(mdl_lcr, 0), 4)
    res_df["LCR_Learn"] = rnd(gb(mdl_lcr, 1), 4)
    res_df["LCR_LearnSE"] = rnd(gs(mdl_lcr, 1), 4)
    res_df["LCR_LearnSlope"] = slope(gb(mdl_lcr, 1))
    res_df["LCR_Rate"] = rnd(gb(mdl_lcr, 2), 4)
    res_df["LCR_RateSE"] = rnd(gs(mdl_lcr, 2), 4)
    res_df["LCR_RateSlope"] = slope(gb(mdl_lcr, 2))
    res_df["LCR_R2"] = pct(gf(mdl_lcr, "R2"))
    res_df["LCR_SEy"] = rnd(gf(mdl_lcr, "SEy"), 4)
    res_df["LCR_F"] = rnd(gf(mdl_lcr, "F"), 2)
    res_df["LCR_df"] = gf(mdl_lcr, "DF")
    res_df["LCR_SSreg"] = rnd(gf(mdl_lcr, "SSreg"), 4)
    res_df["LCR_SSresid"] = rnd(gf(mdl_lcr, "SSresid"), 4)
    res_df["LCR_AdjR2"] = pct(gf(mdl_lcr, "AdjR2"))

    res_df["FitStatus"] = fit_status
    res_df["DataSource"] = data_source

    if "Lot" not in res_df.columns:
        res_df["Lot"] = [
            f"Lot {i+1}" for i in range(len(res_df))
        ]
    if "Year" not in res_df.columns:
        res_df["Year"] = np.nan
    if "ComplexityFactor" not in res_df.columns:
        res_df["ComplexityFactor"] = cfg["DefaultCF"]

    round_cols_4 = [
        "LC_LMP",
        "RT_LMP",
        "LCR_LMP",
        "ComplexityFactor",
    ]
    round_cols_2 = [
        "LC_UnitCost",
        "LC_BaseTotal",
        "LC_AdjTotal",
        "RT_UnitCost",
        "RT_BaseTotal",
        "RT_AdjTotal",
        "LCR_UnitCost",
        "LCR_BaseTotal",
        "LCR_AdjTotal",
    ]
    for c in round_cols_4:
        res_df[c] = res_df[c].apply(lambda v: rnd(v, 4))
    for c in round_cols_2:
        res_df[c] = res_df[c].apply(lambda v: rnd(v, 2))

    res_df["Lot"] = res_df["Lot"].astype(str)
    res_df["Year"] = pd.to_numeric(
        res_df["Year"], errors="coerce"
    ).astype("Int64")
    res_df["Qty"] = pd.to_numeric(
        res_df["Qty"], errors="coerce"
    ).astype("Int64")
    res_df["FirstUnitInLot"] = pd.to_numeric(
        res_df["FirstUnitInLot"], errors="coerce"
    ).astype("Int64")
    res_df["LastUnitInLot"] = pd.to_numeric(
        res_df["LastUnitInLot"], errors="coerce"
    ).astype("Int64")

    rename_dict = {
        "Lot": "Lot",
        "Year": "Fiscal Year",
        "Qty": "Lot Quantity",
        "FirstUnitInLot": "First Unit in Lot",
        "LastUnitInLot": "Last Unit in Lot",
        "ComplexityFactor": "Complexity Factor",
        "LC_LMP": "LC Lot Midpoint (unit no.)",
        "LC_UnitCost": "LC Unit Cost ($K)",
        "LC_BaseTotal": "LC Lot Cost Before Complexity ($)",
        "LC_AdjTotal": "LC Lot Cost After Complexity ($)",
        "RT_LMP": "Rate Lot Midpoint (unit no.)",
        "RT_UnitCost": "Rate Unit Cost ($K)",
        "RT_BaseTotal": "Rate Lot Cost Before Complexity ($)",
        "RT_AdjTotal": "Rate Lot Cost After Complexity ($)",
        "LCR_LMP": "LC+Rate Lot Midpoint (unit no.)",
        "LCR_UnitCost": "LC+Rate Unit Cost ($K)",
        "LCR_BaseTotal": (
            "LC+Rate Lot Cost Before Complexity ($)"
        ),
        "LCR_AdjTotal": (
            "LC+Rate Lot Cost After Complexity ($)"
        ),
        "LC_T1": "LC T1 First Unit Cost ($K)",
        "LC_Icept": "LC Intercept Coeff",
        "LC_IceptSE": "LC Intercept SE",
        "LC_Learn": "LC Learning Coeff",
        "LC_LearnSE": "LC Learning SE",
        "LC_LearnSlope": "LC Learning Slope (%)",
        "LC_R2": "LC R2 (%)",
        "LC_SEy": "LC SEy",
        "LC_F": "LC F",
        "LC_df": "LC df",
        "LC_SSreg": "LC SSreg",
        "LC_SSresid": "LC SSresid",
        "LC_AdjR2": "LC Adj R2 (%)",
        "RT_T1": "Rate T1 First Unit Cost ($K)",
        "RT_Icept": "Rate Intercept Coeff",
        "RT_IceptSE": "Rate Intercept SE",
        "RT_Rate": "Rate Coeff",
        "RT_RateSE": "Rate SE",
        "RT_RateSlope": "Rate Slope (%)",
        "RT_R2": "Rate R2 (%)",
        "RT_SEy": "Rate SEy",
        "RT_F": "Rate F",
        "RT_df": "Rate df",
        "RT_SSreg": "Rate SSreg",
        "RT_SSresid": "Rate SSresid",
        "RT_AdjR2": "Rate Adj R2 (%)",
        "LCR_T1": "LC+Rate T1 First Unit Cost ($K)",
        "LCR_Icept": "LC+Rate Intercept Coeff",
        "LCR_IceptSE": "LC+Rate Intercept SE",
        "LCR_Learn": "LC+Rate Learning Coeff",
        "LCR_LearnSE": "LC+Rate Learning SE",
        "LCR_LearnSlope": "LC+Rate Learning Slope (%)",
        "LCR_Rate": "LC+Rate Rate Coeff",
        "LCR_RateSE": "LC+Rate Rate SE",
        "LCR_RateSlope": "LC+Rate Rate Slope (%)",
        "LCR_R2": "LC+Rate R2 (%)",
        "LCR_SEy": "LC+Rate SEy",
        "LCR_F": "LC+Rate F",
        "LCR_df": "LC+Rate df",
        "LCR_SSreg": "LC+Rate SSreg",
        "LCR_SSresid": "LC+Rate SSresid",
        "LCR_AdjR2": "LC+Rate Adj R2 (%)",
        "FitStatus": "Fit Status",
        "DataSource": "Input Table",
    }

    ordered_cols = list(rename_dict.keys())
    projections_df = res_df[ordered_cols].rename(
        columns=rename_dict
    )

    models_context = {
        "mdl_lc": mdl_lc,
        "mdl_rt": mdl_rt,
        "mdl_lcr": mdl_lcr,
        "fit_q": fit_q,
        "fit_c": fit_c,
        "fit_se": fit_se,
        "n_keep": n_keep,
        "n_unit": n_unit,
        "rate_sd": rate_sd,
        "rate_ok": rate_ok,
        "rate_why": rate_why,
        "t1_lc": t1_lc,
        "b_lc": b_lc,
        "t1_rt": t1_rt,
        "b_rt": b_rt,
        "t1_br": t1_br,
        "b_br": b_br,
        "c_br": c_br,
        "cfg": cfg,
    }

    return projections_df, models_context


