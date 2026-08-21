"""
summary.py - The analyst summary, and the model selection rule.

Turns the three fitted models into the human-readable report an analyst puts in
a basis of estimate, and decides which model is SELECTED.

The selection rule, in order:

1. If the rate coefficient in LC+Rate is significant (``|t| >= TGate``), take
   LC+Rate.
2. Otherwise, if LC could not be fitted but Rate could, take Rate.
3. Otherwise, if the rate slope is significant *and* Rate beats LC by more than
   ``AiccTie`` on AICc, take Rate.
4. Otherwise LC, which is the default.

AICc rather than AIC because these samples are small -- six analogy lots is
normal -- and the correction matters at that size. Where AICc disagrees with
the significance gate the summary says so rather than hiding it, because the
disagreement is exactly what a reviewer needs to see.

Ported unchanged from the original script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cost_core.lotmodel.config import SETTINGS
from cost_core.lotmodel.mathx import lmp_func


def generate_analyst_summary(
    models_context: dict, run_info: dict | None = None
) -> pd.DataFrame:
    ctx = models_context
    cfg = ctx["cfg"]
    ri = run_info or {}

    run_id = ri.get("RunID", cfg["DefaultRunID"])
    program = ri.get("Program", cfg["DefaultProgram"])
    run_label = ri.get("RunLabel", cfg["DefaultRunLabel"])
    base_year = ri.get("BaseYear", cfg["BaseYear"])

    cost_basis_txt = (
        "NOT STATED - declare BaseYear in the RunInfo table"
        if not base_year
        else f"BY{base_year} $K as entered (no escalation applied by the tool)"
    )

    n_keep = ctx["n_keep"]
    n_unit = ctx["n_unit"]
    fit_c = ctx["fit_c"]
    rate_sd = ctx["rate_sd"]
    rate_ok = ctx["rate_ok"]
    rate_why = ctx["rate_why"]

    ln_y = np.log(fit_c)
    ybar = np.mean(ln_y)
    sst = np.sum((ln_y - ybar) ** 2)

    def stat_for(m, k, rate_idx=None):
        if m is None:
            return None
        sse0 = m["SSE"]
        dfe = n_keep - k
        see = np.sqrt(sse0 / dfe) if dfe > 0 else None
        r2 = (1.0 - sse0 / sst) if sst > 0 else None
        adj = (
            (1.0 - (1.0 - r2) * (n_keep - 1) / dfe)
            if (r2 is not None and dfe > 0)
            else None
        )
        cv = (
            np.sqrt(np.exp(see * see) - 1.0)
            if see is not None
            else None
        )

        fit_u = np.exp(m["Fitted"])
        mape = np.mean(np.abs(fit_c - fit_u) / fit_c)
        bias = np.mean(fit_c / fit_u - 1.0)

        kp = k + 1
        sseg = max(sse0, 1e-30)
        aicc = (
            (
                n_keep * np.log(sseg / n_keep)
                + 2 * kp
                + 2 * kp * (kp + 1) / (n_keep - kp - 1)
            )
            if (n_keep - kp - 1 > 0)
            else None
        )

        sec = (
            see * np.sqrt(m["InvDiag"][rate_idx])
            if (rate_idx is not None and see is not None)
            else None
        )
        tc = (
            (m["Beta"][rate_idx] / sec)
            if (sec is not None and sec >= 1e-15)
            else None
        )

        return {
            "SEE": see,
            "R2": r2,
            "Adj": adj,
            "CV": cv,
            "MAPE": mape,
            "Bias": bias,
            "AICc": aicc,
            "T": tc,
        }

    s_lc = stat_for(ctx["mdl_lc"], 2, None)
    s_rt = stat_for(ctx["mdl_rt"], 2, 1)
    s_lcr = stat_for(ctx["mdl_lcr"], 3, 2)

    gs = lambda s, f: s.get(f) if s else None

    t_gate = cfg["TGate"]
    aicc_tie = cfg["AiccTie"]

    lcr_gate = (
        s_lcr is not None
        and gs(s_lcr, "T") is not None
        and abs(gs(s_lcr, "T")) >= t_gate
    )
    rt_gate = (
        s_rt is not None
        and gs(s_rt, "T") is not None
        and abs(gs(s_rt, "T")) >= t_gate
    )

    if lcr_gate:
        sel = "LC+Rate"
    elif (
        ctx["mdl_lc"] is None and ctx["mdl_rt"] is not None
    ):
        sel = "Rate"
    elif (
        ctx["mdl_lc"] is not None
        and rt_gate
        and gs(s_rt, "AICc") is not None
        and gs(s_lc, "AICc") is not None
        and (
            gs(s_rt, "AICc") + aicc_tie < gs(s_lc, "AICc")
        )
    ):
        sel = "Rate"
    elif ctx["mdl_lc"] is not None:
        sel = "LC"
    else:
        sel = None

    aicc_vals = [
        v
        for v in [
            gs(s_lc, "AICc"),
            gs(s_rt, "AICc"),
            gs(s_lcr, "AICc"),
        ]
        if v is not None
    ]
    best_aicc = min(aicc_vals) if aicc_vals else None

    def d_aicc(s):
        if (
            s is None
            or gs(s, "AICc") is None
            or best_aicc is None
        ):
            return None
        return gs(s, "AICc") - best_aicc

    lcr_aicc_disagrees = (
        sel == "LC+Rate"
        and gs(s_lcr, "AICc") is not None
        and gs(s_lc, "AICc") is not None
        and (
            gs(s_lc, "AICc") + aicc_tie
            < gs(s_lcr, "AICc")
        )
    )

    rate_why_not = (
        ""
        if rate_ok
        else (
            "needs >= 4 costed lots"
            if n_keep < 4
            else f"SD(ln qty) {rate_sd:.4f} < {cfg['RateSdFloor']} floor"
        )
    )

    if sel == "LC+Rate":
        sel_note = (
            f"Rate coefficient significant (|t| >= {t_gate})."
            + (
                " Note: AICc favors LC at this sample size - state both in the BOE."
                if lcr_aicc_disagrees
                else ""
            )
        )
    elif sel == "Rate":
        sel_note = f"Slope significant and beats LC by more than {aicc_tie} AICc."
    elif sel == "LC":
        base_msg = "Default model. "
        if not rate_ok:
            base_msg += (
                f"Rate models gated off ({rate_why_not})."
            )
        elif (
            ctx["mdl_lcr"] is not None and not lcr_gate
        ):
            base_msg += f"Rate coefficient not significant (|t| < {t_gate})."
        if gs(s_lc, "AICc") is None:
            base_msg += " n too small for AICc; comparison on coefficient significance only."
        sel_note = base_msg
    else:
        sel_note = "No model could be fitted."

    def fit_txt(m):
        if m is not None:
            return "Yes"
        if not rate_ok:
            return f"No - rate gate ({rate_why_not})"
        return "No - did not converge or singular fit"

    fmt_n = (
        lambda v, f: (
            "n/a"
            if (v is None or pd.isna(v))
            else (
                f"{v:,.2f}"
                if f == "#,##0.00"
                else (
                    f"{v:.4f}"
                    if f == "0.0000"
                    else (
                        f"{v:.6f}"
                        if f == "0.000000"
                        else f"{v:.2f}"
                    )
                )
            )
        )
    )
    fmt_p = (
        lambda v: (
            "n/a"
            if (v is None or pd.isna(v))
            else f"{v * 100:.2f}%"
        )
    )
    fmt_ps = (
        lambda v: (
            "n/a"
            if (v is None or pd.isna(v))
            else f"{v * 100:+.2f}%"
        )
    )

    def mk_col(m, s, b_idx, c_idx):
        dash = "-"
        gc = (
            lambda idx: (
                m["Beta"][idx]
                if (
                    m is not None
                    and idx is not None
                    and len(m["Beta"]) > idx
                )
                else None
            )
        )
        return {
            "Fitted": fit_txt(m),
            "T1": (
                dash
                if m is None
                else fmt_n(
                    np.exp(m["Beta"][0])
                    * cfg["CostUnitScale"],
                    "#,##0.00",
                )
            ),
            "B": (
                dash
                if gc(b_idx) is None
                else fmt_n(gc(b_idx), "0.000000")
            ),
            "BS": (
                dash
                if gc(b_idx) is None
                else fmt_p(2 ** gc(b_idx))
            ),
            "C": (
                dash
                if gc(c_idx) is None
                else fmt_n(gc(c_idx), "0.000000")
            ),
            "CS": (
                dash
                if gc(c_idx) is None
                else fmt_p(2 ** gc(c_idx))
            ),
            "R2": (
                dash
                if m is None
                else fmt_n(gs(s, "R2"), "0.0000")
            ),
            "Adj": (
                dash
                if m is None
                else fmt_n(gs(s, "Adj"), "0.0000")
            ),
            "SEE": (
                dash
                if m is None
                else fmt_n(gs(s, "SEE"), "0.0000")
            ),
            "CV": dash if m is None else fmt_p(gs(s, "CV")),
            "MAPE": (
                dash
                if m is None
                else fmt_p(gs(s, "MAPE"))
            ),
            "Bias": (
                dash
                if m is None
                else fmt_ps(gs(s, "Bias"))
            ),
            "AICc": (
                dash
                if m is None
                else fmt_n(gs(s, "AICc"), "0.00")
            ),
            "DAI": (
                dash
                if m is None
                else fmt_n(d_aicc(s), "0.00")
            ),
            "T": (
                dash
                if m is None
                else fmt_n(gs(s, "T"), "0.00")
            ),
        }

    col_lc = mk_col(ctx["mdl_lc"], s_lc, 1, None)
    col_rt = mk_col(ctx["mdl_rt"], s_rt, None, 1)
    col_br = mk_col(ctx["mdl_lcr"], s_lcr, 1, 2)

    def r5(item, val, a, b, c):
        return {
            "Item": item,
            "Value": val,
            "LC": a,
            "Rate": b,
            "LC+Rate": c,
        }

    rows = [
        r5("Run ID", run_id, "", "", ""),
        r5("Program", program, "", "", ""),
        r5("Run label", run_label, "", "", ""),
        r5("Tool version", cfg["ToolVersion"], "", "", ""),
        r5("Cost basis", cost_basis_txt, "", "", ""),
        r5("Source table", cfg["AnalogyTableName"], "", "", ""),
        r5("Analogy lots in fit", str(n_keep), "", "", ""),
        r5(
            "Quantity-only lots (units held, not fit)",
            str(n_unit - n_keep),
            "",
            "",
            "",
        ),
        r5("SD(ln qty)", fmt_n(rate_sd, "0.0000"), "", "", ""),
        r5(
            "Rate models",
            (
                "enabled"
                if rate_ok
                else f"gated off ({rate_why_not})"
            ),
            "",
            "",
            "",
        ),
        r5("", "", "", "", ""),
        r5(
            "Fitted",
            "",
            col_lc["Fitted"],
            col_rt["Fitted"],
            col_br["Fitted"],
        ),
        r5(
            "SELECTED",
            "",
            "YES" if sel == "LC" else "",
            "YES" if sel == "Rate" else "",
            "YES" if sel == "LC+Rate" else "",
        ),
        r5(
            "T1 ($K)",
            "",
            col_lc["T1"],
            col_rt["T1"],
            col_br["T1"],
        ),
        r5(
            "Learning exponent (b)",
            "",
            col_lc["B"],
            col_rt["B"],
            col_br["B"],
        ),
        r5(
            "Learning curve slope",
            "",
            col_lc["BS"],
            col_rt["BS"],
            col_br["BS"],
        ),
        r5(
            "Rate exponent (c)",
            "",
            col_lc["C"],
            col_rt["C"],
            col_br["C"],
        ),
        r5(
            "Rate slope",
            "",
            col_lc["CS"],
            col_rt["CS"],
            col_br["CS"],
        ),
        r5(
            "R2 (log)",
            "",
            col_lc["R2"],
            col_rt["R2"],
            col_br["R2"],
        ),
        r5(
            "Adj R2",
            "",
            col_lc["Adj"],
            col_rt["Adj"],
            col_br["Adj"],
        ),
        r5(
            "SEE (log)",
            "",
            col_lc["SEE"],
            col_rt["SEE"],
            col_br["SEE"],
        ),
        r5(
            "CV",
            "",
            col_lc["CV"],
            col_rt["CV"],
            col_br["CV"],
        ),
        r5(
            "MAPE",
            "",
            col_lc["MAPE"],
            col_rt["MAPE"],
            col_br["MAPE"],
        ),
        r5(
            "Mean bias",
            "",
            col_lc["Bias"],
            col_rt["Bias"],
            col_br["Bias"],
        ),
        r5(
            "AICc",
            "",
            col_lc["AICc"],
            col_rt["AICc"],
            col_br["AICc"],
        ),
        r5(
            "dAICc",
            "",
            col_lc["DAI"],
            col_rt["DAI"],
            col_br["DAI"],
        ),
        r5(
            "t (rate coefficient)",
            "",
            col_lc["T"],
            col_rt["T"],
            col_br["T"],
        ),
        r5(
            "Selection basis",
            "",
            sel_note if sel in ("LC", None) else "",
            sel_note if sel == "Rate" else "",
            sel_note if sel == "LC+Rate" else "",
        ),
    ]

    return pd.DataFrame(rows)


