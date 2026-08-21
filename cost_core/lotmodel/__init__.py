"""
cost_core.lotmodel - Learning curve / rate analysis on analogy and estimate lots.

The estimating engine from the original spreadsheet-replacement tool, ported
module by module so its numbers are unchanged, plus a layer of additional
statistics the original did not carry.

The original path, numerically identical to the spreadsheet:

    from cost_core.lotmodel import run_lot_cost_model, generate_analyst_summary

    projections, ctx = run_lot_cost_model(analogy_df, estimate_df, overrides)
    summary = generate_analyst_summary(ctx, run_info)

The added path, which reads the same fitted models and reports what the point
estimates alone cannot say -- unbiased refits, prediction intervals, influence
diagnostics and a risk distribution on the buy:

    from cost_core.lotmodel import enrich_run

    extras = enrich_run(ctx, projections, run_info)

``tests/test_lotmodel.py`` holds a golden master: the ported engine must
reproduce the original script's output on the reference data to the last
decimal, so the added statistics can never be confused with a change to the
estimate itself.
"""

from cost_core.lotmodel.chartdata import generate_fit_chart_data
from cost_core.lotmodel.config import SETTINGS
from cost_core.lotmodel.engine import run_lot_cost_model
from cost_core.lotmodel.mathx import (find_col, lmp_func, ols_fit, solve_model,
                                      to_num, track_units)
from cost_core.lotmodel.summary import generate_analyst_summary
from cost_core.lotmodel.workbook import save_complete_excel_workbook

__all__ = [
    "SETTINGS",
    "find_col",
    "generate_analyst_summary",
    "generate_fit_chart_data",
    "lmp_func",
    "ols_fit",
    "run_lot_cost_model",
    "save_complete_excel_workbook",
    "solve_model",
    "to_num",
    "track_units",
]

# The added statistics layer. Imported last because it reads the engine's
# output rather than feeding it.
from cost_core.lotmodel.enrich import (BuyRisk, Enrichment, EnrichmentError,
                                       MethodComparison, compare_fitting_methods,
                                       enrich_run, influence_diagnostics,
                                       projection_intervals, selected_model_name,
                                       simulate_buy)

__all__ += [
    "BuyRisk",
    "Enrichment",
    "EnrichmentError",
    "MethodComparison",
    "compare_fitting_methods",
    "enrich_run",
    "influence_diagnostics",
    "projection_intervals",
    "selected_model_name",
    "simulate_buy",
]
