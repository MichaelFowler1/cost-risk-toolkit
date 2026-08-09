"""
cost_core.reporting - Charts, tables and the assumptions log.

Three things, all of which come out of the same run so that a number on a slide
can be traced back to the row it came from:

* :mod:`~cost_core.reporting.charts` -- publication-quality PNGs sized for a
  slide: the S-curve, the tornado, the cost improvement curve and the CER
  diagnostic panel.
* :mod:`~cost_core.reporting.assumptions` -- the written assumptions and
  provenance log, organised around the GAO guide's four characteristics, which
  separates what was measured from what was assumed.
* :func:`run_full_analysis` -- the whole path end to end, reproducible from a
  seed.
"""

from cost_core.reporting.assumptions import AssumptionLog
from cost_core.reporting.charts import (plot_cer_diagnostics,
                                        plot_learning_curve, plot_s_curve,
                                        plot_summary_table, plot_tornado)
from cost_core.reporting.pipeline import DEMO_RISKS, RunResult, run_full_analysis

__all__ = [
    "AssumptionLog",
    "DEMO_RISKS",
    "RunResult",
    "plot_cer_diagnostics",
    "plot_learning_curve",
    "plot_s_curve",
    "plot_summary_table",
    "plot_tornado",
    "run_full_analysis",
]
