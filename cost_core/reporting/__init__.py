# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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

from __future__ import annotations

import importlib

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

# The workbook writers live in this package too and must import without
# matplotlib, so the chart and pipeline names load lazily (PEP 562) on first
# access rather than when the package is imported. Since 1.0.0 matplotlib is an
# optional extra, so that laziness is what makes `import cost_core.reporting`
# work at all on a bare install, and the names below are exactly the ones that
# do not.
_LAZY = {
    "AssumptionLog": ".assumptions",
    "plot_cer_diagnostics": ".charts",
    "plot_learning_curve": ".charts",
    "plot_s_curve": ".charts",
    "plot_summary_table": ".charts",
    "plot_tornado": ".charts",
    "DEMO_RISKS": ".pipeline",
    "RunResult": ".pipeline",
    "run_full_analysis": ".pipeline",
}


def __getattr__(name: str):
    try:
        module = _LAZY[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
