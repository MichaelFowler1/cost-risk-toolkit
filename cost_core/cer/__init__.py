"""
cost_core.cer - Parametric cost estimating relationships.

Fits cost against technical drivers across a set of past programs, in either
of the two forms in common use, by any of three methods, with the diagnostics
and guardrails a small sample demands.

Typical use::

    from cost_core.cer import fit_cer

    cer = fit_cer(
        programs, response="t1_cost", predictors=["empty_weight_lb"],
        form="log_log", method="mupe",
    )
    cer.equation()
    cer.predict({"empty_weight_lb": [31_000]}, kind="prediction", level=0.80)
    cer.diagnostics().to_frame()

Two things this module insists on. The interval is a *prediction* interval by
default, because a cost estimate forecasts one new program rather than the
mean of the fitted line, and the two differ by exactly the residual scatter.
And a prediction outside the range of the fitting data raises
:class:`ExtrapolationWarning` -- including hidden extrapolation, where every
individual predictor is in range but their combination is not.
"""

from cost_core.cer.diagnostics import Diagnostics, compute_diagnostics
from cost_core.cer.model import (CER, MIN_OBS_PER_PARAM, ExtrapolationWarning,
                                 Form, cer_comparison_table,
                                 compare_cer_forms, compare_cer_methods,
                                 fit_cer)

__all__ = [
    "CER",
    "Diagnostics",
    "ExtrapolationWarning",
    "Form",
    "MIN_OBS_PER_PARAM",
    "cer_comparison_table",
    "compare_cer_forms",
    "compare_cer_methods",
    "compute_diagnostics",
    "fit_cer",
]
