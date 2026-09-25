# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.evm - Earned value management: metrics, earned schedule and
forecasts of the cost and date at completion.

Typical use::

    from cost_core.evm import EvmData, forecast
    data = EvmData.read("cpr.csv")      # period, bcws, bcwp, acwp[, eac][, wbs]
    data.summary(); data.flags()        # CPI, SPI, SPI(t), TCPI, IEACs, warnings
    f = forecast(data, seed=1)
    f.summary()                         # EAC and completion P50/P80, and where
                                        # the contractor's EAC falls on them

See :mod:`cost_core.evm.metrics` for the metrics and the warning signs, and
:mod:`cost_core.evm.forecast` for how the forecast is drawn.
"""

from cost_core.evm.forecast import EvmForecast, forecast
from cost_core.evm.metrics import METRIC_COLUMNS, EvmData, EvmError, earned_schedule

__all__ = [
    "EvmData",
    "EvmError",
    "EvmForecast",
    "METRIC_COLUMNS",
    "earned_schedule",
    "forecast",
]
