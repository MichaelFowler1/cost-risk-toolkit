# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.aoa - Life-cycle cost for an analysis of alternatives.

Typical use::

    from cost_core.aoa import Alternative, CostLine, annual, evaluate, spread
    from cost_core.ingest import InflationTable

    a = Alternative("Upgrade", [
        CostLine("Development", "RDT&E", spread(800, 2027, 4, "bell"),
                 {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.6}),
        CostLine("Operations", "O&S", annual(120, 2031, 2050)),
    ], effectiveness=0.7)
    ...
    result = evaluate([a, b], base_year=2026, discount_rate=0.02,
                      inflation=InflationTable.from_rate(0.021, 2026, 2020, 2060))
    result.summary

See :mod:`cost_core.aoa.lcc` for the reasoning, and
:mod:`cost_core.aoa.growth` for uncertainty drawn from real SAR history.
"""

from cost_core.aoa.growth import (describe_growth, growth_factor,
                                  historical_growth)
from cost_core.aoa.lcc import (PHASES, AoAError, AoAResult, Alternative,
                               CostLine, annual, evaluate, spread)

__all__ = [
    "AoAError",
    "AoAResult",
    "Alternative",
    "CostLine",
    "PHASES",
    "annual",
    "describe_growth",
    "evaluate",
    "growth_factor",
    "historical_growth",
    "spread",
]
