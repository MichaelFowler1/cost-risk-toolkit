# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.portfolio - Which programs to fund, as an integer program.

Needs the optional extra::

    pip install "cost-core[optimize]"

See :mod:`cost_core.portfolio.optimize` for the formulation and the reasoning.
"""

from cost_core.portfolio.optimize import (Candidate, Option, Portfolio,
                                          PortfolioError, PortfolioResult,
                                          budget_risk, candidates_from_aoa,
                                          frontier, marginal_value, solve,
                                          spend_by_year)

__all__ = [
    "Candidate",
    "Option",
    "Portfolio",
    "PortfolioError",
    "PortfolioResult",
    "budget_risk",
    "candidates_from_aoa",
    "frontier",
    "marginal_value",
    "solve",
    "spend_by_year",
]
