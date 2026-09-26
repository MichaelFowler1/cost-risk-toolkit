# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
spec.py - A portfolio decision written down as a JSON file.

::

    {
      "units": "$M",
      "budget": {"2027": 900, "2028": 950, "2029": 1000},
      "exclusive": [["Radar: upgrade", "Radar: new"]],
      "candidates": [
        {"name": "Sustain fleet", "mandatory": true, "category": "Readiness",
         "options": [{"name": "Full", "value": 5,
                      "cost_by_year": {"2027": 300, "2028": 300, "2029": 300}}]},
        {"name": "Missile integration", "requires": ["New missile"], ...}
      ],
      "growth": {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.5},
      "growth_correlation": 0.3,
      "delta": 25
    }

``growth`` (optional) is the cost growth factor :func:`budget_risk` applies
to each funded program; ``delta`` (optional) is the amount :func:`marginal_value`
adds to each year. ``cost_core/examples/portfolio_example.json`` (``ce-core template
portfolio``) is a complete one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from cost_core.portfolio.optimize import Candidate, Option, Portfolio, PortfolioError


def _years(d: Dict[str, Any]) -> Dict[int, float]:
    return {int(y): float(v) for y, v in d.items()}


def load_portfolio(path) -> "tuple[Portfolio, Dict[str, Any]]":
    """Read a spec file: the Portfolio, and the analysis settings beside it."""
    path = Path(path)
    from cost_core.xlspec import read_spec

    spec = read_spec(path)
    for key in ("budget", "candidates"):
        if key not in spec:
            raise PortfolioError(f"{path.name}: missing {key!r}.")
    cands = []
    for c in spec["candidates"]:
        opts = [Option(o["name"], _years(o["cost_by_year"]), float(o["value"]))
                for o in c.get("options", [])]
        cands.append(Candidate(c["name"], opts, bool(c.get("mandatory", False)),
                               tuple(c.get("requires", ())), c.get("category")))
    portfolio = Portfolio(cands, _years(spec["budget"]),
                          [list(g) for g in spec.get("exclusive", [])])
    settings = {k: spec[k] for k in ("units", "growth", "growth_correlation", "delta",
                                    "frontier_scales", "seed", "n_iter") if k in spec}
    # Checked here so a bad setting stops the run with a reason, not partway
    # through the analyses with a traceback.
    if "delta" in settings and not float(settings["delta"]) > 0:
        raise PortfolioError(f"{path.name}: delta, the extra money to test in one year, "
                             f"must be more than 0; got {settings['delta']}.")
    if "growth" in settings:
        from cost_core.monte_carlo import RiskModelError, make_distribution
        try:
            make_distribution(settings["growth"])
        except (RiskModelError, KeyError, TypeError, ValueError) as e:
            raise PortfolioError(f"{path.name}: the growth range: {e}") from None
    rho = float(settings.get("growth_correlation", 0.0))
    if not -1.0 <= rho <= 1.0:
        raise PortfolioError(f"{path.name}: growth_correlation must be between -1 and 1; "
                             f"got {rho:g}.")
    return portfolio, settings
