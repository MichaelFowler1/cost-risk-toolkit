# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
spec.py - An AoA written down as a JSON file.

So an analysis can be kept, reviewed and rerun without writing Python. The
file names the settings and the alternatives; each cost line is phased one of
three ways::

    {"name": "Development", "phase": "RDT&E",
     "spread": {"total": 800, "start": 2027, "years": 4, "profile": "bell"},
     "uncertainty": {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.6}}

    {"name": "Operations", "phase": "O&S",
     "annual": {"amount": 120, "first": 2031, "last": 2050}}

    {"name": "Disposal", "phase": "Disposal", "by_year": {"2051": 40, "2052": 40}}

and the settings are ``base_year``, ``discount_rate`` (a real rate, as a
fraction, required), ``inflation_rate`` (a constant annual rate for then-year
dollars) or ``inflation_csv`` (an index table, see
:class:`cost_core.ingest.InflationTable`), and optionally ``basis``,
``pv_year``, ``n_iter``, ``seed`` and ``units`` (a label such as "$M",
printed on the chart; nothing is converted). ``cost_core/examples/aoa_example.json`` (``ce-core template aoa``) is a complete
one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from cost_core.aoa.lcc import AoAError, AoAResult, Alternative, CostLine, annual, evaluate, spread
from cost_core.ingest.inflation import InflationTable


def _line(d: Dict[str, Any], where: str) -> CostLine:
    ways = [k for k in ("spread", "annual", "by_year") if k in d]
    if len(ways) != 1:
        raise AoAError(f"{where}: give exactly one of spread, annual or by_year.")
    if "spread" in d:
        s = d["spread"]
        years = spread(float(s["total"]), int(s["start"]), int(s["years"]), s.get("profile", "uniform"))
    elif "annual" in d:
        s = d["annual"]
        years = annual(float(s["amount"]), int(s["first"]), int(s["last"]))
    else:
        years = {int(y): float(v) for y, v in d["by_year"].items()}
    return CostLine(d["name"], d["phase"], years, d.get("uncertainty"))


def load_spec(path) -> Dict[str, Any]:
    """Read a spec file into ``evaluate`` keyword arguments."""
    path = Path(path)
    from cost_core.xlspec import read_spec

    spec = read_spec(path)
    for key in ("base_year", "discount_rate", "alternatives"):
        if key not in spec:
            raise AoAError(f"{path.name}: missing {key!r}.")
    alts = []
    for a in spec["alternatives"]:
        lines = [_line(ln, f"{a['name']} / {ln.get('name', '?')}") for ln in a["lines"]]
        alts.append(Alternative(a["name"], lines, a.get("effectiveness"),
                                float(a.get("correlation", 0.3))))
    base_year = int(spec["base_year"])
    years = [y for alt in alts for ln in alt.lines for y in ln.by_year] + [base_year]
    if "inflation_csv" in spec:
        inflation = InflationTable.load(path.parent / spec["inflation_csv"])
    elif "inflation_rate" in spec:
        inflation = InflationTable.from_rate(
            float(spec["inflation_rate"]), base_year=base_year, first_year=min(years),
            last_year=max(years), source=f"constant {float(spec['inflation_rate']):.2%} a year, "
                                          f"from {path.name}")
    else:
        raise AoAError(f"{path.name}: give inflation_rate or inflation_csv.")
    kwargs = {"alternatives": alts, "base_year": base_year, "inflation": inflation,
              "discount_rate": float(spec["discount_rate"])}
    for key, cast in (("basis", str), ("pv_year", int), ("n_iter", int), ("seed", int),
                      ("units", str)):
        if key in spec:
            kwargs[key] = cast(spec[key])
    return kwargs


def run_spec(path) -> AoAResult:
    """Load a spec file and evaluate it."""
    kwargs = load_spec(path)
    alts = kwargs.pop("alternatives")
    return evaluate(alts, **kwargs)
