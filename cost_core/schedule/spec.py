# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
spec.py - A JCL model written down as a JSON file.

::

    {
      "units": "$M",
      "standing_army": 1.2,
      "duration_correlation": 0.5,
      "confidence": 0.7,
      "n_iter": 20000,
      "seed": 1,
      "activities": [
        {"id": "design", "duration": 12, "burn_rate": 1.5,
         "duration_uncertainty": {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.4}},
        {"id": "i&t", "duration": 8, "predecessors": ["bus", ["fsw", 2]], ...}
      ],
      "risks": [
        {"name": "Detector yield", "probability": 0.3, "activities": ["instrument"],
         "delay": {"type": "triangular", "left": 2, "mode": 3, "right": 6}, "cost": 6}
      ]
    }

A predecessor is an id, or ``[id, lag]`` for a lag in months. Durations are in
months; costs in ``units``, which is only a label. ``docs/jcl_example.json``
is a complete one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from cost_core.schedule.jcl import Activity, Project, Risk, ScheduleError


def load_project(path) -> Tuple[Project, Dict[str, Any]]:
    """Read a spec file: the Project, and the analysis settings beside it."""
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if "activities" not in spec:
        raise ScheduleError(f"{path.name}: missing 'activities'.")
    acts = []
    for a in spec["activities"]:
        preds = [p if isinstance(p, str) else (str(p[0]), float(p[1]))
                 for p in a.get("predecessors", [])]
        acts.append(Activity(
            id=a["id"], duration=float(a["duration"]),
            duration_uncertainty=a.get("duration_uncertainty"),
            predecessors=preds,
            fixed_cost=float(a.get("fixed_cost", 0.0)),
            fixed_cost_uncertainty=a.get("fixed_cost_uncertainty"),
            burn_rate=float(a.get("burn_rate", 0.0)),
            name=a.get("name", "")))
    risks = [Risk(r["name"], float(r["probability"]), list(r.get("activities", [])),
                  r.get("delay", 0.0), float(r.get("cost", 0.0)))
             for r in spec.get("risks", [])]
    kwargs = {k: float(spec[k]) for k in ("standing_army", "duration_correlation",
                                          "cost_correlation", "cross_correlation") if k in spec}
    project = Project(acts, risks, name=spec.get("name", path.stem), **kwargs)
    settings = {k: spec[k] for k in ("units", "confidence", "n_iter", "seed") if k in spec}
    return project, settings
