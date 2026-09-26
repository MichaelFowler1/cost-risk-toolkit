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

Or the network can come from a Microsoft Project schedule saved as XML, with
the uncertainty and the risks laid over it::

    {
      "mspdi": "ims.xml",
      "duration_uncertainty": {"type": "triangular", "left": 0.9, "mode": 1.0, "right": 1.3},
      "per_task": {"Thermal vacuum test": {"type": "triangular", "left": 0.9,
                                           "mode": 1.0, "right": 2.0}},
      "standing_army": 1.2,
      "risks": [{"name": "Late GFE", "probability": 0.3,
                 "activities": ["Integrate payload"], "delay": 2}]
    }

The path is relative to the spec. ``per_task`` and a risk's ``activities``
name tasks by their name or their UID; the task costs come from the file.
See :mod:`cost_core.schedule.mspdi` for how the schedule is read.

A predecessor is an id (finish-to-start), ``[id, lag]`` for a lag in months,
``[id, lag, type]`` with ``type`` one of ``FS``, ``SS``, ``FF`` and ``SF``,
or ``{"id": ..., "lag": ..., "type": ...}``. Durations are in
months; costs in ``units``, which is only a label. ``cost_core/examples/jcl_example.json`` (``ce-core template jcl``)
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
    from cost_core.xlspec import read_spec

    spec = read_spec(path)
    if "mspdi" in spec:
        return _load_mspdi(path, spec)
    if "activities" not in spec:
        raise ScheduleError(f"{path.name}: missing 'activities' (or 'mspdi').")
    acts = []
    for a in spec["activities"]:
        # Activity reads every predecessor form itself: an id, [id, lag],
        # [id, lag, type] or {"id", "lag", "type"}.
        preds = [p if isinstance(p, (str, dict)) else tuple(p)
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
    project = Project(acts, risks, name=spec.get("name", path.stem), **_project_kwargs(spec))
    return project, _settings(spec)


def _project_kwargs(spec) -> Dict[str, float]:
    return {k: float(spec[k]) for k in ("standing_army", "duration_correlation",
                                        "cost_correlation", "cross_correlation") if k in spec}


def _settings(spec) -> Dict[str, Any]:
    return {k: spec[k] for k in ("units", "confidence", "n_iter", "seed") if k in spec}


def _load_mspdi(path: Path, spec) -> Tuple[Project, Dict[str, Any]]:
    from cost_core.schedule.mspdi import read_mspdi

    sched = read_mspdi(path.parent / spec["mspdi"])
    detail = sched.detail
    by_name: Dict[str, str] = {}
    for uid, name in zip(detail["uid"], detail["name"]):
        by_name.setdefault(name, f"T{uid}")
    by_uid = {str(uid): f"T{uid}" for uid in detail["uid"]}

    def task(ref) -> str:
        ref = str(ref)
        if ref in by_uid:
            return by_uid[ref]
        if ref in by_name:
            return by_name[ref]
        raise ScheduleError(f"{path.name}: no detail task named or numbered {ref!r} in "
                            f"{spec['mspdi']}.")

    per_task = {int(task(k)[1:]): v for k, v in spec.get("per_task", {}).items()}
    base = sched.to_project(spec.get("duration_uncertainty"), per_task=per_task,
                            remaining=bool(spec.get("remaining", True)))
    risks = [Risk(r["name"], float(r["probability"]), [task(a) for a in r.get("activities", [])],
                  r.get("delay", 0.0), float(r.get("cost", 0.0)))
             for r in spec.get("risks", [])]
    project = Project(base.activities, risks, name=spec.get("name", sched.name),
                      **_project_kwargs(spec))
    settings = _settings(spec)
    settings["notes"] = list(sched.notes)
    settings.setdefault("units", "file currency")
    return project, settings
