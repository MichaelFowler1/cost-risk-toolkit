# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.schedule - Schedule risk and joint cost and schedule confidence.

Typical use::

    from cost_core.schedule import Activity, Project, Risk, simulate
    tri = lambda lo, hi: {"type": "triangular", "left": lo, "mode": 1.0, "right": hi}
    p = Project([
        Activity("design", 12, tri(0.9, 1.4), burn_rate=1.5),
        Activity("build", 18, tri(0.9, 1.5), ["design"], fixed_cost=40, burn_rate=2.0),
        Activity("test", 8, tri(0.9, 1.8), ["build"], burn_rate=2.5),
    ], risks=[Risk("thermal vac retest", 0.2, ["test"], delay=2, cost=3)],
       standing_army=1.2)
    r = simulate(p, seed=1)
    r.summary(); r.frontier(0.7); r.criticality()

See :mod:`cost_core.schedule.jcl` for the model and the reasoning.
"""

from cost_core.schedule.jcl import (Activity, JclResult, Project, Risk,
                                    ScheduleError, critical_path,
                                    point_estimate, simulate)

__all__ = [
    "Activity",
    "JclResult",
    "Project",
    "Risk",
    "ScheduleError",
    "critical_path",
    "point_estimate",
    "simulate",
]
