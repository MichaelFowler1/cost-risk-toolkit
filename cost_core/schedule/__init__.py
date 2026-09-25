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

A schedule kept in Microsoft Project comes in through its XML export::

    from cost_core.schedule import dcma_14_point, read_mspdi, simulate
    ims = read_mspdi("ims.xml")
    dcma_14_point(ims).table          # is the logic fit to simulate?
    r = simulate(ims.to_project(tri(0.9, 1.3)), seed=1)

See :mod:`cost_core.schedule.jcl` for the model and the reasoning,
:mod:`cost_core.schedule.mspdi` for the import and
:mod:`cost_core.schedule.dcma` for the assessment.
"""

from cost_core.schedule.dcma import DcmaResult, dcma_14_point
from cost_core.schedule.jcl import (LINK_TYPES, Activity, JclResult, Link,
                                    Project, Risk, ScheduleError, critical_path,
                                    point_estimate, simulate)
from cost_core.schedule.mspdi import MspdiSchedule, read_mspdi

__all__ = [
    "LINK_TYPES",
    "Activity",
    "DcmaResult",
    "JclResult",
    "Link",
    "MspdiSchedule",
    "Project",
    "Risk",
    "ScheduleError",
    "critical_path",
    "dcma_14_point",
    "point_estimate",
    "read_mspdi",
    "simulate",
]
