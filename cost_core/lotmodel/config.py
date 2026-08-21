"""
config.py - Model settings for the lot cost engine.

These are the knobs the original workbook exposed, carried over unchanged so
that a run of this package reproduces a run of the spreadsheet tool exactly.
The GUI writes over a subset of them per run; anything not overridden keeps
the value here.
"""

from __future__ import annotations

SETTINGS = {
    "AnalogyTableName": "AnalogyLots",
    "EstimateTableName": "EstimateLots",
    "CostUnitScale": 1.0,  # 1 = $K, 1000 = full dollars
    "TotalScale": 1000.0,  # Applied on top of CostUnitScale for totals
    "ToolMatchProjection": True,  # True = Rate & LC+Rate project on lot midpoint
    "DefaultCF": 1.0,
    "FitPriorUnits": 0,
    "FcstPriorUnits": 0,
    "SeedB": -0.152003093,
    "MaxIter": 100,
    "Tol": 1e-9,
    "RateSdFloor": 0.05,
    "SingularTol": 1e-12,
    "TGate": 2.0,
    "AiccTie": 2.0,
    "ToolVersion": "2.0-dev",
    "DefaultRunID": "R001",
    "DefaultProgram": "TEST",
    "DefaultRunLabel": "unlabeled run",
    "BaseYear": "",
}


