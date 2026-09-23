# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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
    # Reproduce a defect kept only for reconciling against legacy workbooks.
    # When True, the Rate model projects on the lot midpoint although it was
    # fitted against lot quantity, and the LC+Rate model drops its qty**c term
    # entirely. Both make projections that do not satisfy the equation the tool
    # prints, and because the rate exponent is negative the error is always
    # upward. Leave this False. See LEGACY_KEY in engine.py.
    "LegacyRateOmission": False,
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
    "ToolVersion": None,  # filled from TOOL_VERSION; see provenance()
    "DefaultRunID": "R001",
    "DefaultProgram": "TEST",
    "DefaultRunLabel": "unlabeled run",
    "BaseYear": "",
}
