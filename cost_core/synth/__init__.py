"""
cost_core.synth - Synthetic CSDR/SRDR contractor cost data.

Generates realistic-looking but entirely invented contractor submissions in the
six shapes an analyst pulls from CADE: DD 1921, DD 1921-1, DD 1921-2, the Cost
and Hour Report (FlexFile), the Quantity Data Report, and the SRDR (DD 2630).

**No real or proprietary data is used anywhere in this package.** Every number
is produced from a seeded pseudo-random generator and a published-format
skeleton. The point is to have something with realistic *pathologies* to clean
-- drifting element names, mixed then-year and base-year dollars, resubmitted
and missing periods, mid-program quantity changes, rate breaks, and hours that
have to be reconciled to dollars through a wrap rate.

Typical use::

    from cost_core.synth import generate_program

    program = generate_program(seed=7)
    program["dd1921_2"]        # the progress curve report
    program.truth.learning_slope   # the answer a fit should recover

The same seed always reproduces the same program exactly.
"""

from cost_core.synth.generator import (Portfolio, PortfolioTruth,
                                       ProgramTruth, SyntheticProgram,
                                       crawford_lot_cost, generate_portfolio,
                                       generate_program)
from cost_core.synth.reports import REPORT_NAMES
from cost_core.synth.spec import (ADAPTATION_WEIGHTS, BASE_WRAP_RATES,
                                  DEFAULT_NAME_VARIANTS, DEFAULT_WBS,
                                  FUNCTIONAL_CATEGORIES, FUNCTIONAL_MIX,
                                  SRDR_ACTIVITY_MIX, TRUE_AIRFRAME_CER,
                                  TRUE_SOFTWARE_CER, InflationAssumption,
                                  PathologyConfig, ProgramSpec, SoftwareSpec,
                                  WBSElement)

__all__ = [
    "ADAPTATION_WEIGHTS",
    "BASE_WRAP_RATES",
    "DEFAULT_NAME_VARIANTS",
    "DEFAULT_WBS",
    "FUNCTIONAL_CATEGORIES",
    "FUNCTIONAL_MIX",
    "InflationAssumption",
    "PathologyConfig",
    "Portfolio",
    "PortfolioTruth",
    "ProgramSpec",
    "ProgramTruth",
    "REPORT_NAMES",
    "SRDR_ACTIVITY_MIX",
    "SoftwareSpec",
    "SyntheticProgram",
    "TRUE_AIRFRAME_CER",
    "TRUE_SOFTWARE_CER",
    "WBSElement",
    "crawford_lot_cost",
    "generate_portfolio",
    "generate_program",
]
