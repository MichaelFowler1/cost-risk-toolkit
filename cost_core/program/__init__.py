"""Several WBS elements priced against one lot schedule.

Each element carries its own analogy history and is fitted on its own; what
they share is the buy schedule. The point estimates add, and the risk is
rolled up through a correlated Monte Carlo rather than summed, because
elements on one programme overrun together. The desktop tool's programme
window drives this package; :mod:`cost_core.program.rollup` holds the
engine and :mod:`cost_core.reporting.program_workbook` writes the workbook.
"""

from __future__ import annotations

from cost_core.program.rollup import (
    DEFAULT_CORRELATION,
    KINDS,
    SPEC_TOLERANCE,
    Element,
    ElementResult,
    Program,
    ProgramError,
    ProgramResult,
    _estimate_frame,
    _lognormal_spec,
    _program_risk,
    _scale_element,
    _scurve,
    _selected_model,
    _tornado,
    buy_profile_sensitivity,
    by_fiscal_year,
    element_summary,
    factor_of,
    fitted,
    flat_amount,
    influence,
    influence_table,
    phase_total,
    price_derived,
    price_element,
    program_summary,
    roll_up,
)
from cost_core.reporting.program_workbook import (
    _sheet_name,
    save_program_workbook,
)

__all__ = [
    "DEFAULT_CORRELATION",
    "KINDS",
    "SPEC_TOLERANCE",
    "Element",
    "ElementResult",
    "Program",
    "ProgramError",
    "ProgramResult",
    "buy_profile_sensitivity",
    "by_fiscal_year",
    "element_summary",
    "factor_of",
    "fitted",
    "flat_amount",
    "influence",
    "influence_table",
    "phase_total",
    "price_derived",
    "price_element",
    "program_summary",
    "roll_up",
    "save_program_workbook",
]
