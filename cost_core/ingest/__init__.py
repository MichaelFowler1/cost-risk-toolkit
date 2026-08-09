"""
cost_core.ingest - ETL from CSDR/SRDR submissions to one normalised table.

Takes the six report shapes an analyst pulls from CADE and lands them in a
single long-format table with full provenance, resolving the four things that
otherwise make the data unusable:

* WBS element names that drift between periods, resolved through a persisted
  crosswalk artifact rather than logic buried in a loader;
* mixed then-year and base-year dollars, deflated through a configurable raw
  index with the original values preserved;
* resubmitted periods, deduplicated to the latest report date with the
  superseded rows kept and marked;
* everything else that does not reconcile, surfaced by validation gates that
  fail the run rather than shipping a plausible number.

Typical use::

    from cost_core.ingest import Crosswalk, InflationTable, normalize

    data = normalize(
        program.reports,
        crosswalk=Crosswalk.load("artifacts/wbs_crosswalk.csv"),
        inflation=InflationTable.load("artifacts/inflation.csv"),
        base_year=2026,
    )
    data.rows                 # the normalised long table
    data.trace(some_row_uid)  # every source row behind one number
    data.validation.to_frame()
"""

from cost_core.ingest.crosswalk import (CROSSWALK_COLUMNS, Crosswalk,
                                        CrosswalkError, ResolvedName,
                                        normalize_key)
from cost_core.ingest.inflation import (DEFAULT_INDEX, INDEX_COLUMNS,
                                        InflationError, InflationTable)
from cost_core.ingest.pipeline import (CATEGORY_ALL, EXTRACTORS, IngestError,
                                       NORMALIZED_COLUMNS,
                                       NormalizedDataset, PROVENANCE_COLUMNS,
                                       RECONCILIATION_RTOL, ValidationGate,
                                       ValidationReport, WBS_PROGRAM_LEVEL,
                                       normalize, normalize_program)

__all__ = [
    "CATEGORY_ALL",
    "CROSSWALK_COLUMNS",
    "Crosswalk",
    "CrosswalkError",
    "DEFAULT_INDEX",
    "EXTRACTORS",
    "INDEX_COLUMNS",
    "IngestError",
    "InflationError",
    "InflationTable",
    "NORMALIZED_COLUMNS",
    "NormalizedDataset",
    "PROVENANCE_COLUMNS",
    "RECONCILIATION_RTOL",
    "ResolvedName",
    "ValidationGate",
    "ValidationReport",
    "WBS_PROGRAM_LEVEL",
    "normalize",
    "normalize_key",
    "normalize_program",
]
