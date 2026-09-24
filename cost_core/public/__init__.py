# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
cost_core.public - Real programs from public government documents.

Everything else in this library runs on data the caller brings or on the
seeded synthetic generator. This subpackage is where real programs come in:
DoD's Selected Acquisition Reports, read into tidy tables with the source file,
page and an arithmetic check behind every number.

Reading PDFs needs the optional extra::

    pip install "cost-core[public]"
"""

from cost_core.public.catalog import CATALOG_COLUMNS, sar_catalog
from cost_core.public.fetch import (Fetched, FetchError, cache_dir, fetch,
                                    wayback_listing)
from cost_core.public.sar import (UNIT_COST_COLUMNS, SarParseError, SarReport,
                                  check_unit_cost, parse_sar_pages, pdf_pages,
                                  read_sar)
from cost_core.public.panel import SarPanel, build_sar_panel

__all__ = [
    "CATALOG_COLUMNS",
    "Fetched",
    "FetchError",
    "SarParseError",
    "SarPanel",
    "SarReport",
    "UNIT_COST_COLUMNS",
    "build_sar_panel",
    "cache_dir",
    "check_unit_cost",
    "fetch",
    "parse_sar_pages",
    "pdf_pages",
    "read_sar",
    "sar_catalog",
    "wayback_listing",
]
