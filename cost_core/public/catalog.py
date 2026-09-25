# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
catalog.py - Which SARs exist, and where to get each one.

DoD publishes released SARs in the Washington Headquarters Services FOIA
reading room, one folder per reporting cycle: ``FY_2014_SARS`` for the December
2014 annual reports, ``June_2025_MSARs`` and ``PB_2027_MSARs`` for the
modernised reports that replaced them. The reading room cannot be listed by a
script (see :mod:`cost_core.public.fetch`), so the catalogue is built from the
Wayback Machine's index of what it has captured under that folder, which is
also where the files themselves come from.

The program name here is a guess from the file name, good enough to pick
files by. The authoritative name is the one :func:`cost_core.public.read_sar`
reads off the report's own page header.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

import pandas as pd

from cost_core.public.fetch import Opener, wayback_listing

#: The reading room folder every released SAR sits under.
READING_ROOM = ("https://www.esd.whs.mil/Portals/54/Documents/FOID/Reading%20Room/"
                "Selected_Acquisition_Reports/")

#: Columns of :func:`sar_catalog`.
CATALOG_COLUMNS = ("program_guess", "cycle", "cycle_year", "kind", "folder",
                   "filename", "url", "capture", "size")

_FOLDER = re.compile(r"^(?:FY_(\d{4})_SARS|(June|PB)_(\d{4})_MSARs)$", re.I)


def _program_guess(filename: str) -> str:
    name = urllib.parse.unquote(filename).rsplit(".", 1)[0]
    name = re.sub(r"^\(U\)\s*", "", name)
    name = re.sub(r"^\d{2}-[FA]-\d{4}_", "", name)          # FOIA case number
    name = re.sub(r"^(DOC_\d+_)?", "", name, flags=re.I)
    name = re.sub(r"^(AF|Army|Navy|DOD|DoD|USSF|SF)_", "", name)
    # Cut at the first sign of the report type or date, however it is glued
    # on: "_SAR_Dec_2017", "-SAR-25_DEC_2010", "December2013SAR", "_2021".
    name = re.split(
        r"(?i)[ _-]*(?:M?SAR(?![a-z])|December|Dec(?=[ _\d-])|June?(?=[ _\d-])"
        r"|FY\s?\d{4}|(?<![\d.])(?:19|20)\d{2}(?!\d))",
        name, maxsplit=1)[0]
    return name.replace("_", " ").strip(" -")


def cycle_from_folder(folder: str) -> "tuple[Optional[str], Optional[int]]":
    """The cycle a reading room folder holds, e.g. ``("Dec 2014", 2014)``
    for ``FY_2014_SARS``, or ``(None, None)`` for any other folder."""
    m = _FOLDER.match(folder)
    if not m:
        return None, None
    if m.group(1):
        return f"Dec {m.group(1)}", int(m.group(1))
    year = int(m.group(3))
    return f"{'Jun' if m.group(2).lower() == 'june' else 'PB'} {year}", year


def sar_catalog(opener: Optional[Opener] = None) -> pd.DataFrame:
    """List every SAR and MSAR the Wayback Machine holds from the reading room.

    One row per file, with the reporting cycle it belongs to:

    * ``cycle``: ``"Dec 2014"`` for an annual SAR, ``"Jun 2025"`` or
      ``"PB 2027"`` for an MSAR;
    * ``kind``: ``"SAR"``, ``"MSAR"`` (the December 2023 cycle was the first
      MSAR, though its folder is named like the SARs before it), or
      ``"combined"`` for a volume binding many reports together;
    * ``capture``: the Wayback timestamp to fetch it at.

    Only the per-cycle folders are listed. The reading room also holds
    compilations of 1984-2007 SARs as single scanned volumes, which this
    parser does not read, and loose FOIA releases of single reports.
    """
    rows = []
    for url, capture, size in wayback_listing(READING_ROOM, opener=opener):
        rel = url.split("Selected_Acquisition_Reports/", 1)[-1]
        if "/" not in rel:
            continue
        folder, filename = rel.split("/", 1)
        cycle, year = cycle_from_folder(folder)
        if cycle is None or not filename.lower().endswith(".pdf"):
            continue
        kind = "MSAR" if "MSAR" in filename.upper() else "SAR"
        if "combined" in filename.lower():
            # FY 2016's cycle was released as three bound volumes of many
            # reports each, which a one-report parser cannot split.
            kind = "combined"
        rows.append((_program_guess(filename), cycle, year, kind, folder,
                     urllib.parse.unquote(filename), url, capture, size))
    cat = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    return cat.sort_values(["cycle_year", "cycle", "program_guess"]).reset_index(drop=True)
