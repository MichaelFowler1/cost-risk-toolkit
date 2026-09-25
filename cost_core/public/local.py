# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
local.py - A catalogue of SAR PDFs already on disk.

:func:`cost_core.public.sar_catalog` lists what the Wayback Machine holds, which
needs the internet. Many places that would use a SAR panel either have no
route to the Archive from the machine doing the analysis, or already hold the
reports: pulled from DAMIR, copied off the reading room by hand, or handed
over on a drive. :func:`local_catalog` lists such a folder in the same shape,
so :func:`cost_core.public.build_sar_panel` reads it exactly as it reads the
downloaded reports, and nothing touches the network.

**Which cycle a file belongs to** comes from the folder first, when the files
were kept under the reading room's own folder names (``FY_2014_SARS``,
``June_2025_MSARs``, ``PB_2027_MSARs``), and otherwise from the file name:
a month and a year ("F-35_SAR_Dec_2017.pdf", "JLENSDecember2013SAR.PDF",
"AAG_MSAR_June_2025.pdf"), or a President's Budget MSAR
("AAG_MSAR_FY2027_PBv2.pdf"). A file whose name gives neither is still
read, with no cycle: the report's own date line, in ``report_label``, says
when it is from.

A file is identified by its path, and the panel records its SHA-256 as it
does for a download, so a number can still be walked back to the exact file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from cost_core.public.catalog import _FOLDER, CATALOG_COLUMNS, _program_guess

# A full month name may be glued to the program name before it
# ("JLENSDecember2013SAR"); an abbreviation must not follow a letter, or
# "ADECS" and "SJUNIT" would read as months.
_MONTH = re.compile(
    r"(?i)(December|June)|(?<![a-z])(Dec|Jun)(?![a-z])")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
#: "PB", "PB2027", "PBv2": a President's Budget MSAR.
_PB = re.compile(r"(?i)(?<![a-z])PB(?:v\d+)?(?![a-z])")


def _cycle_from_folder(folder: str) -> "tuple[Optional[str], Optional[int]]":
    m = _FOLDER.match(folder)
    if not m:
        return None, None
    if m.group(1):
        return f"Dec {m.group(1)}", int(m.group(1))
    year = int(m.group(3))
    return f"{'Jun' if m.group(2).lower() == 'june' else 'PB'} {year}", year


def cycle_from_filename(filename: str) -> "tuple[Optional[str], Optional[int]]":
    """The reporting cycle a SAR's file name gives, or ``(None, None)``.

    ``"Dec 2017"`` for an annual report, ``"Jun 2025"`` for a June MSAR and
    ``"PB 2027"`` for a President's Budget MSAR. A FOIA case number at the
    front ("18-F-1016_") is not taken for a year, and neither is a program
    number such as "DDG_51".

    >>> cycle_from_filename("F-35_SAR_Dec_2017.pdf")
    ('Dec 2017', 2017)
    >>> cycle_from_filename("14-F-0402_DOC_40_JLENSDecember2013SAR.PDF")
    ('Dec 2013', 2013)
    >>> cycle_from_filename("AAG_MSAR_FY2027_PBv2.pdf")
    ('PB 2027', 2027)
    """
    name = re.sub(r"^\d{2}-[FA]-\d{4}_", "", Path(filename).stem)
    years = _YEAR.findall(name)
    if not years:
        return None, None
    year = int(years[-1])
    if _PB.search(name):
        return f"PB {year}", year
    month = _MONTH.search(name)
    if month is None:
        return None, None
    word = (month.group(1) or month.group(2)).lower()
    return f"{'Dec' if word.startswith('dec') else 'Jun'} {year}", year


def local_catalog(directory: Union[str, Path]) -> pd.DataFrame:
    """List the SAR and MSAR PDFs under a folder, in :func:`sar_catalog`'s shape.

    The folder is searched recursively. ``url`` is each file's absolute path,
    which :func:`cost_core.public.fetch` reads in place without a network
    request; ``capture`` is blank.

    Raises:
        FileNotFoundError: ``directory`` does not exist.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no such folder: {root}")
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"):
        cycle, year = _cycle_from_folder(path.parent.name)
        if cycle is None:
            cycle, year = cycle_from_filename(path.name)
        kind = "MSAR" if "MSAR" in path.name.upper() else "SAR"
        if "combined" in path.name.lower():
            kind = "combined"
        rows.append((_program_guess(path.name), cycle, year, kind, path.parent.name,
                     path.name, str(path), None, path.stat().st_size))
    cat = pd.DataFrame(rows, columns=list(CATALOG_COLUMNS))
    cat["cycle_year"] = cat["cycle_year"].astype("Int64")
    return cat.sort_values(["cycle_year", "cycle", "program_guess"],
                           na_position="last").reset_index(drop=True)
