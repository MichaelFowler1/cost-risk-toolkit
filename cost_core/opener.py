# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
opener.py - Work out what a file is, and which command reads it.

``ce-core open my_file`` is for the moment someone has a file and doesn't yet
know the command: an EVM export, a schedule saved from Project, an estimate
workbook. The file's own structure decides, never its name: the sheets a
workbook has, the columns a CSV has, the root of an XML file, the tables of
an IPMDAR dataset. When nothing matches it says what it looked for.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


class OpenError(ValueError):
    """A file ce-core can't place; the message says what it looked for."""


@dataclass
class Plan:
    """What a file is, in words, and the command that reads it."""

    what: str
    argv: List[str]


def _norm(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _evm_columns(columns) -> bool:
    from cost_core.evm.metrics import standard_columns

    try:
        frame = standard_columns(pd.DataFrame(columns=list(columns)))[0]
    except Exception:  # noqa: BLE001 - an ambiguous heading is simply not a match
        return False
    return {"bcws", "bcwp", "acwp"} <= set(frame.columns)


def _lots_columns(columns) -> bool:
    from cost_core.lots import COST_ALIASES, UNIT_ALIASES, _match_column

    return (_match_column(columns, UNIT_ALIASES) is not None
            and _match_column(columns, COST_ALIASES) is not None)


def _is_ipmdar_json(path: Path) -> bool:
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(doc, dict) and any(k in doc for k in ("ReportingCalendar",
                                                            "DatasetMetadata"))


def _spec_kind(path: Path):
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise OpenError(f"{path.name} isn't valid JSON ({e}).") from None
    if not isinstance(doc, dict):
        return None
    for key, kind in (("activities", "jcl"), ("mspdi", "jcl"), ("alternatives", "aoa"),
                      ("candidates", "portfolio")):
        if key in doc:
            return kind
    return None


def _xml_root(path: Path) -> str:
    import xml.etree.ElementTree as ET

    try:
        for _, elem in ET.iterparse(path, events=("start",)):
            return elem.tag.rsplit("}", 1)[-1]
    except ET.ParseError:
        return ""
    return ""


def plan(path, out=None) -> Plan:
    """The command for ``path``, with its results going to ``out`` (by default
    a folder beside the file)."""
    path = Path(path)
    if not path.exists():
        raise OpenError(f"There's nothing at {path}.")
    out = Path(out) if out else path.parent / f"{path.stem} results"
    o = ["--out", str(out)]
    if path.is_dir():
        names = [p.name for p in path.iterdir()]
        if any(n.lower().endswith(".pdf") for n in names):
            return Plan("a folder of PDFs, read as Selected Acquisition Reports",
                        ["sar-panel", "--dir", str(path)] + o)
        if any("reportingcalendar" in _norm(n) for n in names):
            return Plan("an IPMDAR Contract Performance Dataset", ["evm", "--ipmdar", str(path)] + o)
        raise OpenError(f"{path.name} is a folder, and holds neither SAR PDFs nor an IPMDAR "
                        "dataset (its ReportingCalendar table).")
    suffix = path.suffix.lower()
    if suffix == ".zip":
        try:
            names = zipfile.ZipFile(path).namelist()
        except zipfile.BadZipFile:
            raise OpenError(f"{path.name} isn't a readable zip file.") from None
        if any("reportingcalendar" in _norm(n) for n in names):
            return Plan("an IPMDAR Contract Performance Dataset", ["evm", "--ipmdar", str(path)] + o)
        raise OpenError(f"{path.name} is a zip file without an IPMDAR ReportingCalendar table.")
    if suffix == ".json":
        if _is_ipmdar_json(path):
            return Plan("an IPMDAR Contract Performance Dataset", ["evm", "--ipmdar", str(path)] + o)
        kind = _spec_kind(path)
        if kind:
            what = {"jcl": "a JCL spec", "aoa": "an analysis of alternatives spec",
                    "portfolio": "a portfolio spec"}[kind]
            return Plan(what, [kind, "--spec", str(path)] + o)
        raise OpenError(f"{path.name} is JSON, but not an IPMDAR dataset or a JCL, AoA or "
                        "portfolio spec.")
    if suffix == ".xml":
        root = _xml_root(path)
        if root == "Project":
            return Plan("a Microsoft Project schedule saved as XML",
                        ["schedule-check", "--mspdi", str(path)] + o)
        raise OpenError(f"{path.name} is XML but not a Microsoft Project schedule "
                        f"(its root is <{root or '?'}>). From Project: File > Save As > XML.")
    if suffix in (".xlsx", ".xlsm"):
        from cost_core.costrisk import open_workbook

        sheets = open_workbook(path, OpenError)
        names = {_norm(s) for s in sheets}
        if "elements" in names:
            return Plan("a cost estimate with ranges (an Elements sheet)",
                        ["cost-risk", "--data", str(path)] + o)
        for sheet, kind, what in (("activities", "jcl", "a JCL network (an Activities sheet)"),
                                  ("lines", "aoa", "an analysis of alternatives (a Lines sheet)"),
                                  ("candidates", "portfolio",
                                   "a portfolio of candidate programs (a Candidates sheet)")):
            if sheet in names:
                return Plan(what, [kind, "--spec", str(path)] + o)
        first = next(iter(sheets.values()))
        return _by_columns(path, first.columns, o, sheet_note=" on its first sheet")
    if suffix == ".csv":
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except UnicodeDecodeError:
            raise OpenError(f"{path.name} isn't saved as UTF-8 text. In Excel, save it as "
                            "'CSV UTF-8 (Comma delimited)'.") from None
        except ValueError as e:
            raise OpenError(f"{path.name} can't be read as a CSV ({e}).") from None
        return _by_columns(path, columns, o)
    if suffix in (".xls", ".xlsb", ".mpp"):
        raise OpenError(f"{path.name}: save it as {'XML from Microsoft Project (File > Save As)' if suffix == '.mpp' else '.xlsx from Excel'} "
                        "first; cost-core reads the open formats.")
    raise OpenError(f"cost-core doesn't read {suffix or 'extensionless'} files. It opens "
                    ".xlsx and .csv (estimates, EVM data, specs), .xml (Microsoft Project), "
                    ".json and .zip (IPMDAR or a spec) and folders of SAR PDFs.")


def _by_columns(path: Path, columns, o, sheet_note: str = "") -> Plan:
    if _evm_columns(columns):
        return Plan(f"earned value data (BCWS, BCWP and ACWP columns{sheet_note})",
                    ["evm", "--data", str(path)] + o)
    cols = {_norm(c) for c in columns}
    if ("element" in cols or "wbselement" in cols) and ({"low", "min"} & cols):
        return Plan(f"a cost estimate with ranges{sheet_note}",
                    ["cost-risk", "--data", str(path)] + o)
    if _lots_columns(columns):
        raise OpenError(f"{path.name} looks like production lots (units and cost). A learning "
                        "curve needs the dollar year the costs are in, so run:\n"
                        f"  ce-core fit-lots --csv \"{path}\" --dollar-year 2026")
    shown = ", ".join(map(str, list(columns)[:8]))
    raise OpenError(f"{path.name}: its columns ({shown}) don't match anything cost-core reads. "
                    "It expects EVM data (BCWS, BCWP, ACWP), an estimate (Element, Low, "
                    "Most Likely, High) or one of the templates: ce-core template <topic>.")


SEND_TO_NAME = "ce-core (analyse this file).cmd"


def install_send_to() -> Path:
    """A Windows Send To shortcut: right-click a file, Send to, ce-core."""
    import os
    import sys

    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise OpenError("The Send To shortcut is for Windows.")
    folder = Path(appdata) / "Microsoft" / "Windows" / "SendTo"
    if not folder.is_dir():
        raise OpenError(f"There's no Send To folder at {folder}.")
    target = folder / SEND_TO_NAME
    target.write_text(f'@echo off\r\n"{sys.executable}" -m cost_core open %* --pause\r\n',
                      encoding="utf-8")
    return target
