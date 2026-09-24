# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
panel.py - Many SARs into one program-by-report table.

One SAR tells you where one program stood in one year. The dataset worth
having is the panel: every program in every reporting cycle, so a program's
unit cost can be followed from report to report and programs can be compared
with each other. This module walks the catalogue, fetches and reads each
report, and stacks the results, keeping three things together:

* ``unit_cost``: the rows, each tagged with its cycle, program and the source
  file's URL, where it was served from, and its hash;
* ``reports``: one line per report attempted, saying whether it read, how
  many checks it passed, or why it failed;
* ``checks``: every arithmetic check behind every row.

A report that cannot be fetched or read is recorded in ``reports`` and the run
carries on. Across a stratified sample of 133 reports from every cycle 2010 to
2027, 126 read and 1,443 of 1,447 checks held; the failures were truncated
archive copies and reports with no unit cost table, and the failed checks
were errors in the reports themselves (an OCR layer that dropped a decimal
point, a printed percentage that disagrees with its own unit costs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pandas as pd

from cost_core.public.catalog import sar_catalog
from cost_core.public.fetch import fetch
from cost_core.public.sar import read_sar

logger = logging.getLogger(__name__)

REPORT_COLUMNS = ("cycle", "cycle_year", "program_guess", "program", "report_label",
                  "template", "rows", "checks", "failed_checks", "status", "error",
                  "url", "served_from", "sha256")


@dataclass
class SarPanel:
    """Unit cost across many SARs, with the record of how each was read."""

    unit_cost: pd.DataFrame
    reports: pd.DataFrame
    checks: pd.DataFrame

    def to_csv(self, directory) -> "dict[str, str]":
        """Write ``unit_cost.csv``, ``reports.csv`` and ``checks.csv``."""
        from pathlib import Path

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("unit_cost", "reports", "checks"):
            path = out / f"{name}.csv"
            getattr(self, name).to_csv(path, index=False)
            paths[name] = str(path)
        return paths


def _growth(uc: pd.DataFrame) -> pd.DataFrame:
    uc = uc.copy()
    uc["unit_cost_growth_pct"] = (uc["current_unit_cost"] / uc["baseline_unit_cost"] - 1) * 100
    uc["quantity_change_pct"] = (uc["current_quantity"] / uc["baseline_quantity"] - 1) * 100
    return uc


#: Capture times to fall back on when the listed capture is a broken file. The
#: Archive sometimes holds a truncated copy from one crawl and a whole one from
#: another (F-35, December 2019).
FALLBACK_CAPTURES = ("2026", "2022", "2020")


def _fetch_and_read(e):
    """Fetch one catalogue entry and read it, trying other captures of a
    truncated file before giving up."""
    last = None
    for i, stamp in enumerate((str(e["capture"]),) + FALLBACK_CAPTURES):
        got = fetch(e["url"], wayback_timestamp=stamp, try_official=False, refresh=i > 0)
        try:
            return got, read_sar(got.path, program_hint=e["program_guess"],
                                 source={"url": got.url, "served_from": got.served_from,
                                         "sha256": got.sha256})
        except Exception as exc:  # noqa: BLE001
            last = exc
            # Only a damaged file is worth another capture; a report that
            # reads but has no unit cost table will not change.
            damaged = ("EOF" in str(exc) or "PDFSyntax" in type(exc).__name__
                       or "Pdfminer" in type(exc).__name__)
            if not damaged:
                raise
            logger.info("%s: %s; trying another capture", e["url"], exc)
    raise last


def _dedupe(uc: pd.DataFrame) -> pd.DataFrame:
    """Drop rows repeated because a cycle holds the same report twice.

    The reading room sometimes keeps two copies of one report under different
    names (F-35, December 2023: "(U)F-35_MSAR_Dec_2023.pdf" and
    "F-35 MSAR Dec 2023.pdf"). Rows are duplicates when every number and the
    cycle, program, subprogram, measure and comparison agree; the first copy
    is kept. Two copies that differ in any number both stay.
    """
    key = ["cycle", "program", "subprogram", "measure", "comparison", "base_year",
           "baseline_cost", "baseline_quantity", "baseline_unit_cost",
           "current_cost", "current_quantity", "current_unit_cost"]
    return uc.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def build_sar_panel(
    catalog: Optional[pd.DataFrame] = None,
    cycles: Optional[Iterable[str]] = None,
    programs: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> SarPanel:
    """Fetch, read and stack SARs from the catalogue.

    Args:
        catalog: From :func:`cost_core.public.catalog.sar_catalog`; listed
            afresh when omitted.
        cycles: Keep only these cycles, e.g. ``["Dec 2019", "PB 2027"]``.
        programs: Keep only file-name guesses containing one of these,
            case-insensitive, e.g. ``["F-35", "DDG"]``.
        limit: Stop after this many reports, for a quick look.
        progress: Called with one line per report, e.g. ``print``.

    Returns:
        A :class:`SarPanel`. Growth columns are percentages, computed from the
        printed unit costs: ``unit_cost_growth_pct`` is current over baseline
        in the block's base-year dollars, and ``quantity_change_pct`` the same
        for quantity, since much apparent unit cost growth is a quantity cut.
    """
    cat = sar_catalog() if catalog is None else catalog
    cat = cat[cat["kind"] != "combined"]
    if cycles is not None:
        cat = cat[cat["cycle"].isin(list(cycles))]
    if programs is not None:
        pat = "|".join(__import__("re").escape(p) for p in programs)
        cat = cat[cat["program_guess"].str.contains(pat, case=False, regex=True)]
    if limit is not None:
        cat = cat.head(limit)

    rows, checks, reports = [], [], []
    for _, e in cat.iterrows():
        rec = {"cycle": e["cycle"], "cycle_year": e["cycle_year"],
               "program_guess": e["program_guess"], "url": e["url"]}
        try:
            got, report = _fetch_and_read(e)
            rec.update(served_from=got.served_from, sha256=got.sha256)
        except Exception as exc:  # noqa: BLE001 - recorded, the run carries on
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            reports.append(rec)
            if progress:
                progress(f"{e['cycle']:9s} {e['program_guess']:28s} FAILED  {rec['error'][:60]}")
            continue
        tag = {"cycle": e["cycle"], "cycle_year": e["cycle_year"], "program": report.program,
               "report_label": report.report_label, "template": report.template,
               "url": got.url, "sha256": got.sha256}
        failed = int((~report.checks["ok"]).sum())
        uc = report.unit_cost.assign(**tag)
        failed_pages = set(zip(report.checks.loc[~report.checks["ok"], "measure"],
                               report.checks.loc[~report.checks["ok"], "comparison"],
                               report.checks.loc[~report.checks["ok"], "page"]))
        uc["checks_ok"] = [(m, c, p) not in failed_pages
                           for m, c, p in zip(uc["measure"], uc["comparison"], uc["page"])]
        rows.append(uc)
        checks.append(report.checks.assign(**tag))
        rec.update(program=report.program, report_label=report.report_label,
                   template=report.template, rows=len(report.unit_cost),
                   checks=len(report.checks), failed_checks=failed, status="read")
        reports.append(rec)
        if progress:
            progress(f"{e['cycle']:9s} {report.program:28s} {len(report.unit_cost)} rows, "
                     f"{len(report.checks) - failed}/{len(report.checks)} checks")

    front = ["cycle", "cycle_year", "program", "subprogram", "measure", "comparison"]
    unit_cost = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=front)
    unit_cost = _growth(_dedupe(unit_cost)) if rows else unit_cost
    unit_cost = unit_cost[front + [c for c in unit_cost.columns if c not in front]]
    return SarPanel(
        unit_cost=unit_cost,
        reports=pd.DataFrame(reports).reindex(columns=list(REPORT_COLUMNS)),
        checks=pd.concat(checks, ignore_index=True) if checks else pd.DataFrame(),
    )
