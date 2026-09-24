# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
sar.py - Read the unit cost section out of Selected Acquisition Reports.

A SAR is the report DoD sends Congress on each major defense acquisition
program. Of everything in it, the unit cost section is the one worth building
a dataset on first: it is required by statute (the Nunn-McCurdy unit cost
reporting in 10 U.S.C. 4371), so every SAR of every era carries it, and it
compares the current estimate against two baselines, the current one and the
original one. That second comparison is the long view of cost growth: what the
program was supposed to cost per unit when it started, against what it is
expected to cost now, in constant base-year dollars.

**Three templates.** The layout changed twice, and the parser reads all three:

* 2010 to 2019, the DAMIR SAR: "Unit Cost Report", baselines dated as
  "(Mar 2014 APB)", base year as "BY 2002 $M".
* 2021 and 2022, the DAVE SAR: "Current Baseline Compared with Current
  Estimate", often with no base year on the page at all.
* December 2023 on, the modernised MSAR: "(U) Current Estimate Compared with
  Current Baseline", baselines dated 02/05/2020, base year as "Base Year: 2017".

Each has a current-baseline block and an original-baseline block, and in each
block a PAUC (program acquisition unit cost: all acquisition money over all
units) and an APUC (average procurement unit cost: procurement money over
procurement units), each as cost, quantity and unit cost, baseline beside
current estimate.

**Text, not tables.** Parsing works on the text of each page as laid out by
position (``pdfplumber``'s ``layout=True``), one line per printed row, because
the PDFs carry no table structure to lean on and several are scans with an OCR
text layer, where a label can come out as "Item" and a header as "°/.3 Change".
The parser therefore keys on the row order inside a block (cost, quantity,
unit cost) as much as on the labels, and reads only the numbers.

**Checked, not trusted.** Every row is checked against its own arithmetic
(unit cost times quantity against cost, and the reported percentage against
the one the costs imply) and the result of each check travels with the data.
A row that fails is kept and marked, not dropped or repaired: a mismatch is as
often the SAR's own typo as the parser's, and either way a reader should see it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

#: Columns of :attr:`SarReport.unit_cost`, in order.
UNIT_COST_COLUMNS = (
    "subprogram", "comparison", "revised", "measure", "base_year",
    "base_year_source", "baseline_label",
    "baseline_cost", "baseline_quantity", "baseline_unit_cost",
    "current_cost", "current_quantity", "current_unit_cost",
    "reported_pct_change", "page",
)

_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?%?$")
_BLANK = {"-", "--", "—", "N/A", "n/a", "TBD"}
_BASE_YEAR = re.compile(r"(?:Base\s*Year:?\s*|\bBY\s?)((?:19|20)\d{2})")
_APB_DATE = re.compile(r"\((\w{3}\s+\d{4})\s+APB\)")
_SLASH_DATE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
_LONG_DATE = re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|"
                        r"October|November|December)\s+\d{1,2},\s+\d{4})")
#: Footnote and breach marks printed on the end of a number: "$49.732*", "+22.12%†".
_MARKS = "*†‡§¹²³"
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


class SarParseError(ValueError):
    """Raised when a document has no unit cost section the parser can read."""


@dataclass
class SarReport:
    """What was read from one SAR.

    Attributes:
        program: Short program name from the page header, e.g. ``"AAG"``.
        report_label: The report's own date line, e.g. ``"December 2014 SAR"``.
        report_year: Calendar year of the report's as-of date, where readable.
        template: ``"damir"`` (2010-2019), ``"dave"`` (2021-2022) or ``"msar"``.
        unit_cost: One row per program or subprogram, baseline and measure;
            columns :data:`UNIT_COST_COLUMNS`.
        checks: One row per arithmetic check, with ``ok`` saying whether it
            held. See :func:`check_unit_cost`.
        source: Where the PDF came from, when read through :func:`read_sar`.
    """

    program: str
    report_label: str
    report_year: Optional[int]
    template: str
    unit_cost: pd.DataFrame
    checks: pd.DataFrame
    source: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every arithmetic check held."""
        return bool(self.checks["ok"].all()) if len(self.checks) else True


# ---------------------------------------------------------------- tokens ---
def _clean(tok: str) -> str:
    return tok.strip().lstrip("$").rstrip(_MARKS)


def _number(tok: str) -> Optional[float]:
    """A printed number as a float; None for a blank marker like ``--``."""
    tok = _clean(tok)
    if tok in _BLANK:
        return None
    clean = tok.replace(",", "").rstrip("%")
    return float(clean) if clean not in ("", "+", "-", ".") else None


def _is_number(tok: str) -> bool:
    tok = _clean(tok)
    return tok in _BLANK or bool(_NUMBER.match(tok)) and any(c.isdigit() for c in tok)


def _split_row(line: str) -> "tuple[str, list[Optional[float]], Optional[float]]":
    """Split a table row into its label, its numbers, and a percentage.

    Reads numbers left to right after the label and stops at the first word
    that is neither a number nor a blank marker, so trailing text such as
    "Significant Cost Growth" is ignored. Dollar signs, standalone or stuck
    to a number, and footnote marks are dropped.

    A number printed with a ``%`` or an explicit sign (``-6.60%``, ``+13.21``)
    is taken as the row's percentage change and kept out of ``values``, so a
    stray footnote number between the unit cost and the percentage (RQ-4A/B,
    December 2013: ``169.893 1 +13.21``) cannot push it out of place. Costs,
    quantities and unit costs are never printed signed.
    """
    toks = [tok for tok in line.split() if tok != "$"]
    i = 0
    while i < len(toks) and not _is_number(toks[i]):
        i += 1
    label = " ".join(toks[:i])
    values: list = []
    pct = None
    for tok in toks[i:]:
        if not _is_number(tok):
            break
        clean = _clean(tok)
        if clean not in _BLANK and (clean.endswith("%") or clean[:1] in "+-"):
            pct = _number(tok)
        else:
            values.append(_number(tok))
    return label, values, pct


# --------------------------------------------------------------- headers ---
def _template(pages: "list[str]") -> str:
    head = "\n".join(pages[:3])
    if "MSAR" in head or "(U)" in head:
        return "msar"
    if "Defense Acquisition Visibility Environment" in head or re.search(
            r"\bSAR\s+\d{1,2}/\d{1,2}/\d{4}", head) or re.search(r"SAR (DEC|JUN) \d{4}", head):
        return "dave"
    return "damir"


def _header_line(page: str) -> str:
    for line in page.splitlines():
        s = line.strip()
        if s and s != "UNCLASSIFIED":
            return s
    return ""


def _report_label(header: str) -> str:
    for pat in (r"MSAR,\s*\w+\s+\d{1,2},\s*\d{4}", r"\w+\s+\d{1,2},\s*\d{4}\s+SAR",
                r"\w+\s+\d{4}\s+SAR", r"SAR\s+\w{3}\s+\d{4}", r"SAR\s+\d{1,2}/\d{1,2}/\d{4}"):
        m = re.search(pat, header)
        if m:
            return re.sub(r"\s+", " ", m.group(0))
    return ""


def _report_year(label: str) -> Optional[int]:
    years = re.findall(r"(?:19|20)\d{2}", label)
    return int(years[-1]) if years else None


def _program_name(header: str, label: str) -> str:
    name = header.split(label)[0] if label else header
    name = re.split(r"\s{2,}|UNCLASSIFIED", name.strip())[0]
    return name.strip()


def _plausible_name(name: str) -> bool:
    """False for OCR debris such as ``&Tit"?'`` and for section titles."""
    return (bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ./&()+,-]*", name))
            and sum(c.isalnum() for c in name) >= 2
            and name not in ("Unit Cost", "Unit Costs", "Table of Contents"))


# ------------------------------------------------------------- unit cost ---
_PAUC_HEAD = re.compile(r"^(Program Acquisition Unit Cost|PAUC)\b(\s*\(PAUC\))?$")
_APUC_HEAD = re.compile(r"^(Average Procurement Unit Cost|APUC)\b(\s*\(APUC\))?$")
_NOT_SUBPROGRAM = re.compile(
    r"Unit Cost|Baseline|Category|Current|Original|Compared|UCR|Estimate|\$M|"
    r"Report|Item|Change|APB|UNCLASSIFIED|SAR\b|Notes?$|Base-Year|Dollars", re.I)


def _subprogram_name(line: str) -> Optional[str]:
    """A subprogram title such as "F-35 Engine", or None.

    MSARs print it as "(U)F-35 Aircraft Subprogram"; scans add OCR debris
    lines ("IlL", "impirBy") around it, which a real name is told from by
    having a space or hyphen in it.
    """
    name = re.sub(r"^\(U\)\s*", "", line.strip())
    name = re.sub(r"\s+Subprogram$", "", name, flags=re.I)
    if (len(name) < 4 or len(name) > 40 or _NOT_SUBPROGRAM.search(name)
            or not re.search(r"[ -]", name) or _split_row(name)[1]
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ./&()+,-]*", name)):
        return None
    return name


def _comparison(line: str) -> Optional[str]:
    """Which baseline a block header line introduces, if any."""
    if re.search(r"Original\s+(UCR|Baseline|APB)|Compared with Original", line):
        return "original"
    if re.search(r"Current\s+(UCR|Baseline|APB)|Compared with Current Baseline", line):
        return "current"
    return None


def _is_unit_cost_page(page: str) -> bool:
    lines = [l.strip() for l in page.splitlines()]
    has_head = any(_PAUC_HEAD.match(l) for l in lines)
    history = "Unit Cost History" in page and not any(l.startswith("Cost ") for l in lines)
    return has_head and not history and bool(re.search(r"Baseline|UCR", page))


def _row_kind(label: str, position: int) -> str:
    low = label.lower()
    if "quantity" in low:
        return "quantity"
    if "unit cost" in low or low in ("pauc", "apuc", "item"):
        return "unit"
    if "cost" in low:
        return "cost"
    return ("cost", "quantity", "unit")[min(position, 2)]


def _parse_unit_cost_page(page: str, number: int, doc_base_year: Optional[int],
                          program: str) -> "list[dict]":
    rows = []
    comparison = measure = subprogram = baseline_label = None
    base_year: Optional[int] = None
    by_from_header = revised = False
    block_rows = position = 0
    current: dict = {}
    # A base year or a "Revised" printed above a block's header belongs to the
    # block below it, so both are held here until the next header opens.
    pending_by: Optional[int] = None
    pending_revised = False
    lines = [l.strip() for l in page.splitlines() if l.strip()]

    def subprogram_in(candidates):
        for cand in candidates:
            if _comparison(cand) or _PAUC_HEAD.match(cand):
                return None  # the table has started; there was no name
            name = _subprogram_name(cand)
            if name and name != program:
                return name
        return None

    # A subprogram's name prints near the top of the page, just under the
    # page header or the section title, before the first table. Look there
    # rather than at a fixed line, since scans scatter debris in between.
    head = next((k for k, l in enumerate(lines) if l != "UNCLASSIFIED"), 0)
    subprogram = subprogram_in(lines[head + 1:head + 7])
    for k, line in enumerate(lines):
        if line in ("Unit Cost", "(U) Unit Costs", "Unit Costs"):
            continue
        label, values, pct = _split_row(line)
        # A unit cost row can carry a single number when a baseline cell is
        # blank (ITEP, December 2021: "APUC  $ 1.567  0.00%").
        if measure is not None and (len(values) >= 2 or (len(values) == 1 and position >= 2)):
            kind = _row_kind(label, position)
            position += 1
            current[kind] = values
            if kind != "unit":
                continue
            cost = (current.get("cost", []) + [None, None])[:2]
            qty = (current.get("quantity", []) + [None, None])[:2]
            unit = _align_unit(values, cost, qty)
            if pct is None and len(values) > 2:
                pct = values[2]  # printed unsigned, as the DAVE template does for rises
            by, by_source = base_year, "page"
            if by is None:
                by, by_source = doc_base_year, "document"
            rows.append({
                "subprogram": subprogram,
                "comparison": comparison,
                "revised": bool(revised) and comparison == "original",
                "measure": measure,
                "base_year": by,
                "base_year_source": by_source if by is not None else None,
                "baseline_label": baseline_label,
                "baseline_cost": cost[0],
                "baseline_quantity": qty[0],
                "baseline_unit_cost": unit[0],
                "current_cost": cost[1],
                "current_quantity": qty[1],
                "current_unit_cost": unit[1],
                "reported_pct_change": pct,
                "page": number,
            })
            block_rows += 1
            measure, pending_by, pending_revised = None, None, False
            continue
        if _PAUC_HEAD.match(line) or _APUC_HEAD.match(line):
            if comparison is not None:
                measure = "PAUC" if _PAUC_HEAD.match(line) else "APUC"
                current, position = {}, 0
            continue

        by_match = _BASE_YEAR.search(line)
        comp = _comparison(line)
        if comp and (comp != comparison or block_rows):
            comparison, measure, block_rows = comp, None, 0
            baseline_label = None
            base_year = int(by_match.group(1)) if by_match else pending_by
            by_from_header = bool(by_match)
            revised = pending_revised or "Revised" in line
            pending_by, pending_revised = None, False
        elif comparison is not None and block_rows == 0 and measure is None:
            # Still inside the scattered header of the open block.
            if by_match and not by_from_header:
                base_year, by_from_header = int(by_match.group(1)), True
            if "Revised" in line and comparison == "original":
                revised = True
        else:
            if by_match:
                pending_by = int(by_match.group(1))
            if "Revised" in line:
                pending_revised = True
            continue
        m = _APB_DATE.search(line)
        if m:
            baseline_label = m.group(1) + " APB"
        elif baseline_label is None:
            m = _SLASH_DATE.search(line) or _LONG_DATE.search(line)
            if m:
                baseline_label = m.group(1)
    return rows


def _align_unit(values: list, cost: list, qty: list) -> list:
    """Baseline and current unit cost from a row that may be missing one.

    With both printed, they are the first two numbers. With one, it goes to
    the side whose cost over quantity it equals, which is the only evidence a
    row of text leaves about which cell was blank.
    """
    if len(values) >= 2:
        return values[:2]
    (only,) = values
    fits = []
    for side in (0, 1):
        if only is not None and cost[side] and qty[side]:
            fits.append((abs(cost[side] / qty[side] - only) / abs(only or 1), side))
    side = min(fits)[1] if fits else 1
    return [only, None] if side == 0 else [None, only]


def _document_base_year(pages: "list[str]") -> Optional[int]:
    """The base year the document uses most, for blocks that print none.

    Counts "BY 2017"-style mentions anywhere, and the "Base Year" column of
    the 2021-2022 budget table, where the year is the first number on the
    RDT&E, Procurement, PAUC and APUC rows and printed nowhere else.
    """
    years = Counter()
    for page in pages:
        for m in _BASE_YEAR.finditer(page):
            years[int(m.group(1))] += 1
        if "Base" in page and "Budget Estimate" in page:
            for line in page.splitlines():
                label, values, _ = _split_row(line.strip())
                if label in ("RDT&E", "Procurement", "PAUC", "APUC") and len(values) >= 3                         and values[0] is not None and values[0].is_integer()                         and 1970 <= values[0] <= 2040:
                    years[int(values[0])] += 1
    return years.most_common(1)[0][0] if years else None


# ---------------------------------------------------------------- checks ---
def check_unit_cost(unit_cost: pd.DataFrame) -> pd.DataFrame:
    """Check every unit cost row against its own arithmetic.

    Two checks per side of each row:

    * ``unit_times_quantity``: unit cost times quantity against cost. Costs
      print to 0.1 ($M) and unit costs to 0.001, so the allowance is the
      rounding either could carry, scaled by quantity, or 0.1% of the cost,
      whichever is larger. The second covers SARs that print a unit cost to
      fewer places than it shows (HH-60W, December 2022: 81.700 for about
      81.73, 0.04% out); a misread column is out by far more.
    * ``pct_change``: the reported percentage change in unit cost against the
      ones the table implies. SARs are not consistent about how they get it:
      some divide the printed, rounded unit costs (SDB II, December 2017:
      .248 over .236 is the printed 5.08, where the costs give 4.79), some
      the unrounded ones, and the 2021-2022 template rounds away from zero to
      one decimal (-7.80 prints as -7.9). So the check takes the nearer of the
      two implied figures, cost over quantity and unit over unit, and allows
      0.11 points: one step of a one-decimal figure. A misread column is off
      by far more than that.

    Returns a frame of ``check, measure, comparison, subprogram, page,
    expected, found, ok``.
    """
    out = []
    for _, r in unit_cost.iterrows():
        tag = dict(measure=r["measure"], comparison=r["comparison"],
                   subprogram=r["subprogram"], page=r["page"])
        for side in ("baseline", "current"):
            c, q, u = r[f"{side}_cost"], r[f"{side}_quantity"], r[f"{side}_unit_cost"]
            if pd.notna(c) and pd.notna(q) and pd.notna(u) and q:
                allow = max(0.05 + 0.0005 * q + 1e-4 * c, 1e-3 * c)
                out.append({"check": f"{side}_unit_times_quantity", **tag,
                            "expected": c, "found": u * q, "ok": abs(u * q - c) <= allow})
        p = r["reported_pct_change"]
        bc, bq, cc, cq = (r["baseline_cost"], r["baseline_quantity"],
                          r["current_cost"], r["current_quantity"])
        bu, cu = r["baseline_unit_cost"], r["current_unit_cost"]
        implied = []
        if all(pd.notna(v) and v for v in (bc, bq, cc, cq)):
            implied.append(((cc / cq) / (bc / bq) - 1) * 100)
        if pd.notna(bu) and pd.notna(cu) and bu:
            implied.append((cu / bu - 1) * 100)
        if pd.notna(p) and implied:
            best = min(implied, key=lambda v: abs(v - p))
            out.append({"check": "pct_change", **tag, "expected": best,
                        "found": p, "ok": abs(best - p) <= 0.11})
    cols = ["check", "measure", "comparison", "subprogram", "page", "expected", "found", "ok"]
    return pd.DataFrame(out, columns=cols)


# ------------------------------------------------------------------- API ---
def parse_sar_pages(pages: Iterable[str], program_hint: Optional[str] = None) -> SarReport:
    """Parse a SAR given the laid-out text of each of its pages.

    ``pages[i]`` is page ``i + 1``. Separated from :func:`read_sar` so the
    parser can be tested on text alone, without a PDF library installed.

    ``program_hint`` names the program when the page headers do not, as on a
    scan whose OCR turns the header into debris; the catalogue's guess from
    the file name is the usual source. The report's own name wins when it
    reads cleanly.

    Raises:
        SarParseError: No unit cost block was found.
    """
    pages = list(pages)
    template = _template(pages)
    doc_by = _document_base_year(pages)
    uc_pages = [i for i, p in enumerate(pages) if _is_unit_cost_page(p)]
    # The header that names the program is the first one carrying the
    # report's date line; some unit cost pages open on the section title.
    headers = [_header_line(pages[i]) for i in uc_pages] + [_header_line(p) for p in pages]
    header = next((h for h in headers if _report_label(h)), headers[0] if headers else "")
    label = _report_label(header)
    program = _program_name(header, label)
    # Without its date line a header is not known to be the page header at
    # all (MH-139A, June 2024, yields "Threshold"), so the hint wins then too.
    if program_hint and (not label or not _plausible_name(program)):
        program = program_hint
    rows = []
    for i in uc_pages:
        rows.extend(_parse_unit_cost_page(pages[i], i + 1, doc_by, program))
    if not rows:
        raise SarParseError("no unit cost section found")
    unit_cost = pd.DataFrame(rows, columns=list(UNIT_COST_COLUMNS))
    return SarReport(program=program, report_label=label,
                     report_year=_report_year(label), template=template,
                     unit_cost=unit_cost, checks=check_unit_cost(unit_cost))


def pdf_pages(path) -> "list[str]":
    """The laid-out text of every page of a PDF (needs the ``public`` extra)."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "reading SAR PDFs needs pdfplumber: pip install \"cost-core[public]\""
        ) from exc
    with pdfplumber.open(str(path)) as pdf:
        return [page.extract_text(layout=True) or "" for page in pdf.pages]


def read_sar(path, source: Optional[dict] = None,
             program_hint: Optional[str] = None) -> SarReport:
    """Read the unit cost section of a SAR PDF.

    Args:
        path: The PDF.
        source: Provenance to attach, e.g. from :func:`cost_core.public.fetch`.
        program_hint: Name to use if the report's own header is unreadable.
    """
    report = parse_sar_pages(pdf_pages(Path(path)), program_hint=program_hint)
    report.source = dict(source or {"path": str(path)})
    return report
