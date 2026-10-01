# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
template_style.py - Make a template say where the user's numbers go.

The templates hold a worked example, which shows the layout, but not which
cells are the user's to change, which columns are required, or what a heading
means. This module styles a written template the way cost and financial
models are usually marked:

* every cell the user fills in is **yellow**, with empty yellow rows below
  the example for more;
* a **dark blue** heading is a column that must be filled in, a **light blue**
  one is optional, and hovering over a heading shows a note saying what goes
  there, in what units, with an example;
* a column with a fixed set of answers gets a **drop-down list**;
* on a Settings sheet the setting names are grey (leave them) and the values
  yellow, each with its own note and drop-down;
* the Instructions sheet opens with a key to the colours.

Only formatting, notes and validation are added, which no reader looks at, so
a styled template reads exactly as an unstyled one: the tests run every styled
template through its command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

#: (required, note, choices). Choices: a tuple of allowed values, a string
#: holding an Excel list formula, or None. ``strict`` lists refuse anything
#: else; the rest only suggest.
Column = Tuple[bool, str, Optional[object]]

INPUT = "FFF2CC"         # yellow: yours to fill in
REQUIRED = "1F4E79"      # dark blue heading: must be filled in
OPTIONAL = "DDEBF7"      # light blue heading: optional
LABEL = "EDEDED"         # grey: a label, leave it
BLANK_ROWS = 25

DISTRIBUTIONS = ("triangular", "pert", "uniform")
UNITS = ("dollars", "thousands", "millions", "billions")
YES_NO = ("yes", "no")

#: Columns whose drop-down refuses other entries; the rest only suggest.
STRICT = {"Distribution", "Duration Distribution", "Cost Distribution",
          "Delay Distribution", "Use", "Mandatory", "Phase", "Form", "Method", "Basis"}

TABLES: Dict[str, Dict[str, Dict[str, Column]]] = {
    "cost-risk": {
        "Elements": {
            "Element": (True, "The WBS element's name, e.g. 1.0 Air vehicle. Each name once.",
                        None),
            "Point Estimate": (True, "The number in your estimate for this element, in the "
                               "units on the Settings sheet.", None),
            "Low": (False, "The least it could plausibly cost. Leave Low and High blank for "
                    "an element with no uncertainty.", None),
            "Most Likely": (False, "The single most likely cost; blank means the point "
                            "estimate.", None),
            "High": (False, "The most it could plausibly cost.", None),
            "Distribution": (False, "The shape between Low and High: triangular (the "
                             "default), pert, uniform, or lognormal (Low and High are then "
                             "the ends of an 80% range, as a CER gives).",
                             DISTRIBUTIONS + ("lognormal",)),
        },
        "Risks": {
            "Risk": (True, "What might happen, e.g. Qualification test failure.", None),
            "Probability": (True, "The chance it happens: 0.3 or 30%.", None),
            "Low": (False, "The least it would cost if it happens.", None),
            "Most Likely": (True, "The most likely cost if it happens.", None),
            "High": (False, "The most it would cost if it happens.", None),
            "Element": (False, "Optional: the element it belongs to, for the report. Must "
                        "match a name on the Elements sheet.",
                        "=Elements!$A$2:$A$200"),
        },
        "Correlation": {
            "Element A": (True, "One element of a pair that tends to overrun together.",
                          "=Elements!$A$2:$A$200"),
            "Element B": (True, "The other element of the pair.", "=Elements!$A$2:$A$200"),
            "Correlation": (True, "How strongly they move together, -1 to 1 (0.3 is a "
                            "common default; pairs not listed take the Settings value).",
                            None),
        },
    },
    "cer": {
        "Data": {
            "Program": (False, "The past program's name, for the report and diagnostics.",
                        None),
            "Cost": (True, "What the program cost, all in one dollar year and one unit "
                     "(the Settings sheet says which).", None),
            "Use": (False, "yes to fit to this row, no to leave it out without deleting it.",
                    YES_NO),
            "Note": (False, "Why a row is left out, or where the data came from.", None),
            "*": (True, "A technical driver of cost (weight, power, lines of code). Every "
                  "column of numbers here is a driver unless the Settings sheet names them.",
                  None),
        },
        "Estimate": {
            "Program": (True, "The new program to price.", None),
            "*": (True, "Its value for this driver, with the same heading as on the Data "
                  "sheet.", None),
        },
    },
    "phase": {
        "Phasing": {
            "Line": (True, "The estimate line, e.g. Prototype development.", None),
            "Appropriation": (False, "Groups the results: RDT&E, Procurement, O&M or your "
                              "own word.", ("RDT&E", "Procurement", "O&M", "MILCON")),
            "Amount": (True, "The line's total in base-year dollars (the Base Year on the "
                       "Settings sheet).", None),
            "Start": (True, "The first fiscal year it's spent in, e.g. 2027.", None),
            "Years": (True, "How many years it's spent over (or give an End year column "
                      "instead).", None),
            "Profile": (False, "How the money goes out: uniform (default), front, back, "
                        "bell, rayleigh, or one percentage per year such as 10;30;40;20.",
                        ("uniform", "front", "back", "bell", "rayleigh")),
            "Peak": (False, "For rayleigh only: where spending peaks, as a fraction of the "
                     "years (0.38 if blank).", None),
            "Index": (False, "Which index on the Index sheet this line inflates with; blank "
                      "uses the one on the Settings sheet.", None),
        },
        "Index": {
            "index_name": (True, "The index's name, e.g. the appropriation it's for. The "
                           "example's index is INVENTED: replace it with the published one.",
                           None),
            "fiscal_year": (True, "The fiscal year.", None),
            "index_value": (True, "The index for that year (any base; only the ratios "
                            "matter).", None),
        },
    },
    "evm": {
        "Data": {
            "Period": (True, "The month the row reports: 2026-01, 1/31/2026 or 1, 2, 3.",
                       None),
            "Control Account": (False, "Optional label; with it every account is analysed "
                                "as well as the total.", None),
            "Planned Value (BCWS)": (True, "Budgeted cost of work scheduled in the period "
                                     "(not cumulative).", None),
            "Earned Value (BCWP)": (True, "Budgeted cost of work performed in the period.",
                                    None),
            "Actual Cost (ACWP)": (True, "Actual cost of work performed in the period.",
                                   None),
            "Contractor EAC": (False, "Optional: the contractor's estimate at completion "
                               "as reported.", None),
        },
    },
    "jcl": {
        "Activities": {
            "ID": (True, "A short name other rows refer to, e.g. design.", None),
            "Name": (False, "A longer description.", None),
            "Duration": (True, "Months, most likely.", None),
            "Duration Low": (False, "Shortest plausible duration, months.", None),
            "Duration Most Likely": (False, "Most likely duration, months.", None),
            "Duration High": (False, "Longest plausible duration, months.", None),
            "Duration Distribution": (False, "triangular (default), pert or uniform.",
                                      DISTRIBUTIONS),
            "Predecessors": (False, "IDs it waits for, separated by ;, each optionally with "
                             "a link type and lag: design; fsw -4; bus SS +2.", None),
            "Fixed Cost": (False, "Cost spent whatever the duration.", None),
            "Cost Low": (False, "Lowest plausible fixed cost.", None),
            "Cost Most Likely": (False, "Most likely fixed cost.", None),
            "Cost High": (False, "Highest plausible fixed cost.", None),
            "Cost Distribution": (False, "triangular (default), pert or uniform.",
                                  DISTRIBUTIONS),
            "Burn Rate": (False, "Cost per month the activity runs.", None),
        },
        "Risks": {
            "Risk": (True, "What might happen.", None),
            "Probability": (True, "The chance it happens: 0.3 or 30%.", None),
            "Activities": (True, "The activity IDs it delays, separated by ;.", None),
            "Delay Low": (False, "Shortest delay if it happens, months.", None),
            "Delay Most Likely": (False, "Most likely delay, months (or the only delay).",
                                  None),
            "Delay High": (False, "Longest delay, months.", None),
            "Delay Distribution": (False, "triangular (default), pert or uniform.",
                                   DISTRIBUTIONS),
            "Cost": (False, "Cost it adds besides the delay.", None),
        },
    },
    "aoa": {
        "Alternatives": {
            "Alternative": (True, "Each alternative's name, once. The status quo is one of "
                            "them.", None),
            "Effectiveness": (False, "Optional score, on any scale the same for all.", None),
            "Correlation": (False, "How much its cost lines overrun together (0.3 if "
                            "blank).", None),
        },
        "Lines": {
            "Alternative": (True, "Which alternative the line belongs to.",
                            "=Alternatives!$A$2:$A$100"),
            "Line": (True, "The cost line, e.g. Development.", None),
            "Phase": (True, "RDT&E, Procurement, MILCON, O&S or Disposal.",
                      ("RDT&E", "Procurement", "MILCON", "O&S", "Disposal")),
            "Total": (False, "Way 1: a total spread over Years from Start Year.", None),
            "Start Year": (False, "Way 1: the first year of the spread.", None),
            "Years": (False, "Way 1: how many years.", None),
            "Profile": (False, "Way 1: uniform, front, back or bell.",
                        ("uniform", "front", "back", "bell")),
            "Annual Amount": (False, "Way 2: the same amount every year from First Year to "
                              "Last Year.", None),
            "First Year": (False, "Way 2: the first year.", None),
            "Last Year": (False, "Way 2: the last year.", None),
            "Low Factor": (False, "Uncertainty: e.g. 0.9 means 10% under. Blank for none.",
                           None),
            "Most Likely Factor": (False, "Uncertainty: usually 1.0.", None),
            "High Factor": (False, "Uncertainty: e.g. 1.4 means 40% over.", None),
            "Distribution": (False, "triangular (default), pert or uniform.", DISTRIBUTIONS),
        },
        "Phased": {
            "Alternative": (True, "Way 3: the alternative of a line costed year by year.",
                            "=Alternatives!$A$2:$A$100"),
            "Line": (True, "Way 3: the line, as named on the Lines sheet.", None),
            "*": (False, "That fiscal year's cost, in base-year dollars.", None),
        },
    },
    "portfolio": {
        "Candidates": {
            "Candidate": (True, "The program. One row per funding option; the first row "
                          "carries Category, Mandatory and Requires.", None),
            "Category": (False, "A grouping for the results.", None),
            "Mandatory": (False, "yes to always fund it.", YES_NO),
            "Requires": (False, "Candidates it depends on, separated by ;.", None),
            "Option": (True, "The funding option: Full, Reduced rate, Defer...", None),
            "Value": (True, "The benefit score of funding it this way.", None),
            "*": (False, "That year's cost for this option.", None),
        },
        "Budget": {
            "Year": (True, "The fiscal year.", None),
            "Budget": (True, "The money available that year.", None),
        },
        "Exclusive": {
            "Candidates": (False, "Candidates only one of which may be funded, separated by "
                           ";.", None),
        },
    },
}

#: Settings: name -> (note, choices).
SETTINGS: Dict[str, Dict[str, Tuple[str, Optional[object]]]] = {
    "cost-risk": {
        "Units": ("What the costs are in: dollars, thousands, millions, billions or any "
                  "word.", UNITS),
        "Default Correlation": ("How much elements not paired on the Correlation sheet "
                                "move together (0.3 is usual).", None),
        "Iterations": ("How many simulations: 20,000 is plenty.", None),
        "Seed": ("Any whole number; the same seed gives the same answer.", None),
        "Lognormal Range": ("The share between a lognormal element's Low and High: 0.8 "
                            "for 80%.", None),
    },
    "cer": {
        "Cost": ("The heading of the cost column on the Data sheet.", None),
        "Drivers": ("The driver headings, separated by ;. Blank uses every other column of "
                    "numbers.", None),
        "Form": ("log-log (cost = a * driver^b, usual for cost) or linear.",
                 ("log-log", "linear")),
        "Method": ("MUPE (default), OLS or ZMPE; all three are compared anyway.",
                   ("MUPE", "OLS", "ZMPE")),
        "Confidence": ("The prediction interval: 0.8 for 80%.", None),
        "Units": ("What the costs are in.", UNITS),
        "Dollar Year": ("The dollar year of the costs, for the record, e.g. BY2026.", None),
    },
    "phase": {
        "Base Year": ("The dollar year of the amounts, e.g. 2026.", None),
        "Units": ("What the amounts are in.", UNITS),
        "Index": ("The index on the Index sheet that lines without their own use.",
                  "=Index!$A$2:$A$500"),
        "Inflation Rate": ("Instead of an Index sheet: a constant rate, 0.021 for 2.1%.",
                           None),
    },
    "jcl": {
        "Name": ("The project's name, for the report.", None),
        "Units": ("What the costs are in.", UNITS),
        "Standing Army": ("Cost per month of the whole project running.", None),
        "Duration Correlation": ("How much durations move together, 0 to 1.", None),
        "Cost Correlation": ("How much costs move together, 0 to 1.", None),
        "Confidence": ("The joint confidence to report: 0.7 for 70%.", None),
        "Iterations": ("How many simulations.", None),
        "Seed": ("Any whole number; the same seed gives the same answer.", None),
    },
    "aoa": {
        "Base Year": ("The dollar year the cost lines are in.", None),
        "Discount Rate": ("Real discount rate from OMB A-94: 0.02 for 2%.", None),
        "Inflation Rate": ("Annual inflation for then-year dollars: 0.02 for 2%.", None),
        "Units": ("What the costs are in, e.g. BY2026 $M.", None),
        "Basis": ("pv (present value, the A-94 comparison), by or ty.", ("pv", "by", "ty")),
        "Status Quo": ("The alternative the others are measured against: investment, "
                       "savings, payback.", "=Alternatives!$A$2:$A$100"),
        "Iterations": ("How many simulations per alternative.", None),
        "Seed": ("Any whole number; the same seed gives the same answer.", None),
    },
    "portfolio": {
        "Units": ("What the costs and budget are in.", None),
        "Delta": ("The extra money to test in one year.", None),
        "Growth Low": ("Cost growth factor, low: 0.92 means 8% under.", None),
        "Growth Most Likely": ("Cost growth factor, most likely: 1.0.", None),
        "Growth High": ("Cost growth factor, high: 1.45 means 45% over.", None),
        "Growth Correlation": ("How much programs' growth moves together, 0 to 1.", None),
        "Seed": ("Any whole number.", None),
    },
}

KEY = [
    ("Yellow cells", "Yours to fill in. The example rows are there to show the layout: "
     "type over them, and add more rows in the yellow space below.", INPUT, "000000"),
    ("Dark blue heading", "A column that must be filled in.", REQUIRED, "FFFFFF"),
    ("Light blue heading", "An optional column: leave it blank if it doesn't apply.",
     OPTIONAL, "000000"),
    ("Grey cells", "Labels: leave them as they are.", LABEL, "000000"),
    ("Hover over a heading", "A note says what goes in that column, in what units, with an "
     "example. A small arrow in a cell means a drop-down list of the allowed answers.",
     None, "000000"),
]


def _column_spec(table: Dict[str, Column], heading) -> Column:
    return table.get(str(heading).strip()) or table.get("*") or \
        (False, "", None)


def _validation(ws, choices, cells: str, strict: bool):
    from openpyxl.worksheet.datavalidation import DataValidation

    if isinstance(choices, str):
        formula = choices.lstrip("=")
    else:
        formula = '"' + ",".join(choices) + '"'
    dv = DataValidation(type="list", formula1=formula, allow_blank=True,
                        showErrorMessage=strict,
                        errorStyle="stop" if strict else "information",
                        error="Choose one of the listed answers.",
                        errorTitle="Not one of the choices")
    ws.add_data_validation(dv)
    dv.add(cells)


def _note(cell, text: str) -> None:
    from openpyxl.comments import Comment

    if text:
        c = Comment(text, "cost-core")
        c.width, c.height = 320, 110
        cell.comment = c


def style_table(ws, table: Dict[str, Column], blank_rows: int = BLANK_ROWS) -> None:
    """Headings coloured required or optional with a note each; the data area
    yellow, with blank yellow rows below; drop-downs where answers are fixed."""
    from openpyxl.styles import Border, Font, PatternFill, Side

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    last = max(ws.max_row, 1) + blank_rows
    for cell in ws[1]:
        if cell.value is None:
            continue
        required, note, choices = _column_spec(table, cell.value)
        cell.fill = PatternFill("solid", fgColor=REQUIRED if required else OPTIONAL)
        cell.font = Font(bold=True, color="FFFFFF" if required else "1F1F1F")
        _note(cell, ("Required. " if required else "Optional. ") + note)
        col = cell.column_letter
        for row in range(2, last + 1):
            target = ws[f"{col}{row}"]
            target.fill = PatternFill("solid", fgColor=INPUT)
            target.border = border
        if choices:
            _validation(ws, choices, f"{col}2:{col}{last}",
                        str(cell.value).strip() in STRICT)
    ws.sheet_properties.tabColor = "FFC000"


def style_settings(ws, settings: Dict[str, Tuple[str, Optional[object]]]) -> None:
    """Setting names grey with a note; values yellow with a drop-down."""
    from openpyxl.styles import Border, Font, PatternFill, Side

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=REQUIRED)
        cell.font = Font(bold=True, color="FFFFFF")
    for row in range(2, ws.max_row + 1):
        name, value = ws[f"A{row}"], ws[f"B{row}"]
        if name.value is None:
            continue
        note, choices = settings.get(str(name.value).strip(), ("", None))
        name.fill = PatternFill("solid", fgColor=LABEL)
        value.fill = PatternFill("solid", fgColor=INPUT)
        name.border = value.border = border
        _note(name, note)
        if choices:
            _validation(ws, choices, f"B{row}", False)
    ws.sheet_properties.tabColor = "FFC000"


def add_key(ws) -> None:
    """The colour key, at the top of the Instructions sheet."""
    from openpyxl.styles import Alignment, Font, PatternFill

    ws.insert_rows(1, len(KEY) + 2)
    ws["A1"] = "How to fill in this workbook"
    ws["A1"].font = Font(bold=True, size=13)
    for i, (label, text, fill, font) in enumerate(KEY, start=2):
        a, b = ws[f"A{i}"], ws[f"B{i}"]
        a.value, b.value = label, text
        a.font = Font(bold=True, color=font)
        if fill:
            a.fill = PatternFill("solid", fgColor=fill)
        b.alignment = Alignment(wrap_text=True, vertical="top")
    ws.sheet_properties.tabColor = "A6A6A6"


def style_template(path, kind: str) -> Path:
    """Style the template at ``path`` written for ``kind``."""
    from openpyxl import load_workbook

    path = Path(path)
    if path.suffix.lower() not in (".xlsx", ".xlsm") or kind not in TABLES and \
            kind not in SETTINGS:
        return path
    wb = load_workbook(path)
    tables = TABLES.get(kind, {})
    for ws in wb.worksheets:
        if ws.title in tables:
            style_table(ws, tables[ws.title])
        elif ws.title == "Settings" and kind in SETTINGS:
            style_settings(ws, SETTINGS[kind])
        elif ws.title == "Instructions":
            add_key(ws)
    wb.save(path)
    return path


def kinds() -> Sequence[str]:
    return sorted(set(TABLES) | set(SETTINGS))
