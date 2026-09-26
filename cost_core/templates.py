# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
templates.py - Files to fill in, laid out the way each command reads them.

``ce-core template evm`` writes an Excel workbook whose first sheet is the
data, already holding the worked example so the layout is plain, and whose
second sheet says what goes in each column. The spec-driven tools (JCL, AoA,
portfolio) get their worked example as a JSON file to edit, and the lot
model a CSV. A schedule is not typed in at all: it is saved from Microsoft
Project, and :data:`SCHEDULE_HOWTO` says how.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from cost_core.examples import example_path

#: The EVM template's column headings. Human names, which the EVM reader
#: understands (see :data:`cost_core.evm.metrics.COLUMN_ALIASES`).
EVM_HEADINGS = {"period": "Period", "wbs": "Control Account", "bcws": "Planned Value (BCWS)",
                "bcwp": "Earned Value (BCWP)", "acwp": "Actual Cost (ACWP)",
                "eac": "Contractor EAC"}

EVM_INSTRUCTIONS = [
    ("What this workbook is", "The input for `ce-core evm --data my_evm.xlsx`. The Data "
     "sheet holds an invented example program so the layout is plain: replace it with yours."),
    ("One row per", "Reporting period, per control account. Without a Control Account "
     "column, one row per period for the whole program."),
    ("Period", "The month the row reports: 2026-01, 1/31/2026 or a number 1, 2, 3 ... all "
     "work. Periods sort by date."),
    ("Control Account", "Optional. Any label. With it, every account is analysed as well "
     "as the program total."),
    ("Planned Value (BCWS)", "What the baseline planned to be done in that period. Fill "
     "it in for every period to the end of the plan, including the future."),
    ("Earned Value (BCWP)", "The budgeted value of the work actually done in that period. "
     "Leave blank after the latest (status) period."),
    ("Actual Cost (ACWP)", "What that work actually cost in that period. Leave blank after "
     "the status period."),
    ("Contractor EAC", "Optional. The contractor's estimate at completion, if reported."),
    ("Per period, not to date", "Each value is that period's own amount. For cumulative "
     "to-date values instead, run with --cumulative."),
    ("Units", "Any one currency unit throughout: dollars, thousands or millions. Say which "
     "with --units thousands (or dollars, millions) for the labels; nothing is converted."),
    ("Other column names", "PV, EV, AC, 'Planned Value', 'Month', 'WBS' and similar are "
     "recognised too, so an export from another tool may read as it is."),
    ("From an IPMDAR", "No need for this workbook: `ce-core evm --ipmdar the_delivery.zip`."),
]

SCHEDULE_HOWTO = """\
A schedule is not typed in: it comes straight from Microsoft Project.

  1. Open the schedule in Microsoft Project.
  2. File > Save As, and choose "XML Format (*.xml)" as the file type.
  3. Run:  ce-core schedule-check --mspdi my_schedule.xml

Primavera P6 and most other scheduling tools can export the same format:
look for "Microsoft Project XML" or "MSPDI" in their export options.

To see it working first, with an example schedule:  ce-core demo schedule
"""

NEXT_STEPS = {
    "evm": "Open it in Excel. The Data sheet holds an example; replace it with your "
           "program's numbers (the Instructions sheet says what goes where). Then run:\n"
           "  ce-core evm --data {path} --units thousands",
    "cost-risk": "Open it in Excel. It holds a worked example (a ground station upgrade, "
                 "in $M); replace the Elements rows with your estimate, and the Risks and "
                 "Correlation rows with yours or none (the Instructions sheet says what goes "
                 "where). Then run:\n  ce-core cost-risk --data {path}",
    "jcl": "It holds a worked example (a small spacecraft). Edit the activities, their "
           "durations, links, costs and risks (the Instructions sheet says what goes "
           "where), then run:\n  ce-core jcl --spec {path}",
    "aoa": "It holds a worked example of three alternatives. Edit their cost lines (the "
           "Instructions sheet says what goes where), then run:\n  ce-core aoa --spec {path}",
    "portfolio": "It holds a worked example of candidate programs and a budget. Edit them "
                 "(the Instructions sheet says what goes where), "
                 "then run:\n  ce-core portfolio --spec {path}",
    "lots": "One row per production lot: how many units, and what the lot cost (recurring, "
            "in one year's dollars). Replace the example rows, then run:\n"
            "  ce-core fit-lots --csv {path} --dollar-year 2026",
}

_LOTS = pd.DataFrame({"lot": [f"Lot {i}" for i in range(1, 7)],
                      "units": [22, 18, 25, 30, 30, 36],
                      "cost": [96_800_000, 70_200_000, 90_000_000, 100_500_000,
                               96_000_000, 111_600_000]})


def write(topic: str, out) -> Path:
    """Write the template for ``topic`` to ``out``; returns the path."""
    out = Path(out)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    if topic == "evm":
        return _evm(out)
    if topic == "cost-risk":
        from cost_core.costrisk import write_workbook
        return write_workbook(out)
    if topic == "lots":
        _LOTS.to_csv(out, index=False)
        return out
    if topic in ("jcl", "aoa", "portfolio"):
        # A workbook unless a .json is asked for, which scripts still use.
        if out.suffix.lower() == ".json":
            shutil.copyfile(example_path(topic), out)
            return out
        import json

        from cost_core.xlspec import write_workbook
        spec = json.loads(example_path(topic).read_text(encoding="utf-8"))
        return write_workbook(spec, topic, out)
    raise KeyError(f"No template for {topic!r}.")


def _evm(out: Path) -> Path:
    data = pd.read_csv(example_path("evm")).rename(columns=EVM_HEADINGS)
    data = data[[h for h in EVM_HEADINGS.values() if h in data.columns]]
    if out.suffix.lower() == ".csv":
        data.to_csv(out, index=False)
        return out
    from openpyxl.styles import Alignment, Font

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        data.to_excel(xl, sheet_name="Data", index=False)
        pd.DataFrame(EVM_INSTRUCTIONS, columns=["Column or topic", "What to put there"]) \
            .to_excel(xl, sheet_name="Instructions", index=False)
        sheet = xl.sheets["Data"]
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for col, width in zip("ABCDEF", (12, 26, 22, 22, 20, 16)):
            sheet.column_dimensions[col].width = width
        notes = xl.sheets["Instructions"]
        notes.column_dimensions["A"].width = 26
        notes.column_dimensions["B"].width = 100
        for cell in notes[1]:
            cell.font = Font(bold=True)
        for row in notes.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
            row[0].alignment = Alignment(vertical="top")
    return out
