# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
excel_report.py - One Excel workbook per result, for people who work in Excel.

Cost analysts audit, brief and hand on work in Excel, and a reviewer should
not need Python to check a number. So every EVM, JCL, schedule check, AoA
and portfolio run also writes ``report.xlsx``:

* **Summary** first: what the result means in plain words, and the headline
  numbers;
* the tables, one per sheet, with frozen headers, filters and number formats;
* native Excel charts, live and editable, rather than pasted pictures;
* **Assumptions** last: every setting the run used.

Where a figure is simple arithmetic on the data, it is written as a formula,
not a pasted value. The EVM metrics sheet computes CV, SV, CPI, SPI, TCPI and
the CPI-based EAC from the BCWS, BCWP and ACWP columns beside them, so a
reviewer can click a cell and see how it was made. Figures that need an
iteration or a simulation (earned schedule, the forecast percentiles) are
values, and say where they came from.

Only openpyxl is used, which the package already requires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

TITLE = Font(bold=True, size=14)
HEAD = Font(bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
NOTE = Font(italic=True, color="595959")
BOLD = Font(bold=True)

MONEY = "#,##0.00"
MONEY0 = "#,##0"
RATIO = "0.000"
PCT = "0.0%"
NUM = "0.00"


def money_format(units: str = "") -> str:
    """A money format carrying its unit label, e.g. ``$#,##0"K"`` for $K."""
    units = (units or "").strip()
    if units in ("$K", "$M", "$B"):
        return f'"$"#,##0.0"{units[1]}"'
    if units == "$":
        return '"$"#,##0'
    return MONEY0


class ReportWorkbook:
    """A workbook built sheet by sheet, Summary first and Assumptions last."""

    def __init__(self, title: str, subtitle: str = ""):
        self.wb = Workbook()
        self.summary_sheet = self.wb.active
        self.summary_sheet.title = "Summary"
        self.title, self.subtitle = title, subtitle

    # ------------------------------------------------------------- summary
    def summary(self, meaning: Sequence[str], headline: Sequence[Tuple[str, object, str]],
                note: str = "") -> None:
        """The first sheet: the title, the plain-words reading, the numbers.

        ``headline`` is ``(label, value, number_format)`` rows.
        """
        ws = self.summary_sheet
        ws["A1"] = self.title
        ws["A1"].font = TITLE
        if self.subtitle:
            ws["A2"] = self.subtitle
            ws["A2"].font = NOTE
        row = 4
        ws.cell(row=row, column=1, value="What this means").font = BOLD
        for line in meaning:
            row += 1
            cell = ws.cell(row=row, column=1, value=f"- {line}")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
            ws.row_dimensions[row].height = 15 * max(1, len(line) // 95 + 1)
        row += 2
        ws.cell(row=row, column=1, value="Key numbers").font = BOLD
        for label, value, fmt in headline:
            row += 1
            ws.cell(row=row, column=1, value=label)
            cell = ws.cell(row=row, column=2, value=_cell_value(value))
            if fmt:
                cell.number_format = fmt
        if note:
            row += 2
            cell = ws.cell(row=row, column=1, value=note)
            cell.font = NOTE
            cell.alignment = Alignment(wrap_text=True)
            ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
            ws.row_dimensions[row].height = 30
        ws.column_dimensions["A"].width = 58
        ws.column_dimensions["B"].width = 20
        ws.column_dimensions["C"].width = 20
        ws.column_dimensions["D"].width = 20

    # -------------------------------------------------------------- tables
    def table(self, name: str, df: pd.DataFrame, formats: Optional[Dict[str, str]] = None,
              note: str = "") -> Worksheet:
        """A sheet holding ``df``: header styled, frozen and filtered."""
        ws = self.wb.create_sheet(_sheet_name(name))
        start = 1
        if note:
            ws.cell(row=1, column=1, value=note).font = NOTE
            start = 3
        write_frame(ws, df, start_row=start, formats=formats or {})
        return ws

    def assumptions(self, settings: dict, notes: Iterable[str] = ()) -> None:
        """The last sheet: every setting, and every note, one per row."""
        ws = self.wb.create_sheet("Assumptions")
        ws["A1"] = "Setting"
        ws["B1"] = "Value"
        for c in (ws["A1"], ws["B1"]):
            c.font, c.fill = HEAD, HEAD_FILL
        row = 1
        for key, value in settings.items():
            row += 1
            ws.cell(row=row, column=1, value=str(key))
            ws.cell(row=row, column=2, value=_cell_value(value))
        notes = list(notes)
        if notes:
            row += 2
            ws.cell(row=row, column=1, value="Notes").font = BOLD
            for n in notes:
                row += 1
                cell = ws.cell(row=row, column=1, value=n)
                cell.alignment = Alignment(wrap_text=True)
                ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
                ws.row_dimensions[row].height = 15 * max(1, len(n) // 110 + 1)
        ws.column_dimensions["A"].width = 40
        ws.column_dimensions["B"].width = 80
        ws.freeze_panes = "A2"

    def save(self, path) -> Path:
        # Straight lines between the data points: a smoothed curve draws
        # values between the periods that no one computed.
        for ws in self.wb.worksheets:
            for chart in ws._charts:
                for s in chart.series:
                    s.smooth = False
        # Printed for a review, every sheet fits one page wide, landscape.
        for ws in self.wb.worksheets:
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.page_setup.orientation = "landscape"
        from cost_core.reporting import output

        # The marking prints at the top and bottom of every page, and shows
        # on screen at the top of the Summary.
        output.mark_workbook(self.wb)
        if output.marking():
            cell = self.summary_sheet["A3"]
            cell.value = output.marking()
            cell.font = BOLD
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.wb.save(path)
        return path


# ------------------------------------------------------------------ helpers
def _sheet_name(name: str) -> str:
    bad = '[]:*?/\\'
    return "".join("_" if c in bad else c for c in name)[:31]


def _cell_value(value):
    """Something openpyxl will write: numbers as numbers, NaN as blank,
    lists and dicts as compact JSON text."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (str,)) or hasattr(value, "isoformat"):
        return value
    return str(value)


def write_frame(ws: Worksheet, df: pd.DataFrame, start_row: int = 1, start_col: int = 1,
                formats: Optional[Dict[str, str]] = None) -> None:
    """Write ``df`` with a styled header, frozen below it, and a filter."""
    formats = formats or {}
    for j, col in enumerate(df.columns):
        c = ws.cell(row=start_row, column=start_col + j, value=str(col))
        c.font, c.fill = HEAD, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            c = ws.cell(row=start_row + i, column=start_col + j, value=_cell_value(value))
            fmt = formats.get(df.columns[j])
            if fmt:
                c.number_format = fmt
    for j, col in enumerate(df.columns):
        width = max([len(str(col))] + [len(str(v)) for v in df.iloc[:200, j]]) + 2
        ws.column_dimensions[get_column_letter(start_col + j)].width = min(max(width, 10), 48)
    ws.row_dimensions[start_row].height = 30
    ws.freeze_panes = ws.cell(row=start_row + 1, column=start_col)
    if len(df):
        ws.auto_filter.ref = (f"{get_column_letter(start_col)}{start_row}:"
                              f"{get_column_letter(start_col + len(df.columns) - 1)}"
                              f"{start_row + len(df)}")


def _line_chart(title: str, y_title: str, x_title: str) -> LineChart:
    ch = LineChart()
    ch.title, ch.y_axis.title, ch.x_axis.title = title, y_title, x_title
    ch.height, ch.width = 9, 18
    return ch


def _scatter(title: str, x_title: str, y_title: str) -> ScatterChart:
    ch = ScatterChart()
    ch.title, ch.x_axis.title, ch.y_axis.title = title, x_title, y_title
    ch.style = 13
    ch.height, ch.width = 9, 18
    return ch


# ---------------------------------------------------------------------- EVM
def evm_workbook(data, fc, path, units: str = "") -> Path:
    """EVM status, the metrics as live formulas, the forecast and warnings."""
    from cost_core import plain

    m = data.metrics()
    status = str(data.periods[data.status - 1])
    fmt = money_format(units)
    rb = ReportWorkbook(f"Earned value: {data.name}",
                        f"Status period {status}. Costs in {units or 'the units of the data'}.")
    last = m.iloc[-1]
    q = lambda p: float(np.quantile(fc.eac, p))  # noqa: E731
    headline = [
        ("Budget at completion (BAC)", data.bac, fmt),
        ("Percent complete", last.pct_complete, PCT),
        ("CPI", last.cpi, RATIO), ("SPI", last.spi, RATIO), ("SPI(t)", last.spi_t, RATIO),
        ("Schedule variance, periods (SV(t))", last.sv_t, NUM),
        ("Forecast final cost, P50", q(0.5), fmt), ("Forecast final cost, P80", q(0.8), fmt),
        ("Forecast finish period, P50", float(np.quantile(fc.finish, 0.5)), NUM),
        ("Forecast finish period, P80", float(np.quantile(fc.finish, 0.8)), NUM),
        ("Planned duration, periods", data.planned_duration, "0"),
        ("Chance BAC is enough", fc.confidence_of_cost(data.bac), PCT),
    ]
    if fc.contractor_eac is not None and np.isfinite(fc.contractor_eac):
        headline += [("Contractor EAC", fc.contractor_eac, fmt),
                     ("Chance the contractor EAC holds", fc.confidence_of_cost(fc.contractor_eac),
                      PCT)]
    rb.summary(plain.evm(data, fc, units, where="on the Warning signs sheet"), headline,
               note="The Metrics sheet computes CV, SV, CPI, SPI, TCPI and the CPI-based EAC with "
                    "Excel formulas from the BCWS, BCWP and ACWP beside them; click a cell to "
                    "see how it was made.")
    _evm_metrics_sheet(rb.wb, data, m, fmt)
    flags = data.flags()
    rb.table("Warning signs", flags.assign(raised=flags["raised"].map({True: "YES", False: ""})))
    pct = fc.percentiles((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    ws = rb.table("Forecast", pct, {"confidence": "0%", "eac": fmt, "finish": NUM},
                  note="Simulated from the program's own record; see the Assumptions sheet.")
    ch = _scatter("Final cost S-curve", "Final cost", "Confidence")
    s = Series(Reference(ws, min_col=1, min_row=4, max_row=3 + len(pct)),
               Reference(ws, min_col=2, min_row=4, max_row=3 + len(pct)), title="EAC")
    ch.series.append(s)
    ch.y_axis.number_format = "0%"
    ws.add_chart(ch, "F3")
    if data.accounts:
        rows = []
        for name, acct in data.accounts.items():
            r = acct.metrics().iloc[-1]
            rows.append({"account": name, "BAC": acct.bac, "BCWS": r.bcws, "BCWP": r.bcwp,
                         "ACWP": r.acwp, "CV": r.cv, "SV": r.sv, "CPI": r.cpi, "SPI": r.spi,
                         "SPI(t)": r.spi_t, "IEAC (CPI)": r.ieac_cpi})
        rb.table("Accounts", pd.DataFrame(rows).sort_values("CPI"),
                 {c: fmt for c in ("BAC", "BCWS", "BCWP", "ACWP", "CV", "SV", "IEAC (CPI)")}
                 | {"CPI": RATIO, "SPI": RATIO, "SPI(t)": RATIO},
                 note="Status period values per control account, worst CPI first.")
    from cost_core.evm.checks import data_checks, thresholds, variance_breaches

    limits = thresholds(data)
    rb.table("Data checks", data_checks(data),
             note="Questions to ask about the data before trusting the metrics. A negative "
                  "period value can be a legitimate correction.")
    rb.table("Variance reports", variance_breaches(data, **limits),
             {"cv": fmt, "sv": fmt, "cv_pct": "0.0", "sv_pct": "0.0", "cpi": RATIO,
              "spi": RATIO},
             note="Accounts whose cumulative variance breaks the thresholds "
                  f"({limits or 'the defaults: 10% of BCWP for cost, 10% of BCWS for schedule'}).")
    rb.assumptions({"status period": status, "BAC": data.bac, "simulated completions": fc.n_iter,
                    "seed": fc.seed, "units": units,
                    "variance thresholds": limits or "10% cost, 10% schedule (defaults)"},
                   list(data.notes) + list(fc.notes))
    return rb.save(path)


def _evm_metrics_sheet(wb: Workbook, data, m: pd.DataFrame, fmt: str) -> None:
    """Cumulative values as data, the arithmetic metrics as formulas."""
    ws = wb.create_sheet("Metrics")
    ws["A1"] = "BAC"
    ws["A1"].font = BOLD
    ws["C1"] = float(data.bac)
    ws["C1"].number_format = fmt
    ws["E1"] = ("Blue columns are the data; the rest are Excel formulas on them and on BAC, "
                "except ES, which needs the baseline's shape and is a computed value.")
    ws["E1"].font = NOTE
    heads = ["Period", "Time", "BCWS (cum)", "BCWP (cum)", "ACWP (cum)", "CV", "SV", "CPI",
             "SPI", "% complete", "TCPI (to BAC)", "IEAC (CPI)", "ES", "SPI(t)",
             "Contractor EAC"]
    top = 3
    for j, h in enumerate(heads, start=1):
        c = ws.cell(row=top, column=j, value=h)
        c.font, c.fill = HEAD, HEAD_FILL
        c.alignment = Alignment(wrap_text=True)
    for i, r in enumerate(m.itertuples(index=False), start=top + 1):
        ws.cell(row=i, column=1, value=_cell_value(r.period))
        ws.cell(row=i, column=2, value=float(r.time))
        for col, v in ((3, r.bcws), (4, r.bcwp), (5, r.acwp)):
            c = ws.cell(row=i, column=col, value=float(v))
            c.number_format = fmt
            c.font = Font(color="1F4E79")
        f = {
            6: f"=D{i}-E{i}", 7: f"=D{i}-C{i}",
            8: f'=IF(E{i}=0,"",D{i}/E{i})', 9: f'=IF(C{i}=0,"",D{i}/C{i})',
            10: f"=D{i}/$C$1", 11: f'=IF($C$1-E{i}=0,"",($C$1-D{i})/($C$1-E{i}))',
            12: f'=IF(OR(H{i}="",H{i}=0),"",E{i}+($C$1-D{i})/H{i})',
        }
        for col, formula in f.items():
            ws.cell(row=i, column=col, value=formula)
        ws.cell(row=i, column=13, value=_cell_value(r.es))
        ws.cell(row=i, column=14, value=f'=IF(B{i}=0,"",M{i}/B{i})')
        ws.cell(row=i, column=15, value=_cell_value(r.eac))
        for col in (6, 7, 12, 15):
            ws.cell(row=i, column=col).number_format = fmt
        for col in (8, 9, 11, 14):
            ws.cell(row=i, column=col).number_format = RATIO
        ws.cell(row=i, column=10).number_format = PCT
        ws.cell(row=i, column=13).number_format = NUM
    for j, w in enumerate((12, 7, 14, 14, 14, 13, 13, 8, 8, 10, 10, 15, 8, 8, 14), start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.row_dimensions[top].height = 30
    ws.freeze_panes = ws.cell(row=top + 1, column=2)
    n = len(m)
    ch = _line_chart("Cumulative BCWS, BCWP and ACWP", "Cost", "Period")
    for col in (3, 4, 5):
        ch.add_data(Reference(ws, min_col=col, min_row=top, max_row=top + n), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=top + 1, max_row=top + n))
    ws.add_chart(ch, f"B{top + n + 3}")
    ch2 = _line_chart("CPI, SPI and SPI(t)", "Index", "Period")
    lo = float(np.nanmin(m[["cpi", "spi", "spi_t"]].to_numpy()))
    hi = float(np.nanmax(m[["cpi", "spi", "spi_t"]].to_numpy()))
    ch2.y_axis.scaling.min = max(0.0, np.floor(min(lo, 1.0) * 10 - 1) / 10)
    ch2.y_axis.scaling.max = np.ceil(max(hi, 1.0) * 10 + 0.5) / 10
    for col in (8, 9, 14):
        ch2.add_data(Reference(ws, min_col=col, min_row=top, max_row=top + n), titles_from_data=True)
    ch2.set_categories(Reference(ws, min_col=1, min_row=top + 1, max_row=top + n))
    ws.add_chart(ch2, f"J{top + n + 3}")


# ---------------------------------------------------------------------- JCL
def jcl_workbook(result, project, confidence: float, path, units: str = "",
                 critical_path: Optional[pd.DataFrame] = None) -> Path:
    from cost_core import plain

    fmt = money_format(units)
    rb = ReportWorkbook(f"Joint cost and schedule confidence: {project.name}",
                        f"{result.n_iter:,} simulated projects. Costs in {units or 'the units of the spec'}.")
    cost_p = float(np.quantile(result.cost, confidence))
    fin_p = float(np.quantile(result.finish, confidence))
    headline = [("Plan: finish (months)", result.point_finish, NUM),
                ("Plan: cost", result.point_cost, fmt),
                ("Joint confidence of the plan", result.point_jcl, PCT),
                (f"Cost P{confidence * 100:.0f}", cost_p, fmt),
                (f"Schedule P{confidence * 100:.0f} (months)", fin_p, NUM),
                ("Joint confidence of those two together", result.joint(cost_p, fin_p), PCT)]
    rb.summary(plain.jcl(result, confidence, units), headline)
    rb.table("Summary table", result.summary(confidence))
    front = result.frontier(confidence, points=40)
    ws = rb.table(f"Frontier {confidence * 100:.0f}%", front,
                  {"finish": NUM, "cost": fmt, "joint": PCT},
                  note=f"Budget and date pairs that each reach {confidence:.0%} joint confidence.")
    ch = _scatter(f"{confidence:.0%} joint confidence frontier", "Finish (months)", "Budget")
    ch.series.append(Series(Reference(ws, min_col=2, min_row=4, max_row=3 + len(front)),
                            Reference(ws, min_col=1, min_row=4, max_row=3 + len(front)),
                            title="Frontier"))
    ws.add_chart(ch, "F3")
    rb.table("Criticality", result.criticality(),
             {"criticality": PCT, "duration_mean": NUM, "rank_corr_finish": NUM,
              "rank_corr_cost": NUM})
    if critical_path is not None:
        rb.table("Critical path", critical_path)
    rb.assumptions({"n_iter": result.n_iter, "seed": result.seed, "confidence": confidence,
                    "standing army per month": project.standing_army,
                    "duration correlation": project.duration_correlation,
                    "cost correlation": project.cost_correlation,
                    "risks": [r.name for r in project.risks], "units": units}, result.notes)
    return rb.save(path)


# --------------------------------------------------------------------- DCMA
def dcma_workbook(result, schedule, path) -> Path:
    from cost_core import plain

    rb = ReportWorkbook(f"DCMA 14-point assessment: {schedule.name}",
                        f"{len(schedule.detail)} detail tasks, {len(schedule.links)} links.")
    t = result.table.copy()
    t["result"] = t["passed"].map({True: "pass", False: "FAIL"}).fillna("n/a")
    rb.summary(plain.dcma(result, schedule),
               [("Checks passed", result.passed, "0"), ("Checks failed", result.failed, "0"),
                ("Not assessable from this file", 14 - result.passed - result.failed, "0")])
    rb.table("Checks", t[["check", "name", "count", "base", "value", "threshold", "result",
                          "note"]], {"value": RATIO, "threshold": RATIO})
    failing = pd.DataFrame([{"check": k, "task": n} for k, v in result.tasks.items() for n in v],
                           columns=["check", "task"])
    rb.table("Failing tasks", failing, note="Every task that failed each check.")
    rb.table("Tasks", schedule.tasks)
    rb.table("Links", schedule.links)
    rb.assumptions({"status date": schedule.status_date, "minutes per day": schedule.minutes_per_day,
                    "days per month": schedule.days_per_month}, schedule.notes)
    return rb.save(path)


# --------------------------------------------------------------- cost risk
def cost_risk_workbook(result, path) -> Path:
    from cost_core import plain

    units = result.units
    money = money_format(units)
    sim = result.sim
    rb = ReportWorkbook("Cost risk analysis",
                        f"{sim.n_iter:,} simulations; costs in {plain.units_label(units) or 'the units entered'}.")
    rb.summary(plain.cost_risk(result, plain.units_label(units)),
               [("Point estimate", result.point_estimate, money),
                ("Confidence of the point estimate", sim.point_estimate_percentile / 100.0, PCT),
                ("P50", sim.p50, money), ("P80", sim.p80, money), ("P90", sim.p90, money),
                ("Reserve to P80", sim.p80 - result.point_estimate, money),
                ("Coefficient of variation", sim.cv, PCT)])
    conf = result.confidence_table()
    ws = rb.table("Confidence", conf, {"confidence": PCT, "cost": money, "reserve": money,
                                       "reserve_pct": PCT})
    ch = _scatter("Total cost S-curve", "Total cost", "Confidence")
    n = len(conf)
    ch.series.append(Series(Reference(ws, min_col=1, min_row=2, max_row=1 + n),
                            Reference(ws, min_col=2, min_row=2, max_row=1 + n),
                            title="Total cost"))
    ws.add_chart(ch, "F2")
    rb.table("Reserve allocation", result.allocation(0.8),
             {"point_estimate": money, "own_p80": money, "allocated_p80": money,
              "reserve": money, "share_of_reserve": PCT},
             note="The total's P80 shared across elements and risks: what each averages in "
                  "the simulations whose total lands at the P80. The allocated column adds up "
                  "to the P80; the own P80 column does not, because percentiles don't add.")
    rb.table("Drivers", result.drivers(), {"std_dev": money, "covariance_with_total": NUM,
                                           "variance_share": PCT})
    rb.table("Elements", result.element_table(),
             {c: money for c in ("point_estimate", "low", "most_likely", "high", "mean",
                                 "p50", "p80")})
    risks = result.risk_table()
    if len(risks):
        rb.table("Risks", risks, {"probability": PCT} | {
            c: money for c in ("low", "most_likely", "high", "mean_if_it_happens",
                               "expected_cost")})
    rb.table("Correlation", result.correlation_matrix().rename_axis("element").reset_index(),
             {c: "0.00" for c in result.correlation_matrix().columns})
    rb.table("Correlation effect", result.impact.to_frame(),
             {"independent": money, "correlated": money, "ratio": "0.000"})
    rb.table("Convergence", sim.convergence(), {"p80": money, "relative_change": "0.000%"})
    rb.assumptions(result.assumptions)
    return rb.save(path)


# ---------------------------------------------------------------------- AoA
def aoa_workbook(result, path) -> Path:
    from cost_core import plain

    units = str(result.assumptions.get("units") or "")
    rb = ReportWorkbook("Analysis of alternatives: life-cycle cost",
                        f"{result.assumptions.get('n_iter', 0):,} draws per alternative; "
                        f"{result.assumptions.get('basis', '')} dollars.")
    s = result.summary
    headline = [(f"{r.alternative}: P50", r.p50, MONEY0) for r in s.itertuples()]
    rb.summary(plain.aoa(result, units), headline)
    rb.table("Alternatives", s, {c: MONEY0 for c in ("by", "ty", "pv", "mean", "p50", "p80")}
             | {"p_cheapest": PCT})
    rb.table("Cost lines", result.lines)
    sc = result.s_curves().reset_index()
    ws = rb.table("S-curves", sc, {c: MONEY0 for c in sc.columns if c != sc.columns[0]})
    ch = _scatter("Life-cycle cost S-curves", "Life-cycle cost", "Confidence (percentile)")
    n = len(sc)
    for j in range(2, len(sc.columns) + 1):
        # Confidence up the side, each alternative's cost along the bottom.
        ch.series.append(Series(Reference(ws, min_col=1, min_row=2, max_row=1 + n),
                                Reference(ws, min_col=j, min_row=2, max_row=1 + n),
                                title=str(sc.columns[j - 1])))
    ws.add_chart(ch, f"{get_column_letter(len(sc.columns) + 2)}2")
    rb.assumptions(result.assumptions)
    return rb.save(path)


# ---------------------------------------------------------------- portfolio
def portfolio_workbook(result, portfolio, path, units: str = "", tables: Optional[Dict[str, pd.DataFrame]] = None,
                       risk: Optional[pd.DataFrame] = None) -> Path:
    from cost_core import plain

    rb = ReportWorkbook("Portfolio: which programs to fund",
                        f"Costs in {units or 'the units of the spec'}.")
    rb.summary(plain.portfolio(result, len(portfolio.candidates), risk, units),
               [("Total value", result.value, NUM),
                ("Programs funded", len(result.funded), "0"),
                ("Candidates", len(portfolio.candidates), "0")])
    rb.table("Selected", result.selected, {"value": NUM, "cost": MONEY})
    ws = rb.table("Spend by year", result.spend)
    if len(result.spend) and {"year", "budget"} <= set(result.spend.columns):
        ch = _line_chart("Spend against budget", "Cost", "Year")
        cols = [j + 1 for j, c in enumerate(result.spend.columns) if c in ("budget", "spend")]
        for col in cols:
            ch.add_data(Reference(ws, min_col=col, min_row=1, max_row=1 + len(result.spend)),
                        titles_from_data=True)
        ych = list(result.spend.columns).index("year") + 1
        ch.set_categories(Reference(ws, min_col=ych, min_row=2, max_row=1 + len(result.spend)))
        ws.add_chart(ch, f"{get_column_letter(len(result.spend.columns) + 2)}2")
    for name, df in (tables or {}).items():
        rb.table(name, df)
    if risk is not None:
        rb.table("Budget risk", risk, {"p_over_budget": PCT})
    rb.assumptions({"units": units})
    return rb.save(path)
