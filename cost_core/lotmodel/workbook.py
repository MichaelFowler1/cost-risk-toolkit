"""
workbook.py - The Excel deliverable, with embedded scatter charts.

Writes the three sheets the original tool produced -- Analyst_Summary,
Estimate_Projections and Fit_Chart_Data -- with native Excel scatter charts so
the workbook stays live and editable rather than carrying pasted images.

Ported unchanged from the original script. Extra sheets carrying the added
statistics are appended by :mod:`cost_core.lotmodel.enrich`, so an analyst who
only wants the original three still gets exactly those.
"""

from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.chart.label import DataLabelList


def save_complete_excel_workbook(
    filename: str,
    projections_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    chart_df: pd.DataFrame,
):
    """Write all 3 tables and embed native Excel scatter plots."""
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer, sheet_name="Analyst_Summary", index=False
        )
        projections_df.to_excel(
            writer,
            sheet_name="Estimate_Projections",
            index=False,
        )
        chart_df.to_excel(
            writer, sheet_name="Fit_Chart_Data", index=False
        )

    wb = openpyxl.load_workbook(filename)
    ws = wb["Fit_Chart_Data"]
    max_r = len(chart_df) + 1  # 1-indexed including header

    def build_scatter_chart(
        title: str,
        x_col: int,
        actual_col: int,
        est_col: int,
        chart_cell: str,
        x_axis_title_text: str,
        y_axis_title_text: str = "Unit Cost / AUC ($K)",
    ):
        chart = ScatterChart()
        chart.title = title
        chart.style = 13

        # Text labels on axes
        chart.x_axis.title = x_axis_title_text
        chart.y_axis.title = y_axis_title_text

        # Numeric tick marks & labels ('out' is the valid openpyxl value)
        chart.x_axis.tickLblPos = "nextTo"
        chart.y_axis.tickLblPos = "nextTo"
        chart.x_axis.majorTickMark = "out"
        chart.y_axis.majorTickMark = "out"

        # Chart dimensions
        chart.width = 16
        chart.height = 10

        x_values = Reference(
            ws, min_col=x_col, min_row=2, max_row=max_r
        )
        y_actual = Reference(
            ws, min_col=actual_col, min_row=1, max_row=max_r
        )
        y_est = Reference(
            ws, min_col=est_col, min_row=1, max_row=max_r
        )

        # Actuals: Markers with data labels showing numbers
        s_act = Series(
            values=y_actual,
            xvalues=x_values,
            title_from_data=True,
        )
        s_act.marker.symbol = "circle"
        s_act.marker.size = 7
        s_act.graphicalProperties.line.noFill = True

        # Show actual cost values on scatter points
        s_act.dataLabels = DataLabelList()
        s_act.dataLabels.showVal = True

        # Estimates: Smooth Fitted Curve (No markers)
        s_est = Series(
            values=y_est,
            xvalues=x_values,
            title_from_data=True,
        )
        s_est.marker.symbol = "none"
        s_est.smooth = True

        chart.series.append(s_act)
        chart.series.append(s_est)
        ws.add_chart(chart, chart_cell)

    # Chart 1: Learning Curve (LC) Fit -> X = LC Midpoint (Col 6), Y = Actual (Col 5) vs LC_Est (Col 7)
    build_scatter_chart(
        title="Learning Curve Fit: Actual vs Estimated AUC",
        x_col=6,
        actual_col=5,
        est_col=7,
        chart_cell="B10",
        x_axis_title_text="LC Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 2: Rate Model Fit -> X = Lot Qty (Col 2), Y = Actual (Col 5) vs Rate_Est (Col 10)
    build_scatter_chart(
        title="Rate Model Fit: Actual vs Estimated AUC",
        x_col=2,
        actual_col=5,
        est_col=10,
        chart_cell="L10",
        x_axis_title_text="Analogy Lot Quantity (Units / Lot)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 3: LC + Rate Fit -> X = LC+Rate Midpoint (Col 13), Y = Actual (Col 5) vs LCR_Est (Col 14)
    build_scatter_chart(
        title="LC+Rate Model Fit: Actual vs Estimated AUC",
        x_col=13,
        actual_col=5,
        est_col=14,
        chart_cell="V10",
        x_axis_title_text="LC+Rate Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    wb.save(filename)



