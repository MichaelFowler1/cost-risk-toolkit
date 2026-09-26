# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
lot_workbook.py - The Excel workbook the lot cost engine writes.

Analyst_Summary, Estimate_Projections and Fit_Chart_Data, plus the optional
Risk_Summary, Risk_Intervals and Risk_SCurve sheets when the risk frames are
supplied, all with native Excel charts so the workbook stays live and editable
rather than carrying pasted images.

Ported unchanged from the original script.
"""

from __future__ import annotations

import openpyxl
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.chart.series import SeriesLabel
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText
from openpyxl.drawing.text import (
    CharacterProperties,
    Paragraph,
    ParagraphProperties,
    RichTextProperties,
)
from openpyxl.utils import get_column_letter
import pandas as pd

#: A default Excel column is roughly 1.72 cm wide. Chart anchors are spaced off
#: this so widening a chart cannot silently overlap the one beside it.
_COL_CM = 1.72


def _nice_bounds(lo: float, hi: float, pad_frac: float = 0.15):
    """Axis bounds padded off the data and rounded to a readable step.

    Framing an axis on the data is what makes a tight band or a narrow
    distribution legible, but the raw min and max give labels like 298 and
    51.43M. Rounding outwards to a round step keeps both.
    """
    import math

    span = hi - lo
    if not math.isfinite(span) or span <= 0:
        span = abs(hi) or 1.0
    lo -= span * pad_frac
    hi += span * pad_frac
    width = hi - lo
    step = 10.0 ** math.floor(math.log10(width))
    for mult in (1, 2, 2.5, 5, 10):
        if width / (step * mult) <= 8:
            step *= mult
            break
    return math.floor(lo / step) * step, math.ceil(hi / step) * step


def _money_axis_fmt(lo: float, hi: float) -> str:
    """Excel number format for a money axis covering lo to hi.

    Each comma in an Excel format divides by a thousand. The decimal place
    appears only for a narrow spread, where rounding to whole units prints
    the same label several times in a row.
    """
    span = abs(hi - lo)
    if abs(hi) >= 1e9:
        return '#,##0.0,,,"B"' if span < 1e10 else '#,##0,,,"B"'
    if abs(hi) >= 1e6:
        return '#,##0.0,,"M"' if span < 1e7 else '#,##0,,"M"'
    if abs(hi) >= 1e3:
        return '#,##0.0,"K"' if span < 1e4 else '#,##0,"K"'
    return "#,##0"


def _money_short(value: float) -> str:
    """Compact money for a chart label: $250.0M rather than 250,000,000."""
    v = float(value)
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cut:
            return f"${v / cut:,.1f}{suffix}"
    return f"${v:,.0f}"


def _chart_anchor(index: int, width_cm: float, start_col: int = 2,
                  row: int = 10) -> str:
    """Anchor cell for the nth chart in a left-to-right row of charts."""
    step = int(width_cm / _COL_CM) + 4  # +4 columns of breathing room
    return f"{get_column_letter(start_col + index * step)}{row}"


def _format_chart(
    chart,
    title: str,
    x_title: str,
    y_title: str,
    width: float = 18,
    height: float = 11,
):
    """Apply the same axis, title and legend treatment to every chart.

    Two Excel quirks are handled here. openpyxl writes ``delete="1"`` onto a
    freshly created axis, which tells Excel to hide that axis completely, tick
    numbers and all; and a chart title defaults to overlaying the plot rather
    than sitting above it. The manual plot-area layout then reserves room on
    the left and bottom so the axis titles do not land on top of the tick
    numbers.
    """
    chart.title = title
    chart.style = 13
    chart.title.overlay = False

    chart.x_axis.title = x_title
    chart.y_axis.title = y_title

    for axis in (chart.x_axis, chart.y_axis):
        axis.delete = False
        axis.tickLblPos = "nextTo"
        axis.majorTickMark = "out"
        axis.minorTickMark = "none"
        axis.numFmt = "#,##0"

    chart.width = width
    chart.height = height

    # Keep the legend off the data.
    if chart.legend is not None:
        chart.legend.position = "b"
        chart.legend.overlay = False

    chart.layout = Layout(
        manualLayout=ManualLayout(
            xMode="edge",
            yMode="edge",
            x=0.11,
            y=0.13,
            w=0.86,
            h=0.68,
        )
    )


def save_complete_excel_workbook(
    filename: str,
    projections_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    chart_df: pd.DataFrame,
    risk_summary_df: pd.DataFrame | None = None,
    risk_intervals_df: pd.DataFrame | None = None,
    risk_scurve_df: pd.DataFrame | None = None,
):
    """Write the tables and embed native Excel scatter plots.

    The three risk frames are optional: they are present only when cost_core
    is installed and the risk analysis ran.
    """
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
        if risk_summary_df is not None:
            risk_summary_df.to_excel(
                writer, sheet_name="Risk_Summary", index=False
            )
        if risk_intervals_df is not None:
            risk_intervals_df.to_excel(
                writer, sheet_name="Risk_Intervals", index=False
            )
        if risk_scurve_df is not None:
            risk_scurve_df.to_excel(
                writer, sheet_name="Risk_SCurve", index=False
            )

    wb = openpyxl.load_workbook(filename)
    ws = wb["Fit_Chart_Data"]
    max_r = len(chart_df) + 1  # 1-indexed including header

    # Data labels inherit the source cell's number format, so format the
    # actual-AUC column and the labels come out with a thousands separator.
    for row in range(2, max_r + 1):
        ws.cell(row=row, column=5).number_format = "#,##0.00"

    fit_chart_width = 18

    def build_scatter_chart(
        title: str,
        x_col: int,
        actual_col: int,
        est_col: int,
        slot: int,
        x_axis_title_text: str,
        y_axis_title_text: str = "Unit Cost / AUC ($K)",
    ):
        chart = ScatterChart()
        _format_chart(
            chart,
            title,
            x_axis_title_text,
            y_axis_title_text,
            fit_chart_width,
            11,
        )

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

        # Print the actual AUC beside each marker.
        #
        # The attribute is dLbls. A Series has no `dataLabels` alias, unlike a
        # chart, so assigning to `series.dataLabels` sets a stray Python
        # attribute that never reaches the XML -- no error, no labels. The
        # other show flags are written explicitly because Excel treats an
        # absent flag as inherited rather than false.
        labels = DataLabelList()
        labels.showVal = True
        labels.showSerName = False
        labels.showCatName = False
        labels.showLegendKey = False
        labels.showBubbleSize = False
        labels.showPercent = False
        labels.dLblPos = "t"  # above the marker, clear of the fitted line
        # openpyxl can only write formatCode here, never sourceLinked, so
        # Excel falls back to the source cell's format. The AUC column is
        # formatted below to match.
        labels.numFmt = "#,##0.00"
        # 8pt, because the rate chart puts lots of equal quantity almost on
        # top of each other and full-size labels collide.
        labels.txPr = RichText(
            bodyPr=RichTextProperties(),
            p=[
                Paragraph(
                    pPr=ParagraphProperties(
                        defRPr=CharacterProperties(sz=800)
                    ),
                    endParaRPr=CharacterProperties(sz=800),
                )
            ],
        )
        s_act.dLbls = labels

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
        ws.add_chart(chart, _chart_anchor(slot, fit_chart_width))

    # Chart 1: Learning Curve (LC) Fit -> X = LC Midpoint (Col 6), Y = Actual (Col 5) vs LC_Est (Col 7)
    build_scatter_chart(
        title="Learning Curve Fit: Actual vs Estimated AUC",
        x_col=6,
        actual_col=5,
        est_col=7,
        slot=0,
        x_axis_title_text="LC Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 2: Rate Model Fit -> X = Lot Qty (Col 2), Y = Actual (Col 5) vs Rate_Est (Col 10)
    build_scatter_chart(
        title="Rate Model Fit: Actual vs Estimated AUC",
        x_col=2,
        actual_col=5,
        est_col=10,
        slot=1,
        x_axis_title_text="Analogy Lot Quantity (Units / Lot)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 3: LC + Rate Fit -> X = LC+Rate Midpoint (Col 13), Y = Actual (Col 5) vs LCR_Est (Col 14)
    build_scatter_chart(
        title="LC+Rate Model Fit: Actual vs Estimated AUC",
        x_col=13,
        actual_col=5,
        est_col=14,
        slot=2,
        x_axis_title_text="LC+Rate Lot Midpoint (Unit Number)",
        y_axis_title_text="Unit Cost / AUC ($K)",
    )

    # Chart 4: the forecast with its prediction band, on the risk sheet.
    if risk_intervals_df is not None and len(risk_intervals_df):
        wsr = wb["Risk_Intervals"]
        last_r = len(risk_intervals_df) + 1

        # Find the columns by header rather than by position. cost_core owns
        # the shape of this frame, and it has changed once already.
        headers = {
            str(c.value): c.column
            for c in next(wsr.iter_rows(min_row=1, max_row=1))
        }
        x_name = next(
            (n for n in ("Last Unit in Lot", "Fiscal Year", "Lot Quantity")
             if n in headers),
            None,
        )
        y_names = [
            n for n in ("Unit Cost ($K)", "Unit Cost Lower", "Unit Cost Upper")
            if n in headers
        ]
        if x_name is None or len(y_names) < 2:
            from cost_core.reporting.output import mark_workbook
            mark_workbook(wb)
            wb.save(filename)
            return

        band = ScatterChart()
        _format_chart(
            band,
            "Forecast Unit Cost with Prediction Interval",
            x_name,
            "Unit Cost ($K)",
            20,
            11,
        )

        # Excel starts a value axis at zero, which squeezes a tight prediction
        # band into what looks like one thick line. Frame the axis on the band
        # itself so its width is actually readable.
        lo_y, hi_y = _nice_bounds(
            float(risk_intervals_df["Unit Cost Lower"].min()),
            float(risk_intervals_df["Unit Cost Upper"].max()),
        )
        band.y_axis.scaling.min = max(0.0, lo_y)
        band.y_axis.scaling.max = hi_y

        xs = Reference(
            wsr, min_col=headers[x_name], min_row=2, max_row=last_r
        )
        for name in y_names:
            dashed = name != "Unit Cost ($K)"
            s = Series(
                values=Reference(
                    wsr, min_col=headers[name], min_row=1, max_row=last_r
                ),
                xvalues=xs,
                title_from_data=True,
            )
            s.marker.symbol = "none" if dashed else "circle"
            s.smooth = False
            if dashed:
                s.graphicalProperties.line.dashStyle = "dash"
            band.series.append(s)
        wsr.add_chart(band, "P2")

    # Chart 5: the S-curve. Cost on the x axis, cumulative probability on the
    # y axis, which is the orientation everyone reads a P80 off.
    if risk_scurve_df is not None and len(risk_scurve_df):
        wss = wb["Risk_SCurve"]
        last_s = len(risk_scurve_df) + 1
        for row in range(2, last_s + 1):
            wss.cell(row=row, column=1).number_format = "0%"
            wss.cell(row=row, column=2).number_format = "#,##0"

        curve = ScatterChart()
        _format_chart(
            curve,
            "Cost S-Curve: Probability the Buy Comes In At or Below",
            "Buy Total ($)",
            "Cumulative Probability",
            20,
            11,
        )
        curve.y_axis.numFmt = "0%"
        # Whole dollars would give an axis of unreadable 9-digit labels that
        # collide; each comma is a division by a thousand in an Excel format.
        # The decimal place appears only for a narrow spread, where rounding
        # to whole units would print the same label several times over.
        lo_x = float(risk_scurve_df["Buy Total ($)"].min())
        hi_x = float(risk_scurve_df["Buy Total ($)"].max())
        curve.x_axis.numFmt = _money_axis_fmt(lo_x, hi_x)
        # The P50 and P80 markers are named in the legend rather than by data
        # labels beside them. Excel can only place a label immediately next to
        # its point, and the curve runs through the point, so on a steep
        # S-curve every available position puts the line through the text.
        # The legend sits below the plot where nothing can overlap it.
        if curve.legend is not None:
            curve.legend.position = "b"
            curve.legend.overlay = False
        # Probability is bounded, and Excel otherwise draws an axis to 120%.
        curve.y_axis.scaling.min = 0
        curve.y_axis.scaling.max = 1
        # Zoom to where the distribution actually sits, rather than showing
        # 500M of empty space because Excel starts every axis at zero.
        nice_lo, nice_hi = _nice_bounds(lo_x, hi_x)
        curve.x_axis.scaling.min = max(0.0, nice_lo)
        curve.x_axis.scaling.max = nice_hi
        s = Series(
            values=Reference(wss, min_col=1, min_row=1, max_row=last_s),
            xvalues=Reference(wss, min_col=2, min_row=2, max_row=last_s),
            title_from_data=True,
        )
        s.marker.symbol = "none"
        s.smooth = True
        s.tx = SeriesLabel(v="Buy total distribution")
        curve.series.append(s)

        # Call out P50 and P80. Each is its own one-point series, named for
        # the percentile and its cost, so the legend carries both while the
        # marker shows where it falls on the curve.
        pct = risk_scurve_df["Percentile"].round(4)
        # Left to its own palette Excel gives these two more shades of the
        # curve's green, so they disappear into it. Fixed colours from the
        # Okabe-Ito set, which stays distinguishable for the common colour
        # vision deficiencies, plus a different shape each so the two are
        # still telling apart in greyscale or on a photocopy.
        marks = [
            ("P50", 0.50, "0072B2", "circle"),      # blue
            ("P80", 0.80, "D55E00", "diamond"),     # vermillion
        ]
        for i, (name, level, colour, symbol) in enumerate(marks):
            hit = risk_scurve_df.loc[pct == round(level, 4), "Buy Total ($)"]
            if hit.empty:
                continue
            cost = float(hit.iloc[0])
            y_col = 4 + i * 2  # D, then F
            x_col = y_col + 1  # E, then G
            wss.cell(row=1, column=y_col, value=f"{name}  {_money_short(cost)}")
            wss.cell(row=2, column=y_col, value=level).number_format = "0%"
            wss.cell(row=1, column=x_col, value=f"{name} cost")
            wss.cell(
                row=2, column=x_col, value=cost
            ).number_format = "#,##0"

            mark = Series(
                values=Reference(wss, min_col=y_col, min_row=1, max_row=2),
                xvalues=Reference(wss, min_col=x_col, min_row=2, max_row=2),
                title_from_data=True,
            )
            mark.marker.symbol = symbol
            mark.marker.size = 11
            # Filled in the marker's own colour with a white keyline, so it
            # reads as a marker sitting on the curve rather than a kink in it.
            marker_style = GraphicalProperties(solidFill=colour)
            marker_style.line.solidFill = "FFFFFF"
            marker_style.line.width = 19050  # 1.5pt, in EMU
            mark.marker.graphicalProperties = marker_style
            # No connecting line: these are single points.
            mark.graphicalProperties.line.noFill = True
            curve.series.append(mark)

        wss.add_chart(curve, "I2")

    from cost_core.reporting.output import mark_workbook
    mark_workbook(wb)
    wb.save(filename)
