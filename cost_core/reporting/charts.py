"""
charts.py - Publication-quality figures for a cost briefing.

Every function here writes a PNG sized and weighted for a slide: readable at
projector distance, no chartjunk, and no colour that carries meaning on its own
(each series is distinguishable in greyscale, because printed handouts happen).

Charts are always saved, never shown. That is deliberate -- a figure that only
exists on screen cannot be attached to the run that produced it, and every
figure here is meant to be reproducible from its seed alongside the assumptions
log.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

# Charts are written to file, never displayed, so a non-interactive backend is
# always correct here. Respect an explicit choice if the caller made one.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

logger = logging.getLogger(__name__)

#: Figure DPI. 200 is enough for a projected slide and for print at this size.
DPI = 200

#: Colours chosen to stay distinguishable in greyscale and to colour-blind
#: viewers: they differ in lightness, not only in hue.
INK = "#1a1a1a"
PRIMARY = "#1f4e79"      # deep blue
SECONDARY = "#c0504d"    # brick red
ACCENT = "#e8a33d"       # amber
MUTED = "#9aa5b1"        # grey
GRID = "#d6dbe1"

_STYLE = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
    "figure.autolayout": False,
}


def _money(value: float, _pos: float | None = None) -> str:
    """Format a value as dollars, scaled to whatever reads cleanly."""
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"${value / 1e6:,.1f}M"
    if magnitude >= 1e3:
        return f"${value / 1e3:,.0f}K"
    return f"${value:,.0f}"


def _money_formatter(values) -> FuncFormatter:
    """A dollar formatter with enough decimals for *this* axis range.

    A fixed number of decimals produces duplicate tick labels whenever the
    range is narrow -- an axis reading "$4M $4M $5M $6M $6M $6M" is worse than
    no labels at all, because it looks like data. So the decimal count is
    chosen from the span actually being plotted.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    span = float(np.ptp(finite)) if finite.size else 0.0
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0

    if peak >= 1e9:
        scale, suffix = 1e9, "B"
    elif peak >= 1e6:
        scale, suffix = 1e6, "M"
    elif peak >= 1e3:
        scale, suffix = 1e3, "K"
    else:
        scale, suffix = 1.0, ""

    scaled_span = span / scale
    decimals = 0 if scaled_span >= 8 else (1 if scaled_span >= 0.8 else 2)

    def fmt(value: float, _pos: float | None = None) -> str:
        return f"${value / scale:,.{decimals}f}{suffix}"

    return FuncFormatter(fmt)


def _plain(value: float, _pos: float | None = None) -> str:
    """Tick label without scientific notation, for a log axis of plain units."""
    return f"{value:,.0f}" if abs(value) >= 1 else f"{value:g}"


def _label_log_axis(axis, formatter) -> None:
    """Label a log axis on both its major and minor ticks.

    Necessary because cost data usually spans less than one decade, so the
    only *major* tick in range may be a single power of ten and every visible
    label is a minor one. Matplotlib formats those with its own scientific
    notation unless told otherwise, which is where "4 x 10^6" comes from
    instead of "$4.0M".
    """
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(formatter)
    axis.set_minor_locator(
        matplotlib.ticker.LogLocator(base=10.0, subs=(2.0, 3.0, 5.0, 7.0), numticks=12)
    )


def _titles(ax, title: str, subtitle: str | None = None) -> None:
    """Set a title with an optional subtitle above the axes, without collision.

    The title is pushed up by enough padding to leave room for the subtitle
    underneath it; both are left-aligned to the axes so they read as one block.
    """
    ax.set_title(title, loc="left", pad=26 if subtitle else 12)
    if subtitle:
        ax.text(
            0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=10,
            color=MUTED, va="bottom", ha="left",
        )


def _finish(fig, path: str | Path) -> Path:
    """Save and close. Closing matters: a long run otherwise leaks figures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path


# ==========================================================================
# S-curve
# ==========================================================================
def plot_s_curve(
    result,
    path: str | Path,
    *,
    comparison=None,
    comparison_label: str = "Independent (no correlation)",
    title: str = "Total cost S-curve",
    subtitle: str | None = None,
) -> Path:
    """Cumulative distribution of total cost, with the thresholds marked.

    The single most useful chart in a cost risk briefing, because it answers
    the only question that matters at funding time: what confidence level does
    this amount of money buy? The point estimate is marked with the percentile
    it actually sits at, which is usually a good deal lower than the audience
    expects.

    Args:
        result: A :class:`~cost_core.monte_carlo.RiskSimulationResult`.
        comparison: Optional second result plotted behind the first --
            typically the same model sampled without correlation, which makes
            the variance understatement visible rather than merely stated.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.6))

        if comparison is not None:
            ordered = np.sort(comparison.totals)
            probability = np.arange(1, ordered.size + 1) / ordered.size
            ax.plot(
                ordered, probability * 100.0, color=MUTED, linewidth=2.0,
                linestyle="--", label=comparison_label, zorder=2,
            )

        ordered = np.sort(result.totals)
        probability = np.arange(1, ordered.size + 1) / ordered.size
        ax.plot(
            ordered, probability * 100.0, color=PRIMARY, linewidth=2.6,
            label="Correlated", zorder=3,
        )

        for level, colour, style in (
            (50, ACCENT, ":"), (80, SECONDARY, "-."), (90, INK, ":")
        ):
            value = float(np.percentile(result.totals, level))
            ax.plot(
                [value, value], [0, level], color=colour, linestyle=style,
                linewidth=1.4, zorder=1,
            )
            ax.plot([ordered.min(), value], [level, level], color=colour,
                    linestyle=style, linewidth=1.4, zorder=1)
            ax.annotate(
                f"P{level}  {_money(value)}",
                xy=(value, level), xytext=(6, -14), textcoords="offset points",
                color=colour, fontsize=10, fontweight="bold",
            )

        point = result.point_estimate
        percentile = result.point_estimate_percentile
        ax.scatter(
            [point], [percentile], s=90, color="white", edgecolor=SECONDARY,
            linewidth=2.2, zorder=5,
        )
        ax.annotate(
            f"Point estimate {_money(point)}\nsits at the {percentile:.0f}th percentile",
            xy=(point, percentile), xytext=(12, 26), textcoords="offset points",
            fontsize=10, color=SECONDARY, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": SECONDARY, "linewidth": 1.2},
        )

        ax.set_xlabel("Total cost")
        ax.set_ylabel("Confidence level (%)")
        ax.set_ylim(0, 100)
        ax.set_xlim(ordered.min(), ordered.max())
        ax.xaxis.set_major_formatter(_money_formatter(ordered))
        ax.grid(axis="both", alpha=0.6)
        _titles(ax, title, subtitle)
        if comparison is not None:
            ax.legend(loc="lower right")
        fig.tight_layout()
        return _finish(fig, path)


# ==========================================================================
# Tornado
# ==========================================================================
def plot_tornado(
    result,
    path: str | Path,
    *,
    top_n: int = 12,
    title: str = "Contribution to total cost variance",
) -> Path:
    """Variance contribution by element and discrete risk.

    Ranked on the covariance decomposition rather than on input spread, so the
    bars sum to 100% and an element that is only moderately variable but moves
    with everything else gets the weight it deserves. Discrete risks are
    shaded differently because they are a different kind of thing and usually
    call for a different response.
    """
    frame = result.tornado().head(top_n).iloc[::-1]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, max(3.4, 0.42 * len(frame) + 1.8)))

        colours = [
            PRIMARY if kind == "element" else ACCENT for kind in frame["kind"]
        ]
        bars = ax.barh(
            frame["component"], frame["variance_share"] * 100.0,
            color=colours, edgecolor=INK, linewidth=0.5, height=0.72,
        )
        for bar, share in zip(bars, frame["variance_share"]):
            ax.annotate(
                f"{share * 100.0:.1f}%",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=10,
            )

        ax.set_xlabel("Share of total variance (%)")
        ax.grid(axis="x", alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, max(frame["variance_share"].max() * 100.0 * 1.18, 5))

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=PRIMARY),
            plt.Rectangle((0, 0), 1, 1, color=ACCENT),
        ]
        if "discrete risk" in set(frame["kind"]):
            ax.legend(handles, ["WBS element", "Discrete risk"], loc="lower right")
        fig.tight_layout()
        return _finish(fig, path)


# ==========================================================================
# Learning curve
# ==========================================================================
def plot_learning_curve(
    curve_fit,
    actual_lots: pd.DataFrame,
    path: str | Path,
    *,
    forecast_lots: np.ndarray | None = None,
    level: float = 0.80,
    title: str = "Production cost improvement curve",
) -> Path:
    """Observed lot averages, the fitted curve, and forecast intervals.

    Log-log axes, because that is where a power law is a straight line and
    where a departure from one is visible. The forecast band is a *prediction*
    interval: the range a new lot is expected to fall in, not the range the
    fitted line lies in.
    """
    from cost_core.learning_curve import _as_lot_array

    observed = _as_lot_array(actual_lots)
    quantity = (observed[:, 1] - observed[:, 0] + 1).astype(float)
    midpoint = np.sqrt(observed[:, 0] * observed[:, 1])
    actual_avg = actual_lots["lot_cost"].to_numpy() / quantity

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(9.0, 5.6))

        ax.scatter(
            midpoint, actual_avg, s=64, color="white", edgecolor=PRIMARY,
            linewidth=2.0, zorder=4, label="Reported lot average",
        )

        span_hi = float(observed[:, 1].max())
        if forecast_lots is not None and len(forecast_lots):
            span_hi = max(span_hi, float(np.max(np.asarray(forecast_lots))))
        smooth = np.unique(
            np.round(np.geomspace(1, max(span_hi, 2), 220)).astype(int)
        )
        ax.plot(
            smooth, curve_fit.model.lot_average(smooth, smooth),
            color=PRIMARY, linewidth=2.4, zorder=3,
            label=(
                f"{curve_fit.theory.value.title()} "
                f"{curve_fit.method.upper()} fit, "
                f"{curve_fit.slope:.1%} slope"
            ),
        )

        spread = list(actual_avg)
        if forecast_lots is not None and len(forecast_lots):
            forecast = curve_fit.forecast_lots(
                forecast_lots, level=level, kind="prediction"
            )
            fmid = np.sqrt(forecast["first_unit"] * forecast["last_unit"])
            # Error bars rather than a filled band: the interval belongs to
            # each specific forecast lot, and shading between three widely
            # spaced points draws a shape that implies values in between.
            ax.errorbar(
                fmid, forecast["lot_average"],
                yerr=np.vstack(
                    [
                        forecast["lot_average"] - forecast["lot_average_lower"],
                        forecast["lot_average_upper"] - forecast["lot_average"],
                    ]
                ),
                fmt="D", markersize=7, color=SECONDARY, ecolor=SECONDARY,
                elinewidth=1.8, capsize=6, capthick=1.8, zorder=4,
                label=f"Forecast lot, {level:.0%} prediction interval",
            )
            spread += list(forecast["lot_average_lower"])
            spread += list(forecast["lot_average_upper"])

        for brk in curve_fit.breaks:
            ax.axvline(
                brk.at_unit, color=ACCENT, linestyle="--", linewidth=1.6, zorder=2
            )
            ax.annotate(
                f"rate break\nunit {brk.at_unit}",
                xy=(brk.at_unit, max(spread)), xytext=(5, -8),
                textcoords="offset points", color=ACCENT, fontsize=9,
                fontweight="bold", va="top",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Lot midpoint unit (log scale)")
        ax.set_ylabel("Average unit cost (log scale)")
        _label_log_axis(ax.yaxis, _money_formatter(spread))
        _label_log_axis(ax.xaxis, FuncFormatter(_plain))
        ax.grid(which="both", alpha=0.5)
        ax.set_axisbelow(True)
        _titles(
            ax, title,
            f"CV {curve_fit.cv:.1%}  |  standard error "
            f"{_money(curve_fit.standard_error)}  |  "
            f"{curve_fit.result.n_obs} lots, {curve_fit.result.df} df",
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        return _finish(fig, path)


# ==========================================================================
# CER diagnostics
# ==========================================================================
def plot_cer_diagnostics(
    cer, path: str | Path, *, title: str = "CER fit and diagnostics"
) -> Path:
    """Four panels: the fit, residuals, leverage/influence, and a Q-Q plot.

    The residual and leverage panels are the ones that matter on a
    twenty-program sample. A CER can look excellent on the first panel while
    one program sets the entire slope, and only the third panel shows it.
    """
    from cost_core.cer.model import Form

    diag = cer.diagnostics()
    first = cer.predictors[0]
    x = cer.fitting_data[first].to_numpy()
    y = cer.fitting_data[cer.response].to_numpy()

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

        # --- 1. fit with prediction band, over the first predictor
        ax = axes[0, 0]
        ax.scatter(x, y, s=52, color="white", edgecolor=PRIMARY, linewidth=1.8,
                   zorder=3, label="Programs")
        if len(cer.predictors) == 1:
            grid = np.linspace(x.min(), x.max(), 120)
            band = cer.predict(
                {first: grid}, kind="prediction", warn_on_extrapolation=False
            )
            ax.plot(grid, band["fit"], color=PRIMARY, linewidth=2.2, zorder=2,
                    label="CER")
            ax.fill_between(
                grid, band["lower"], band["upper"], color=PRIMARY, alpha=0.14,
                zorder=1, label="80% prediction interval",
            )
        money = _money_formatter(np.concatenate([y, cer.result.fitted]))
        if cer.form is Form.LOG_LOG:
            ax.set_xscale("log")
            ax.set_yscale("log")
            _label_log_axis(ax.yaxis, money)
            _label_log_axis(ax.xaxis, FuncFormatter(_plain))
        else:
            ax.yaxis.set_major_formatter(money)
        ax.set_xlabel(first)
        ax.set_ylabel(cer.response)
        ax.set_title("Fit", loc="left", fontsize=11)
        ax.grid(alpha=0.5, which="both")
        ax.legend(fontsize=9)

        # --- 2. residuals against fitted
        ax = axes[0, 1]
        ax.axhline(0.0, color=MUTED, linewidth=1.2)
        ax.scatter(cer.result.fitted, diag.residuals, s=52, color="white",
                   edgecolor=SECONDARY, linewidth=1.8)
        ax.set_xlabel("Fitted value")
        ax.set_ylabel("Residual (fitting scale)")
        ax.xaxis.set_major_formatter(_money_formatter(cer.result.fitted))
        ax.set_title("Residuals", loc="left", fontsize=11)
        ax.grid(alpha=0.5)

        # --- 3. leverage against influence
        ax = axes[1, 0]
        ax.scatter(diag.leverage, diag.cooks_distance, s=52, color="white",
                   edgecolor=ACCENT, linewidth=1.8)
        ax.axvline(diag.leverage_threshold, color=MUTED, linestyle="--",
                   linewidth=1.2)
        ax.axhline(diag.cooks_threshold, color=MUTED, linestyle="--",
                   linewidth=1.2)
        for label, lev, cook in zip(
            diag.labels, diag.leverage, diag.cooks_distance
        ):
            if lev > diag.leverage_threshold or cook > diag.cooks_threshold:
                ax.annotate(label, xy=(lev, cook), xytext=(5, 4),
                            textcoords="offset points", fontsize=8)
        ax.set_xlabel("Leverage")
        ax.set_ylabel("Cook's distance")
        ax.set_title("Influence", loc="left", fontsize=11)
        ax.grid(alpha=0.5)

        # --- 4. normal Q-Q of the standardised residuals
        ax = axes[1, 1]
        ordered = np.sort(diag.standardized_residuals)
        n = ordered.size
        from scipy import stats as _stats

        theoretical = _stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.scatter(theoretical, ordered, s=52, color="white",
                   edgecolor=PRIMARY, linewidth=1.8)
        limits = [
            min(theoretical.min(), ordered.min()),
            max(theoretical.max(), ordered.max()),
        ]
        ax.plot(limits, limits, color=MUTED, linewidth=1.2, linestyle="--")
        ax.set_xlabel("Theoretical quantile")
        ax.set_ylabel("Standardised residual")
        ax.set_title("Normal Q-Q", loc="left", fontsize=11)
        ax.grid(alpha=0.5)

        fig.suptitle(
            f"{title}: {cer.equation()}", x=0.008, y=0.995, ha="left",
            va="top", fontsize=13, fontweight="bold",
        )
        fig.text(
            0.008, 0.952,
            f"{cer.method.upper()}  |  n={cer.result.n_obs}, df={cer.df}  |  "
            f"SE {_money(cer.standard_error)}  |  CV {cer.cv:.1%}",
            fontsize=10, color=MUTED, va="top",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        return _finish(fig, path)


# ==========================================================================
# Summary table
# ==========================================================================
def plot_summary_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    title: str = "Summary",
    column_formats: dict[str, str] | None = None,
) -> Path:
    """Render a DataFrame as a slide-ready table image.

    A rendered table alongside the CSV, because the number that ends up on the
    slide should come from the same run as the charts rather than being
    retyped from a spreadsheet.
    """
    formatted = frame.copy()
    for column, spec in (column_formats or {}).items():
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda v, s=spec: format(v, s) if pd.notna(v) else ""
            )
    formatted = formatted.astype(str)

    n_rows, n_cols = formatted.shape
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(
            figsize=(max(6.0, 2.1 * n_cols), 0.36 * (n_rows + 1) + 0.9)
        )
        ax.axis("off")

        # bbox rather than loc: it makes the table fill the axes exactly, so
        # the figure height set above is the table height and there is no
        # block of dead white space underneath.
        table = ax.table(
            cellText=formatted.to_numpy(),
            colLabels=[str(c).replace("_", " ") for c in formatted.columns],
            cellLoc="right",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            if row == 0:
                cell.set_facecolor(PRIMARY)
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f4f6f8")

        ax.set_title(title, loc="left", pad=14)
        fig.tight_layout()
        return _finish(fig, path)
