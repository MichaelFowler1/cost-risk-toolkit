"""
lots.py - The simple front door: units and cost, one row per lot.

This is the input shape an analyst actually has in front of them when they pull
a production history off a DD 1921 series or out of a spreadsheet: for each
lot, how many units and what it cost. Two columns.

    units,cost
    22,96800000
    18,70200000
    25,90000000

Everything else is derived. Lot 1 is units 1-22, lot 2 is units 23-40, and so
on by running total, which is what turns a flat list of lots into a position on
a learning curve. The fitting itself is the existing engine in
:mod:`cost_core.learning_curve`; this module is the part that takes a user's
two columns, checks the things that silently ruin a curve, and hands over
something well formed.

**The three ways this input goes wrong quietly**, all of which are checked
here rather than left to the analyst to remember:

*Nonrecurring cost mixed into the lot totals.* Nonrecurring is front-loaded, so
including it makes early lots look expensive and the curve looks steeper than
the production process really is. The direction of the error flatters nobody:
it overstates future savings. ``cost_basis`` has to be declared, and declaring
``"total"`` warns.

*Then-year dollars across several fiscal years.* Escalation pushes later lots
up while learning pushes them down, so the fitted slope comes out biased toward
100% -- less learning than actually happened. This module fits what it is
given and records the declared dollar basis; it does not deflate. Supply
constant-year dollars, or normalise first through
:mod:`cost_core.ingest.inflation`.

*Too few lots.* Degrees of freedom are ``lots - 2``. Two lots interpolate
exactly and are refused. Three gives one degree of freedom and an interval so
wide it cannot support a decision. Five is the practical floor.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from cost_core.fitting import FitError
from cost_core.learning_curve import (METHODS, CurveFit, RateBreak, Theory,
                                      _model_from_theta, comparison_table,
                                      compare_methods, compare_theories,
                                      fit_curve, retransformation_report)

logger = logging.getLogger(__name__)

CostBasis = Literal["recurring", "total"]

#: Below this many lots the fit is reported but flagged hard. Three is the
#: absolute floor (one degree of freedom); five is where an interval starts to
#: mean something.
COMFORTABLE_LOTS = 5

#: Column headers accepted for the quantity column, lowercased and stripped.
UNIT_ALIASES: tuple[str, ...] = (
    "units", "unit", "quantity", "qty", "lot_quantity", "lot_qty",
    "units_in_lot", "quantity_in_lot", "buy_quantity", "n_units",
)

#: Column headers accepted for the cost column.
COST_ALIASES: tuple[str, ...] = (
    "cost", "total_cost", "lot_cost", "lot_total_cost", "amount", "dollars",
    "recurring_cost", "cost_dollars", "total", "lot_total",
)

#: Optional column naming the lot, carried through to the report only.
LABEL_ALIASES: tuple[str, ...] = ("lot", "lot_number", "lot_no", "label", "name")


class LotInputError(ValueError):
    """Raised when the supplied lot data cannot support a curve.

    A ValueError by design: bad production history should stop the analysis,
    not produce a slope that looks reasonable on a chart.
    """


def _match_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    """Find the first column whose normalised name is a known alias."""
    normalised = {
        str(c).strip().lower().replace(" ", "_").replace("-", "_"): c
        for c in columns
    }
    for alias in aliases:
        if alias in normalised:
            return normalised[alias]
    return None


@dataclass
class LotSeries:
    """A production history: quantity and cost for each lot, in order.

    Attributes:
        quantities: Units in each lot, in build order.
        costs: Total cost of each lot, matching ``cost_basis``.
        cost_basis: ``"recurring"`` (correct for a learning curve) or
            ``"total"`` (includes nonrecurring, and warns).
        first_unit: Unit number the first lot starts at. Defaults to 1. Set it
            higher when the programme has a prior buy the curve has already
            learned through -- otherwise the fitted first-unit cost is the
            cost of a unit that was never built.
        dollar_year: Fiscal year the constant dollars are stated in.
            **Required.** No index is applied -- costs are taken as already
            normalised -- but "constant dollars" is meaningless without saying
            constant relative to *when*, and an output nobody can place in a
            year cannot be compared to anything or reused later. Refusing to
            proceed without it is cheaper than discovering a year later that
            nobody recorded it.
        labels: Optional lot names for reporting.
        quantity_definition: What a "unit" means on this programme --
            delivered, completed, accepted. Recorded, never inferred, because
            the three differ and the difference moves the curve.
        program: Optional programme name for reports.
    """

    quantities: np.ndarray
    costs: np.ndarray
    cost_basis: CostBasis = "recurring"
    first_unit: int = 1
    dollar_year: int | None = None
    labels: tuple[str, ...] = ()
    quantity_definition: str = "unspecified"
    program: str = "unnamed program"

    def __post_init__(self) -> None:
        self.quantities = np.atleast_1d(np.asarray(self.quantities))
        self.costs = np.atleast_1d(np.asarray(self.costs, dtype=float))
        self._validate()
        if not self.labels:
            self.labels = tuple(f"Lot {i}" for i in range(1, self.n_lots + 1))

    # ---------------------------------------------------------- validation
    def _validate(self) -> None:
        n_q, n_c = self.quantities.size, self.costs.size
        if n_q != n_c:
            raise LotInputError(
                f"Got {n_q} quantities and {n_c} costs. Every lot needs both."
            )
        if n_q == 0:
            raise LotInputError("No lots supplied.")

        if self.dollar_year is None:
            raise LotInputError(
                "dollar_year is required. Costs are treated as already in "
                "constant dollars and no inflation index is applied, but "
                "constant dollars are constant relative to a specific year -- "
                "without it the fitted first-unit cost cannot be compared to "
                "anything, escalated to a budget year, or reused. Pass "
                "dollar_year=2026 (or whichever year your data is stated in)."
            )
        try:
            self.dollar_year = int(self.dollar_year)
        except (TypeError, ValueError):
            raise LotInputError(
                f"dollar_year must be a fiscal year like 2026; got "
                f"{self.dollar_year!r}."
            ) from None
        if not 1900 <= self.dollar_year <= 2200:
            raise LotInputError(
                f"dollar_year {self.dollar_year} is not a plausible fiscal "
                f"year."
            )

        if self.cost_basis not in ("recurring", "total"):
            raise LotInputError(
                f"cost_basis must be 'recurring' or 'total'; got "
                f"{self.cost_basis!r}. It has to be declared because a learning "
                f"curve fitted to totals that include nonrecurring cost reads a "
                f"steeper slope than the production process really has."
            )

        # Quantities must be whole units: half an aircraft is not a data point.
        as_float = np.asarray(self.quantities, dtype=float)
        if np.any(as_float <= 0):
            bad = [
                f"lot {i + 1} = {v:g}"
                for i, v in enumerate(as_float) if v <= 0
            ]
            raise LotInputError(
                f"Lot quantities must be positive; found {', '.join(bad)}."
            )
        if np.any(np.abs(as_float - np.round(as_float)) > 1e-9):
            bad = [
                f"lot {i + 1} = {v:g}"
                for i, v in enumerate(as_float)
                if abs(v - round(v)) > 1e-9
            ]
            raise LotInputError(
                f"Lot quantities must be whole units; found {', '.join(bad)}."
            )
        self.quantities = np.round(as_float).astype(int)

        if np.any(self.costs <= 0):
            bad = [
                f"lot {i + 1} = {v:,.2f}"
                for i, v in enumerate(self.costs) if v <= 0
            ]
            raise LotInputError(
                f"Lot costs must be positive; found {', '.join(bad)}. A "
                f"learning curve is fitted in log space, which is undefined at "
                f"or below zero."
            )

        if int(self.first_unit) < 1:
            raise LotInputError(
                f"first_unit must be 1 or greater; got {self.first_unit}."
            )
        self.first_unit = int(self.first_unit)

        if self.labels and len(self.labels) != n_q:
            raise LotInputError(
                f"Got {len(self.labels)} labels for {n_q} lots."
            )

    # ------------------------------------------------------------ geometry
    @property
    def n_lots(self) -> int:
        return int(self.quantities.size)

    @property
    def total_units(self) -> int:
        return int(self.quantities.sum())

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def df(self) -> int:
        """Residual degrees of freedom a two-parameter curve would have."""
        return self.n_lots - 2

    def unit_ranges(self) -> np.ndarray:
        """(n, 2) array of first and last unit for each lot.

        This is the step that turns a flat list of lots into positions on a
        curve: lot boundaries are the running total of the quantities, offset
        by ``first_unit``. Lots are assumed contiguous and in build order,
        which is what "these are my lots, in order" means.
        """
        cumulative = np.cumsum(self.quantities)
        last = cumulative + self.first_unit - 1
        first = last - self.quantities + 1
        return np.column_stack([first, last]).astype(int)

    def cumulative_average(self) -> np.ndarray:
        """Cumulative cost divided by cumulative units, lot by lot."""
        return np.cumsum(self.costs) / np.cumsum(self.quantities)

    def check_constant_dollars(self, *, warn: bool = True) -> list[str]:
        """Look for the signature of then-year data labelled as constant.

        A learning curve means cumulative average cost falls monotonically. If
        it *rises* between lots, the most common explanation by some distance
        is escalation still in the numbers -- then-year dollars handed over as
        constant. That is worth catching here because the error is invisible
        downstream: the fit succeeds, and the only symptom is a slope biased
        toward 100%, which reads as "this programme did not learn much" rather
        than as a data problem.

        A warning rather than a refusal, because a genuine increase happens --
        a rate break, a design change, a second source standing up, a lot built
        after a long gap. The point is to make the analyst look, not to
        overrule them.

        Returns:
            A list of human-readable findings, empty when the series falls
            monotonically.
        """
        findings: list[str] = []
        cum_avg = self.cumulative_average()
        rises = np.flatnonzero(np.diff(cum_avg) > 0)

        if rises.size:
            detail = ", ".join(
                f"{self.labels[i]} -> {self.labels[i + 1]} "
                f"({cum_avg[i]:,.0f} to {cum_avg[i + 1]:,.0f}, "
                f"{(cum_avg[i + 1] / cum_avg[i] - 1) * 100:+.1f}%)"
                for i in rises
            )
            findings.append(
                f"Cumulative average cost RISES at {rises.size} point(s) in "
                f"{self.program}: {detail}. On a learning curve it should fall "
                f"monotonically. The usual cause is then-year dollars supplied "
                f"as constant FY{self.dollar_year} dollars, which biases the "
                f"fitted slope toward 100% and understates the learning that "
                f"actually occurred. A genuine increase is possible -- a rate "
                f"break, design change or production gap -- so confirm which "
                f"before relying on the slope."
            )

        # A lot average rising is a weaker but earlier signal: the cumulative
        # average can keep falling for a while after individual lots turn up.
        lot_avg = self.costs / self.quantities
        lot_rises = np.flatnonzero(np.diff(lot_avg) > 0)
        if lot_rises.size and not rises.size:
            detail = ", ".join(
                f"{self.labels[i]} -> {self.labels[i + 1]} "
                f"({(lot_avg[i + 1] / lot_avg[i] - 1) * 100:+.1f}%)"
                for i in lot_rises
            )
            findings.append(
                f"Cumulative average still falls, but individual lot average "
                f"cost rises at: {detail}. Worth checking against programme "
                f"history before fitting."
            )

        if warn:
            for finding in findings:
                warnings.warn(finding, RuntimeWarning, stacklevel=3)
                logger.warning(finding)
        return findings

    def dollar_basis_note(self) -> str:
        """The sentence the assumptions log records about escalation.

        Written out in full because "no index applied" needs to read as a
        decision somebody made, not as a step that was skipped.
        """
        return (
            f"Costs were supplied already normalised to constant FY"
            f"{self.dollar_year} dollars, as declared by the analyst on "
            f"ingest. No inflation index was applied by this tool and no "
            f"escalation assumption is embedded in the fit. Any error in the "
            f"upstream normalisation passes through unaltered; the fitted "
            f"slope and first-unit cost are stated in FY{self.dollar_year} "
            f"dollars and must be escalated before comparison with a budget "
            f"in any other year."
        )

    def to_frame(self) -> pd.DataFrame:
        """The input plus everything derived from it."""
        ranges = self.unit_ranges()
        return pd.DataFrame(
            {
                "lot": list(self.labels),
                "units": self.quantities,
                "cost": self.costs,
                "first_unit": ranges[:, 0],
                "last_unit": ranges[:, 1],
                "cumulative_units": np.cumsum(self.quantities),
                "lot_average_cost": self.costs / self.quantities,
                "cumulative_average_cost": self.cumulative_average(),
            }
        )

    # ----------------------------------------------------------- the fit
    def fit(
        self,
        *,
        theory: Theory | str = Theory.CRAWFORD,
        method: str = "ols",
        breaks: Iterable[RateBreak] = (),
        allow_small_sample: bool = True,
    ) -> CurveFit:
        """Fit a learning curve to this production history.

        Args:
            theory: Crawford unit theory (the usual reading of lot cost) or
                Wright cumulative average.
            method: ``"ols"``, ``"mupe"`` or ``"zmpe"``.
            breaks: Rate breaks to model explicitly.
            allow_small_sample: If False, refuse rather than warn when there
                are too few lots to support the model.

        Raises:
            LotInputError: With fewer than three lots, or when the sample is
                too small and ``allow_small_sample`` is False.
        """
        if self.n_lots < 3:
            raise LotInputError(
                f"{self.n_lots} lot(s) cannot support a learning curve. Two "
                f"points and two parameters interpolate exactly, leaving no "
                f"way to estimate a standard error or an interval -- a perfect "
                f"fit that means nothing. Three lots is the minimum and five "
                f"is the practical floor."
            )

        if self.cost_basis == "total":
            message = (
                f"Lot costs for {self.program} are declared as TOTAL cost, "
                f"which includes nonrecurring. Nonrecurring cost is "
                f"front-loaded and does not follow a learning curve, so it "
                f"makes the early lots look expensive and the fitted slope "
                f"come out steeper than the production process really is -- "
                f"overstating future savings. Fit on recurring cost only "
                f"where the data allows it."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            logger.warning(message)

        # Look for escalation still sitting in "constant" dollars before
        # fitting, so the analyst sees it alongside the slope rather than
        # after quoting it.
        self.check_constant_dollars()

        if self.n_lots < COMFORTABLE_LOTS:
            message = (
                f"{self.n_lots} lots gives {self.df} degree(s) of freedom, "
                f"below the {COMFORTABLE_LOTS} lots normally wanted. The slope "
                f"will carry a very wide interval and a single unusual lot can "
                f"set it. Treat the point estimate as indicative and read the "
                f"per-lot errors before relying on it."
            )
            if not allow_small_sample:
                raise LotInputError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        ranges = self.unit_ranges()
        fit = fit_curve(
            theory=theory,
            method=method,
            lots=ranges,
            lot_costs=self.costs,
            breaks=tuple(breaks),
        )
        logger.info(
            "%s: fitted %s %s curve on %d lots (%d units), slope %.2f%%, "
            "T1 %.4g, CV %.1f%%",
            self.program, str(theory), method.upper(), self.n_lots,
            self.total_units, fit.slope * 100.0, fit.t1, fit.cv * 100.0,
        )
        return fit

    # --------------------------------------------------------------- I/O
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        units_col: str | None = None,
        cost_col: str | None = None,
        label_col: str | None = None,
        **kwargs: Any,
    ) -> "LotSeries":
        """Build from a DataFrame, matching column names loosely.

        Real files say ``Qty``, ``Lot Quantity``, ``Units``, ``Total Cost``.
        The aliases in :data:`UNIT_ALIASES` and :data:`COST_ALIASES` cover the
        common spellings; anything else is named explicitly by the caller
        rather than guessed at.

        Raises:
            LotInputError: If either required column cannot be identified, or
                a named column is absent.
        """
        for name, supplied in (("units", units_col), ("cost", cost_col)):
            if supplied is not None and supplied not in frame.columns:
                raise LotInputError(
                    f"No column {supplied!r} for {name} in the file. Found: "
                    f"{list(frame.columns)}."
                )

        units_col = units_col or _match_column(frame.columns, UNIT_ALIASES)
        cost_col = cost_col or _match_column(frame.columns, COST_ALIASES)

        if units_col is None or cost_col is None:
            missing = []
            if units_col is None:
                missing.append(f"quantity (tried {list(UNIT_ALIASES[:5])}...)")
            if cost_col is None:
                missing.append(f"cost (tried {list(COST_ALIASES[:5])}...)")
            raise LotInputError(
                f"Could not identify the {' and '.join(missing)} column(s) in "
                f"{list(frame.columns)}. Rename the columns or pass "
                f"units_col= and cost_col= explicitly."
            )

        label_col = label_col or _match_column(frame.columns, LABEL_ALIASES)
        subset = frame[[units_col, cost_col]].dropna()
        if len(subset) < len(frame):
            logger.warning(
                "Dropped %d row(s) with a missing quantity or cost.",
                len(frame) - len(subset),
            )

        labels: tuple[str, ...] = ()
        if label_col is not None:
            labels = tuple(str(v) for v in frame.loc[subset.index, label_col])

        return cls(
            quantities=subset[units_col].to_numpy(),
            costs=_parse_currency(subset[cost_col]),
            labels=labels,
            **kwargs,
        )

    @classmethod
    def read(cls, path: str | Path, *, sheet: str | int = 0, **kwargs: Any) -> "LotSeries":
        """Read a lot series from CSV or Excel.

        Raises:
            FileNotFoundError: If the file is missing.
            LotInputError: On an unsupported extension, or if Excel support is
                requested without openpyxl installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No lot data file at {path}.")

        suffix = path.suffix.lower()
        if suffix in (".csv", ".txt"):
            frame = pd.read_csv(path)
        elif suffix in (".xlsx", ".xlsm"):
            try:
                frame = pd.read_excel(path, sheet_name=sheet)
            except ImportError as exc:
                raise LotInputError(
                    f"Reading {path.name} needs openpyxl, which is not "
                    f"installed. Run 'pip install openpyxl', or save the sheet "
                    f"as CSV."
                ) from exc
        else:
            raise LotInputError(
                f"Unsupported file type {suffix!r}. Supply a .csv or .xlsx."
            )

        logger.info("Read %d row(s) from %s", len(frame), path)
        return cls.from_frame(frame, **kwargs)


def _parse_currency(values: pd.Series) -> np.ndarray:
    """Coerce a cost column to float, tolerating $ signs and thousands commas.

    Spreadsheets export currency as text more often than not, and a column that
    silently becomes NaN is worse than one that fails.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values.to_numpy(dtype=float)

    cleaned = (
        values.astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (1,234) is negative
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].head(3).tolist()
        raise LotInputError(
            f"Could not read {int(parsed.isna().sum())} cost value(s) as "
            f"numbers, e.g. {bad}. Remove any text or footnote markers from "
            f"the cost column."
        )
    return parsed.to_numpy(dtype=float)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
@dataclass
class LotFitReport:
    """A fitted curve and everything a reviewer will ask about it."""

    series: LotSeries
    fit: CurveFit
    by_method: dict[str, CurveFit] = field(default_factory=dict)
    by_theory: dict[str, CurveFit] = field(default_factory=dict)

    @property
    def per_lot(self) -> pd.DataFrame:
        """Actual against fitted for each lot, with the percentage error.

        The most useful diagnostic on a short series: it shows *which* lot the
        curve misses, which is usually a programmatic event somebody remembers
        -- a second source, a design change, a cold line.
        """
        frame = self.series.to_frame()
        ranges = self.series.unit_ranges()
        # Derived after the fit, never used by it -- see the note on
        # LotFitReport for why the fit works on exact lot averages instead.
        frame["lot_midpoint"] = self.fit.lot_midpoint(ranges[:, 0], ranges[:, 1])
        frame["fitted_average"] = self.fit.result.fitted
        frame["fitted_lot_cost"] = frame["fitted_average"] * frame["units"]
        frame["residual"] = frame["cost"] - frame["fitted_lot_cost"]
        frame["percent_error"] = self.fit.result.percent_errors * 100.0
        return frame

    def curvature(self) -> tuple[float, float]:
        """Quadratic term in the log-log residuals, and its t statistic.

        A correctly specified learning curve is a straight line in log-log
        space, so its residuals should show no systematic bend. Escalation
        left in supposedly-constant dollars produces a very distinctive one:
        escalation compounds with *time* while learning compounds with *log
        quantity*, and since quantity grows faster than linearly in lot index,
        the mismatch shows up as convexity.

        This is far more sensitive than checking whether cumulative average
        cost rises. Escalation has to exceed roughly 10% a year before it
        overwhelms learning enough to turn the cumulative average upward,
        whereas the bend is detectable from about 2% -- which matters, because
        2-6% is the realistic range and it is exactly the range that passes a
        level check while shifting the fitted slope by several points.

        Returns:
            ``(coefficient, t_statistic)``. Both NaN when there are too few
            lots to estimate a quadratic (fewer than four).
        """
        arr = self.series.unit_ranges()
        x = np.log(np.sqrt(arr[:, 0] * arr[:, 1]))
        observed = np.log(self.series.costs / self.series.quantities)
        residual = observed - np.log(self.fit.result.fitted)

        design = np.column_stack([np.ones(x.size), x, x**2])
        dof = x.size - design.shape[1]
        if dof < 1:
            return float("nan"), float("nan")

        beta, *_ = np.linalg.lstsq(design, residual, rcond=None)
        fitted_resid = design @ beta
        sigma2 = float(np.sum((residual - fitted_resid) ** 2) / dof)
        if sigma2 <= 0:
            return float(beta[2]), float("nan")
        se = float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[2, 2]))
        return float(beta[2]), float(beta[2] / se) if se > 0 else float("nan")

    def check_curve_shape(self, *, threshold: float = 2.5, warn: bool = True) -> list[str]:
        """Flag a systematic bend in the residuals.

        Escalation is the most common cause but not the only one -- a rate
        break, a design change or a production gap bends the curve too. The
        finding is therefore phrased as "this is not a single clean curve",
        with escalation named as the first thing to rule out, rather than as a
        diagnosis.
        """
        findings: list[str] = []
        coefficient, t_stat = self.curvature()

        if not np.isfinite(t_stat):
            if self.series.n_lots < 4:
                findings.append(
                    f"Only {self.series.n_lots} lots, too few to test the "
                    f"shape of the residuals; a systematic bend from "
                    f"escalation or a rate break could not be detected here."
                )
        elif abs(t_stat) > threshold:
            direction = "upward (convex)" if coefficient > 0 else "downward (concave)"
            extra = (
                " Convexity is the signature of escalation left in dollars "
                "declared constant: it compounds with time while learning "
                "compounds with log quantity. Rule that out first, then look "
                "for a rate break, design change or production gap."
                if coefficient > 0 else
                " Check for a rate break or a change in the production "
                "process partway through the series."
            )
            findings.append(
                f"The residuals bend {direction} (quadratic t = {t_stat:.1f}), "
                f"so these lots are not one clean learning curve.{extra}"
            )

        if warn:
            for finding in findings:
                warnings.warn(finding, RuntimeWarning, stacklevel=3)
                logger.warning(finding)
        return findings

    def diagnostics(self) -> list[str]:
        """Every data-quality finding, in one list."""
        return [
            *self.series.check_constant_dollars(warn=False),
            *self.check_curve_shape(warn=False),
        ]

    def forecast(
        self, quantities: Iterable[int], *, level: float = 0.80
    ) -> pd.DataFrame:
        """Forecast the next lots, continuing from the last unit built.

        Prediction intervals, not confidence intervals: the question is what a
        *new lot* will cost, not where the fitted line lies.
        """
        quantities = [int(q) for q in quantities]
        if any(q <= 0 for q in quantities):
            raise LotInputError("Forecast lot quantities must be positive.")

        cursor = int(self.series.unit_ranges()[-1, 1])
        spans = []
        for q in quantities:
            spans.append((cursor + 1, cursor + q))
            cursor += q
        return self.fit.forecast_lots(
            np.array(spans), level=level, kind="prediction"
        )

    def equation(self) -> str:
        """The fitted curve written out, for quoting or re-use elsewhere."""
        return self.fit.equation()

    def price_lot_plan(
        self, quantities: Iterable[int], *, first_unit: int = 1
    ) -> pd.DataFrame:
        """Price an arbitrary lot plan with this curve, from unit 1 by default.

        The curve used as an estimating relationship rather than as a forecast
        of its own programme: hand it a buy profile and it produces the
        learning curve table an analyst would build by hand, lot midpoints and
        all.

        The intended use is analogy -- pricing a programme that has no cost
        history of its own using the slope from one that does. That is a
        judgement about whether the two programmes are similar enough in
        product and process for the slope to carry, and it belongs in the
        assumptions log, which is why :func:`build_assumption_log` records it
        as an assumption rather than a result whenever a plan is priced.

        Every row carries the source programme's name and its lot count.
        Without that, an output folder holds one table saying the curve was
        fitted to six lots and another showing five priced lots, with nothing
        to say they describe different programmes -- which reads as a
        truncation bug rather than as two different things.
        """
        priced = self.fit.price_lots(quantities, first_unit=first_unit)
        priced.insert(0, "priced_by_analogy_from", self.series.program)
        priced.insert(1, "source_lots_fitted", self.series.n_lots)
        return priced

    def simulate_forecast(
        self,
        quantities: Iterable[int],
        *,
        n_iter: int = 20_000,
        seed: int | None = None,
        include_residual: bool = True,
        residual_correlation: float = 0.30,
    ) -> "ForecastSimulation":
        """Monte Carlo the cost of a future buy, using only the fitted curve.

        The rest of this library simulates risk across WBS elements from
        distributions an analyst elicits. Lot data supports something
        different and, for a production buy, often more defensible: the
        uncertainty is *measured from the programme's own history* rather than
        judged, so there is nothing to argue about except the data.

        Two sources are propagated, and they are not the same thing:

        **Parameter uncertainty.** The slope and first-unit cost are estimates
        from a handful of lots, and a short series pins them down loosely. Each
        iteration draws a (log T1, b) pair from the fitted covariance, so the
        simulation explores every curve the data cannot rule out. On four lots
        this dominates.

        **Residual scatter.** Even given the true curve, an individual lot
        lands off it. Each iteration applies a multiplicative shock drawn from
        the fitted residual spread. This is what makes the answer a
        *prediction* about a real future lot rather than a statement about
        where the fitted line sits.

        What this deliberately does *not* include: schedule risk, requirement
        changes, rate changes not present in the history, or anything else the
        past lots never experienced. It is production cost risk conditional on
        the programme continuing as it has been, which is a narrower and more
        honest claim than a full risk model.

        Args:
            quantities: Units in each future lot.
            n_iter: Iterations.
            seed: Fixed seed. A P80 that moves between runs is not defensible.
            include_residual: Include lot-to-lot scatter. Setting this False
                gives the uncertainty in the fitted *curve* alone, which is a
                confidence statement, not a prediction.
            residual_correlation: Correlation between the residual shocks of
                different future lots. **Not zero by default**, for exactly the
                reason set out in :mod:`cost_core.monte_carlo`: consecutive
                lots on one programme share a workforce, a supply base and a
                schedule, so when one comes in high the next usually does too.
                Treating them as independent lets the shocks cancel and
                understates the variance of the total buy. Parameter
                uncertainty is already perfectly correlated across lots -- one
                curve is drawn per iteration and applied to all of them --
                which is correct and usually the dominant term.

        Raises:
            LotInputError: On a non-positive quantity, too few iterations, or
                a correlation outside the range a matrix of this size allows.
        """
        quantities = [int(q) for q in quantities]
        if not quantities or any(q <= 0 for q in quantities):
            raise LotInputError("Forecast lot quantities must all be positive.")
        if n_iter < 2:
            raise LotInputError(
                f"Need at least 2 iterations to form a distribution; got {n_iter}."
            )

        spans = self._forecast_spans(quantities)
        rng = np.random.default_rng(seed)

        theta_hat = self.fit.result.theta
        covariance = self.fit.result.cov
        draws = rng.multivariate_normal(theta_hat, covariance, size=n_iter)

        breaks = self.fit.model.breaks
        theory = self.fit.theory
        totals = np.empty(n_iter, dtype=float)
        per_lot = np.empty((n_iter, len(spans)), dtype=float)

        for i, theta in enumerate(draws):
            model = _model_from_theta(theta, theory, breaks)
            costs = model.lot_cost(spans[:, 0], spans[:, 1])
            per_lot[i] = costs
            totals[i] = costs.sum()

        if include_residual:
            # Multiplicative, because the fit's residuals are proportional,
            # and correlated across lots, because they are not independent
            # events on one production line.
            n_lots = per_lot.shape[1]
            sigma = self.fit.result.sigma
            if n_lots == 1 or residual_correlation == 0.0:
                log_shocks = rng.standard_normal(per_lot.shape) * sigma
            else:
                from cost_core.monte_carlo import uniform_correlation

                corr = uniform_correlation(n_lots, residual_correlation)
                try:
                    chol = np.linalg.cholesky(corr)
                except np.linalg.LinAlgError:  # pragma: no cover - guarded above
                    chol = np.linalg.cholesky(corr + np.eye(n_lots) * 1e-10)
                log_shocks = (rng.standard_normal(per_lot.shape) @ chol.T) * sigma
            per_lot = per_lot * np.exp(log_shocks)
            totals = per_lot.sum(axis=1)

        point = float(
            np.sum(self.fit.model.lot_cost(spans[:, 0], spans[:, 1]))
        )
        logger.info(
            "%s: simulated %d future lots (%d units) over %d iterations; "
            "point estimate %.4g sits at the %.1fth percentile",
            self.series.program, len(spans), sum(quantities), n_iter,
            point, float(np.mean(totals <= point) * 100.0),
        )
        return ForecastSimulation(
            totals=totals,
            per_lot=per_lot,
            quantities=tuple(quantities),
            spans=spans,
            point_estimate=point,
            seed=seed,
            included_residual=include_residual,
            residual_correlation=residual_correlation,
            n_history_lots=self.series.n_lots,
            program=self.series.program,
            dollar_year=self.series.dollar_year,
        )

    def _forecast_spans(self, quantities: Iterable[int]) -> np.ndarray:
        """Unit ranges for future lots, continuing from the last unit built."""
        cursor = int(self.series.unit_ranges()[-1, 1])
        spans = []
        for q in quantities:
            spans.append((cursor + 1, cursor + int(q)))
            cursor += int(q)
        return np.array(spans, dtype=int)

    def summary(self) -> pd.DataFrame:
        """Headline numbers, in the order they should be read.

        R squared is last, and carries a caveat. On a learning curve fitted to
        lot averages the points are strongly trended by construction, so R
        squared is high for almost any downward-sloping model and does not
        discriminate between one that forecasts well and one that does not.
        """
        lo, hi = self.fit.slope_interval
        rows = [
            ("lots", self.series.n_lots),
            ("units", self.series.total_units),
            ("degrees_of_freedom", self.fit.df),
            ("slope", self.fit.slope),
            ("slope_lower_80", lo),
            ("slope_upper_80", hi),
            ("first_unit_cost_t1", self.fit.t1),
            ("standard_error", self.fit.standard_error),
            ("cv", self.fit.cv),
            ("mean_percent_error", self.fit.result.mean_percent_error),
            ("worst_lot_percent_error", float(
                np.max(np.abs(self.fit.result.percent_errors)) * 100.0
            )),
            ("r_squared_read_last", self.fit.r_squared),
        ]
        return pd.DataFrame(rows, columns=["statistic", "value"])

    def narrative(self) -> str:
        """A few sentences a reviewer can read without the tables."""
        lo, hi = self.fit.slope_interval
        worst = int(np.argmax(np.abs(self.fit.result.percent_errors)))
        parts = [
            f"{self.series.program}: {self.fit.theory.value.title()} curve "
            f"fitted by {self.fit.method.upper()} to {self.series.n_lots} lots "
            f"covering {self.series.total_units} units, in constant FY"
            f"{self.series.dollar_year} dollars. Slope "
            f"{self.fit.slope:.2%} (80% interval {lo:.1%} to {hi:.1%}), "
            f"first-unit cost {self.fit.t1:,.0f}, standard error "
            f"{self.fit.standard_error:,.0f}, CV {self.fit.cv:.1%} on "
            f"{self.fit.df} degree{'s' if self.fit.df != 1 else ''} of freedom.",
            f"The curve misses {self.series.labels[worst]} by "
            f"{self.fit.result.percent_errors[worst] * 100:+.1f}%, the largest "
            f"departure in the series.",
        ]
        parts.extend(self.diagnostics())
        if self.series.cost_basis == "total":
            parts.append(
                "Costs were declared as totals including nonrecurring, so this "
                "slope is steeper than the production process alone."
            )
        if self.series.n_lots < COMFORTABLE_LOTS:
            parts.append(
                f"With only {self.series.n_lots} lots this is an indicative "
                f"fit; the interval on the slope is too wide to separate it "
                f"from neighbouring curves."
            )
        return " ".join(parts)

    def method_comparison(self) -> pd.DataFrame:
        return comparison_table(self.by_method) if self.by_method else pd.DataFrame()

    def theory_comparison(self) -> pd.DataFrame:
        return comparison_table(self.by_theory) if self.by_theory else pd.DataFrame()

    def retransformation(self):
        """Bias of the naive OLS retransformation, measured on this dataset."""
        if "ols" not in self.by_method:
            raise FitError(
                "Retransformation bias needs the OLS fit; run analyse_lots "
                "with compare=True."
            )
        return retransformation_report(self.by_method)


@dataclass
class ForecastSimulation:
    """Distribution of the cost of a future buy, from the fitted curve.

    Exposes the same vocabulary as the WBS-level simulator in
    :mod:`cost_core.monte_carlo` -- ``totals``, ``point_estimate``,
    ``point_estimate_percentile`` -- so the same S-curve chart draws it.
    """

    totals: np.ndarray
    per_lot: np.ndarray
    quantities: tuple[int, ...]
    spans: np.ndarray
    point_estimate: float
    seed: int | None
    included_residual: bool
    residual_correlation: float = 0.0
    n_history_lots: int = 0
    program: str = "unnamed program"
    dollar_year: int | None = None

    @property
    def n_iter(self) -> int:
        return int(self.totals.size)

    @property
    def mean(self) -> float:
        return float(np.mean(self.totals))

    @property
    def std(self) -> float:
        return float(np.std(self.totals, ddof=1))

    @property
    def cv(self) -> float:
        return float(self.std / self.mean) if self.mean else float("nan")

    @property
    def p50(self) -> float:
        return float(np.percentile(self.totals, 50))

    @property
    def p80(self) -> float:
        return float(np.percentile(self.totals, 80))

    @property
    def p90(self) -> float:
        return float(np.percentile(self.totals, 90))

    def percentile_of(self, value: float) -> float:
        return float(np.mean(self.totals <= value) * 100.0)

    @property
    def point_estimate_percentile(self) -> float:
        return self.percentile_of(self.point_estimate)

    def summary(self) -> pd.DataFrame:
        rows = [
            ("iterations", self.n_iter),
            ("future_lots", len(self.quantities)),
            ("future_units", int(sum(self.quantities))),
            ("point_estimate", self.point_estimate),
            ("point_estimate_percentile", self.point_estimate_percentile),
            ("mean", self.mean),
            ("std_dev", self.std),
            ("cv", self.cv),
            ("p50", self.p50),
            ("p80", self.p80),
            ("p90", self.p90),
            ("reserve_to_p80", self.p80 - self.point_estimate),
            ("reserve_to_p80_pct",
             100.0 * (self.p80 / self.point_estimate - 1.0)
             if self.point_estimate else float("nan")),
        ]
        return pd.DataFrame(rows, columns=["statistic", "value"])

    def narrative(self) -> str:
        basis = (
            f"curve uncertainty plus lot-to-lot scatter correlated at "
            f"{self.residual_correlation:.2f} across future lots"
            if self.included_residual
            else "curve uncertainty only (a confidence statement, not a "
                 "prediction about a real lot)"
        )
        year = f" FY{self.dollar_year}" if self.dollar_year else ""
        return (
            f"{self.program}: {len(self.quantities)} future lot(s) totalling "
            f"{sum(self.quantities)} units. Point estimate "
            f"{self.point_estimate:,.0f}{year} sits at the "
            f"{self.point_estimate_percentile:.0f}th percentile; P50 "
            f"{self.p50:,.0f}, P80 {self.p80:,.0f}, P90 {self.p90:,.0f}. "
            f"Risk reserve to P80 is {self.p80 - self.point_estimate:,.0f} "
            f"({100 * (self.p80 / self.point_estimate - 1):.1f}%). CV "
            f"{self.cv:.1%}. Uncertainty propagated: {basis}. Measured from "
            f"the programme's own {self.n_history_lots}-lot history rather "
            f"than from elicited distributions."
        )


def build_assumption_log(
    report: "LotFitReport",
    source: str | Path | None = None,
    priced_plan: pd.DataFrame | None = None,
    priced_from_unit: int = 1,
):
    """Assemble the written assumptions log for a lot-based fit.

    Records the dollar basis as an explicit decision, the quantity definition
    the analyst declared, every diagnostic, and the methodological choices --
    mapped to the four characteristics of a reliable estimate in the GAO Cost
    Estimating and Assessment Guide.
    """
    from cost_core.reporting.assumptions import AssumptionLog

    series, fit = report.series, report.fit
    log = AssumptionLog(
        title=f"Learning curve assumptions and provenance - {series.program}"
    )

    log.section(
        "1. Source data",
        f"- Source: {source if source else 'supplied in memory'}\n"
        f"- {series.n_lots} lots covering {series.total_units} units, "
        f"first unit numbered {series.first_unit}\n"
        f"- Cost basis declared: **{series.cost_basis}**\n"
        f"- Quantity definition declared: **{series.quantity_definition}**\n"
        f"- Dollars: constant **FY{series.dollar_year}**",
    ).table("1.1 Lots as supplied and derived", series.to_frame())

    # --- the escalation decision, recorded as a decision
    findings = report.diagnostics()
    coefficient, t_stat = report.curvature()
    body = series.dollar_basis_note()
    body += (
        "\n\nTwo checks were run for escalation left in the data. The first "
        "asks whether cumulative average cost ever rises, which it should not "
        "on a learning curve. The second tests the log-log residuals for a "
        "systematic bend, since escalation compounds with time while learning "
        "compounds with log quantity and the mismatch shows up as convexity."
    )
    if findings:
        body += "\n\n**Findings:**\n" + "\n".join(f"- {f}" for f in findings)
    else:
        body += (
            f"\n\nNeither check fired: cumulative average cost falls "
            f"monotonically, and the residual curvature is not significant "
            f"(quadratic t = {t_stat:.1f}). This is consistent with the "
            f"constant-dollar declaration."
        )
    body += (
        "\n\n**Limits of these checks.** The level check only fires once "
        "escalation is severe enough to overwhelm learning, which on a typical "
        "profile takes about 10% a year. The curvature test is sensitive from "
        "roughly 2%, but a rate break or design change bends the residuals the "
        "same way, so it identifies a departure from a single clean curve "
        "rather than escalation specifically. Neither can separate moderate "
        "escalation from genuinely slower learning without a fiscal year "
        "attached to each lot."
    )
    log.section("2. Dollar basis and escalation", body)
    log.assume(
        "Dollar basis",
        f"Costs are constant FY{series.dollar_year} dollars; no inflation "
        f"index applied by this tool.",
        "Declared by the analyst on ingest. Normalisation, if any was needed, "
        "happened upstream and is not verifiable from this input.",
    )
    log.assume(
        "Quantity definition",
        f"A 'unit' means: {series.quantity_definition}.",
        "Declared by the analyst. Delivered, completed and accepted counts "
        "differ, and the difference shifts every point on the curve.",
    )
    log.assume(
        "Lot sequencing",
        f"Lots are contiguous and in build order, starting at unit "
        f"{series.first_unit}.",
        "Implied by supplying lots as an ordered list. A prior buy the curve "
        "has already learned through would need a higher first unit.",
    )
    if series.cost_basis == "total":
        log.assume(
            "Nonrecurring cost included",
            "Lot costs include nonrecurring cost.",
            "Declared by the analyst. Nonrecurring is front-loaded and does "
            "not follow the curve, so the fitted slope is steeper than the "
            "production process alone.",
        )

    log.section(
        "3. The fitted equation",
        f"**{fit.equation()}**\n\n"
        f"Stated in constant FY{series.dollar_year} dollars, with `x` the "
        f"cumulative unit number counting from the start of production. Under "
        f"{fit.theory.value} theory this prices "
        + (
            "an individual unit; the cost of a lot is the sum over the units "
            "it contains."
            if fit.theory is Theory.CRAWFORD else
            "the cumulative *average* through quantity x; the cost of a lot is "
            "the difference between the cumulative totals at its endpoints. "
            "Reading it as a unit cost is the most common way a borrowed curve "
            "produces a wrong answer."
        ),
    ).table("3.1 Coefficients", fit.equation_detail())

    log.section(
        "4. Curve fit",
        f"- Theory: **{fit.theory.value}**\n"
        f"- Method: **{fit.method.upper()}**\n"
        f"- Slope **{fit.slope:.2%}**, first-unit cost "
        f"{fit.t1:,.0f} (FY{series.dollar_year})\n"
        f"- Standard error {fit.standard_error:,.0f}, CV {fit.cv:.1%}, "
        f"{fit.df} degrees of freedom\n\n"
        f"{report.narrative()}",
    ).table("4.1 Headline statistics", report.summary()).table(
        "4.2 Per-lot fit quality", report.per_lot[
            ["lot", "units", "lot_average_cost", "fitted_average", "percent_error"]
        ]
    )

    if report.by_method:
        log.table("4.3 Fitting methods compared", report.method_comparison())
        try:
            log.table(
                "4.4 Retransformation bias measured on this dataset",
                report.retransformation().to_frame(),
            )
        except FitError:  # pragma: no cover - only when compare=False
            pass
    if report.by_theory:
        log.table("4.5 Theories compared", report.theory_comparison())

    if priced_plan is not None:
        total = float(priced_plan["lot_cost"].sum())
        units = int(priced_plan["units"].sum())
        log.section(
            "5. Curve applied to another lot plan (analogy)",
            f"The fitted equation was applied to a lot plan of "
            f"{list(priced_plan['units'])}, priced from unit "
            f"{priced_from_unit}: {units} units for "
            f"{total:,.0f} in constant FY{series.dollar_year} dollars.\n\n"
            f"The `lot_midpoint` column is the algebraic midpoint -- the unit "
            f"whose cost equals the lot average. It is solved for exactly here "
            f"rather than approximated, because the lot average is itself "
            f"exact, and it is the figure to check the curve against by hand.\n\n"
            f"**This is an analogy, and its validity is a judgement, not a "
            f"result.** The slope carries across only if the two programmes are "
            f"similar enough in product, process, production rate and "
            f"contractor. Nothing in the data can confirm that; the estimate "
            f"inherits the uncertainty of the fitted curve *plus* whatever "
            f"error the analogy itself introduces, and the second is not "
            f"quantified anywhere in this document.",
        ).table("5.1 Priced lot plan", priced_plan)
        log.assume(
            "Analogy",
            f"The {fit.slope:.2%} slope fitted to {series.program} applies to "
            f"the priced lot plan.",
            "Analyst judgement that the two programmes are comparable in "
            "product, process, rate and contractor. Not testable from this "
            "data, and not included in any interval reported here.",
        )
        log.gao(
            "Credible",
            "Where the curve was applied by analogy, the analogy is recorded "
            "as an untested assumption rather than presented as a fitted "
            "result.",
        )

    log.section(
        "6. On R squared",
        "R squared is reported last and should not be used as a validity "
        "check on this data. Lot average cost falls monotonically against "
        "cumulative quantity by construction, so almost any downward-sloping "
        "model returns a high R squared. It measures how tightly the points "
        "hug the fitted line, which a wrong model can do perfectly well: "
        "fitting Crawford-generated data as a Wright curve returns R squared "
        "above 0.99 with a demonstrably wrong forecast. Standard error, in "
        "dollars, and the per-lot percentage errors are the numbers to argue "
        "with.",
    )

    log.gao(
        "Comprehensive",
        f"All {series.n_lots} reported lots included; none excluded from the "
        f"fit.",
    ).gao(
        "Well-documented",
        f"Dollar basis, quantity definition and lot sequencing each recorded "
        f"as declared assumptions with their basis; source data reproduced in "
        f"full in table 1.1.",
    ).gao(
        "Accurate",
        f"Three fitting methods compared rather than one assumed, with the "
        f"retransformation bias of naive OLS measured on this dataset."
        if report.by_method else
        "Single fitting method applied; run with compare=True to measure the "
        "retransformation bias.",
    ).gao(
        "Credible",
        f"Slope reported with an 80% interval ({fit.slope_interval[0]:.1%} to "
        f"{fit.slope_interval[1]:.1%}) on {fit.df} degrees of freedom, and "
        f"per-lot errors shown so the reader can see which lots the curve "
        f"misses.",
    )
    return log


def analyse_lots(
    series: LotSeries,
    *,
    theory: Theory | str = Theory.CRAWFORD,
    method: str = "ols",
    breaks: Iterable[RateBreak] = (),
    compare: bool = True,
) -> LotFitReport:
    """Fit a lot series and assemble the full diagnostic report.

    Args:
        series: The production history.
        theory: Headline theory.
        method: Headline fitting method.
        breaks: Rate breaks to model explicitly.
        compare: Also fit the other two methods and the other theory, so the
            headline number can be shown against its alternatives rather than
            asserted on its own.
    """
    fit = series.fit(theory=theory, method=method, breaks=breaks)

    by_method: dict[str, CurveFit] = {}
    by_theory: dict[str, CurveFit] = {}
    if compare:
        ranges = series.unit_ranges()
        with warnings.catch_warnings():
            # The small-sample and cost-basis warnings already fired once on
            # the headline fit; repeating them per variant is noise.
            warnings.simplefilter("ignore", RuntimeWarning)
            by_method = compare_methods(
                theory=theory, lots=ranges, lot_costs=series.costs,
                breaks=tuple(breaks),
            )
            by_theory = compare_theories(
                method=method, lots=ranges, lot_costs=series.costs,
                breaks=tuple(breaks),
            )

    return LotFitReport(
        series=series, fit=fit, by_method=by_method, by_theory=by_theory
    )
