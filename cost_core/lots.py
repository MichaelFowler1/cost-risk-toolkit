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
a learning curve.

**The fitting is the lot cost model in :mod:`cost_core.lotmodel`** -- the same
engine the desktop tool runs, so the command line and the window give the same
answer for the same lots. Three candidate models are fitted against the lot
midpoint and one is selected:

    LC        ln(unit cost) = ln(T1) + b * ln(lot midpoint)
    Rate      ln(unit cost) = ln(T1) + c * ln(lot quantity)
    LC+Rate   both terms together

The midpoint is the unit whose cost equals the lot average, and under a power
curve it depends on the slope being fitted -- so the fit iterates to a fixed
point rather than solving in one pass. Selection goes to LC+Rate when its rate
coefficient is significant, to Rate when the rate slope is significant and
beats LC by more than the AICc tie threshold, and to LC otherwise.

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

*Too few lots.* A two-parameter model on this data has ``lots - 2`` degrees of
freedom. Two lots interpolate exactly and are refused. Three gives one degree
of freedom and an interval too wide to support a decision. Five is the
practical floor, and LC+Rate needs more still because it spends a third
parameter.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from cost_core.fitting import FitError
from cost_core.lotmodel import (generate_analyst_summary,
                                generate_fit_chart_data, run_lot_cost_model)
from cost_core.lotmodel.enrich import (BuyRisk, EnrichmentError,
                                       compare_fitting_methods,
                                       influence_diagnostics,
                                       projection_intervals, selected_model_name,
                                       simulate_buy)
from cost_core.lotmodel.mathx import lmp_func

logger = logging.getLogger(__name__)

#: Residual correlation across forecast lots, re-exported so callers of this
#: module do not have to reach into the lotmodel package for it.
from cost_core.lotmodel.enrich import DEFAULT_LOT_CORRELATION  # noqa: E402

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

    # ----------------------------------------------------------- the fit
    def to_analogy_frame(self) -> pd.DataFrame:
        """The history in the shape the lot cost engine expects.

        Unit cost rather than lot cost, because the engine fits ``ln(AUC)``.
        Lot order is the build order the series was given in, which is what
        the running unit count depends on.
        """
        return pd.DataFrame({
            "Lot": np.arange(1, self.n_lots + 1),
            "Qty": self.quantities.astype(float),
            "AUC": self.costs / self.quantities,
        })

    def estimate_frame(self, quantities: Iterable[int],
                       complexity: float = 1.0) -> pd.DataFrame:
        """A forecast buy in the shape the engine expects.

        Raises:
            LotInputError: If any quantity is not a positive whole number.
        """
        quantities = [int(q) for q in quantities]
        if not quantities or any(q <= 0 for q in quantities):
            raise LotInputError(
                f"Forecast lot quantities must all be positive whole units; "
                f"got {quantities}.")
        return pd.DataFrame({
            "Lot": np.arange(1, len(quantities) + 1),
            "Qty": [float(q) for q in quantities],
            "Complexity": [float(complexity)] * len(quantities),
        })

    # ----------------------------------------------------------- the fit
    def fit(
        self,
        *,
        forecast: Iterable[int] | None = None,
        complexity: float = 1.0,
        t_gate: float = 2.0,
        aicc_tie: float = 2.0,
        allow_small_sample: bool = True,
        program: str | None = None,
        legacy_rate_omission: bool = False,
    ) -> "LotModelFit":
        """Fit LC, Rate and LC+Rate to this history and select between them.

        Args:
            forecast: Units in each future lot. When omitted the history's own
                quantities are priced instead, which back-casts the lots the
                model was fitted on -- useful on its own, and it keeps the
                engine's estimate table non-empty.
            complexity: Complexity factor applied to every forecast lot.
            t_gate: Significance cutoff on the rate coefficient.
            aicc_tie: How much better on AICc Rate must be to beat LC.
            allow_small_sample: If False, refuse rather than warn when there
                are too few lots to support the model.
            program: Name carried into reports.
            legacy_rate_omission: Reproduce the original tool, which projected
                Rate on the lot midpoint and LC+Rate without its rate factor,
                so its projected costs did not satisfy the equation it
                printed. Only for reproducing a workbook built by it.

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
                f"is the practical floor.")

        if self.cost_basis == "total":
            message = (
                f"Lot costs for {self.program} are declared as TOTAL cost, "
                f"which includes nonrecurring. Nonrecurring cost is "
                f"front-loaded and does not follow a learning curve, so it "
                f"makes the early lots look expensive and the fitted slope "
                f"come out steeper than the production process really is -- "
                f"overstating future savings. Fit on recurring cost only "
                f"where the data allows it.")
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            logger.warning(message)

        # Escalation still sitting in "constant" dollars, before fitting, so
        # the analyst sees it alongside the slope rather than after quoting it.
        self.check_constant_dollars()

        if self.n_lots < COMFORTABLE_LOTS:
            message = (
                f"{self.n_lots} lots gives {self.n_lots - 2} degree(s) of "
                f"freedom on a two-parameter model, below the "
                f"{COMFORTABLE_LOTS} lots normally wanted. The slope will "
                f"carry a very wide interval and a single unusual lot can set "
                f"it. Treat the point estimate as indicative and read the "
                f"per-lot errors before relying on it.")
            if not allow_small_sample:
                raise LotInputError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        quantities = list(forecast) if forecast is not None else list(self.quantities)
        estimate = self.estimate_frame(quantities, complexity)

        prior_fit = self.first_unit - 1
        prior_fcst = (prior_fit + self.total_units) if forecast is not None else prior_fit

        overrides = {
            "TGate": float(t_gate),
            "AiccTie": float(aicc_tie),
            "FitPriorUnits": int(prior_fit),
            "FcstPriorUnits": int(prior_fcst),
            "CostUnitScale": 1.0,
            "TotalScale": 1.0,
            "DefaultCF": float(complexity),
            "LegacyRateOmission": bool(legacy_rate_omission),
        }

        projections, ctx = run_lot_cost_model(
            self.to_analogy_frame(), estimate, overrides)
        summary = generate_analyst_summary(ctx, {
            "RunID": "", "Program": program or self.program,
            "RunLabel": "", "BaseYear": f"FY{self.dollar_year}"})

        fit = LotModelFit(series=self, projections=projections, ctx=ctx,
                          summary=summary, chart=generate_fit_chart_data(ctx),
                          forecast_quantities=quantities,
                          is_forecast=forecast is not None,
                          complexity=float(complexity))
        logger.info(
            "%s: %s selected on %d lots, slope %s, T1 %.4g",
            self.program, fit.selected_model, self.n_lots,
            f"{fit.slope:.2%}" if fit.slope is not None else "n/a", fit.t1)
        return fit


@dataclass
class LotModelFit:
    """The three fitted models, the one selected, and what they price.

    A thin wrapper over the engine's output. Everything here reads the models
    the engine fitted; nothing recomputes them, so the numbers a caller sees
    are the numbers the desktop tool would show for the same lots.
    """

    series: "LotSeries"
    projections: pd.DataFrame
    ctx: dict
    summary: pd.DataFrame
    chart: pd.DataFrame
    forecast_quantities: list[int]
    is_forecast: bool
    complexity: float = 1.0

    # ------------------------------------------------------------ selection
    @property
    def selected_model(self) -> str:
        """Which of LC, Rate or LC+Rate the selection rule chose."""
        return selected_model_name(self.summary)

    @property
    def selection_note(self) -> str:
        """Why this model was chosen, in the engine's own words.

        The engine writes the reason into the selected model's own column
        rather than the shared Value column, so read it from there.
        """
        row = self.summary[self.summary["Item"] == "Selection basis"]
        if row.empty:
            return ""
        note = str(row.iloc[0].get(self.selected_model, "")).strip()
        if not note:
            note = str(row.iloc[0].get("Value", "")).strip()
        return note

    def _summary_value(self, item: str, column: str = "Value") -> str:
        row = self.summary[self.summary["Item"] == item]
        return str(row.iloc[0][column]) if not row.empty else ""

    # ------------------------------------------------------- coefficients
    @property
    def t1(self) -> float:
        """Theoretical first-unit cost under the selected model."""
        return {"LC": self.ctx.get("t1_lc"), "Rate": self.ctx.get("t1_rt"),
                "LC+Rate": self.ctx.get("t1_br")}[self.selected_model]

    @property
    def b(self) -> float | None:
        """Learning exponent, or None when the selected model has no LC term."""
        return {"LC": self.ctx.get("b_lc"), "Rate": None,
                "LC+Rate": self.ctx.get("b_br")}[self.selected_model]

    @property
    def c(self) -> float | None:
        """Rate exponent, or None when the selected model has no rate term."""
        return {"LC": None, "Rate": self.ctx.get("b_rt"),
                "LC+Rate": self.ctx.get("c_br")}[self.selected_model]

    @property
    def slope(self) -> float | None:
        """Learning slope, ``2 ** b``. None when there is no learning term."""
        return None if self.b is None else float(2.0 ** self.b)

    @property
    def rate_slope(self) -> float | None:
        """Rate slope, ``2 ** c``: the cost effect of doubling the lot size."""
        return None if self.c is None else float(2.0 ** self.c)

    @property
    def n_obs(self) -> int:
        return int(self.ctx.get("n_keep", 0))

    @property
    def n_params(self) -> int:
        return {"LC": 2, "Rate": 2, "LC+Rate": 3}[self.selected_model]

    @property
    def df(self) -> int:
        """Residual degrees of freedom on the selected model."""
        return self.n_obs - self.n_params

    @property
    def sigma(self) -> float:
        """Standard error of the estimate, on the log scale."""
        model = {"LC": "mdl_lc", "Rate": "mdl_rt",
                 "LC+Rate": "mdl_lcr"}[self.selected_model]
        return float(self.ctx[model]["SEy"])

    @property
    def cv(self) -> float:
        """Coefficient of variation implied by the log-space scatter."""
        return float(np.sqrt(np.exp(self.sigma ** 2) - 1.0))

    @property
    def r_squared(self) -> float:
        """Reported for completeness, and last. See the module notes."""
        model = {"LC": "mdl_lc", "Rate": "mdl_rt",
                 "LC+Rate": "mdl_lcr"}[self.selected_model]
        return float(self.ctx[model]["R2"])

    def equation(self, *, precision: int = 6) -> str:
        """The selected model written out, ready to quote or re-use."""
        terms = [f"{self.t1:,.2f}"]
        if self.b is not None:
            terms.append(f"midpoint^({self.b:.{precision}f})")
        if self.c is not None:
            terms.append(f"qty^({self.c:.{precision}f})")
        return "Unit Cost = " + " * ".join(terms)

    def equation_detail(self) -> pd.DataFrame:
        """Every coefficient a reader needs to rebuild the model by hand."""
        rows = [
            ("selected_model", self.selected_model),
            ("selection_note", self.selection_note),
            ("equation", self.equation()),
            ("T1_first_unit_cost", self.t1),
            ("b_learning_exponent", self.b if self.b is not None else ""),
            ("learning_slope", self.slope if self.slope is not None else ""),
            ("c_rate_exponent", self.c if self.c is not None else ""),
            ("rate_slope", self.rate_slope if self.rate_slope is not None else ""),
            ("lots_fitted", self.n_obs),
            ("parameters", self.n_params),
            ("degrees_of_freedom", self.df),
            ("SEE_log", self.sigma),
            ("cv", self.cv),
            ("r_squared_read_last", self.r_squared),
        ]
        return pd.DataFrame(rows, columns=["term", "value"])

    def model_comparison(self) -> pd.DataFrame:
        """LC, Rate and LC+Rate side by side, as the engine reported them.

        The models that were not selected are shown too, because the first
        question a reviewer asks is what the alternatives said.
        """
        wanted = ["Fitted", "SELECTED", "T1 ($K)", "Learning exponent (b)",
                  "Learning curve slope", "Rate exponent (c)", "Rate slope",
                  "R2 (log)", "Adj R2", "SEE (log)", "CV", "MAPE",
                  "Mean bias", "AICc", "dAICc", "t (rate coeff)"]
        rows = [r for _, r in self.summary.iterrows() if r["Item"] in wanted]
        return pd.DataFrame(rows)[["Item", "LC", "Rate", "LC+Rate"]].reset_index(
            drop=True)

    def lot_midpoints(self) -> np.ndarray:
        """The midpoint each analogy lot was priced at, under the selection."""
        column = {"LC": "LC Lot Midpoint", "Rate": "LC Lot Midpoint",
                  "LC+Rate": "LC+Rate Lot Midpoint"}[self.selected_model]
        if column in self.chart.columns:
            return self.chart[column].to_numpy(dtype=float)
        b = self.b if self.b is not None else 0.0
        spans = self.series.unit_ranges()
        return np.array([lmp_func(s, e, q, b) for (s, e), q
                         in zip(spans, self.series.quantities)], dtype=float)


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
    """A fitted lot cost model and everything a reviewer will ask about it.

    The fit itself comes from :mod:`cost_core.lotmodel`; this adds the layer
    that says how much confidence the numbers carry. None of it feeds back into
    the estimate.
    """

    series: LotSeries
    fit: LotModelFit
    level: float = 0.80
    _methods: Any = field(default=None, repr=False)
    _influence: Any = field(default=None, repr=False)

    # -------------------------------------------------------------- the fit
    @property
    def selected_model(self) -> str:
        return self.fit.selected_model

    def equation(self) -> str:
        return self.fit.equation()

    @property
    def per_lot(self) -> pd.DataFrame:
        """Actual against fitted for each analogy lot, with the midpoint.

        The most useful diagnostic on a short series: it names which lot the
        model misses, and that lot usually maps to something a programme
        manager remembers.
        """
        _, _, fitted = self._fitted()
        actual = self.series.costs / self.series.quantities
        frame = self.series.to_frame()
        frame["lot_midpoint"] = self.fit.lot_midpoints()
        frame["fitted_unit_cost"] = fitted
        frame["fitted_lot_cost"] = fitted * frame["units"]
        frame["residual"] = frame["cost"] - frame["fitted_lot_cost"]
        frame["percent_error"] = (actual - fitted) / fitted * 100.0
        return frame

    def model_comparison(self) -> pd.DataFrame:
        return self.fit.model_comparison()

    def _fitted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Design matrix, log response and fitted unit costs, full precision.

        Rebuilt from the model the engine fitted rather than read off the
        chart sheet, which rounds to cents for display. Rounding is invisible
        in a report and quite visible to a residual diagnostic.
        """
        from cost_core.lotmodel.enrich import _design

        design, y_log, _ = _design(self.fit.ctx, self.selected_model)
        beta = np.linalg.lstsq(design, y_log, rcond=None)[0]
        return design, y_log, np.exp(design @ beta)

    # ------------------------------------------------------- added statistics
    def methods(self):
        """OLS against MUPE and ZMPE on the selected model's own regressors."""
        if self._methods is None:
            self._methods = compare_fitting_methods(self.fit.ctx,
                                                    self.selected_model)
        return self._methods

    def influence(self) -> pd.DataFrame:
        """Leverage and Cook's distance for each analogy lot."""
        if self._influence is None:
            self._influence = influence_diagnostics(
                self.fit.ctx, self.selected_model,
                labels=list(self.series.labels))
        return self._influence

    def intervals(self, level: float | None = None) -> pd.DataFrame:
        """Prediction intervals on every priced lot."""
        return projection_intervals(self.fit.ctx, self.fit.projections,
                                    self.selected_model,
                                    level=self.level if level is None else level)

    def forecast(self, quantities: Iterable[int] | None = None, *,
                 level: float | None = None,
                 complexity: float | None = None) -> pd.DataFrame:
        """Price future lots with prediction intervals.

        With no argument this prices whatever the fit was given. Passing
        quantities refits nothing -- it re-prices under the same selected
        model, continuing from the last unit already built.
        """
        if quantities is None:
            return self.intervals(level=level)
        refit = self.series.fit(
            forecast=quantities,
            complexity=self.fit.complexity if complexity is None else complexity,
            program=self.series.program)
        return projection_intervals(
            refit.ctx, refit.projections, refit.selected_model,
            level=self.level if level is None else level)

    def price_lot_plan(self, quantities: Iterable[int], *,
                       first_unit: int = 1,
                       complexity: float | None = None) -> pd.DataFrame:
        """Price an arbitrary buy profile with this model, from unit 1.

        The model used as an estimating relationship rather than as a forecast
        of its own programme. The intended use is analogy: pricing a programme
        with no cost history using the slope from one that has it, which is a
        judgement about whether the two are comparable and is recorded in the
        assumptions log as an untested assumption.
        """
        quantities = [int(q) for q in quantities]
        if not quantities or any(q <= 0 for q in quantities):
            raise LotInputError(
                f"Lot quantities must all be positive whole units; got "
                f"{quantities}.")
        if first_unit < 1:
            raise LotInputError(
                f"first_unit must be 1 or greater; got {first_unit}.")

        cf = self.fit.complexity if complexity is None else complexity
        estimate = self.series.estimate_frame(quantities, cf)
        overrides = {
            "TGate": 2.0, "AiccTie": 2.0,
            "FitPriorUnits": self.series.first_unit - 1,
            "FcstPriorUnits": int(first_unit) - 1,
            "CostUnitScale": 1.0, "TotalScale": 1.0, "DefaultCF": float(cf),
            # Whatever the fit was run under, so the priced plan and the
            # equation printed beside it cannot come from different formulas.
            "LegacyRateOmission": bool(
                self.fit.ctx["cfg"]["LegacyRateOmission"]),
        }
        projections, _ = run_lot_cost_model(
            self.series.to_analogy_frame(), estimate, overrides)

        model = self.selected_model
        priced = projections[[
            "Lot", "Lot Quantity", "First Unit in Lot", "Last Unit in Lot",
            f"{model} Lot Midpoint (unit no.)", f"{model} Unit Cost ($K)",
            f"{model} Lot Cost After Complexity ($)",
        ]].copy()
        priced.columns = ["lot", "units", "first_unit", "last_unit",
                          "lot_midpoint", "unit_cost", "lot_cost"]
        priced.insert(0, "priced_by_analogy_from", self.series.program)
        priced.insert(1, "source_lots_fitted", self.series.n_lots)
        priced["cumulative_units"] = priced["units"].cumsum()
        priced["cumulative_cost"] = priced["lot_cost"].cumsum()
        return priced

    def simulate(self, quantities: Iterable[int] | None = None, *,
                 n_iter: int = 20_000, seed: int | None = 0,
                 lot_correlation: float = DEFAULT_LOT_CORRELATION) -> BuyRisk:
        """Monte Carlo the total of the priced lots.

        Parameter uncertainty is drawn once per iteration from the fitted
        covariance and applied to every lot; residual scatter is drawn per lot
        and correlated across them. Both are t-distributed on the fit's degrees
        of freedom, because sigma is estimated rather than known.
        """
        if quantities is None:
            ctx, projections = self.fit.ctx, self.fit.projections
        else:
            refit = self.series.fit(forecast=quantities,
                                    complexity=self.fit.complexity,
                                    program=self.series.program)
            ctx, projections = refit.ctx, refit.projections
        return simulate_buy(ctx, projections, self.selected_model,
                            n_iter=n_iter, seed=seed,
                            lot_correlation=lot_correlation)

    # ---------------------------------------------------------- diagnostics
    def curvature(self) -> tuple[float, float]:
        """Quadratic term in the log-log residuals, and its t statistic.

        A correctly specified model leaves no systematic bend in its residuals,
        so a significant quadratic term says these lots are not one clean
        curve -- a rate break, a design change, a production gap, or a model
        that does not fit.

        A note on what it does *not* do here. Fitted against the lot midpoint,
        this test is a poor detector of escalation: the fitted slope moves with
        the escalation, the midpoint moves with the slope, and the trend is
        largely absorbed rather than left in the residuals. Measured on a
        six-lot series, the quadratic t barely moves between 0% and 20% a year.
        Escalation detection here rests on the level check in
        :meth:`LotSeries.check_constant_dollars`, which needs roughly 10% a
        year before it bites. Below that, the only reliable answer is a fiscal
        year attached to each lot.

        Returns:
            ``(coefficient, t_statistic)``. Both NaN when there are too few
            lots to estimate a quadratic, or when the fit is so close that the
            residuals carry nothing but rounding error.
        """
        x = np.log(self.fit.lot_midpoints())
        _, observed, fitted_levels = self._fitted()
        residual = observed - np.log(fitted_levels)

        design = np.column_stack([np.ones(x.size), x, x ** 2])
        dof = x.size - design.shape[1]
        if dof < 1:
            return float("nan"), float("nan")

        # A model that passes through every point leaves residuals at the
        # floating-point floor. Fitting a quadratic to those returns a large t
        # from pure rounding, which would flag the cleanest possible data as
        # bent. Nothing is there to test.
        scale = float(np.max(np.abs(observed))) or 1.0
        if float(np.max(np.abs(residual))) <= 1e-9 * scale:
            return float("nan"), float("nan")

        beta, *_ = np.linalg.lstsq(design, residual, rcond=None)
        sigma2 = float(np.sum((residual - design @ beta) ** 2) / dof)
        if sigma2 <= 0:
            return float(beta[2]), float("nan")
        se = float(np.sqrt(sigma2 * np.linalg.inv(design.T @ design)[2, 2]))
        return float(beta[2]), (float(beta[2] / se) if se > 0 else float("nan"))

    def check_curve_shape(self, *, threshold: float = 2.5,
                          warn: bool = True) -> list[str]:
        """Flag a systematic bend in the residuals."""
        findings: list[str] = []
        coefficient, t_stat = self.curvature()

        if not np.isfinite(t_stat):
            if self.series.n_lots < 4:
                findings.append(
                    f"Only {self.series.n_lots} lots, too few to test the "
                    f"shape of the residuals; a systematic bend from "
                    f"escalation or a rate break could not be detected here.")
        elif abs(t_stat) > threshold:
            direction = "upward (convex)" if coefficient > 0 else "downward (concave)"
            findings.append(
                f"The residuals bend {direction} (quadratic t = {t_stat:.1f}), "
                f"so these lots are not one clean curve. Look for a rate "
                f"break, a design change or a production gap partway through "
                f"the series. Note this test is not a reliable escalation "
                f"detector under a midpoint fit -- the slope and the midpoint "
                f"move with the escalation and absorb it.")

        if warn:
            for finding in findings:
                warnings.warn(finding, RuntimeWarning, stacklevel=3)
                logger.warning(finding)
        return findings

    def diagnostics(self) -> list[str]:
        """Every data-quality finding, in one list."""
        notes = [*self.series.check_constant_dollars(warn=False),
                 *self.check_curve_shape(warn=False)]
        influential = self.influence().loc[
            self.influence()["Influential"], "Lot"].tolist()
        if influential:
            notes.append(
                f"Lot(s) {', '.join(map(str, influential))} exceed the "
                f"conventional Cook's distance flag and are setting this fit. "
                f"Confirm each belongs in the sample before relying on the "
                f"slope.")
        return notes

    # -------------------------------------------------------------- output
    def summary(self) -> pd.DataFrame:
        """Headline numbers, in the order they should be read."""
        rows = [
            ("selected_model", self.selected_model),
            ("lots", self.series.n_lots),
            ("units", self.series.total_units),
            ("parameters", self.fit.n_params),
            ("degrees_of_freedom", self.fit.df),
            ("T1_first_unit_cost", self.fit.t1),
            ("learning_slope", self.fit.slope if self.fit.slope is not None else ""),
            ("rate_slope", self.fit.rate_slope
             if self.fit.rate_slope is not None else ""),
            ("SEE_log", self.fit.sigma),
            ("cv", self.fit.cv),
            ("worst_lot_percent_error",
             float(np.max(np.abs(self.per_lot["percent_error"].to_numpy())))),
            ("ols_understates_mean_pct", self.methods().percent_understated),
            ("r_squared_read_last", self.fit.r_squared),
        ]
        return pd.DataFrame(rows, columns=["statistic", "value"])

    def narrative(self) -> str:
        """A few sentences a reviewer can read without the tables."""
        per_lot = self.per_lot
        worst = int(np.argmax(np.abs(per_lot["percent_error"].to_numpy())))
        slope = (f"{self.fit.slope:.2%} learning slope"
                 if self.fit.slope is not None else "no learning term")
        rate = (f", {self.fit.rate_slope:.2%} rate slope"
                if self.fit.rate_slope is not None else "")
        parts = [
            f"{self.series.program}: {self.selected_model} selected on "
            f"{self.series.n_lots} lots covering {self.series.total_units} "
            f"units, in constant FY{self.series.dollar_year} dollars. {slope}"
            f"{rate}, first-unit cost {self.fit.t1:,.0f}, SEE {self.fit.sigma:.4f} "
            f"on the log scale, CV {self.fit.cv:.1%}, {self.fit.df} degree"
            f"{'s' if self.fit.df != 1 else ''} of freedom.",
            f"{self.fit.selection_note}".strip(),
            f"The model misses {per_lot['lot'].iloc[worst]} by "
            f"{per_lot['percent_error'].iloc[worst]:+.1f}%, the largest "
            f"departure in the series.",
        ]
        parts.extend(self.diagnostics())
        if self.series.cost_basis == "total":
            parts.append(
                "Costs were declared as totals including nonrecurring, so this "
                "slope is steeper than the production process alone.")
        if self.series.n_lots < COMFORTABLE_LOTS:
            parts.append(
                f"With only {self.series.n_lots} lots this is an indicative "
                f"fit; the interval on the slope is too wide to separate it "
                f"from neighbouring curves.")
        return " ".join(p for p in parts if p)


def analyse_lots(series: LotSeries, *, forecast: Iterable[int] | None = None,
                 complexity: float = 1.0, level: float = 0.80,
                 t_gate: float = 2.0, aicc_tie: float = 2.0,
                 legacy_rate_omission: bool = False) -> LotFitReport:
    """Fit a lot series and assemble the full diagnostic report.

    Args:
        series: The production history.
        forecast: Units in each future lot. Omitted, the history's own
            quantities are re-priced.
        complexity: Complexity factor on the priced lots.
        level: Coverage for the prediction intervals.
        t_gate: Significance cutoff on the rate coefficient.
        aicc_tie: How much better on AICc Rate must be to beat LC.
        legacy_rate_omission: Reproduce the original tool's projections, which
            did not satisfy the equation it printed. See ``LotSeries.fit``.
    """
    fit = series.fit(forecast=forecast, complexity=complexity,
                     legacy_rate_omission=legacy_rate_omission,
                     t_gate=t_gate, aicc_tie=aicc_tie)
    return LotFitReport(series=series, fit=fit, level=level)


def build_assumption_log(report: LotFitReport,
                         source: str | Path | None = None,
                         priced_plan: pd.DataFrame | None = None,
                         priced_from_unit: int = 1):
    """Assemble the written assumptions log for a lot cost model run.

    Records the dollar basis as an explicit decision, the quantity definition
    the analyst declared, the model selection and why, every diagnostic, and
    the methodological choices -- mapped to the four characteristics of a
    reliable estimate in the GAO Cost Estimating and Assessment Guide.
    """
    from cost_core.reporting.assumptions import AssumptionLog

    series, fit = report.series, report.fit
    log = AssumptionLog(
        title=f"Lot cost model assumptions and provenance - {series.program}")

    log.section(
        "1. Source data",
        f"- Source: {source if source else 'supplied in memory'}\n"
        f"- {series.n_lots} analogy lots covering {series.total_units} units, "
        f"first unit numbered {series.first_unit}\n"
        f"- Cost basis declared: **{series.cost_basis}**\n"
        f"- Quantity definition declared: **{series.quantity_definition}**\n"
        f"- Dollars: constant **FY{series.dollar_year}**",
    ).table("1.1 Lots as supplied and derived", series.to_frame())

    # --- the escalation decision, recorded as a decision
    findings = report.diagnostics()
    _, t_stat = report.curvature()
    body = series.dollar_basis_note()
    body += (
        "\n\nTwo checks were run for escalation left in the data. The first "
        "asks whether cumulative average cost ever rises, which it should not "
        "on a learning curve. The second tests the log-log residuals for a "
        "systematic bend, since escalation compounds with time while learning "
        "compounds with log quantity and the mismatch shows up as convexity.")
    if findings:
        body += "\n\n**Findings:**\n" + "\n".join(f"- {f}" for f in findings)
    else:
        body += (
            f"\n\nNeither check fired: cumulative average cost falls "
            f"monotonically, and the residual curvature is not significant "
            f"(quadratic t = {t_stat:.1f}). This is consistent with the "
            f"constant-dollar declaration.")
    body += (
        "\n\n**Limits of these checks.** The level check only fires once "
        "escalation is severe enough to overwhelm learning, which on a typical "
        "profile takes about 10% a year. The curvature test does not fill that "
        "gap under a midpoint fit: the fitted slope moves with the escalation "
        "and the midpoint moves with the slope, so the trend is absorbed "
        "rather than left in the residuals, and the quadratic term barely "
        "responds. It is a test that these lots are one clean curve, not a "
        "test for escalation. Below roughly 10% a year, moderate escalation "
        "and genuinely slower learning cannot be told apart without a fiscal "
        "year attached to each lot.")
    log.section("2. Dollar basis and escalation", body)

    log.assume(
        "Dollar basis",
        f"Costs are constant FY{series.dollar_year} dollars; no inflation "
        f"index applied by this tool.",
        "Declared by the analyst on ingest. Normalisation, if any was needed, "
        "happened upstream and is not verifiable from this input.")
    log.assume(
        "Quantity definition",
        f"A 'unit' means: {series.quantity_definition}.",
        "Declared by the analyst. Delivered, completed and accepted counts "
        "differ, and the difference shifts every point on the curve.")
    log.assume(
        "Lot sequencing",
        f"Lots are contiguous and in build order, starting at unit "
        f"{series.first_unit}.",
        "Implied by supplying lots as an ordered list. A prior buy the model "
        "has already learned through would need a higher first unit.")
    if series.cost_basis == "total":
        log.assume(
            "Nonrecurring cost included",
            "Lot costs include nonrecurring cost.",
            "Declared by the analyst. Nonrecurring is front-loaded and does "
            "not follow the curve, so the fitted slope is steeper than the "
            "production process alone.")

    # --- the model, and why this one
    slope = (f"{fit.slope:.2%}" if fit.slope is not None else "n/a")
    rate = (f"{fit.rate_slope:.2%}" if fit.rate_slope is not None else "n/a")
    log.section(
        "3. Model selected",
        f"**{fit.selected_model}** — {fit.selection_note}\n\n"
        f"**{fit.equation()}**\n\n"
        f"Unit cost in constant FY{series.dollar_year} dollars, with the "
        f"midpoint the unit whose cost equals the lot average. Because that "
        f"midpoint depends on the slope being fitted, the fit iterates to a "
        f"fixed point rather than solving in one pass.\n\n"
        f"- Learning slope {slope}, rate slope {rate}\n"
        f"- T1 {fit.t1:,.2f}, SEE {fit.sigma:.4f} on the log scale, "
        f"CV {fit.cv:.1%}\n"
        f"- {fit.n_obs} lots, {fit.n_params} parameters, {fit.df} degrees of "
        f"freedom\n\n"
        f"Three models were fitted and all three priced every lot, so the "
        f"alternatives are on the record rather than discarded.",
    ).table("3.1 Models compared", fit.model_comparison()
            ).table("3.2 Coefficients", fit.equation_detail()
                    ).table("3.3 Per-lot fit quality", report.per_lot[
                        ["lot", "units", "lot_midpoint", "lot_average_cost",
                         "fitted_unit_cost", "percent_error"]])

    # --- the added statistics
    methods = report.methods()
    log.section(
        "4. Retransformation bias",
        f"The engine fits ln(unit cost) by ordinary least squares and then "
        f"exponentiates back to dollars. That step is biased: with log-space "
        f"errors of variance s², the retransformed value estimates the "
        f"*median* and understates the *mean* by exp(s²/2).\n\n"
        f"On this data that factor is **{methods.theoretical_factor:.5f}**, an "
        f"understatement of **{methods.percent_understated:.3f}%** before any "
        f"risk analysis begins. Duan's nonparametric smearing estimate agrees "
        f"at {methods.smearing_factor:.5f}, so the lognormal assumption is not "
        f"doing the work.\n\n"
        f"MUPE and ZMPE refit the same regressors under a proportional-error "
        f"loss and drive the mean percentage error to zero. MUPE places the "
        f"curve {(methods.mupe_over_ols - 1) * 100:+.3f}% relative to OLS and "
        f"ZMPE {(methods.zmpe_over_ols - 1) * 100:+.3f}%.",
    ).table("4.1 Fitting methods compared", methods.frame)

    influence = report.influence()
    log.section(
        "5. Influence",
        f"With {series.n_lots} analogy lots a single lot can set the slope "
        f"while every summary statistic still looks healthy. Leverage says "
        f"which lot is unusual in the predictors; Cook's distance says which "
        f"is actually moving the fit. The conventional flags are 2p/n and 4/n, "
        f"and they are flags rather than verdicts -- the largest or smallest "
        f"lot in a sample has high leverage by construction.",
    ).table("5.1 Leverage and influence", influence)

    intervals = report.intervals()
    log.section(
        "6. Prediction intervals",
        f"Each priced lot carries a {report.level:.0%} **prediction** "
        f"interval: the range a single new lot is expected to fall in, not the "
        f"range the fitted line lies in. The two differ by exactly the "
        f"residual variance, and that term does not shrink with more analogy "
        f"lots. The multiplier is a t on {fit.df} degrees of freedom, because "
        f"sigma is estimated rather than known.",
    ).table("6.1 Priced lots with intervals", intervals)

    if priced_plan is not None:
        total = float(priced_plan["lot_cost"].sum())
        units = int(priced_plan["units"].sum())
        log.section(
            "7. Model applied to another lot plan (analogy)",
            f"The selected model was applied to a lot plan of "
            f"{list(priced_plan['units'])}, priced from unit "
            f"{priced_from_unit}: {units} units for {total:,.0f} in constant "
            f"FY{series.dollar_year} dollars.\n\n"
            f"**This is an analogy, and its validity is a judgement, not a "
            f"result.** The slope carries across only if the two programmes "
            f"are similar enough in product, process, production rate and "
            f"contractor. Nothing in the data can confirm that; the estimate "
            f"inherits the uncertainty of the fitted model *plus* whatever "
            f"error the analogy itself introduces, and the second is not "
            f"quantified anywhere in this document.",
        ).table("7.1 Priced lot plan", priced_plan)
        log.assume(
            "Analogy",
            f"The {fit.selected_model} model fitted to {series.program} "
            f"applies to the priced lot plan.",
            "Analyst judgement that the two programmes are comparable in "
            "product, process, rate and contractor. Not testable from this "
            "data, and not included in any interval reported here.")
        log.gao(
            "Credible",
            "Where the model was applied by analogy, the analogy is recorded "
            "as an untested assumption rather than presented as a fitted "
            "result.")

    log.section(
        "8. On R squared",
        "R squared is reported last and should not be used as a validity "
        "check on this data. Unit cost falls monotonically against the lot "
        "midpoint by construction, so almost any downward-sloping model "
        "returns a high R squared. It measures how tightly the points hug the "
        "fitted line, which a wrong model can do perfectly well. The standard "
        "error of the estimate, the per-lot percentage errors and the "
        "influence table are the numbers to argue with.")

    log.gao(
        "Comprehensive",
        f"All {series.n_lots} reported lots included in the fit; three "
        f"candidate models fitted and all three priced every lot."
    ).gao(
        "Well-documented",
        "Dollar basis, quantity definition and lot sequencing each recorded as "
        "declared assumptions with their basis; source data reproduced in full "
        "in table 1.1; the selection rule and its outcome stated in section 3."
    ).gao(
        "Accurate",
        f"Retransformation bias of the log-space fit measured at "
        f"{methods.percent_understated:.3f}% on this dataset and reported "
        f"against MUPE and ZMPE refits, rather than left in the estimate."
    ).gao(
        "Credible",
        f"Prediction intervals on every priced lot at {report.level:.0%}, "
        f"influence diagnostics naming any lot that sets the fit, and a "
        f"selection note stating why this model was chosen over the other two.")
    return log
