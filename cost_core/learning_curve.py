"""
learning_curve.py - Learning and cost-improvement curves.

Two theories of what "the curve" describes, both in use and not
interchangeable:

**Wright (cumulative average theory).** The *cumulative average* cost of the
first x units follows ``T1 * x^b``. Total cost through x is therefore
``T1 * x^(b+1)``, and the cost of an individual unit is the difference between
successive totals.

**Crawford (unit theory).** The cost of *unit* x itself follows ``T1 * x^b``,
and cumulative cost is the sum over units.

The same data fitted under the two theories gives different slopes and, more
to the point, different forecasts -- Wright's cumulative-average form falls
faster in total than Crawford's unit form for the same nominal slope. Which one
applies is a property of the production process, not a modelling preference, so
:func:`fit_curve` makes the caller choose and :func:`compare_theories` reports
both side by side rather than letting one be assumed.

Three fitting methods are available through :mod:`cost_core.fitting`: log-log
OLS, MUPE and ZMPE. The OLS fit is retained because it is what everyone does,
and :func:`retransformation_report` measures how much it understates the mean
by exponentiating a log-space regression. Forecasts come with prediction
intervals; standard error and coefficient of variation are reported in
preference to R^2, which on log-log cost data is close to 1 for almost any
model and so discriminates between none of them.

The original :class:`LearningCurveModel` API is unchanged and still exported.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from cost_core import fitting
from cost_core.fitting import (METHODS, FitError, FitResult, ModelSpec,
                               predict_with_interval)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LearningCurveModel:
    """
    Representation of a Wright's Learning Curve model.
    
    Attributes:
        slope: The learning slope (e.g., 0.85 for an 85% curve).
        reference_quantity: The quantity at which the reference cost was observed.
        reference_cost: The cost of the unit at the reference quantity.
    """
    slope: float
    reference_quantity: float
    reference_cost: float

    @property
    def learning_exponent(self) -> float:
        """Calculates the 'b' parameter (slope constant)."""
        return np.log2(self.slope)

    def predict_unit_cost(self, quantity: float) -> float:
        """Predicts the cost of a specific unit index using the power law."""
        # Cost(x) = T1 * x^b
        # First, solve for T1 (Theoretical First Unit): T1 = RefCost / RefQty^b
        t1 = self.reference_cost / (self.reference_quantity ** self.learning_exponent)
        return t1 * (quantity ** self.learning_exponent)


def fit_learning_curve(
    df: pd.DataFrame, 
    quantity_col: str = "unit_quantity", 
    cost_col: str = "unit_cost"
) -> LearningCurveModel:
    """
    Fits a learning curve model to historical data using log-log linear regression.
    
    Args:
        df: DataFrame containing historical production data.
        quantity_col: Column name for unit quantities.
        cost_col: Column name for unit costs.
        
    Returns:
        LearningCurveModel: The fitted model parameters.
    """
    # 1. Validation
    if len(df) < 2:
        raise ValueError("At least 2 data points are required to fit a learning curve.")
    
    if (df[quantity_col] <= 0).any() or (df[cost_col] <= 0).any():
        raise ValueError("Quantities and costs must be positive, non-zero values.")

    logger.info(f"Fitting learning curve on {len(df)} data points.")

    # 2. Linear Regression in Log-Log space
    # log(y) = log(a) + b * log(x)
    log_x = np.log2(df[quantity_col].values)
    log_y = np.log2(df[cost_col].values)
    
    slope_b, intercept_log_a, _, _, _ = stats.linregress(log_x, log_y)
    
    # 3. Derive model parameters
    # slope = 2^b
    learning_slope = 2**slope_b
    
    # Use the mean of the data as the reference point for the model
    ref_qty = df[quantity_col].mean()
    ref_cost = (2**intercept_log_a) * (ref_qty ** slope_b)

    model = LearningCurveModel(
        slope=float(learning_slope),
        reference_quantity=float(ref_qty),
        reference_cost=float(ref_cost)
    )
    
    logger.info(f"Model fit complete: Slope={model.slope:.2%}")
    return model


def forecast_costs(
    model: LearningCurveModel, 
    quantities: list[float] | np.ndarray
) -> pd.DataFrame:
    """
    Generates a cost forecast for a range of quantities.
    
    Returns:
        pd.DataFrame: Contains 'quantity', 'unit_cost', and 'total_cost'.
    """
    q_array = np.sort(np.array(quantities))
    
    unit_costs = np.array([model.predict_unit_cost(q) for q in q_array])
    total_costs = q_array * unit_costs
    
    return pd.DataFrame({
        "quantity": q_array,
        "unit_cost": unit_costs,
        "total_cost": total_costs
    })


# ==========================================================================
# Theories, rate breaks, and the curve model
# ==========================================================================
class Theory(str, Enum):
    """Which quantity the power law describes."""

    WRIGHT = "wright"      # cumulative average cost follows T1 * x^b
    CRAWFORD = "crawford"  # the cost of unit x follows T1 * x^b

    @classmethod
    def parse(cls, value: "Theory | str") -> "Theory":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise FitError(
                f"Unknown learning-curve theory {value!r}. "
                f"Allowed: {[t.value for t in cls]}."
            ) from None


@dataclass(frozen=True)
class RateBreak:
    """A discontinuity in the curve at a known unit.

    Production is not a smooth process. A second source comes on line, a
    design change lands, the line goes cold for two years and restarts with a
    different crew. Each of those puts a step in the curve, and a fit that
    ignores it does not average the step away -- it tilts the *slope* to
    accommodate it, which is much worse, because the tilt then propagates into
    every forecast lot.

    Attributes:
        at_unit: First unit affected. Everything from here on carries the step.
        step_factor: Multiplicative cost shock. ``None`` means estimate it from
            the data, adding one parameter to the fit.
        learning_loss: Fraction of accumulated learning lost at the break, as
            in a production gap where the workforce has turned over. A value
            of 0.3 means the curve backs up by 30% of the units built so far.
            This is an analyst assumption, not a fitted quantity: with a
            handful of lots there is rarely enough information to separate a
            level shift from a loss of learning, and pretending otherwise
            produces two parameters that trade off against each other.
        label: Free text carried into the assumptions log.
    """

    at_unit: int
    step_factor: float | None = None
    learning_loss: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.at_unit < 2:
            raise FitError(
                f"A rate break must fall at unit 2 or later; got {self.at_unit}. "
                f"A break at unit 1 is not a break, it is the first unit cost."
            )
        if self.step_factor is not None and self.step_factor <= 0:
            raise FitError(
                f"step_factor must be positive; got {self.step_factor}."
            )
        if not 0.0 <= self.learning_loss < 1.0:
            raise FitError(
                f"learning_loss is a fraction of accumulated learning and must "
                f"be in [0, 1); got {self.learning_loss}."
            )

    @property
    def is_estimated(self) -> bool:
        return self.step_factor is None


@dataclass(frozen=True)
class CurveModel:
    """A fitted learning curve under one theory.

    Attributes:
        theory: Wright or Crawford.
        t1: Theoretical first-unit cost.
        b: Learning exponent. The slope is ``2 ** b``.
        breaks: Any rate breaks or production gaps, in unit order.
    """

    theory: Theory
    t1: float
    b: float
    breaks: tuple[RateBreak, ...] = ()

    @property
    def slope(self) -> float:
        """The learning slope, e.g. 0.85 for an 85% curve."""
        return float(2.0**self.b)

    @property
    def is_smooth(self) -> bool:
        """True when no break or gap disturbs the closed forms."""
        return not self.breaks

    # ------------------------------------------------------- break handling
    def _effective_units(self, units: np.ndarray) -> np.ndarray:
        """Units shifted back by any accumulated loss of learning."""
        eff = np.asarray(units, dtype=float).copy()
        for brk in self.breaks:
            if brk.learning_loss <= 0.0:
                continue
            lost = brk.learning_loss * (brk.at_unit - 1)
            eff = np.where(units >= brk.at_unit, eff - lost, eff)
        # A curve cannot back up past its own first unit.
        return np.maximum(eff, 1.0)

    def _step_factor(self, units: np.ndarray) -> np.ndarray:
        """Cumulative product of every step in force at each unit."""
        factor = np.ones_like(np.asarray(units, dtype=float))
        for brk in self.breaks:
            k = 1.0 if brk.step_factor is None else brk.step_factor
            factor = np.where(units >= brk.at_unit, factor * k, factor)
        return factor

    # ------------------------------------------------------------ the curve
    def unit_cost(self, units) -> np.ndarray:
        """Cost of each individual unit.

        Raises:
            FitError: If any unit index is below 1.
        """
        units = np.atleast_1d(np.asarray(units, dtype=float))
        if np.any(units < 1.0):
            raise FitError(
                f"Unit indices start at 1; got a minimum of {units.min()}."
            )
        eff = self._effective_units(units)

        if self.theory is Theory.CRAWFORD:
            base = self.t1 * eff**self.b
        else:
            # Wright: the cost of unit x is the increment in cumulative total,
            # T1 * (x^(b+1) - (x-1)^(b+1)), which collapses to T1 at x = 1.
            exponent = self.b + 1.0
            base = self.t1 * (eff**exponent - np.maximum(eff - 1.0, 0.0) ** exponent)

        return base * self._step_factor(units)

    def cum_total(self, quantity) -> np.ndarray:
        """Cumulative cost of units 1..quantity."""
        quantities = np.atleast_1d(np.asarray(quantity, dtype=float))
        if self.theory is Theory.WRIGHT and self.is_smooth:
            return self.t1 * quantities ** (self.b + 1.0)
        return np.array(
            [float(np.sum(self.unit_cost(np.arange(1, int(q) + 1)))) for q in quantities]
        )

    def cum_average(self, quantity) -> np.ndarray:
        """Average cost of the first ``quantity`` units."""
        quantities = np.atleast_1d(np.asarray(quantity, dtype=float))
        if self.theory is Theory.WRIGHT and self.is_smooth:
            return self.t1 * quantities**self.b
        return self.cum_total(quantities) / quantities

    def lot_cost(self, first_unit, last_unit) -> np.ndarray:
        """Total cost of the units in a lot, inclusive of both endpoints."""
        first = np.atleast_1d(np.asarray(first_unit, dtype=int))
        last = np.atleast_1d(np.asarray(last_unit, dtype=int))
        if first.shape != last.shape:
            raise FitError(
                f"Got {first.size} first-unit values and {last.size} last-unit "
                f"values; lots need both."
            )
        if np.any(last < first):
            raise FitError("A lot cannot end before it begins.")

        if self.theory is Theory.WRIGHT and self.is_smooth:
            # Exact difference of cumulative totals: cheaper and more accurate
            # than summing telescoping increments in floating point.
            exponent = self.b + 1.0
            return self.t1 * (
                last.astype(float) ** exponent
                - (first.astype(float) - 1.0) ** exponent
            )
        return np.array(
            [
                float(np.sum(self.unit_cost(np.arange(f, l + 1))))
                for f, l in zip(first, last)
            ]
        )

    def lot_average(self, first_unit, last_unit) -> np.ndarray:
        """Average unit cost within a lot -- what a 1921-2 actually reports."""
        first = np.atleast_1d(np.asarray(first_unit, dtype=int))
        last = np.atleast_1d(np.asarray(last_unit, dtype=int))
        return self.lot_cost(first, last) / (last - first + 1).astype(float)

    def describe(self) -> dict[str, object]:
        """Flat summary for the assumptions log."""
        return {
            "theory": self.theory.value,
            "t1": self.t1,
            "b": self.b,
            "slope": self.slope,
            "breaks": [
                {
                    "at_unit": brk.at_unit,
                    "step_factor": brk.step_factor,
                    "learning_loss": brk.learning_loss,
                    "label": brk.label,
                }
                for brk in self.breaks
            ],
        }


# ==========================================================================
# Model specifications for the shared estimator
# ==========================================================================
def _break_param_names(breaks: tuple[RateBreak, ...]) -> tuple[str, ...]:
    return tuple(
        f"log_step_at_{brk.at_unit}" for brk in breaks if brk.is_estimated
    )


def _model_from_theta(
    theta: np.ndarray, theory: Theory, breaks: tuple[RateBreak, ...]
) -> CurveModel:
    """Rebuild a CurveModel from the parameter vector.

    Step factors are carried as logs so the optimiser cannot walk them
    negative, which would mean a rate break that makes units cost less than
    nothing.
    """
    resolved, k = [], 2
    for brk in breaks:
        if brk.is_estimated:
            resolved.append(replace(brk, step_factor=float(np.exp(theta[k]))))
            k += 1
        else:
            resolved.append(brk)
    return CurveModel(
        theory=theory,
        t1=float(np.exp(theta[0])),
        b=float(theta[1]),
        breaks=tuple(resolved),
    )


def _unit_spec(theory: Theory, breaks: tuple[RateBreak, ...]) -> ModelSpec:
    """Fit against per-unit observations.

    Under Crawford the observation is the cost of that unit; under Wright it is
    the cumulative average through that quantity. Both are the same power law
    in form, which is exactly why the two theories are so easy to confuse:
    the *fit* looks identical and only the interpretation -- and therefore
    every forecast -- differs.
    """

    def predict(theta: np.ndarray, X) -> np.ndarray:
        model = _model_from_theta(theta, theory, breaks)
        units = np.asarray(X, dtype=float)
        if theory is Theory.CRAWFORD:
            return model.unit_cost(units)
        return model.cum_average(units)

    def initial(X, y) -> np.ndarray:
        units = np.asarray(X, dtype=float)
        b0 = np.log2(0.85)
        if units.size >= 2 and np.ptp(units) > 0:
            slope, intercept = np.polyfit(np.log(units), np.log(y), 1)
            b0 = float(slope)
            log_t1 = float(intercept)
        else:  # pragma: no cover - guarded by the n >= 2 check in fit()
            log_t1 = float(np.log(np.mean(y)))
        return np.array([log_t1, b0, *np.zeros(len(_break_param_names(breaks)))])

    return ModelSpec(
        name=f"{theory.value}-unit-curve",
        param_names=("log_t1", "b", *_break_param_names(breaks)),
        predict=predict,
        link="log",
        log_scale_index=0,
        initial=initial,
    )


def _lot_spec(theory: Theory, breaks: tuple[RateBreak, ...]) -> ModelSpec:
    """Fit against lot averages, using exact lot costs.

    The traditional shortcut is to price a lot at its "lot midpoint" -- an
    approximate unit index whose cost stands in for the lot average. That
    approximation was worth making with a slide rule. Here the exact lot
    average is available in closed form under Wright and by summation under
    Crawford, so the shortcut buys nothing and costs accuracy on the early
    lots, where lots are small and the curve is steepest.
    """

    def predict(theta: np.ndarray, X) -> np.ndarray:
        model = _model_from_theta(theta, theory, breaks)
        arr = np.asarray(X, dtype=int)
        return model.lot_average(arr[:, 0], arr[:, 1])

    def initial(X, y) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        midpoints = np.sqrt(arr[:, 0] * arr[:, 1])  # geometric, positive by construction
        b0, log_t1 = np.log2(0.85), float(np.log(np.mean(y)))
        if arr.shape[0] >= 2 and np.ptp(midpoints) > 0:
            slope, intercept = np.polyfit(np.log(midpoints), np.log(y), 1)
            b0, log_t1 = float(slope), float(intercept)
        return np.array([log_t1, b0, *np.zeros(len(_break_param_names(breaks)))])

    return ModelSpec(
        name=f"{theory.value}-lot-curve",
        param_names=("log_t1", "b", *_break_param_names(breaks)),
        predict=predict,
        link="log",
        log_scale_index=0,
        initial=initial,
    )


# ==========================================================================
# Fit result
# ==========================================================================
@dataclass(frozen=True)
class CurveFit:
    """A fitted curve plus everything needed to defend and forecast from it."""

    model: CurveModel
    result: FitResult
    theory: Theory
    method: str
    granularity: str  # "unit" or "lot"
    breaks: tuple[RateBreak, ...] = ()
    _X: object = field(default=None, repr=False)

    # ------------------------------------------------------------- reporting
    @property
    def slope(self) -> float:
        return self.model.slope

    @property
    def t1(self) -> float:
        return self.model.t1

    @property
    def standard_error(self) -> float:
        """Standard error of the estimate, in dollars."""
        return self.result.standard_error

    @property
    def cv(self) -> float:
        """Coefficient of variation: the relative spread of the residuals.

        Reported instead of leaning on R^2. A log-log cost fit will show an
        R^2 of 0.98 whether or not the model is any good, because cost and
        quantity are both large and both trending; the CV says how wide the
        scatter actually is, in a unit a reviewer can argue with.
        """
        return self.result.cv

    @property
    def r_squared(self) -> float:
        """Available for completeness. See :attr:`cv` for why it is not the
        headline number."""
        return self.result.r_squared

    @property
    def slope_interval(self) -> tuple[float, float]:
        """An 80% interval on the learning slope itself."""
        se_b = self.result.param_se["b"]
        tcrit = float(stats.t.ppf(0.90, self.result.df))
        return (
            float(2.0 ** (self.model.b - tcrit * se_b)),
            float(2.0 ** (self.model.b + tcrit * se_b)),
        )

    def summary(self) -> pd.DataFrame:
        return self.result.summary()

    # ------------------------------------------------------------ forecasting
    def forecast_lots(
        self,
        lots: pd.DataFrame | np.ndarray,
        *,
        level: float = 0.80,
        kind: str = "prediction",
    ) -> pd.DataFrame:
        """Forecast future lots with intervals.

        Args:
            lots: Either a DataFrame with ``first_unit`` and ``last_unit``
                columns, or an (n, 2) array of the same.
            level: Interval coverage.
            kind: ``"prediction"`` for a new lot (the default and almost
                always what is wanted), ``"confidence"`` for the mean of the
                fitted relationship.

        Returns:
            DataFrame with quantity, forecast lot average and lot cost, and the
            interval on both.

        Raises:
            FitError: If the lot definition is malformed, or the fit was made
                on unit data (whose parameter uncertainty does not transfer to
                a lot-average prediction without refitting).
        """
        arr = _as_lot_array(lots)
        quantity = (arr[:, 1] - arr[:, 0] + 1).astype(float)

        if self.granularity == "lot":
            interval = predict_with_interval(
                self.result, arr, level=level, kind=kind
            )
            avg, lower, upper = (
                interval["fit"].to_numpy(),
                interval["lower"].to_numpy(),
                interval["upper"].to_numpy(),
            )
        else:
            # A unit-granularity fit predicts unit or cumulative-average cost.
            # Rather than pretend its covariance applies to a lot average,
            # scale the point estimate exactly and widen it by the fitted CV.
            avg = self.model.lot_average(arr[:, 0], arr[:, 1])
            tcrit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, self.result.df))
            spread = tcrit * self.result.sigma
            lower, upper = avg * np.exp(-spread), avg * np.exp(spread)

        return pd.DataFrame(
            {
                "first_unit": arr[:, 0],
                "last_unit": arr[:, 1],
                "quantity": quantity.astype(int),
                "lot_average": avg,
                "lot_average_lower": lower,
                "lot_average_upper": upper,
                "lot_cost": avg * quantity,
                "lot_cost_lower": lower * quantity,
                "lot_cost_upper": upper * quantity,
                "level": level,
                "kind": kind,
            }
        )

    def forecast_units(
        self, units, *, level: float = 0.80, kind: str = "prediction"
    ) -> pd.DataFrame:
        """Forecast individual unit costs with intervals."""
        units = np.atleast_1d(np.asarray(units, dtype=float))
        if self.granularity == "unit":
            interval = predict_with_interval(
                self.result, units, level=level, kind=kind
            )
            out = interval.rename(
                columns={"fit": "unit_cost", "lower": "lower", "upper": "upper"}
            )
            out.insert(0, "unit", units.astype(int))
            return out

        point = self.model.unit_cost(units)
        tcrit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, self.result.df))
        spread = tcrit * self.result.sigma
        return pd.DataFrame(
            {
                "unit": units.astype(int),
                "unit_cost": point,
                "lower": point * np.exp(-spread),
                "upper": point * np.exp(spread),
                "level": level,
                "kind": kind,
            }
        )

    def to_legacy_model(self) -> "LearningCurveModel":
        """Interoperate with the original :class:`LearningCurveModel` API."""
        return LearningCurveModel(
            slope=self.model.slope,
            reference_quantity=1.0,
            reference_cost=self.model.t1,
        )


def _as_lot_array(lots) -> np.ndarray:
    """Coerce a lot definition to an (n, 2) integer array of unit ranges."""
    if isinstance(lots, pd.DataFrame):
        missing = [c for c in ("first_unit", "last_unit") if c not in lots.columns]
        if missing:
            raise FitError(
                f"Lot table is missing column(s) {missing}. Got "
                f"{list(lots.columns)}."
            )
        arr = lots[["first_unit", "last_unit"]].to_numpy()
    else:
        arr = np.asarray(lots)
    arr = np.atleast_2d(arr)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise FitError(
            f"Lots must be an (n, 2) array of first and last unit; got shape "
            f"{arr.shape}."
        )
    arr = arr.astype(int)
    if np.any(arr[:, 0] < 1):
        raise FitError("Unit indices start at 1.")
    if np.any(arr[:, 1] < arr[:, 0]):
        raise FitError("A lot cannot end before it begins.")
    return arr


# ==========================================================================
# Fitting entry points
# ==========================================================================
def fit_curve(
    *,
    theory: Theory | str = Theory.CRAWFORD,
    method: str = "ols",
    units=None,
    costs=None,
    lots: pd.DataFrame | np.ndarray | None = None,
    lot_costs=None,
    breaks: tuple[RateBreak, ...] | list[RateBreak] = (),
    allow_low_df: bool = True,
) -> CurveFit:
    """Fit a learning curve under an explicit theory and method.

    Supply either unit-level data (``units`` and ``costs``) or lot-level data
    (``lots`` and ``lot_costs``), not both.

    Under Crawford, ``costs`` are the costs of the individual units named in
    ``units``. Under Wright, they are the *cumulative average* cost through
    each quantity in ``units``. Passing unit costs and asking for Wright will
    fit cleanly and forecast wrongly, which is why the theory is a required
    keyword with no default that flatters either choice.

    Args:
        theory: Wright or Crawford.
        method: ``"ols"``, ``"mupe"`` or ``"zmpe"``.
        units: Unit indices or cumulative quantities.
        costs: Unit costs or cumulative averages, matching ``theory``.
        lots: DataFrame with ``first_unit``/``last_unit``, or an (n, 2) array.
        lot_costs: *Total* cost of each lot. Converted to lot averages
            internally, since that is the scale the curve predicts.
        breaks: Rate breaks or production gaps to model explicitly.
        allow_low_df: Passed through to the estimator.

    Raises:
        FitError: On conflicting or missing data, an unknown theory or method,
            or a break that falls outside the observed unit range.
    """
    theory = Theory.parse(theory)
    breaks = tuple(sorted(breaks, key=lambda b: b.at_unit))
    method = str(method).lower()
    if method not in METHODS:
        raise FitError(f"Unknown method {method!r}. Allowed: {list(METHODS)}.")

    have_units = units is not None and costs is not None
    have_lots = lots is not None and lot_costs is not None
    if have_units == have_lots:
        raise FitError(
            "Supply exactly one of (units, costs) or (lots, lot_costs); got "
            f"units={'set' if units is not None else 'unset'}, "
            f"lots={'set' if lots is not None else 'unset'}."
        )

    if have_units:
        X = np.atleast_1d(np.asarray(units, dtype=float))
        y = np.atleast_1d(np.asarray(costs, dtype=float))
        if X.shape != y.shape:
            raise FitError(
                f"Got {X.size} quantities and {y.size} costs; these must "
                f"correspond one to one."
            )
        max_unit = float(X.max())
        spec = _unit_spec(theory, breaks)
        granularity = "unit"
    else:
        X = _as_lot_array(lots)
        totals = np.atleast_1d(np.asarray(lot_costs, dtype=float))
        if totals.size != X.shape[0]:
            raise FitError(
                f"Got {X.shape[0]} lots and {totals.size} lot costs."
            )
        quantity = (X[:, 1] - X[:, 0] + 1).astype(float)
        y = totals / quantity
        max_unit = float(X[:, 1].max())
        spec = _lot_spec(theory, breaks)
        granularity = "lot"

    for brk in breaks:
        if brk.at_unit > max_unit:
            raise FitError(
                f"Rate break at unit {brk.at_unit} falls beyond the last "
                f"observed unit ({int(max_unit)}); there is no data on either "
                f"side of it to estimate a step from."
            )

    result = fitting.fit(spec, X, y, method=method, allow_low_df=allow_low_df)
    model = _model_from_theta(result.theta, theory, breaks)

    logger.info(
        "Fitted %s %s curve by %s: slope=%.2f%%, T1=%.0f, CV=%.3f",
        theory.value,
        granularity,
        method.upper(),
        model.slope * 100.0,
        model.t1,
        result.cv,
    )
    return CurveFit(
        model=model,
        result=result,
        theory=theory,
        method=method,
        granularity=granularity,
        breaks=model.breaks,
        _X=X,
    )


def fit_from_progress_report(
    lot_table: pd.DataFrame,
    *,
    theory: Theory | str = Theory.CRAWFORD,
    method: str = "ols",
    breaks: tuple[RateBreak, ...] | list[RateBreak] = (),
    cost_col: str = "lot_cost",
) -> CurveFit:
    """Fit directly from a normalised DD 1921-2 table.

    Accepts the output of
    :meth:`cost_core.ingest.NormalizedDataset.learning_curve_input`, which
    carries ``first_unit``, ``last_unit`` and the lot cost in base-year
    dollars.

    Raises:
        FitError: If required columns are absent.
    """
    required = {"first_unit", "last_unit", cost_col}
    missing = sorted(required - set(lot_table.columns))
    if missing:
        raise FitError(
            f"Progress-curve table is missing column(s) {missing}. Got "
            f"{list(lot_table.columns)}."
        )
    return fit_curve(
        theory=theory,
        method=method,
        lots=lot_table,
        lot_costs=lot_table[cost_col].to_numpy(),
        breaks=breaks,
    )


# ==========================================================================
# Comparison and diagnostics
# ==========================================================================
def compare_theories(
    *, method: str = "ols", **kwargs
) -> dict[str, CurveFit]:
    """Fit the same data under both theories.

    Reporting one theory alone invites the question "what does the other one
    say", and the honest answer is usually "meaningfully different". Making
    the comparison cheap makes it routine.
    """
    return {
        theory.value: fit_curve(theory=theory, method=method, **kwargs)
        for theory in Theory
    }


def compare_methods(
    *, theory: Theory | str = Theory.CRAWFORD, **kwargs
) -> dict[str, CurveFit]:
    """Fit the same data by OLS, MUPE and ZMPE."""
    return {
        method: fit_curve(theory=theory, method=method, **kwargs)
        for method in METHODS
    }


def comparison_table(fits: dict[str, CurveFit]) -> pd.DataFrame:
    """Flat side-by-side summary of several fits, for a slide or a log."""
    rows = []
    for label, curve in fits.items():
        lo, hi = curve.slope_interval
        rows.append(
            {
                "label": label,
                "theory": curve.theory.value,
                "method": curve.method.upper(),
                "slope": curve.slope,
                "slope_lower_80": lo,
                "slope_upper_80": hi,
                "t1": curve.t1,
                "std_error": curve.standard_error,
                "cv": curve.cv,
                "mean_pct_error": curve.result.mean_percent_error,
                "df": curve.result.df,
                "r_squared": curve.r_squared,
            }
        )
    return pd.DataFrame(rows)


def retransformation_report(fits: dict[str, CurveFit]) -> fitting.RetransformationBias:
    """Measure how much the naive OLS retransformation understates the mean.

    Args:
        fits: Output of :func:`compare_methods`, containing at least ``"ols"``.

    Raises:
        FitError: If no OLS fit is present to measure.
    """
    if "ols" not in fits:
        raise FitError(
            "Retransformation bias is a property of the OLS fit; none was "
            f"supplied. Got: {sorted(fits)}."
        )
    return fitting.retransformation_bias(
        fits["ols"].result,
        fits.get("mupe").result if "mupe" in fits else None,
        fits.get("zmpe").result if "zmpe" in fits else None,
    )


def detect_rate_breaks(
    lot_table: pd.DataFrame,
    *,
    theory: Theory | str = Theory.CRAWFORD,
    cost_col: str = "lot_cost",
    threshold: float = 0.15,
) -> list[RateBreak]:
    """Suggest where a rate break may sit, from residuals of a smooth fit.

    Suggestions only, and deliberately so. A break is a *physical* event -- a
    second source, a design change, a cold line -- and the analyst should be
    able to name it. Fitting a step wherever the residuals happen to jump will
    always improve the fit and will sometimes be fitting noise.

    Args:
        threshold: Relative residual above which a lot is flagged.

    Returns:
        Candidate breaks with unestimated step factors, ordered by unit.
    """
    smooth = fit_from_progress_report(
        lot_table, theory=theory, method="ols", cost_col=cost_col
    )
    errors = smooth.result.percent_errors
    arr = _as_lot_array(lot_table)

    candidates = []
    for i in range(1, len(errors)):
        jump = errors[i] - errors[i - 1]
        if abs(jump) >= threshold:
            candidates.append(
                RateBreak(
                    at_unit=int(arr[i, 0]),
                    label=f"residual jump of {jump:+.1%} at lot {i + 1}",
                )
            )
    if candidates:
        logger.info(
            "Found %d candidate rate break(s); these are suggestions and "
            "should be corroborated with programmatic history before use.",
            len(candidates),
        )
    return candidates


if __name__ == "__main__":
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO)

    # Demo Data: 85% Learning Curve (Theoretical)
    data = pd.DataFrame({
        "unit_quantity": [1, 2, 4, 8, 16],
        "unit_cost": [100.0, 85.0, 72.25, 61.41, 52.20]
    })

    print("--- Historical Data ---")
    print(data)

    # Fit Model
    lc_model = fit_learning_curve(data)
    print(f"\nFitted Slope: {lc_model.slope:.4f}")

    # Forecast
    future_units = np.array([32, 64, 128])
    forecast_df = forecast_costs(lc_model, future_units)

    print("\n--- Forecasted Data ---")
    print(forecast_df.to_string(index=False))