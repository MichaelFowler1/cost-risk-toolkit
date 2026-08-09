"""
model.py - Parametric cost estimating relationships.

A CER predicts the cost of a new program from its technical characteristics,
fitted across a handful of past programs. Two things about that sentence drive
the whole design of this module.

**"A new program."** The interval that matters is the *prediction* interval --
the range a single new observation is expected to fall in -- not the
confidence interval on the mean of the fitted relationship. The two are
routinely confused, and the confusion always runs the same direction: the
confidence interval is narrower, it shrinks toward zero as the sample grows,
and quoting it makes an estimate look far more certain than it is. A cost
estimate is a forecast of one program, so :meth:`CER.predict` takes ``kind``
explicitly and defaults to ``"prediction"``. The variance identity
``Var_pred = Var_mean + sigma^2`` is asserted in the tests; the extra term is
the scatter of programs about the line, and no amount of additional data makes
it go away.

**"A handful."** CER datasets are small. Six programs and three predictors
leaves two degrees of freedom, which is not enough to estimate anything you
would want to defend, and the arithmetic will not complain. So this module
applies guardrails before it reports: zero degrees of freedom is refused
outright by the estimator, fewer than three warns, and fewer than three
observations per parameter warns separately, because that ratio is what
governs whether the coefficients are separable at all.

Both functional forms in common use are supported:

``log_log``   ``y = a * x_1^b_1 * ... * x_k^b_k``, fitted in log space. The
              usual choice for cost, because it gives constant elasticities and
              multiplicative error, which is how cost actually scatters.
``linear``    ``y = b_0 + b_1 x_1 + ... + b_k x_k``, additive error.

Each can be fitted by OLS, MUPE or ZMPE. Under OLS the error structure follows
the form -- additive for linear, multiplicative for log-log. MUPE and ZMPE are
always proportional-error methods; see :mod:`cost_core.fitting` for why that
matters and what the naive OLS retransformation costs.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from cost_core import fitting
from cost_core.cer.diagnostics import Diagnostics, compute_diagnostics
from cost_core.fitting import (METHODS, FitError, FitResult, ModelSpec,
                               predict_with_interval)

logger = logging.getLogger(__name__)

#: Observations per estimated parameter below which the fit is flagged. Three
#: is the usual rule of thumb; below it the coefficients start trading off
#: against each other and their individual values stop meaning much even when
#: the overall fit looks fine.
MIN_OBS_PER_PARAM = 3.0


class ExtrapolationWarning(UserWarning):
    """Raised when a prediction point falls outside the fitting data.

    Its own class so callers can escalate it to an error with
    ``warnings.simplefilter("error", ExtrapolationWarning)`` -- worth doing in
    an automated pipeline, where nobody is reading the log.
    """


class Form(str, Enum):
    """CER functional form."""

    LOG_LOG = "log_log"
    LINEAR = "linear"

    @classmethod
    def parse(cls, value: "Form | str") -> "Form":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise FitError(
                f"Unknown CER form {value!r}. Allowed: {[f.value for f in cls]}."
            ) from None


# --------------------------------------------------------------------------
# model specifications
# --------------------------------------------------------------------------
def _log_log_spec(predictors: tuple[str, ...]) -> ModelSpec:
    """y = exp(theta_0) * prod(x_j ** theta_j)."""

    def predict(theta: np.ndarray, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.exp(theta[0] + np.log(arr) @ np.asarray(theta[1:]))

    def jacobian(theta: np.ndarray, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        f = predict(theta, arr)
        return np.column_stack([f, f[:, None] * np.log(arr)])

    def initial(X, y) -> np.ndarray:
        arr = np.log(np.asarray(X, dtype=float))
        design = np.column_stack([np.ones(arr.shape[0]), arr])
        beta, *_ = np.linalg.lstsq(design, np.log(np.asarray(y, dtype=float)), rcond=None)
        return beta

    return ModelSpec(
        name=f"log-log CER ({', '.join(predictors)})",
        param_names=("log_a", *(f"b_{p}" for p in predictors)),
        predict=predict,
        link="log",
        log_scale_index=0,
        initial=initial,
        jacobian=jacobian,
    )


def _linear_spec(predictors: tuple[str, ...]) -> ModelSpec:
    """y = theta_0 + sum(theta_j * x_j)."""

    def predict(theta: np.ndarray, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return theta[0] + arr @ np.asarray(theta[1:])

    def jacobian(theta: np.ndarray, X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(arr.shape[0]), arr])

    def initial(X, y) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(arr.shape[0]), arr])
        beta, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
        return beta

    return ModelSpec(
        name=f"linear CER ({', '.join(predictors)})",
        param_names=("intercept", *(f"b_{p}" for p in predictors)),
        predict=predict,
        link="identity",
        log_scale_index=None,
        initial=initial,
        jacobian=jacobian,
    )


# --------------------------------------------------------------------------
# the CER
# --------------------------------------------------------------------------
@dataclass
class CER:
    """A fitted cost estimating relationship."""

    form: Form
    method: str
    result: FitResult
    response: str
    predictors: tuple[str, ...]
    fitting_data: pd.DataFrame = field(repr=False)
    labels: tuple[str, ...] = ()

    # ----------------------------------------------------------- properties
    @property
    def coefficients(self) -> dict[str, float]:
        return self.result.params

    @property
    def standard_error(self) -> float:
        """Standard error of the estimate, in the units of the response."""
        return self.result.standard_error

    @property
    def cv(self) -> float:
        """Coefficient of variation of the estimate."""
        return self.result.cv

    @property
    def df(self) -> int:
        return self.result.df

    @property
    def r_squared(self) -> float:
        """Reported for completeness. See the module docstring and
        :attr:`cv` for why it does not lead."""
        return self.result.r_squared

    @property
    def obs_per_param(self) -> float:
        return self.result.n_obs / self.result.n_params

    @property
    def predictor_ranges(self) -> dict[str, tuple[float, float]]:
        """Min and max of each predictor in the fitting data."""
        return {
            name: (
                float(self.fitting_data[name].min()),
                float(self.fitting_data[name].max()),
            )
            for name in self.predictors
        }

    def summary(self) -> pd.DataFrame:
        return self.result.summary()

    def diagnostics(self) -> Diagnostics:
        """Leverage, influence and residuals for the fitting sample."""
        return compute_diagnostics(self.result, self.labels or None)

    # ---------------------------------------------------------- extrapolation
    def _extrapolation_check(self, X_new: np.ndarray) -> pd.DataFrame:
        """Flag prediction points outside the fitting data.

        Two checks, because they catch different mistakes. The per-predictor
        range check is obvious and catches most cases. The leverage check
        catches *hidden* extrapolation: a point whose every individual value
        sits inside the observed range, but whose combination does not occur
        anywhere in the sample -- a light aircraft that is also very fast, when
        every program in the data is either light and slow or heavy and fast.
        The CER has no information about that corner, and only the leverage
        ratio will say so.
        """
        ranges = self.predictor_ranges
        notes: list[str] = []
        outside: list[bool] = []

        for row in X_new:
            breaches = []
            for value, name in zip(row, self.predictors):
                lo, hi = ranges[name]
                if value < lo:
                    breaches.append(f"{name}={value:,.4g} below observed min {lo:,.4g}")
                elif value > hi:
                    breaches.append(f"{name}={value:,.4g} above observed max {hi:,.4g}")
            outside.append(bool(breaches))
            notes.append("; ".join(breaches))

        # Hidden extrapolation: leverage of the new point against the fit.
        jac_fit = fitting.design_matrix(self.result)
        gram_inv = np.linalg.pinv(jac_fit.T @ jac_fit)
        max_train_leverage = float(
            np.max(np.einsum("ij,jk,ik->i", jac_fit, gram_inv, jac_fit))
        )
        jac_new = fitting._jacobian(
            self.result.spec,
            self.result.theta,
            X_new,
            on_log_scale=fitting.fitting_scale(self.result) == "log",
        )
        new_leverage = np.einsum("ij,jk,ik->i", jac_new, gram_inv, jac_new)
        ratio = new_leverage / max(max_train_leverage, 1e-300)

        for i, r in enumerate(ratio):
            if r > 1.0 and not outside[i]:
                outside[i] = True
                notes[i] = (
                    f"inside every individual predictor range, but this "
                    f"combination of predictors is {r:.1f}x more extreme than "
                    f"anything in the fitting data (hidden extrapolation)"
                )

        return pd.DataFrame(
            {
                "outside_fitting_range": outside,
                "leverage_ratio": ratio,
                "extrapolation_note": notes,
            }
        )

    # ------------------------------------------------------------- predicting
    def predict(
        self,
        new_data: pd.DataFrame | np.ndarray | dict,
        *,
        kind: str = "prediction",
        level: float = 0.80,
        warn_on_extrapolation: bool = True,
    ) -> pd.DataFrame:
        """Estimate cost for new programs, with an interval.

        Args:
            new_data: DataFrame with the predictor columns, a dict of the
                same, or an array with columns in ``self.predictors`` order.
            kind: ``"prediction"`` for the range a single new program is
                expected to fall in -- the right choice for a cost estimate,
                and the default. ``"confidence"`` for the range the *mean*
                relationship lies in, which is a statement about the fitted
                line and not about any program.
            level: Interval coverage.
            warn_on_extrapolation: Emit :class:`ExtrapolationWarning` when a
                point falls outside the fitting data.

        Returns:
            DataFrame with ``fit``, ``lower``, ``upper``, ``se``, the interval
            ``kind`` and ``level``, plus the extrapolation columns.

        Raises:
            FitError: On an unknown ``kind`` or missing predictor columns.
        """
        if kind not in ("prediction", "confidence"):
            raise FitError(
                f"kind must be 'prediction' (a new program) or 'confidence' "
                f"(the mean relationship); got {kind!r}. These are not "
                f"interchangeable -- see the module docstring."
            )

        X_new = self._coerce(new_data)
        interval = predict_with_interval(
            self.result, X_new, level=level, kind=kind
        )
        extrapolation = self._extrapolation_check(X_new)
        out = pd.concat([interval, extrapolation], axis=1)

        if warn_on_extrapolation and out["outside_fitting_range"].any():
            offenders = out.loc[
                out["outside_fitting_range"], "extrapolation_note"
            ].tolist()
            warnings.warn(
                f"{len(offenders)} of {len(out)} prediction point(s) fall "
                f"outside the fitting data: {offenders}. The CER has no "
                f"evidence out there; the interval reflects only the "
                f"uncertainty of the fitted form, not the risk that the form "
                f"itself stops holding.",
                ExtrapolationWarning,
                stacklevel=2,
            )
        return out

    def _coerce(self, new_data) -> np.ndarray:
        """Bring caller input into an (n, k) array in predictor order."""
        if isinstance(new_data, dict):
            new_data = pd.DataFrame(
                {k: np.atleast_1d(v) for k, v in new_data.items()}
            )
        if isinstance(new_data, pd.DataFrame):
            missing = [p for p in self.predictors if p not in new_data.columns]
            if missing:
                raise FitError(
                    f"New data is missing predictor column(s) {missing}. "
                    f"Expected {list(self.predictors)}, got "
                    f"{list(new_data.columns)}."
                )
            arr = new_data[list(self.predictors)].to_numpy(dtype=float)
        else:
            arr = np.atleast_2d(np.asarray(new_data, dtype=float))

        if arr.shape[1] != len(self.predictors):
            raise FitError(
                f"Expected {len(self.predictors)} predictor(s) "
                f"{list(self.predictors)}; got {arr.shape[1]} column(s)."
            )
        if self.form is Form.LOG_LOG and np.any(arr <= 0):
            raise FitError(
                "A log-log CER is undefined at or below zero for any "
                "predictor."
            )
        return arr

    # ----------------------------------------------------------------- output
    def describe(self) -> dict[str, object]:
        """Flat summary for the assumptions log."""
        return {
            "form": self.form.value,
            "method": self.method,
            "response": self.response,
            "predictors": list(self.predictors),
            "equation": self.equation(),
            "n_obs": self.result.n_obs,
            "n_params": self.result.n_params,
            "df": self.df,
            "obs_per_param": self.obs_per_param,
            "standard_error": self.standard_error,
            "cv": self.cv,
            "r_squared": self.r_squared,
            "predictor_ranges": {
                k: list(v) for k, v in self.predictor_ranges.items()
            },
            "coefficients": self.coefficients,
        }

    def equation(self) -> str:
        """The fitted relationship, written out."""
        theta = self.result.theta
        if self.form is Form.LOG_LOG:
            terms = "".join(
                f" * {p}^{theta[i + 1]:.4f}" for i, p in enumerate(self.predictors)
            )
            return f"{self.response} = {np.exp(theta[0]):,.4g}{terms}"
        terms = "".join(
            f" {'+' if theta[i + 1] >= 0 else '-'} {abs(theta[i + 1]):,.4g}*{p}"
            for i, p in enumerate(self.predictors)
        )
        return f"{self.response} = {theta[0]:,.4g}{terms}"


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit_cer(
    data: pd.DataFrame,
    response: str,
    predictors: list[str] | tuple[str, ...],
    *,
    form: Form | str = Form.LOG_LOG,
    method: str = "ols",
    label_col: str | None = None,
    allow_low_df: bool = True,
) -> CER:
    """Fit a cost estimating relationship.

    Args:
        data: One row per past program.
        response: Column to predict, typically a cost or an effort.
        predictors: Technical driver columns.
        form: ``"log_log"`` or ``"linear"``.
        method: ``"ols"``, ``"mupe"`` or ``"zmpe"``.
        label_col: Column holding program names, used to label diagnostics.
        allow_low_df: If False, refuse rather than warn on inadequate degrees
            of freedom. Worth setting in an automated pipeline.

    Raises:
        FitError: On missing columns, an unknown form or method, non-positive
            values where the form requires positives, or too few observations
            to identify the parameters.
    """
    form = Form.parse(form)
    method = str(method).lower()
    if method not in METHODS:
        raise FitError(f"Unknown method {method!r}. Allowed: {list(METHODS)}.")

    predictors = tuple(predictors)
    if not predictors:
        raise FitError("A CER needs at least one predictor.")

    missing = [c for c in (response, *predictors) if c not in data.columns]
    if missing:
        raise FitError(
            f"Data is missing column(s) {missing}. Got {list(data.columns)}."
        )

    frame = data[[response, *predictors]].dropna()
    if len(frame) < len(data):
        logger.warning(
            "Dropped %d row(s) with missing values; fitting on %d.",
            len(data) - len(frame),
            len(frame),
        )

    X = frame[list(predictors)].to_numpy(dtype=float)
    y = frame[response].to_numpy(dtype=float)

    if form is Form.LOG_LOG:
        if np.any(X <= 0):
            bad = [p for i, p in enumerate(predictors) if np.any(X[:, i] <= 0)]
            raise FitError(
                f"A log-log CER takes logs of its predictors, which is "
                f"undefined at or below zero. Offending column(s): {bad}. "
                f"Use the linear form, or shift the predictor on purpose and "
                f"say so."
            )
        if np.any(y <= 0):
            raise FitError(
                f"A log-log CER takes the log of {response!r}, which is "
                f"undefined at or below zero."
            )

    spec = (
        _log_log_spec(predictors) if form is Form.LOG_LOG else _linear_spec(predictors)
    )
    result = fitting.fit(spec, X, y, method=method, allow_low_df=allow_low_df)

    # Small-sample guardrail beyond the estimator's degrees-of-freedom check.
    ratio = result.n_obs / result.n_params
    if ratio < MIN_OBS_PER_PARAM:
        message = (
            f"{result.n_obs} observations for {result.n_params} parameters is "
            f"{ratio:.1f} per parameter, below the {MIN_OBS_PER_PARAM:.0f} "
            f"normally required. The coefficients are not well separated: the "
            f"overall fit may look sound while no individual predictor's "
            f"effect is estimable. Drop a predictor or find more programs."
        )
        if not allow_low_df:
            raise FitError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    labels = (
        tuple(str(x) for x in data.loc[frame.index, label_col])
        if label_col and label_col in data.columns
        else tuple(str(i) for i in frame.index)
    )

    logger.info(
        "Fitted %s by %s on %d programs: SE=%.4g, CV=%.3f, df=%d",
        spec.name,
        method.upper(),
        result.n_obs,
        result.standard_error,
        result.cv,
        result.df,
    )

    return CER(
        form=form,
        method=method,
        result=result,
        response=response,
        predictors=predictors,
        fitting_data=frame,
        labels=labels,
    )


def compare_cer_methods(
    data: pd.DataFrame,
    response: str,
    predictors: list[str] | tuple[str, ...],
    *,
    form: Form | str = Form.LOG_LOG,
    **kwargs,
) -> dict[str, CER]:
    """Fit the same CER by OLS, MUPE and ZMPE."""
    return {
        method: fit_cer(
            data, response, predictors, form=form, method=method, **kwargs
        )
        for method in METHODS
    }


def compare_cer_forms(
    data: pd.DataFrame,
    response: str,
    predictors: list[str] | tuple[str, ...],
    *,
    method: str = "ols",
    **kwargs,
) -> dict[str, CER]:
    """Fit the same data as both a log-log and a linear CER."""
    return {
        form.value: fit_cer(
            data, response, predictors, form=form, method=method, **kwargs
        )
        for form in Form
    }


def cer_comparison_table(cers: dict[str, CER]) -> pd.DataFrame:
    """Side-by-side summary of several CERs.

    Standard error and CV lead; R^2 is present but last, because comparing
    candidate CERs on R^2 alone is what produces a model that fits the sample
    beautifully and the next program badly.
    """
    rows = []
    for label, cer in cers.items():
        rows.append(
            {
                "label": label,
                "form": cer.form.value,
                "method": cer.method.upper(),
                "equation": cer.equation(),
                "std_error": cer.standard_error,
                "cv": cer.cv,
                "mean_pct_error": cer.result.mean_percent_error,
                "n_obs": cer.result.n_obs,
                "df": cer.df,
                "obs_per_param": cer.obs_per_param,
                "r_squared": cer.r_squared,
            }
        )
    return pd.DataFrame(rows)
