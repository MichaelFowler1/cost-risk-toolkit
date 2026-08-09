"""
diagnostics.py - Residual, leverage and influence diagnostics for a CER.

A CER is usually fitted to somewhere between six and twenty programs. At that
size a single unusual program can set the slope on its own, and the summary
statistics will not say so: the R^2 stays high, the standard error stays
plausible, and the one point doing all the work is invisible. Leverage and
influence are the diagnostics that find it.

Three quantities, all computed on the scale the fit actually minimised on --
log space for a log-log fit, percentage-error space for MUPE and ZMPE, level
space for a linear OLS:

**Leverage** (``h_ii``) is how unusual observation *i* is in the *predictor*
space, ignoring its response entirely. It is bounded in [0, 1] and the
leverages sum to exactly the number of parameters, which is asserted in the
tests -- it falls straight out of the hat matrix being a projection.

**Cook's distance** combines leverage with the size of the residual: how much
the whole fitted surface moves if observation *i* is dropped. Computed in
closed form, and the tests check it against an actual leave-one-out refit,
because the closed form is exact and there is no reason to accept an
approximation of something verifiable.

**DFFITS** is the same idea scaled to a single fitted value.

The conventional thresholds (``2p/n`` for leverage, ``4/n`` for Cook's D) are
flags, not verdicts. A high-leverage program is often the most informative one
in the sample -- the biggest airframe in a weight-based CER has high leverage
by construction and dropping it would be indefensible. The diagnostics exist to
make sure that judgement gets made deliberately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cost_core import fitting
from cost_core.fitting import FitResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Diagnostics:
    """Per-observation regression diagnostics.

    Attributes:
        residuals: Residuals on the fitting scale.
        standardized_residuals: Residuals divided by their own standard error,
            which varies with leverage -- a high-leverage point pulls the line
            toward itself and so has an artificially small raw residual.
        leverage: ``h_ii``, the diagonal of the hat matrix.
        cooks_distance: Influence on the whole fitted surface.
        dffits: Influence on the observation's own fitted value.
        labels: Row labels, typically program names.
        n_obs, n_params, df: Sizes.
        sigma: Residual scale on the fitting scale.
    """

    residuals: np.ndarray
    standardized_residuals: np.ndarray
    leverage: np.ndarray
    cooks_distance: np.ndarray
    dffits: np.ndarray
    labels: tuple[str, ...]
    n_obs: int
    n_params: int
    df: int
    sigma: float

    # ------------------------------------------------------------ thresholds
    @property
    def leverage_threshold(self) -> float:
        """``2p/n``: the conventional flag for an unusual predictor value."""
        return 2.0 * self.n_params / self.n_obs

    @property
    def cooks_threshold(self) -> float:
        """``4/n``: the conventional flag for an influential observation."""
        return 4.0 / self.n_obs

    @property
    def high_leverage(self) -> list[str]:
        return [
            self.labels[i]
            for i in np.flatnonzero(self.leverage > self.leverage_threshold)
        ]

    @property
    def influential(self) -> list[str]:
        return [
            self.labels[i]
            for i in np.flatnonzero(self.cooks_distance > self.cooks_threshold)
        ]

    def to_frame(self) -> pd.DataFrame:
        """One row per observation, sorted by influence."""
        frame = pd.DataFrame(
            {
                "observation": list(self.labels),
                "residual": self.residuals,
                "std_residual": self.standardized_residuals,
                "leverage": self.leverage,
                "cooks_distance": self.cooks_distance,
                "dffits": self.dffits,
                "high_leverage": self.leverage > self.leverage_threshold,
                "influential": self.cooks_distance > self.cooks_threshold,
            }
        )
        return frame.sort_values("cooks_distance", ascending=False).reset_index(
            drop=True
        )

    def narrative(self) -> str:
        """A sentence for the assumptions log."""
        parts = [
            f"{self.n_obs} observations, {self.n_params} parameters, "
            f"{self.df} degrees of freedom."
        ]
        if self.high_leverage:
            parts.append(
                f"High leverage (>{self.leverage_threshold:.2f}): "
                f"{', '.join(self.high_leverage)}."
            )
        if self.influential:
            parts.append(
                f"Influential (Cook's D >{self.cooks_threshold:.2f}): "
                f"{', '.join(self.influential)}. These set the fit; confirm "
                f"each belongs in the sample before relying on it."
            )
        if not self.high_leverage and not self.influential:
            parts.append("No observation exceeds the conventional flags.")
        return " ".join(parts)


def compute_diagnostics(
    result: FitResult, labels: list[str] | tuple[str, ...] | None = None
) -> Diagnostics:
    """Compute leverage and influence for a fit.

    Args:
        result: Any fit from :mod:`cost_core.fitting`.
        labels: Row labels, usually program names. Defaults to indices.

    Raises:
        ValueError: If ``labels`` does not match the number of observations.
    """
    n, p = result.n_obs, result.n_params
    if labels is None:
        labels = tuple(str(i) for i in range(n))
    elif len(labels) != n:
        raise ValueError(
            f"Got {len(labels)} labels for {n} observations."
        )

    jac = fitting.design_matrix(result)
    residuals = fitting.fitting_residuals(result)

    # Hat matrix diagonal, via the pseudo-inverse so a rank-deficient design
    # still reports rather than raising.
    gram_inv = np.linalg.pinv(jac.T @ jac)
    leverage = np.einsum("ij,jk,ik->i", jac, gram_inv, jac)
    leverage = np.clip(leverage, 0.0, 1.0)

    sigma = result.sigma
    one_minus_h = np.maximum(1.0 - leverage, 1e-12)

    # A fit that passes exactly through every point -- noiseless synthetic
    # data, or an interpolating model -- has residuals and a sigma that are
    # both at the floating-point floor. Standardising one by the other is 0/0,
    # and the O(1) garbage it produces would be flagged as influential. The
    # honest answer is that a perfect fit has no influential observations, so
    # the residual-based diagnostics are reported as zero. Leverage is
    # unaffected: it depends only on the predictors.
    scale = float(np.max(np.abs(fitting.design_matrix(result) @ result.theta)))
    degenerate = sigma <= 1e-10 * max(scale, 1.0)
    if degenerate:
        logger.info(
            "Residual scale is at the floating-point floor (sigma=%.3g); the "
            "fit is effectively exact, so influence measures are reported as "
            "zero rather than as ratios of rounding error.",
            sigma,
        )
        standardized = np.zeros_like(residuals)
        cooks = np.zeros_like(residuals)
    else:
        standardized = residuals / (sigma * np.sqrt(one_minus_h))
        cooks = (standardized**2 / p) * (leverage / one_minus_h)

    # DFFITS uses the leave-one-out sigma, which is available in closed form.
    if degenerate:
        dffits = np.zeros_like(residuals)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            s_minus_i = np.sqrt(
                np.maximum(
                    (result.df * sigma**2 - residuals**2 / one_minus_h)
                    / max(result.df - 1, 1),
                    0.0,
                )
            )
            dffits = np.where(
                s_minus_i > 0,
                residuals
                / (s_minus_i * np.sqrt(one_minus_h))
                * np.sqrt(leverage / one_minus_h),
                0.0,
            )

    return Diagnostics(
        residuals=residuals,
        standardized_residuals=standardized,
        leverage=leverage,
        cooks_distance=cooks,
        dffits=np.nan_to_num(dffits),
        labels=tuple(str(x) for x in labels),
        n_obs=n,
        n_params=p,
        df=result.df,
        sigma=sigma,
    )
