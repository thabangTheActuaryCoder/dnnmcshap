"""Residual-augmented Monte Carlo sensitivity analysis (Algorithm 3.2).

Incorporates model uncertainty by augmenting base MC predictions with
centred training residuals and fixed perturbation bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constants import SEED


@dataclass
class SensitivityResult:
    """Stores residual-augmented sensitivity outputs."""

    residuals_centred: np.ndarray
    sigma_e: float
    y_augmented: np.ndarray
    y_upper: np.ndarray
    y_lower: np.ndarray


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute centred training residuals and residual standard deviation.

    Parameters
    ----------
    y_true : np.ndarray
        Actual training targets (de-normalised).
    y_pred : np.ndarray
        DNN predictions on training data (de-normalised).

    Returns
    -------
    residuals_centred : np.ndarray
        Mean-centred residuals.
    sigma_e : float
        Standard deviation of centred residuals.
    """
    residuals = y_true - y_pred
    residuals_centred = residuals - residuals.mean()
    sigma_e = float(residuals_centred.std())
    return residuals_centred, sigma_e


def augment_predictions(
    y_mc: np.ndarray,
    residuals_centred: np.ndarray,
    sigma_e: float,
    *,
    seed: int = SEED,
) -> SensitivityResult:
    """Augment base MC predictions with bootstrap-resampled residuals.

    The augmented distribution is:
        ``Y_res = Y_hat + epsilon``
    where epsilon is drawn (with replacement) from the centred training
    residuals. Fixed perturbation bands at +/-0.5*sigma_e are also computed.

    Parameters
    ----------
    y_mc : np.ndarray
        Base MC predicted costs, shape ``(B,)``.
    residuals_centred : np.ndarray
        Centred training residuals.
    sigma_e : float
        Residual standard deviation.
    seed : int
        Random seed for bootstrap resampling.

    Returns
    -------
    SensitivityResult
    """
    rng = np.random.default_rng(seed)
    eps_sample = rng.choice(residuals_centred, size=len(y_mc), replace=True)

    return SensitivityResult(
        residuals_centred=residuals_centred,
        sigma_e=sigma_e,
        y_augmented=y_mc + eps_sample,
        y_upper=y_mc + 0.5 * sigma_e,
        y_lower=y_mc - 0.5 * sigma_e,
    )
