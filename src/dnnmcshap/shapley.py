"""SHAP GradientExplainer computation and dummy-to-raw aggregation.

Computes Shapley values on the 22 encoded features and aggregates
them to the 11 raw variables using the dummy-group mapping.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import shap
import torch

from ._constants import (
    DEFAULT_SHAP_BACKGROUND,
    DEFAULT_SHAP_BATCH,
    RAW_TO_ENCODED,
    RAW_VARIABLES,
    SEED,
)
from .model import FunnelDNN
from .standardisation import Standardiser


@dataclass
class ShapResult:
    """Stores SHAP computation results."""

    shap_values_22: np.ndarray
    shap_values_11: np.ndarray
    phi_0: float
    efficiency_error: np.ndarray


def sample_background(
    X_train_std: np.ndarray,
    n_background: int = DEFAULT_SHAP_BACKGROUND,
    *,
    seed: int = SEED,
) -> torch.Tensor:
    """Sample background data for the SHAP explainer.

    Parameters
    ----------
    X_train_std : np.ndarray
        Standardised training features, shape ``(n_train, 22)``.
    n_background : int
        Number of background samples.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Background tensor, shape ``(n_background, 22)``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train_std), n_background, replace=False)
    return torch.tensor(X_train_std[idx], dtype=torch.float32)


def aggregate_dummy_to_raw(shap_values_22: np.ndarray) -> np.ndarray:
    """Aggregate 22-column SHAP values to 11 raw variables.

    For each raw variable, the SHAP values of its dummy columns are
    summed.

    Parameters
    ----------
    shap_values_22 : np.ndarray
        SHAP values, shape ``(n, 22)``.

    Returns
    -------
    np.ndarray
        Aggregated SHAP values, shape ``(n, 11)``.
    """
    n = shap_values_22.shape[0]
    shap_values_11 = np.zeros((n, 11), dtype=np.float32)
    for j, (var_name, col_indices) in enumerate(RAW_TO_ENCODED.items()):
        shap_values_11[:, j] = shap_values_22[:, col_indices].sum(axis=1)
    return shap_values_11


def compute_shap_values(
    model: FunnelDNN,
    X_foreground_std: np.ndarray,
    X_background: torch.Tensor,
    *,
    batch_size: int = DEFAULT_SHAP_BATCH,
    verbose: bool = False,
) -> ShapResult:
    """Compute SHAP values using GradientExplainer.

    The model is copied to CPU for SHAP computation (GradientExplainer
    performs better on CPU than MPS).

    Parameters
    ----------
    model : FunnelDNN
        Trained model.
    X_foreground_std : np.ndarray
        Standardised foreground features, shape ``(n, 22)``.
    X_background : torch.Tensor
        Background tensor for the explainer.
    batch_size : int
        Process foreground data in batches of this size.
    verbose : bool
        Print progress.

    Returns
    -------
    ShapResult
    """
    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    X_bg = X_background.cpu() if X_background.device.type != "cpu" else X_background
    X_fg = torch.tensor(X_foreground_std, dtype=torch.float32)

    explainer = shap.GradientExplainer(model_cpu, X_bg)

    n = len(X_foreground_std)
    n_batches = (n + batch_size - 1) // batch_size
    shap_values_list = []

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        sv = explainer.shap_values(X_fg[start:end])
        if isinstance(sv, list):
            sv = sv[0]
        if isinstance(sv, torch.Tensor):
            sv = sv.cpu().numpy()
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv.squeeze(-1)
        shap_values_list.append(sv)
        if verbose:
            print(f"  SHAP batch {i + 1}/{n_batches} ({end - start} profiles)")

    shap_values_22 = np.concatenate(shap_values_list, axis=0)

    # Compute phi_0 (expected model output on background)
    with torch.no_grad():
        phi_0 = model_cpu(X_bg).mean().item()

    # Efficiency check
    with torch.no_grad():
        y_norm = model_cpu(X_fg).numpy().flatten()
    reconstructed = phi_0 + shap_values_22.sum(axis=1)
    efficiency_error = np.abs(reconstructed - y_norm)

    # Aggregate to raw variables
    shap_values_11 = aggregate_dummy_to_raw(shap_values_22)

    return ShapResult(
        shap_values_22=shap_values_22,
        shap_values_11=shap_values_11,
        phi_0=phi_0,
        efficiency_error=efficiency_error,
    )
