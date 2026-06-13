"""Checkpoint save/load and CSV export utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ._constants import ENCODED_FEATURES, RAW_VARIABLES
from .model import FunnelDNN
from .standardisation import Standardiser


def save_checkpoint(
    path: str | Path,
    model: FunnelDNN,
    standardiser: Standardiser,
    *,
    architecture: dict[str, Any] | None = None,
    hyperparameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    training_history: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint in ``.pth`` format.

    The checkpoint includes the model state dict, architecture metadata,
    standardisation parameters, hyperparameters and evaluation metrics.

    Parameters
    ----------
    path : str or Path
        Output file path.
    model : FunnelDNN
        Trained model.
    standardiser : Standardiser
        Fitted standardiser.
    architecture : dict, optional
        Architecture metadata (input_dim, hidden_layers, activation, bias).
    hyperparameters : dict, optional
        Training hyperparameters.
    metrics : dict, optional
        Evaluation metrics.
    training_history : dict, optional
        Training and validation loss/R-squared histories.
    """
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "standardisation": standardiser.to_dict(),
        "features": list(ENCODED_FEATURES),
    }
    if architecture is not None:
        checkpoint["architecture"] = architecture
    if hyperparameters is not None:
        checkpoint["hyperparameters"] = hyperparameters
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if training_history is not None:
        checkpoint["training_history"] = training_history

    torch.save(checkpoint, str(path))


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[FunnelDNN, Standardiser, dict[str, Any]]:
    """Load a model checkpoint.

    Parameters
    ----------
    path : str or Path
        Path to ``.pth`` checkpoint file.
    device : torch.device or str
        Device to load model onto.

    Returns
    -------
    model : FunnelDNN
        Model with loaded weights in eval mode.
    standardiser : Standardiser
        Reconstructed standardiser.
    metadata : dict
        Remaining checkpoint data (architecture, hyperparameters, metrics, etc.).
    """
    ckpt = torch.load(str(path), map_location=device, weights_only=False)

    arch = ckpt.get("architecture", {})
    model = FunnelDNN(
        input_dim=arch.get("input_dim", 22),
        hidden_layers=arch.get("hidden_layers", [256, 128, 64, 32, 16]),
        activation=arch.get("activation", "CELU"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model = model.to(device)

    std_data = ckpt["standardisation"]
    # Handle both list and ndarray formats
    if isinstance(std_data.get("X_mean"), np.ndarray):
        std_dict = {
            "X_mean": std_data["X_mean"].tolist(),
            "X_std": std_data["X_std"].tolist(),
            "y_mean": float(std_data["y_mean"]),
            "y_std": float(std_data["y_std"]),
        }
    else:
        std_dict = std_data
    standardiser = Standardiser.from_dict(std_dict)

    metadata = {
        k: v for k, v in ckpt.items()
        if k not in ("model_state_dict", "standardisation")
    }
    return model, standardiser, metadata


def export_mc_results(
    path: str | Path,
    df_mc: pd.DataFrame,
    y_mc: np.ndarray,
    y_augmented: np.ndarray | None = None,
    shap_values_11: np.ndarray | None = None,
) -> None:
    """Export MC profiles with predictions and SHAP values to CSV.

    Parameters
    ----------
    path : str or Path
        Output CSV path.
    df_mc : pd.DataFrame
        Raw MC profiles.
    y_mc : np.ndarray
        Base MC predictions.
    y_augmented : np.ndarray, optional
        Residual-augmented predictions.
    shap_values_11 : np.ndarray, optional
        Aggregated SHAP values (11 raw variables).
    """
    df_export = df_mc.copy()
    df_export["predicted_cost"] = y_mc
    if y_augmented is not None:
        df_export["predicted_cost_augmented"] = y_augmented
    if shap_values_11 is not None:
        for j, var in enumerate(RAW_VARIABLES):
            df_export[f"shap_{var}"] = shap_values_11[:, j]
    df_export.to_csv(str(path), index=False)


def export_shap_tables(
    path: str | Path,
    global_shap: pd.DataFrame,
    conditional_results: list | None = None,
    bootstrap_ci: pd.DataFrame | None = None,
) -> None:
    """Export SHAP importance tables to CSV.

    Parameters
    ----------
    path : str or Path
        Output CSV path.
    global_shap : pd.DataFrame
        Global SHAP importance table.
    conditional_results : list, optional
        List of ConditionalResult objects.
    bootstrap_ci : pd.DataFrame, optional
        Bootstrap CI table to merge.
    """
    table = global_shap.copy()
    if conditional_results is not None:
        for cond in conditional_results:
            name = cond.threshold_name
            for col in ["Cond Mean Signed", "Cond Mean |SHAP|"]:
                merge_col = f"{col} ({name})"
                lookup = dict(zip(cond.summary["Variable"], cond.summary[col]))
                table[merge_col] = table["Variable"].map(lookup)
    if bootstrap_ci is not None:
        for col in bootstrap_ci.columns:
            if col != "Variable":
                lookup = dict(zip(bootstrap_ci["Variable"], bootstrap_ci[col]))
                table[col] = table["Variable"].map(lookup)
    table.to_csv(str(path), index=False)
