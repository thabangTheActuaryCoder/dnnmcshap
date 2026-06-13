"""Evaluation metrics for DNN predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Standard R-squared: ``1 - SS_res / SS_tot``."""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return (1 - ss_res / ss_tot).item()


def r2_dot(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Dot-product R-squared (squared correlation).

    Used during training on normalised data for computational speed.
    """
    num = torch.dot(y_true.squeeze(), y_pred.squeeze()) ** 2
    den = torch.dot(y_true.squeeze(), y_true.squeeze()) * torch.dot(
        y_pred.squeeze(), y_pred.squeeze()
    )
    if den == 0:
        return 0.0
    return (num / den).item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred)).item()


@dataclass
class EvaluationResult:
    """Container for model evaluation metrics on a single dataset split."""

    r2: float
    rmse: float
    mae: float


def evaluate_model(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> EvaluationResult:
    """Compute R-squared, RMSE and MAE in one call.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth values (de-normalised).
    y_pred : torch.Tensor
        Model predictions (de-normalised).

    Returns
    -------
    EvaluationResult
    """
    return EvaluationResult(
        r2=r2_score(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
    )
