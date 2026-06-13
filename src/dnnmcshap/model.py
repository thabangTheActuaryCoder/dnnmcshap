"""FunnelDNN architecture and scoring utilities.

Implements the bias-free funnel architecture
(22 -> 256 -> 128 -> 64 -> 32 -> 16 -> 1) used for health insurance
cost prediction.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ._constants import HIDDEN_LAYERS, INPUT_DIM
from .standardisation import Standardiser


class FunnelDNN(nn.Module):
    """Feed-forward DNN with funnel (decreasing-width) architecture.

    All linear layers are bias-free following the dissertation specification.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 22).
    hidden_layers : list[int]
        Widths of hidden layers (default ``[256, 128, 64, 32, 16]``).
    activation : str
        Activation function name: ``"CELU"``, ``"GELU"`` or ``"Tanh"``.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_layers: list[int] | None = None,
        activation: str = "CELU",
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = list(HIDDEN_LAYERS)

        act_map = {
            "CELU": nn.CELU(),
            "GELU": nn.GELU(),
            "Tanh": nn.Tanh(),
        }
        if activation not in act_map:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(act_map.keys())}."
            )
        act_fn = act_map[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim, bias=False))
            layers.append(copy.deepcopy(act_fn))
            prev_dim = h_dim
        # Output layer (linear, no activation, no bias)
        layers.append(nn.Linear(prev_dim, 1, bias=False))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def auto_device() -> torch.device:
    """Select the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_dnn(
    model: FunnelDNN,
    X_encoded_raw: np.ndarray,
    standardiser: Standardiser,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    """Score profiles through the DNN and return de-normalised predictions.

    Parameters
    ----------
    model : FunnelDNN
        Trained model in eval mode.
    X_encoded_raw : np.ndarray
        Reference-category encoded features, shape ``(n, 22)``.
    standardiser : Standardiser
        Fitted standardiser for input and output transformations.
    device : torch.device, optional
        Device to run inference on. Defaults to model's current device.

    Returns
    -------
    np.ndarray
        De-normalised predictions, shape ``(n,)``.
    """
    if device is None:
        device = next(model.parameters()).device

    X_std = standardiser.transform_X(X_encoded_raw)
    X_t = torch.tensor(X_std, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        preds_norm = model(X_t).cpu().numpy().flatten()

    return standardiser.inverse_transform_y(preds_norm)
