"""Input and response standardisation for DNN training.

Standardises features using training-set statistics only to prevent
data leakage. Zero-variance columns receive ``X_std = 1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StandardisationParams:
    """Stores the mean and standard deviation arrays for X and y."""

    X_mean: np.ndarray
    X_std: np.ndarray
    y_mean: float
    y_std: float


class Standardiser:
    """Fit/transform standardiser for model inputs and response.

    Fits on training data only and applies the same transformation
    to validation and test sets.
    """

    def __init__(self) -> None:
        self._params: StandardisationParams | None = None

    @property
    def params(self) -> StandardisationParams:
        """Return fitted parameters or raise if not yet fitted."""
        if self._params is None:
            raise RuntimeError("Standardiser has not been fitted yet.")
        return self._params

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "Standardiser":
        """Compute standardisation parameters from training data.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature array, shape ``(n, p)``.
        y_train : np.ndarray
            Training response array, shape ``(n,)``.

        Returns
        -------
        Standardiser
            ``self``, for method chaining.
        """
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        # Prevent division by zero for constant (e.g. binary) columns
        X_std[X_std == 0] = 1.0

        self._params = StandardisationParams(
            X_mean=X_mean,
            X_std=X_std,
            y_mean=float(y_train.mean()),
            y_std=float(y_train.std()),
        )
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Standardise feature array using fitted parameters."""
        p = self.params
        return ((X - p.X_mean) / p.X_std).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Standardise response array using fitted parameters."""
        p = self.params
        return ((y - p.y_mean) / p.y_std).astype(np.float32)

    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """De-normalise predictions back to original scale."""
        p = self.params
        return y_norm * p.y_std + p.y_mean

    def to_dict(self) -> dict[str, Any]:
        """Serialise parameters to a dictionary (for checkpoint saving)."""
        p = self.params
        return {
            "X_mean": p.X_mean.tolist(),
            "X_std": p.X_std.tolist(),
            "y_mean": p.y_mean,
            "y_std": p.y_std,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Standardiser":
        """Reconstruct a fitted Standardiser from a dictionary."""
        obj = cls()
        obj._params = StandardisationParams(
            X_mean=np.array(d["X_mean"], dtype=np.float32),
            X_std=np.array(d["X_std"], dtype=np.float32),
            y_mean=float(d["y_mean"]),
            y_std=float(d["y_std"]),
        )
        return obj
