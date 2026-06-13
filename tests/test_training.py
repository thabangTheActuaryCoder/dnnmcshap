"""Tests for dnnmcshap.training."""

import torch
import pytest

from dnnmcshap._constants import SEED
from dnnmcshap.model import FunnelDNN
from dnnmcshap.training import TrainingResult, train_model


class TestTrainModel:
    def test_returns_training_result(self, synthetic_encoded, fitted_standardiser):
        X, y = synthetic_encoded
        std = fitted_standardiser
        X_std = std.transform_X(X)
        y_std = std.transform_y(y)

        X_t = torch.tensor(X_std, dtype=torch.float32)
        y_t = torch.tensor(y_std, dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(SEED)
        model = FunnelDNN(input_dim=22, hidden_layers=[16, 8], activation="CELU")

        result = train_model(
            model, X_t, y_t, X_t, y_t,
            lr=0.01, batch_size=32,
            max_epochs=10, patience=5, val_every=2,
            device=torch.device("cpu"),
        )

        assert isinstance(result, TrainingResult)
        assert result.best_epoch > 0
        assert len(result.train_losses) > 0
        assert len(result.val_epochs) > 0
        assert result.best_model_state is not None

    def test_early_stopping(self, synthetic_encoded, fitted_standardiser):
        """With tiny patience, training should stop early."""
        X, y = synthetic_encoded
        std = fitted_standardiser
        X_std = std.transform_X(X)
        y_std = std.transform_y(y)

        X_t = torch.tensor(X_std, dtype=torch.float32)
        y_t = torch.tensor(y_std, dtype=torch.float32).unsqueeze(1)

        torch.manual_seed(SEED)
        model = FunnelDNN(input_dim=22, hidden_layers=[16, 8])

        result = train_model(
            model, X_t, y_t, X_t, y_t,
            lr=0.01, batch_size=32,
            max_epochs=200, patience=2, val_every=1,
            device=torch.device("cpu"),
        )
        # Should stop before max epochs
        assert len(result.train_losses) < 200
