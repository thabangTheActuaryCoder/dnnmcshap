"""Tests for dnnmcshap.model."""

import numpy as np
import pytest
import torch

from dnnmcshap.model import FunnelDNN, auto_device, score_dnn


class TestFunnelDNN:
    def test_default_architecture(self):
        model = FunnelDNN()
        assert model.network is not None
        # Should have 5 hidden + 1 output = 6 linear layers
        linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 6

    def test_bias_free(self):
        model = FunnelDNN()
        for module in model.network:
            if isinstance(module, torch.nn.Linear):
                assert module.bias is None

    def test_forward_shape(self):
        model = FunnelDNN()
        x = torch.randn(10, 22)
        out = model(x)
        assert out.shape == (10, 1)

    def test_custom_hidden_layers(self):
        model = FunnelDNN(hidden_layers=[8, 4])
        linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3  # 2 hidden + 1 output

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unsupported activation"):
            FunnelDNN(activation="ReLU")

    def test_all_activations(self):
        for act in ["CELU", "GELU", "Tanh"]:
            model = FunnelDNN(activation=act)
            x = torch.randn(5, 22)
            out = model(x)
            assert out.shape == (5, 1)


class TestAutoDevice:
    def test_returns_device(self):
        device = auto_device()
        assert isinstance(device, torch.device)


class TestScoreDNN:
    def test_score_returns_array(self, trained_small_model, synthetic_encoded,
                                 fitted_standardiser):
        X, _ = synthetic_encoded
        preds = score_dnn(trained_small_model, X, fitted_standardiser,
                          device=torch.device("cpu"))
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))
