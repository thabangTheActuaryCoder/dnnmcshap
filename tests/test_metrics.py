"""Tests for dnnmcshap.metrics."""

import pytest
import torch

from dnnmcshap.metrics import evaluate_model, mae, r2_dot, r2_score, rmse


class TestMetrics:
    def test_r2_perfect_prediction(self):
        y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_zero_variance(self):
        y = torch.tensor([5.0, 5.0, 5.0])
        p = torch.tensor([5.0, 5.0, 5.0])
        assert r2_score(y, p) == 0.0

    def test_r2_dot_perfect(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert r2_dot(y, y) == pytest.approx(1.0, abs=1e-5)

    def test_rmse_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_positive(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor([1.5, 2.5, 3.5])
        assert rmse(y, p) > 0

    def test_mae_zero(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor([2.0, 3.0, 4.0])
        assert mae(y, p) == pytest.approx(1.0)


class TestEvaluateModel:
    def test_returns_result(self):
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        p = torch.tensor([1.1, 2.1, 2.9, 4.1])
        result = evaluate_model(y, p)
        assert result.r2 > 0.9
        assert result.rmse > 0
        assert result.mae > 0
