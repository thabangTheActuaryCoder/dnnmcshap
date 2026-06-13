"""Tests for dnnmcshap.shapley."""

import numpy as np
import pytest
import torch

from dnnmcshap._constants import SEED
from dnnmcshap.shapley import aggregate_dummy_to_raw, sample_background


class TestAggregateDummyToRaw:
    def test_output_shape(self):
        sv22 = np.random.randn(50, 22).astype(np.float32)
        sv11 = aggregate_dummy_to_raw(sv22)
        assert sv11.shape == (50, 11)

    def test_continuous_pass_through(self):
        """Single-column variables (age, gender, bmi, children, smoker) pass through."""
        sv22 = np.zeros((10, 22), dtype=np.float32)
        sv22[:, 0] = 1.0  # age
        sv11 = aggregate_dummy_to_raw(sv22)
        np.testing.assert_allclose(sv11[:, 0], 1.0)

    def test_dummy_aggregation(self):
        """Region dummies (cols 5,6,7) should sum for raw variable index 5."""
        sv22 = np.zeros((10, 22), dtype=np.float32)
        sv22[:, 5] = 0.1
        sv22[:, 6] = 0.2
        sv22[:, 7] = 0.3
        sv11 = aggregate_dummy_to_raw(sv22)
        np.testing.assert_allclose(sv11[:, 5], 0.6, atol=1e-6)


class TestSampleBackground:
    def test_output_shape(self):
        X = np.random.randn(500, 22).astype(np.float32)
        bg = sample_background(X, n_background=100, seed=SEED)
        assert bg.shape == (100, 22)
        assert isinstance(bg, torch.Tensor)

    def test_deterministic(self):
        X = np.random.randn(200, 22).astype(np.float32)
        bg1 = sample_background(X, n_background=50, seed=42)
        bg2 = sample_background(X, n_background=50, seed=42)
        torch.testing.assert_close(bg1, bg2)
