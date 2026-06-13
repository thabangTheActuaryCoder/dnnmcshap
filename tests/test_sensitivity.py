"""Tests for dnnmcshap.sensitivity."""

import numpy as np
import pytest

from dnnmcshap.sensitivity import augment_predictions, compute_residuals


class TestComputeResiduals:
    def test_centred_mean_zero(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])
        residuals, sigma_e = compute_residuals(y_true, y_pred)
        assert abs(residuals.mean()) < 1e-10
        assert sigma_e > 0

    def test_sigma_e_value(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        residuals, sigma_e = compute_residuals(y_true, y_pred)
        assert sigma_e == pytest.approx(0.0)


class TestAugmentPredictions:
    def test_output_shape(self):
        y_mc = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        residuals = np.array([-5.0, 5.0, -3.0, 3.0, 0.0])
        sigma_e = 4.0
        result = augment_predictions(y_mc, residuals, sigma_e)
        assert result.y_augmented.shape == y_mc.shape
        assert result.y_upper.shape == y_mc.shape
        assert result.y_lower.shape == y_mc.shape

    def test_perturbation_bands(self):
        y_mc = np.array([100.0, 200.0])
        residuals = np.array([0.0])
        sigma_e = 10.0
        result = augment_predictions(y_mc, residuals, sigma_e)
        np.testing.assert_allclose(result.y_upper, y_mc + 5.0)
        np.testing.assert_allclose(result.y_lower, y_mc - 5.0)
