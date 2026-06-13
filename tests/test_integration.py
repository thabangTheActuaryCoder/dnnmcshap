"""Integration tests for the dnnmcshap package."""

import numpy as np
import pytest
import torch

from dnnmcshap import (
    FunnelDNN,
    Standardiser,
    encode_profiles,
    estimate_marginals,
    evaluate_model,
    generate_mc_profiles,
    score_dnn,
)
from dnnmcshap._constants import SEED
from dnnmcshap.sensitivity import augment_predictions, compute_residuals
from dnnmcshap.shapley import aggregate_dummy_to_raw


class TestEndToEnd:
    """Test that the core workflow runs without errors on synthetic data."""

    def test_encode_standardise_score(self, synthetic_raw, synthetic_encoded,
                                      trained_small_model, fitted_standardiser):
        X, y = synthetic_encoded
        model = trained_small_model
        std = fitted_standardiser

        # Score should return array of correct length
        preds = score_dnn(model, X, std, device=torch.device("cpu"))
        assert preds.shape == (len(X),)
        assert not np.any(np.isnan(preds))

    def test_mc_generate_encode_score(self, synthetic_raw, trained_small_model,
                                      fitted_standardiser):
        df_raw = synthetic_raw.drop(columns=["charges"])
        marginals = estimate_marginals(df_raw)

        rng = np.random.default_rng(SEED)
        df_mc = generate_mc_profiles(20, marginals, rng=rng)
        X_mc = encode_profiles(df_mc)

        preds = score_dnn(
            trained_small_model, X_mc, fitted_standardiser,
            device=torch.device("cpu"),
        )
        assert preds.shape == (20,)
        assert not np.any(np.isnan(preds))

    def test_sensitivity_workflow(self, synthetic_encoded, trained_small_model,
                                  fitted_standardiser):
        X, y = synthetic_encoded
        preds = score_dnn(
            trained_small_model, X, fitted_standardiser,
            device=torch.device("cpu"),
        )
        residuals, sigma_e = compute_residuals(y, preds)
        assert abs(residuals.mean()) < 1e-3
        assert sigma_e > 0

        y_mc = preds[:10]
        result = augment_predictions(y_mc, residuals, sigma_e)
        assert result.y_augmented.shape == (10,)

    def test_evaluate_on_predictions(self, synthetic_encoded,
                                     trained_small_model, fitted_standardiser):
        X, y = synthetic_encoded
        preds = score_dnn(
            trained_small_model, X, fitted_standardiser,
            device=torch.device("cpu"),
        )
        result = evaluate_model(
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(preds, dtype=torch.float32),
        )
        assert result.rmse > 0
        assert result.mae > 0

    def test_shap_aggregation(self):
        """Verify that dummy-to-raw aggregation preserves total SHAP."""
        sv22 = np.random.randn(100, 22).astype(np.float32)
        sv11 = aggregate_dummy_to_raw(sv22)

        total_22 = sv22.sum(axis=1)
        total_11 = sv11.sum(axis=1)
        np.testing.assert_allclose(total_22, total_11, atol=1e-5)
