"""Tests for dnnmcshap.pipeline."""

import numpy as np
import pytest
import torch

from dnnmcshap.pipeline import PipelineConfig, PipelineResult, run_pipeline


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.mc_samples == 100_000
        assert config.activation == "CELU"
        assert config.seed == 42


class TestRunPipeline:
    @pytest.mark.slow
    def test_small_pipeline(self, synthetic_raw, synthetic_encoded):
        """Run a small pipeline on synthetic data (slow test)."""
        X, y = synthetic_encoded
        n = len(X)
        split = int(0.7 * n)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        df_raw = synthetic_raw.drop(columns=["charges"])

        config = PipelineConfig(
            activation="CELU",
            lr=0.01,
            batch_size=32,
            max_epochs=5,
            patience=3,
            val_every=1,
            mc_samples=50,
            shap_background=20,
            shap_batch=25,
            bootstrap_r=50,
        )

        result = run_pipeline(
            X_train, y_train, X_val, y_val, df_raw,
            config=config,
            device=torch.device("cpu"),
            verbose=False,
        )

        assert isinstance(result, PipelineResult)
        assert len(result.y_mc) == 50
        assert result.shap_result.shap_values_11.shape == (50, 11)
        assert len(result.conditional_results) == 3
        assert result.global_shap is not None
