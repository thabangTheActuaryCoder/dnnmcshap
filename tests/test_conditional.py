"""Tests for dnnmcshap.conditional."""

import numpy as np
import pytest

from dnnmcshap._constants import RAW_VARIABLES
from dnnmcshap.conditional import (
    BootstrapCIResult,
    ConditionalResult,
    bootstrap_conditional_ci,
    compute_conditional_shap,
)


class TestComputeConditionalShap:
    def test_returns_list(self):
        sv11 = np.random.randn(1000, 11).astype(np.float32)
        y_mc = np.random.randn(1000).astype(np.float64) * 1000 + 5000
        results = compute_conditional_shap(sv11, y_mc)
        assert isinstance(results, list)
        assert len(results) == 3  # tau_90, tau_95, tau_99

    def test_conditional_result_fields(self):
        sv11 = np.random.randn(1000, 11).astype(np.float32)
        y_mc = np.random.randn(1000).astype(np.float64) * 1000 + 5000
        results = compute_conditional_shap(sv11, y_mc)
        for r in results:
            assert isinstance(r, ConditionalResult)
            assert r.n_high_cost > 0
            assert len(r.summary) == 11
            assert "Variable" in r.summary.columns

    def test_custom_thresholds(self):
        sv11 = np.random.randn(500, 11).astype(np.float32)
        y_mc = np.random.randn(500).astype(np.float64) * 1000 + 5000
        results = compute_conditional_shap(
            sv11, y_mc, thresholds={"tau_50": 0.50}
        )
        assert len(results) == 1
        assert results[0].threshold_name == "tau_50"


class TestBootstrapConditionalCI:
    def test_returns_result(self):
        sv11 = np.random.randn(500, 11).astype(np.float32)
        y_mc = np.random.randn(500).astype(np.float64) * 1000 + 5000
        result = bootstrap_conditional_ci(
            sv11, y_mc, n_resamples=100, seed=42
        )
        assert isinstance(result, BootstrapCIResult)
        assert result.n_resamples == 100
        assert len(result.ci_table) == 11

    def test_ci_lower_less_than_upper(self):
        sv11 = np.random.randn(500, 11).astype(np.float32)
        y_mc = np.random.randn(500).astype(np.float64) * 1000 + 5000
        result = bootstrap_conditional_ci(
            sv11, y_mc, n_resamples=200, seed=42
        )
        ci_lo = result.ci_table.iloc[:, 2].values
        ci_hi = result.ci_table.iloc[:, 3].values
        assert np.all(ci_lo <= ci_hi)
