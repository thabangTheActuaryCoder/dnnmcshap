"""Tests for dnnmcshap.montecarlo."""

import numpy as np
import pytest

from dnnmcshap._constants import (
    AGE_MAX,
    AGE_MIN,
    BMI_MAX,
    BMI_MIN,
    CHILDREN_MAX,
    CHILDREN_MIN,
)
from dnnmcshap.montecarlo import (
    MarginalParams,
    estimate_marginals,
    generate_mc_profiles,
)


class TestEstimateMarginals:
    def test_returns_marginal_params(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        assert isinstance(params, MarginalParams)
        assert 0 < params.p_male < 1
        assert 0 < params.p_smoker < 1

    def test_probabilities_sum_to_one(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        assert params.region_probs.sum() == pytest.approx(1.0)
        assert params.med_hist_probs.sum() == pytest.approx(1.0)
        assert params.cov_probs.sum() == pytest.approx(1.0)


class TestGenerateMCProfiles:
    def test_output_shape(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        df_mc = generate_mc_profiles(50, params, rng=np.random.default_rng(42))
        assert len(df_mc) == 50
        assert set(df_mc.columns) == {
            "age", "gender", "bmi", "children", "smoker",
            "region", "medical_history", "family_medical_history",
            "exercise_frequency", "occupation", "coverage_level",
        }

    def test_age_bounds(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        df_mc = generate_mc_profiles(1000, params, rng=np.random.default_rng(42))
        assert df_mc["age"].min() >= AGE_MIN
        assert df_mc["age"].max() <= AGE_MAX

    def test_bmi_bounds(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        df_mc = generate_mc_profiles(1000, params, rng=np.random.default_rng(42))
        assert df_mc["bmi"].min() >= BMI_MIN
        assert df_mc["bmi"].max() <= BMI_MAX

    def test_children_bounds(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        df_mc = generate_mc_profiles(1000, params, rng=np.random.default_rng(42))
        assert df_mc["children"].min() >= CHILDREN_MIN
        assert df_mc["children"].max() <= CHILDREN_MAX

    def test_deterministic_with_seed(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        params = estimate_marginals(df)
        df1 = generate_mc_profiles(10, params, rng=np.random.default_rng(99))
        df2 = generate_mc_profiles(10, params, rng=np.random.default_rng(99))
        assert df1.equals(df2)
