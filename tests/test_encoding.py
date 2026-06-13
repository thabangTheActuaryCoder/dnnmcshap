"""Tests for dnnmcshap.encoding."""

import numpy as np
import pandas as pd
import pytest

from dnnmcshap.encoding import encode_dataframe, encode_profiles


class TestEncodeProfiles:
    def test_output_shape(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        result = encode_profiles(df)
        assert result.shape == (len(df), 22)

    def test_dtype_float32(self, synthetic_raw):
        df = synthetic_raw.drop(columns=["charges"])
        result = encode_profiles(df)
        assert result.dtype == np.float32

    def test_binary_encoding_gender(self):
        df = pd.DataFrame({
            "age": [30], "gender": ["male"], "bmi": [25.0],
            "children": [1], "smoker": ["no"], "region": ["northeast"],
            "medical_history": ["None"], "family_medical_history": ["None"],
            "exercise_frequency": ["Rarely"], "occupation": ["Unemployed"],
            "coverage_level": ["Basic"],
        })
        X = encode_profiles(df)
        assert X[0, 1] == 1.0  # gender = male

        df["gender"] = "female"
        X = encode_profiles(df)
        assert X[0, 1] == 0.0  # gender = female

    def test_reference_category_all_zeros(self):
        """When all categoricals are reference, dummy columns should be 0."""
        df = pd.DataFrame({
            "age": [30], "gender": ["female"], "bmi": [25.0],
            "children": [1], "smoker": ["no"], "region": ["northeast"],
            "medical_history": ["None"], "family_medical_history": ["None"],
            "exercise_frequency": ["Rarely"], "occupation": ["Unemployed"],
            "coverage_level": ["Basic"],
        })
        X = encode_profiles(df)
        # All dummy columns (indices 5-21) should be 0
        assert np.all(X[0, 5:] == 0.0)

    def test_region_dummies(self):
        df = pd.DataFrame({
            "age": [30], "gender": ["male"], "bmi": [25.0],
            "children": [0], "smoker": ["no"], "region": ["southwest"],
            "medical_history": ["None"], "family_medical_history": ["None"],
            "exercise_frequency": ["Rarely"], "occupation": ["Unemployed"],
            "coverage_level": ["Basic"],
        })
        X = encode_profiles(df)
        assert X[0, 5] == 1.0  # southwest
        assert X[0, 6] == 0.0  # northwest
        assert X[0, 7] == 0.0  # southeast


class TestEncodeDataframe:
    def test_output_has_charges(self, synthetic_raw):
        result = encode_dataframe(synthetic_raw)
        assert "charges" in result.columns

    def test_output_columns(self, synthetic_raw):
        result = encode_dataframe(synthetic_raw)
        assert len(result.columns) == 23  # 22 features + charges

    def test_row_count(self, synthetic_raw):
        result = encode_dataframe(synthetic_raw)
        assert len(result) == len(synthetic_raw)
