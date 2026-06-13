"""Tests for dnnmcshap.standardisation."""

import numpy as np
import pytest

from dnnmcshap.standardisation import Standardiser


class TestStandardiser:
    def test_fit_sets_params(self, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)
        assert std.is_fitted
        assert std.params.X_mean.shape == (22,)
        assert std.params.X_std.shape == (22,)

    def test_not_fitted_raises(self):
        std = Standardiser()
        with pytest.raises(RuntimeError):
            std.params

    def test_transform_X_zero_mean(self, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)
        X_std = std.transform_X(X)
        # Mean of standardised should be near 0
        assert np.allclose(X_std.mean(axis=0), 0.0, atol=0.01)

    def test_transform_y_zero_mean(self, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)
        y_std = std.transform_y(y)
        assert abs(y_std.mean()) < 0.01

    def test_inverse_transform_y(self, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)
        y_std = std.transform_y(y)
        y_back = std.inverse_transform_y(y_std)
        assert np.allclose(y_back, y, atol=0.1)

    def test_zero_variance_handling(self):
        """Columns with zero variance should get std=1.0."""
        X = np.array([[1, 0], [1, 1], [1, 0]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        std = Standardiser().fit(X, y)
        assert std.params.X_std[0] == 1.0  # constant column

    def test_to_dict_from_dict_roundtrip(self, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)
        d = std.to_dict()
        std2 = Standardiser.from_dict(d)
        assert np.allclose(std.params.X_mean, std2.params.X_mean)
        assert np.allclose(std.params.X_std, std2.params.X_std)
        assert std.params.y_mean == pytest.approx(std2.params.y_mean)
        assert std.params.y_std == pytest.approx(std2.params.y_std)
