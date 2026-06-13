"""Shared test fixtures for dnnmcshap tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dnnmcshap._constants import (
    COVERAGE_LEVEL_LEVELS,
    EXERCISE_FREQUENCY_LEVELS,
    FAMILY_MEDICAL_HISTORY_LEVELS,
    MEDICAL_HISTORY_LEVELS,
    OCCUPATION_LEVELS,
    REGION_LEVELS,
    SEED,
)
from dnnmcshap.encoding import encode_profiles
from dnnmcshap.model import FunnelDNN
from dnnmcshap.standardisation import Standardiser


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_raw(rng):
    """Synthetic raw dataset with 100 rows and 11 + charges columns."""
    n = 100
    df = pd.DataFrame({
        "age": rng.integers(18, 66, size=n),
        "gender": rng.choice(["male", "female"], size=n),
        "bmi": rng.uniform(18.0, 50.0, size=n),
        "children": rng.integers(0, 6, size=n),
        "smoker": rng.choice(["yes", "no"], size=n),
        "region": rng.choice(REGION_LEVELS, size=n),
        "medical_history": rng.choice(MEDICAL_HISTORY_LEVELS, size=n),
        "family_medical_history": rng.choice(
            FAMILY_MEDICAL_HISTORY_LEVELS, size=n
        ),
        "exercise_frequency": rng.choice(EXERCISE_FREQUENCY_LEVELS, size=n),
        "occupation": rng.choice(OCCUPATION_LEVELS, size=n),
        "coverage_level": rng.choice(COVERAGE_LEVEL_LEVELS, size=n),
    })
    # Simple additive charges formula for testing
    charges = (
        100.0 * df["age"].values
        + 200.0 * df["bmi"].values
        + 5000.0 * (df["smoker"] == "yes").astype(float).values
        + rng.normal(0, 500, size=n)
    ).astype(np.float32)
    df["charges"] = charges
    return df


@pytest.fixture
def synthetic_encoded(synthetic_raw):
    """Encoded features and target from synthetic data."""
    df = synthetic_raw.copy()
    X = encode_profiles(df.drop(columns=["charges"]))
    y = df["charges"].values.astype(np.float32)
    return X, y


@pytest.fixture
def fitted_standardiser(synthetic_encoded):
    """Standardiser fitted on synthetic data."""
    X, y = synthetic_encoded
    return Standardiser().fit(X, y)


@pytest.fixture
def small_model():
    """Small FunnelDNN for fast testing."""
    torch.manual_seed(SEED)
    return FunnelDNN(input_dim=22, hidden_layers=[16, 8], activation="CELU")


@pytest.fixture
def trained_small_model(small_model, synthetic_encoded, fitted_standardiser):
    """Quickly trained small model (5 epochs)."""
    X, y = synthetic_encoded
    std = fitted_standardiser

    X_std = std.transform_X(X)
    y_std = std.transform_y(y)

    X_t = torch.tensor(X_std, dtype=torch.float32)
    y_t = torch.tensor(y_std, dtype=torch.float32).unsqueeze(1)

    model = small_model
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for _ in range(5):
        optimiser.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimiser.step()

    model.eval()
    return model
