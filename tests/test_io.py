"""Tests for dnnmcshap.io."""

import numpy as np
import pandas as pd
import pytest
import torch

from dnnmcshap._constants import SEED
from dnnmcshap.io import (
    export_mc_results,
    export_shap_tables,
    load_checkpoint,
    save_checkpoint,
)
from dnnmcshap.model import FunnelDNN
from dnnmcshap.standardisation import Standardiser


class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path, synthetic_encoded):
        X, y = synthetic_encoded
        std = Standardiser().fit(X, y)

        torch.manual_seed(SEED)
        model = FunnelDNN(input_dim=22, hidden_layers=[16, 8], activation="CELU")

        path = tmp_path / "test_model.pth"
        save_checkpoint(
            path, model, std,
            architecture={
                "input_dim": 22,
                "hidden_layers": [16, 8],
                "activation": "CELU",
            },
            hyperparameters={"lr": 0.001},
            metrics={"test_r2": 0.95},
        )

        model2, std2, meta = load_checkpoint(path)

        assert isinstance(model2, FunnelDNN)
        assert std2.is_fitted
        assert "hyperparameters" in meta
        assert meta["hyperparameters"]["lr"] == 0.001

        # Predictions should match
        x = torch.randn(5, 22)
        model.eval()
        model2.eval()
        with torch.no_grad():
            p1 = model(x)
            p2 = model2(x)
        torch.testing.assert_close(p1, p2)


class TestExport:
    def test_export_mc_results(self, tmp_path):
        df_mc = pd.DataFrame({
            "age": [30, 40, 50],
            "gender": ["male", "female", "male"],
            "bmi": [25.0, 30.0, 22.0],
            "children": [1, 2, 0],
            "smoker": ["no", "yes", "no"],
            "region": ["northeast", "southwest", "northwest"],
            "medical_history": ["None", "Diabetes", "None"],
            "family_medical_history": ["None", "None", "Heart disease"],
            "exercise_frequency": ["Rarely", "Frequently", "Never"],
            "occupation": ["Student", "White collar", "Unemployed"],
            "coverage_level": ["Basic", "Premium", "Standard"],
        })
        y_mc = np.array([5000.0, 8000.0, 3000.0])
        path = tmp_path / "mc_export.csv"
        export_mc_results(path, df_mc, y_mc)
        df_loaded = pd.read_csv(path)
        assert "predicted_cost" in df_loaded.columns
        assert len(df_loaded) == 3

    def test_export_shap_tables(self, tmp_path):
        df_global = pd.DataFrame({
            "Variable": ["age", "smoker"],
            "Mean Signed SHAP": [0.1, 0.5],
            "Mean |SHAP|": [0.2, 0.6],
        })
        path = tmp_path / "shap_table.csv"
        export_shap_tables(path, df_global)
        df_loaded = pd.read_csv(path)
        assert len(df_loaded) == 2
