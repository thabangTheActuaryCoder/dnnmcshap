"""Tests for dnnmcshap.plotting."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from dnnmcshap.plotting import (
    plot_actual_vs_predicted,
    plot_mc_convergence,
    plot_mc_distribution,
    plot_residual_diagnostics,
    plot_training_curves,
)


class TestPlotFunctions:
    def test_plot_training_curves(self):
        fig = plot_training_curves(
            train_losses=[1.0, 0.8, 0.6, 0.5, 0.4],
            val_epochs=[1, 3, 5],
            val_losses=[1.1, 0.7, 0.45],
            train_r2s=[0.5, 0.7, 0.85],
            val_r2s=[0.4, 0.65, 0.8],
            best_epoch=5,
        )
        assert isinstance(fig, plt.Figure)

    def test_plot_actual_vs_predicted(self):
        y_act = np.random.randn(100).astype(np.float64) * 1000 + 5000
        y_pred = y_act + np.random.randn(100) * 100
        fig = plot_actual_vs_predicted(y_act, y_pred, r2=0.95, set_name="Test")
        assert isinstance(fig, plt.Figure)

    def test_plot_residual_diagnostics(self):
        y_act = np.random.randn(200).astype(np.float64) * 1000 + 5000
        y_pred = y_act + np.random.randn(200) * 100
        fig = plot_residual_diagnostics(y_act, y_pred)
        assert isinstance(fig, plt.Figure)

    def test_plot_mc_distribution(self):
        y_mc = np.random.randn(500).astype(np.float64) * 1000 + 5000
        fig = plot_mc_distribution(y_mc)
        assert isinstance(fig, plt.Figure)

    def test_plot_mc_convergence(self):
        df = pd.DataFrame({
            "B": [100, 500, 1000],
            "Mean": [5000, 5100, 5050],
            "SD": [1000, 990, 995],
            "P50": [4800, 4900, 4850],
            "P90": [6500, 6400, 6450],
            "P95": [7000, 6900, 6950],
            "P99": [8000, 7900, 7950],
        })
        fig = plot_mc_convergence(df)
        assert isinstance(fig, plt.Figure)

    def test_save_path(self, tmp_path):
        y_mc = np.random.randn(100) * 1000 + 5000
        path = tmp_path / "test_fig.png"
        fig = plot_mc_distribution(y_mc, save_path=str(path))
        assert path.exists()
