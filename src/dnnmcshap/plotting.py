"""Publication-quality visualisation functions.

Every function returns a :class:`matplotlib.figure.Figure` and accepts
an optional *save_path* argument for persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats

from ._constants import ENCODED_FEATURES, RAW_VARIABLES, TAIL_THRESHOLDS


def _setup_style() -> None:
    sns.set_style("white")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.grid": False,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def _save_and_return(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_phase_comparison(
    val_epochs_dict: dict[str, list[int]],
    val_losses_dict: dict[str, list[float]],
    val_r2s_dict: dict[str, list[float]],
    phase_name: str,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot validation loss and R-squared curves for a hyperparameter phase."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label in val_epochs_dict:
        axes[0].plot(val_epochs_dict[label], val_losses_dict[label],
                     label=label, marker="o", markersize=3)
        axes[1].plot(val_epochs_dict[label], val_r2s_dict[label],
                     label=label, marker="o", markersize=3)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Loss (MSE)")
    axes[0].set_title(f"{phase_name}: Validation Loss")
    axes[0].legend()
    sns.despine(ax=axes[0])

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation R-squared")
    axes[1].set_title(f"{phase_name}: Validation R-squared")
    axes[1].legend()
    sns.despine(ax=axes[1])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_training_curves(
    train_losses: list[float],
    val_epochs: list[int],
    val_losses: list[float],
    train_r2s: list[float],
    val_r2s: list[float],
    best_epoch: int,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss/R-squared for the final model."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_epochs = range(1, len(train_losses) + 1)
    axes[0].plot(all_epochs, train_losses, label="Training Loss",
                 color="#2196F3", alpha=0.7)
    axes[0].plot(val_epochs, val_losses, label="Validation Loss",
                 color="#F44336", marker="o", markersize=4)
    axes[0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                    label=f"Best epoch (e*={best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    sns.despine(ax=axes[0])

    axes[1].plot(val_epochs, train_r2s, label="Training R-squared",
                 color="#2196F3", marker="o", markersize=4)
    axes[1].plot(val_epochs, val_r2s, label="Validation R-squared",
                 color="#F44336", marker="o", markersize=4)
    axes[1].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                    label=f"Best epoch (e*={best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R-squared")
    axes[1].set_title("Training and Validation R-squared")
    axes[1].legend()
    sns.despine(ax=axes[1])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_actual_vs_predicted(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    r2: float,
    set_name: str = "Test",
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Sorted line plot of actual vs predicted charges."""
    _setup_style()
    sort_idx = np.argsort(y_actual)
    y_act_sorted = y_actual[sort_idx]
    y_prd_sorted = y_predicted[sort_idx]

    n_plot = min(10000, len(y_act_sorted))
    step = max(1, len(y_act_sorted) // n_plot)
    plot_idx = np.arange(0, len(y_act_sorted), step)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_idx, y_act_sorted[plot_idx], color="tab:blue", linewidth=1.2,
            label="Actual Charges", alpha=0.8)
    ax.plot(plot_idx, y_prd_sorted[plot_idx], color="tab:orange", linewidth=1.2,
            label="Predicted Charges", alpha=0.8)
    ax.set_xlabel("Observation Index (sorted by actual charges)")
    ax.set_ylabel("Charges (R)")
    ax.set_title(f"Actual vs Predicted - {set_name} Set (R-squared = {r2:.4f})")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_residual_diagnostics(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    *,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Four-panel residual diagnostics (residuals vs fitted, histogram, QQ, order)."""
    _setup_style()
    residuals = y_actual - y_predicted
    fitted = y_predicted

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    n_plot = min(10000, len(residuals))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(residuals), n_plot, replace=False)

    # (a) Residuals vs Fitted
    axes[0, 0].scatter(fitted[idx], residuals[idx], alpha=0.3, s=5, color="#2196F3")
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Fitted Values (R)")
    axes[0, 0].set_ylabel("Residuals (R)")
    axes[0, 0].set_title("(a) Residuals vs Fitted Values")
    sns.despine(ax=axes[0, 0])

    # (b) Histogram
    axes[0, 1].hist(residuals, bins=80, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Residual (R)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("(b) Distribution of Residuals")
    sns.despine(ax=axes[0, 1])

    # (c) QQ Plot
    std_res = (residuals - residuals.mean()) / residuals.std()
    qq_sample = rng.choice(std_res, min(5000, len(std_res)), replace=False)
    qq_sample.sort()
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(qq_sample)))
    axes[1, 0].scatter(theoretical, qq_sample, alpha=0.3, s=5, color="#2196F3")
    axes[1, 0].plot([-4, 4], [-4, 4], "r--", linewidth=1)
    axes[1, 0].set_xlabel("Theoretical Quantiles")
    axes[1, 0].set_ylabel("Sample Quantiles")
    axes[1, 0].set_title("(c) Q-Q Plot of Residuals")
    sns.despine(ax=axes[1, 0])

    # (d) Residuals vs Order
    idx_seq = rng.choice(len(residuals), n_plot, replace=False)
    idx_seq.sort()
    axes[1, 1].scatter(idx_seq, residuals[idx_seq], alpha=0.3, s=5, color="#2196F3")
    axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes[1, 1].set_xlabel("Observation Index")
    axes[1, 1].set_ylabel("Residual (R)")
    axes[1, 1].set_title("(d) Residuals vs Observation Order")
    sns.despine(ax=axes[1, 1])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_violin_distributions(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    r2: float,
    set_name: str = "Test",
    *,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Violin plot of actual vs predicted charge distributions."""
    _setup_style()
    rng = np.random.default_rng(seed)
    n_violin = min(50000, len(y_actual))
    v_idx = rng.choice(len(y_actual), n_violin, replace=False)

    df_violin = pd.DataFrame({
        "Charges (R)": np.concatenate([y_actual[v_idx], y_predicted[v_idx]]),
        "Type": ["Actual"] * len(v_idx) + ["Predicted"] * len(v_idx),
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(data=df_violin, x="Type", y="Charges (R)", ax=ax,
                   palette=["tab:blue", "tab:orange"], inner="quartile", cut=0)
    ax.set_title(f"{set_name} Set (R-squared = {r2:.4f})")
    ax.set_xlabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_pca_diagnostics(
    pc_coords: np.ndarray,
    colour_values: np.ndarray,
    explained_variance: tuple[float, float],
    title: str,
    cbar_label: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """PCA scatter plot coloured by target or predicted values."""
    _setup_style()
    ev1, ev2 = explained_variance

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        pc_coords[:, 0], pc_coords[:, 1],
        c=colour_values, cmap="viridis", s=6, alpha=0.5,
        vmin=vmin, vmax=vmax, edgecolors="none",
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(f"PC1 ({ev1:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev2:.1f}% var)")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_correlation_heatmaps(
    corr_matrix: pd.DataFrame,
    title: str,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Correlation heatmap with viridis colourmap."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix.values, cmap="viridis", vmin=-1, vmax=1,
                   aspect="auto")
    n = len(corr_matrix)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_matrix.index, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_mc_convergence(
    df_conv: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """MC convergence diagnostics across pilot sizes."""
    _setup_style()
    stat_cols = ["Mean", "SD", "P50", "P90", "P95", "P99"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for idx_s, stat_name in enumerate(stat_cols):
        ax = axes[idx_s // 3, idx_s % 3]
        ax.plot(df_conv["B"], df_conv[stat_name], marker="o", color="#2196F3")
        ax.set_xlabel("B (sample size)")
        ax.set_ylabel(stat_name)
        ax.set_title(f"MC Convergence: {stat_name}")
        ax.set_xscale("log")
        sns.despine(ax=ax)

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_mc_distribution(
    y_mc: np.ndarray,
    *,
    y_train: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram and ECDF of MC predicted costs."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(y_mc, bins=80, color="#2196F3", edgecolor="white", alpha=0.7,
                 density=True)
    axes[0].set_xlabel("Predicted Cost (R)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("MC Predicted Cost Distribution")
    sns.despine(ax=axes[0])

    sorted_y = np.sort(y_mc)
    ecdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    axes[1].plot(sorted_y, ecdf, color="#2196F3", linewidth=1.5)
    axes[1].set_xlabel("Predicted Cost (R)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Empirical CDF of MC Predicted Costs")
    sns.despine(ax=axes[1])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_sensitivity(
    y_mc: np.ndarray,
    y_augmented: np.ndarray,
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Overlay histograms of base, augmented, and perturbation bands."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(y_mc, bins=80, alpha=0.5, label="Base MC", color="#2196F3",
            density=True)
    ax.hist(y_augmented, bins=80, alpha=0.5, label="Augmented", color="#F44336",
            density=True)
    ax.axvline(np.median(y_upper), color="green", linestyle="--",
               label="+0.5 sigma_e")
    ax.axvline(np.median(y_lower), color="orange", linestyle="--",
               label="-0.5 sigma_e")
    ax.set_xlabel("Predicted Cost (R)")
    ax.set_ylabel("Density")
    ax.set_title("Residual-Augmented MC Sensitivity")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_shap_global(
    df_global: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of global SHAP importance."""
    _setup_style()
    df_plot = df_global.sort_values("Mean |SHAP|", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(df_plot["Variable"], df_plot["Mean |SHAP|"],
                 color="#2196F3", edgecolor="white")
    axes[0].set_xlabel("Mean |SHAP|")
    axes[0].set_title("Global SHAP Importance (|SHAP|)")
    sns.despine(ax=axes[0])

    axes[1].barh(df_plot["Variable"], df_plot["Mean Signed SHAP"],
                 color="#F44336", edgecolor="white")
    axes[1].axvline(x=0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Mean Signed SHAP")
    axes[1].set_title("Global SHAP Direction (Signed)")
    sns.despine(ax=axes[1])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_shap_beeswarm(
    shap_values_22: np.ndarray,
    X_foreground_std: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """SHAP beeswarm plot on the 22 encoded features."""
    _setup_style()
    if feature_names is None:
        feature_names = list(ENCODED_FEATURES)

    explanation = shap.Explanation(
        values=shap_values_22,
        base_values=np.full(len(shap_values_22), 0.0),
        data=X_foreground_std,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    shap.plots.beeswarm(explanation, max_display=22, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_conditional_shap(
    conditional_results: list,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar charts of conditional SHAP importance at each threshold."""
    _setup_style()
    n_thresholds = len(conditional_results)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(6 * n_thresholds, 6))
    if n_thresholds == 1:
        axes = [axes]

    for idx, cond in enumerate(conditional_results):
        df_plot = cond.summary.sort_values("Cond Mean |SHAP|", ascending=True)
        axes[idx].barh(df_plot["Variable"], df_plot["Cond Mean |SHAP|"],
                       color="#2196F3", edgecolor="white")
        axes[idx].set_xlabel("Cond Mean |SHAP|")
        axes[idx].set_title(f"{cond.threshold_name} (n={cond.n_high_cost})")
        sns.despine(ax=axes[idx])

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_shap_waterfall(
    shap_values_profile: np.ndarray,
    phi_0: float,
    feature_values: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    title: str = "SHAP Waterfall",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """SHAP waterfall plot for a single profile."""
    _setup_style()
    if feature_names is None:
        feature_names = list(RAW_VARIABLES)

    explanation = shap.Explanation(
        values=shap_values_profile,
        base_values=phi_0,
        data=feature_values,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=11, show=False)
    fig = plt.gcf()
    plt.title(title)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_bootstrap_ci(
    ci_result: Any,
    *,
    top_n: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of bootstrap distributions for top features."""
    _setup_style()
    df = ci_result.ci_table
    top_vars = df["Variable"].head(top_n).tolist()

    n_cols = min(top_n, 2)
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).flatten() if top_n > 1 else [axes]

    for i, var in enumerate(top_vars):
        var_idx = RAW_VARIABLES.index(var)
        ax = axes_flat[i]
        ax.hist(ci_result.boot_means[:, var_idx], bins=50,
                color="#2196F3", edgecolor="white", alpha=0.8)
        row = df[df["Variable"] == var].iloc[0]
        ci_lo_col = [c for c in df.columns if "CI Lower" in c][0]
        ci_hi_col = [c for c in df.columns if "CI Upper" in c][0]
        ax.axvline(row[ci_lo_col], color="red", linestyle="--")
        ax.axvline(row[ci_hi_col], color="red", linestyle="--")
        ax.set_title(f"{var}")
        ax.set_xlabel("Bootstrap Mean |SHAP|")
        sns.despine(ax=ax)

    for j in range(len(top_vars), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return _save_and_return(fig, save_path)


def plot_hyperparameter_scatter(
    config_labels: list[str],
    config_val_losses: list[float],
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of validation loss across all hyperparameter configurations."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(16, 7))

    x_pos = list(range(len(config_labels)))
    ax.scatter(x_pos, config_val_losses, color="blue", s=80, zorder=5)

    min_loss = min(config_val_losses)
    min_idx = config_val_losses.index(min_loss)
    ax.axhline(y=min_loss, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=min_idx, color="black", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Best Parameters")
    ax.set_ylabel("Error")
    ax.set_title("Validation Loss")
    sns.despine(ax=ax)
    fig.tight_layout()
    return _save_and_return(fig, save_path)
