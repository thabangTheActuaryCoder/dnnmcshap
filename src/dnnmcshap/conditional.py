"""Conditional Shapley analysis at tail thresholds with bootstrap CIs.

Computes conditional mean SHAP values for high-cost subsets and
nonparametric bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._constants import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_R,
    RAW_VARIABLES,
    SEED,
    TAIL_THRESHOLDS,
)


@dataclass
class ConditionalResult:
    """Conditional SHAP summary for a single threshold."""

    threshold_name: str
    tau: float
    u_value: float
    n_high_cost: int
    summary: pd.DataFrame


@dataclass
class BootstrapCIResult:
    """Bootstrap confidence interval results."""

    threshold_name: str
    n_samples: int
    n_resamples: int
    alpha: float
    ci_table: pd.DataFrame
    boot_means: np.ndarray


def compute_conditional_shap(
    shap_values_11: np.ndarray,
    y_mc: np.ndarray,
    *,
    thresholds: dict[str, float] | None = None,
) -> list[ConditionalResult]:
    """Compute conditional SHAP summaries at tail thresholds.

    Parameters
    ----------
    shap_values_11 : np.ndarray
        Aggregated SHAP values, shape ``(B, 11)``.
    y_mc : np.ndarray
        MC predicted costs, shape ``(B,)``.
    thresholds : dict, optional
        Threshold names to quantile levels. Defaults to
        ``TAIL_THRESHOLDS``.

    Returns
    -------
    list[ConditionalResult]
    """
    if thresholds is None:
        thresholds = dict(TAIL_THRESHOLDS)

    results = []
    for name, tau in thresholds.items():
        u = np.percentile(y_mc, tau * 100)
        mask = y_mc >= u
        sv_cond = shap_values_11[mask]

        cond_mean_signed = sv_cond.mean(axis=0)
        cond_mean_abs = np.abs(sv_cond).mean(axis=0)

        df = pd.DataFrame({
            "Variable": RAW_VARIABLES,
            "Cond Mean Signed": cond_mean_signed,
            "Cond Mean |SHAP|": cond_mean_abs,
        }).sort_values("Cond Mean |SHAP|", ascending=False).reset_index(drop=True)

        results.append(ConditionalResult(
            threshold_name=name,
            tau=tau,
            u_value=float(u),
            n_high_cost=int(mask.sum()),
            summary=df,
        ))

    return results


def bootstrap_conditional_ci(
    shap_values_11: np.ndarray,
    y_mc: np.ndarray,
    *,
    threshold_name: str = "tau_99",
    tau: float = 0.99,
    n_resamples: int = DEFAULT_BOOTSTRAP_R,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
    seed: int = SEED,
) -> BootstrapCIResult:
    """Compute bootstrap confidence intervals for conditional mean |SHAP|.

    Uses nonparametric percentile bootstrap with *n_resamples* resamples
    of the high-cost subset profiles.

    Parameters
    ----------
    shap_values_11 : np.ndarray
        Aggregated SHAP values, shape ``(B, 11)``.
    y_mc : np.ndarray
        MC predicted costs, shape ``(B,)``.
    threshold_name : str
        Label for the threshold.
    tau : float
        Quantile level (e.g. 0.99).
    n_resamples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level for confidence intervals.
    seed : int
        Random seed.

    Returns
    -------
    BootstrapCIResult
    """
    u = np.percentile(y_mc, tau * 100)
    mask = y_mc >= u
    sv_abs = np.abs(shap_values_11[mask])
    n = sv_abs.shape[0]

    rng = np.random.default_rng(seed)
    boot_means = np.zeros((n_resamples, 11), dtype=np.float64)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_means[r, :] = sv_abs[idx].mean(axis=0)

    ci_lower = np.percentile(boot_means, 100 * (alpha / 2), axis=0)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    boot_se = boot_means.std(axis=0)
    point_est = sv_abs.mean(axis=0)

    ci_table = pd.DataFrame({
        "Variable": RAW_VARIABLES,
        f"Mean |SHAP| ({threshold_name})": point_est,
        f"CI Lower ({100*alpha/2:.1f}%)": ci_lower,
        f"CI Upper ({100*(1-alpha/2):.1f}%)": ci_upper,
        "Boot SE": boot_se,
        "CI Width": ci_upper - ci_lower,
    }).sort_values(
        f"Mean |SHAP| ({threshold_name})", ascending=False
    ).reset_index(drop=True)

    return BootstrapCIResult(
        threshold_name=threshold_name,
        n_samples=n,
        n_resamples=n_resamples,
        alpha=alpha,
        ci_table=ci_table,
        boot_means=boot_means,
    )
