"""End-to-end pipeline orchestrator.

Runs all 14 steps of the DNN-Monte Carlo-Shapley framework:
data preparation, training, MC simulation, SHAP attribution,
conditional analysis and sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ._constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETAS,
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_R,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MC_SAMPLES,
    DEFAULT_PATIENCE,
    DEFAULT_SHAP_BACKGROUND,
    DEFAULT_SHAP_BATCH,
    DEFAULT_VAL_EVERY,
    INPUT_DIM,
    RAW_VARIABLES,
    SEED,
    TAIL_THRESHOLDS,
)
from .conditional import (
    BootstrapCIResult,
    ConditionalResult,
    bootstrap_conditional_ci,
    compute_conditional_shap,
)
from .encoding import encode_profiles
from .model import FunnelDNN, auto_device, score_dnn
from .montecarlo import MarginalParams, estimate_marginals, generate_mc_profiles
from .sensitivity import SensitivityResult, augment_predictions, compute_residuals
from .shapley import ShapResult, compute_shap_values, sample_background
from .standardisation import Standardiser
from .training import TrainingResult, train_model


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline run."""

    # Training
    activation: str = "CELU"
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    val_every: int = DEFAULT_VAL_EVERY
    optimiser_name: str = "NAdam"
    betas: tuple[float, float] = DEFAULT_BETAS
    seed: int = SEED

    # MC
    mc_samples: int = DEFAULT_MC_SAMPLES

    # SHAP
    shap_background: int = DEFAULT_SHAP_BACKGROUND
    shap_batch: int = DEFAULT_SHAP_BATCH

    # Bootstrap
    bootstrap_r: int = DEFAULT_BOOTSTRAP_R
    bootstrap_alpha: float = DEFAULT_BOOTSTRAP_ALPHA

    # Thresholds
    thresholds: dict[str, float] = field(
        default_factory=lambda: dict(TAIL_THRESHOLDS)
    )

    # Output
    output_dir: str | None = None


@dataclass
class PipelineResult:
    """Stores all outputs from a full pipeline run."""

    standardiser: Standardiser
    model: FunnelDNN
    training_result: TrainingResult
    marginals: MarginalParams
    df_mc: pd.DataFrame
    y_mc: np.ndarray
    sensitivity: SensitivityResult
    shap_result: ShapResult
    global_shap: pd.DataFrame
    conditional_results: list[ConditionalResult]
    bootstrap_ci: BootstrapCIResult


def run_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    df_raw: pd.DataFrame,
    *,
    config: PipelineConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Run the complete DNN-MC-SHAP pipeline end-to-end.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Raw (un-standardised) encoded training data.
    X_val, y_val : np.ndarray
        Raw encoded validation data.
    df_raw : pd.DataFrame
        Raw training data with original categorical columns
        (for marginal estimation).
    config : PipelineConfig, optional
        Pipeline configuration. Uses defaults if ``None``.
    device : torch.device, optional
        Compute device. Defaults to :func:`auto_device`.
    verbose : bool
        Print progress.

    Returns
    -------
    PipelineResult
    """
    if config is None:
        config = PipelineConfig()
    if device is None:
        device = auto_device()

    # Step 1: Standardise
    if verbose:
        print("Step 1/10: Standardising data...")
    standardiser = Standardiser().fit(X_train, y_train)
    X_tr_std = standardiser.transform_X(X_train)
    y_tr_std = standardiser.transform_y(y_train)
    X_va_std = standardiser.transform_X(X_val)
    y_va_std = standardiser.transform_y(y_val)

    X_tr_t = torch.tensor(X_tr_std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_std, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va_std, dtype=torch.float32)
    y_va_t = torch.tensor(y_va_std, dtype=torch.float32).unsqueeze(1)

    # Step 2: Train model
    if verbose:
        print("Step 2/10: Training DNN...")
    torch.manual_seed(config.seed)
    model = FunnelDNN(input_dim=INPUT_DIM, activation=config.activation)
    training_result = train_model(
        model, X_tr_t, y_tr_t, X_va_t, y_va_t,
        optimiser_name=config.optimiser_name,
        lr=config.lr,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        val_every=config.val_every,
        betas=config.betas,
        device=device,
        verbose=verbose,
    )
    model.load_state_dict(training_result.best_model_state)
    model = model.to(device)
    model.eval()

    # Step 3: Estimate marginals
    if verbose:
        print("Step 3/10: Estimating marginals...")
    marginals = estimate_marginals(df_raw)

    # Step 4: Generate MC profiles
    if verbose:
        print(f"Step 4/10: Generating {config.mc_samples:,} MC profiles...")
    rng = np.random.default_rng(config.seed)
    df_mc = generate_mc_profiles(config.mc_samples, marginals, rng=rng)
    X_mc_encoded = encode_profiles(df_mc)

    # Step 5: Score MC profiles
    if verbose:
        print("Step 5/10: Scoring MC profiles...")
    y_mc = score_dnn(model, X_mc_encoded, standardiser, device=device)

    # Step 6: Sensitivity analysis
    if verbose:
        print("Step 6/10: Running sensitivity analysis...")
    y_train_pred = score_dnn(model, X_train, standardiser, device=device)
    residuals_centred, sigma_e = compute_residuals(y_train, y_train_pred)
    sensitivity = augment_predictions(
        y_mc, residuals_centred, sigma_e, seed=config.seed + 1
    )

    # Step 7: SHAP values
    if verbose:
        print("Step 7/10: Computing SHAP values...")
    X_mc_std = standardiser.transform_X(X_mc_encoded)
    X_bg = sample_background(X_tr_std, config.shap_background, seed=config.seed)
    shap_result = compute_shap_values(
        model, X_mc_std, X_bg, batch_size=config.shap_batch, verbose=verbose
    )

    # Step 8: Global SHAP summary
    if verbose:
        print("Step 8/10: Computing global SHAP summary...")
    mean_signed = shap_result.shap_values_11.mean(axis=0)
    mean_abs = np.abs(shap_result.shap_values_11).mean(axis=0)
    global_shap = pd.DataFrame({
        "Variable": RAW_VARIABLES,
        "Mean Signed SHAP": mean_signed,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    # Step 9: Conditional SHAP
    if verbose:
        print("Step 9/10: Computing conditional SHAP...")
    conditional_results = compute_conditional_shap(
        shap_result.shap_values_11, y_mc, thresholds=config.thresholds
    )

    # Step 10: Bootstrap CIs
    if verbose:
        print("Step 10/10: Computing bootstrap CIs...")
    bootstrap_ci = bootstrap_conditional_ci(
        shap_result.shap_values_11, y_mc,
        threshold_name="tau_99", tau=0.99,
        n_resamples=config.bootstrap_r, alpha=config.bootstrap_alpha,
        seed=config.seed,
    )

    if verbose:
        print("Pipeline complete.")

    return PipelineResult(
        standardiser=standardiser,
        model=model,
        training_result=training_result,
        marginals=marginals,
        df_mc=df_mc,
        y_mc=y_mc,
        sensitivity=sensitivity,
        shap_result=shap_result,
        global_shap=global_shap,
        conditional_results=conditional_results,
        bootstrap_ci=bootstrap_ci,
    )
