"""dnnmcshap - DNN-Monte Carlo-Shapley framework for health insurance cost analysis.

A three-stage framework combining Deep Neural Network prediction,
Monte Carlo simulation and Shapley value attribution for understanding
health insurance cost drivers.

Reference
---------
Baloyi, T. B. J. (2026). *Understanding Health Insurance Cost Predictions
Using Deep Neural Networks, Monte Carlo Simulation and Shapley Values*.
MSc Dissertation, University of the Free State.
Supervisor: Mr J. Blomerous (FASSA).
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core
from .conditional import (
    BootstrapCIResult,
    ConditionalResult,
    bootstrap_conditional_ci,
    compute_conditional_shap,
)
from .encoding import encode_dataframe, encode_profiles
from .io import (
    export_mc_results,
    export_shap_tables,
    load_checkpoint,
    save_checkpoint,
)
from .metrics import EvaluationResult, evaluate_model, mae, r2_dot, r2_score, rmse
from .model import FunnelDNN, auto_device, score_dnn
from .montecarlo import (
    MarginalParams,
    estimate_marginals,
    generate_mc_profiles,
    run_convergence_diagnostics,
)
from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .sensitivity import SensitivityResult, augment_predictions, compute_residuals
from .shapley import (
    ShapResult,
    aggregate_dummy_to_raw,
    compute_shap_values,
    sample_background,
)
from .standardisation import Standardiser, StandardisationParams
from .training import PhaseResult, TrainingResult, run_hyperparameter_comparison, train_model

__all__ = [
    # Version
    "__version__",
    # Encoding
    "encode_profiles",
    "encode_dataframe",
    # Standardisation
    "Standardiser",
    "StandardisationParams",
    # Metrics
    "r2_score",
    "r2_dot",
    "rmse",
    "mae",
    "evaluate_model",
    "EvaluationResult",
    # Model
    "FunnelDNN",
    "auto_device",
    "score_dnn",
    # Training
    "train_model",
    "run_hyperparameter_comparison",
    "TrainingResult",
    "PhaseResult",
    # Monte Carlo
    "MarginalParams",
    "estimate_marginals",
    "generate_mc_profiles",
    "run_convergence_diagnostics",
    # Sensitivity
    "compute_residuals",
    "augment_predictions",
    "SensitivityResult",
    # Shapley
    "compute_shap_values",
    "aggregate_dummy_to_raw",
    "sample_background",
    "ShapResult",
    # Conditional
    "compute_conditional_shap",
    "bootstrap_conditional_ci",
    "ConditionalResult",
    "BootstrapCIResult",
    # IO
    "save_checkpoint",
    "load_checkpoint",
    "export_mc_results",
    "export_shap_tables",
    # Pipeline
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]
