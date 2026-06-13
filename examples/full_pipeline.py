"""Full pipeline example using run_pipeline() on synthetic data."""

import numpy as np
import pandas as pd
import torch

import dnnmcshap as dm
from dnnmcshap._constants import (
    COVERAGE_LEVEL_LEVELS,
    EXERCISE_FREQUENCY_LEVELS,
    FAMILY_MEDICAL_HISTORY_LEVELS,
    MEDICAL_HISTORY_LEVELS,
    OCCUPATION_LEVELS,
    REGION_LEVELS,
)

# Generate synthetic data
rng = np.random.default_rng(42)
n = 300

df_raw = pd.DataFrame({
    "age": rng.integers(18, 66, size=n),
    "gender": rng.choice(["male", "female"], size=n),
    "bmi": rng.uniform(18.0, 50.0, size=n),
    "children": rng.integers(0, 6, size=n),
    "smoker": rng.choice(["yes", "no"], size=n, p=[0.2, 0.8]),
    "region": rng.choice(REGION_LEVELS, size=n),
    "medical_history": rng.choice(MEDICAL_HISTORY_LEVELS, size=n),
    "family_medical_history": rng.choice(FAMILY_MEDICAL_HISTORY_LEVELS, size=n),
    "exercise_frequency": rng.choice(EXERCISE_FREQUENCY_LEVELS, size=n),
    "occupation": rng.choice(OCCUPATION_LEVELS, size=n),
    "coverage_level": rng.choice(COVERAGE_LEVEL_LEVELS, size=n),
})
df_raw["charges"] = (
    100.0 * df_raw["age"]
    + 200.0 * df_raw["bmi"]
    + 5000.0 * (df_raw["smoker"] == "yes").astype(float)
    + rng.normal(0, 500, size=n)
).astype(np.float32)

# Encode and split
X = dm.encode_profiles(df_raw.drop(columns=["charges"]))
y = df_raw["charges"].values.astype(np.float32)

split = int(0.7 * n)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Configure for fast execution
config = dm.PipelineConfig(
    activation="CELU",
    lr=0.001,
    batch_size=64,
    max_epochs=20,
    patience=5,
    val_every=2,
    mc_samples=500,
    shap_background=50,
    shap_batch=250,
    bootstrap_r=200,
)

# Run end-to-end
result = dm.run_pipeline(
    X_train, y_train, X_val, y_val,
    df_raw.drop(columns=["charges"]),
    config=config,
    device=torch.device("cpu"),
    verbose=True,
)

print("\n--- Pipeline Results ---")
print(f"Training best epoch: {result.training_result.best_epoch}")
print(f"Training best val R2: {result.training_result.best_val_r2:.4f}")
print(f"MC samples: {len(result.y_mc)}")
print(f"SHAP efficiency (max error): {result.shap_result.efficiency_error.max():.6f}")
print(f"\nGlobal SHAP importance:")
print(result.global_shap.to_string(index=False))
print(f"\nConditional results: {len(result.conditional_results)} thresholds")
for cr in result.conditional_results:
    print(f"  {cr.threshold_name}: n={cr.n_high_cost}, u={cr.u_value:.2f}")
print(f"\nBootstrap CI (tau_99):")
print(result.bootstrap_ci.ci_table.to_string(index=False))
