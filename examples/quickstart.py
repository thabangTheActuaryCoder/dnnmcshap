"""Quick start example for dnnmcshap on synthetic data.

Demonstrates the core workflow: generate synthetic data, train a small
model, run MC simulation, and compute SHAP values.
"""

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

# -- 1. Generate synthetic data --
rng = np.random.default_rng(42)
n = 500

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
# Simple additive charges
df_raw["charges"] = (
    100.0 * df_raw["age"]
    + 200.0 * df_raw["bmi"]
    + 5000.0 * (df_raw["smoker"] == "yes").astype(float)
    + rng.normal(0, 500, size=n)
).astype(np.float32)

# -- 2. Encode and split --
X = dm.encode_profiles(df_raw.drop(columns=["charges"]))
y = df_raw["charges"].values.astype(np.float32)

split = int(0.7 * n)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# -- 3. Standardise --
std = dm.Standardiser().fit(X_train, y_train)

# -- 4. Train --
torch.manual_seed(42)
model = dm.FunnelDNN(input_dim=22, hidden_layers=[64, 32, 16], activation="CELU")

X_tr_t = torch.tensor(std.transform_X(X_train), dtype=torch.float32)
y_tr_t = torch.tensor(std.transform_y(y_train), dtype=torch.float32).unsqueeze(1)
X_va_t = torch.tensor(std.transform_X(X_val), dtype=torch.float32)
y_va_t = torch.tensor(std.transform_y(y_val), dtype=torch.float32).unsqueeze(1)

result = dm.train_model(
    model, X_tr_t, y_tr_t, X_va_t, y_va_t,
    lr=0.001, batch_size=64, max_epochs=50, patience=5, val_every=2,
    device=torch.device("cpu"), verbose=True,
)
model.load_state_dict(result.best_model_state)
model.eval()

print(f"\nBest epoch: {result.best_epoch}, Val R2: {result.best_val_r2:.4f}")

# -- 5. Evaluate --
preds = dm.score_dnn(model, X_val, std, device=torch.device("cpu"))
metrics = dm.evaluate_model(
    torch.tensor(y_val), torch.tensor(preds, dtype=torch.float32)
)
print(f"Test R2: {metrics.r2:.4f}, RMSE: {metrics.rmse:.2f}, MAE: {metrics.mae:.2f}")

# -- 6. MC simulation --
marginals = dm.estimate_marginals(df_raw.drop(columns=["charges"]))
df_mc = dm.generate_mc_profiles(1000, marginals, rng=np.random.default_rng(42))
X_mc = dm.encode_profiles(df_mc)
y_mc = dm.score_dnn(model, X_mc, std, device=torch.device("cpu"))

print(f"\nMC predictions: mean={y_mc.mean():.2f}, std={y_mc.std():.2f}")

# -- 7. SHAP values --
X_bg = dm.sample_background(std.transform_X(X_train), n_background=50)
shap_result = dm.compute_shap_values(
    model, std.transform_X(X_mc), X_bg, batch_size=500, verbose=True
)

print(f"\nSHAP efficiency error (max): {shap_result.efficiency_error.max():.6f}")

# -- 8. Global importance --
mean_abs = np.abs(shap_result.shap_values_11).mean(axis=0)
from dnnmcshap._constants import RAW_VARIABLES
for var, val in sorted(zip(RAW_VARIABLES, mean_abs), key=lambda x: -x[1]):
    print(f"  {var:30s} {val:.4f}")

print("\nQuickstart complete.")
