# dnnmcshap

A three-stage framework combining **Deep Neural Network** prediction,
**Monte Carlo** simulation and **Shapley** value attribution for
understanding health insurance cost drivers.

Developed as part of an MSc dissertation at the University of the Free
State, supervised by Dr. JM Blomerous (FASSA).

## Installation

### Python

```bash
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

### R

The `rdnnmcshap` R package wraps the Python package via `reticulate`:

```r
# Install Python package first
reticulate::py_install("dnnmcshap", pip = TRUE)

# Install R package from source
install.packages("rdnnmcshap", repos = NULL, type = "source")
```

## Quick start (Python)

```python
import dnnmcshap as dm
import numpy as np
import torch

# Load a trained checkpoint
model, std, meta = dm.load_checkpoint("model.pth")

# Generate MC profiles
marginals = dm.estimate_marginals(df_raw)
df_mc = dm.generate_mc_profiles(100_000, marginals)
X_mc = dm.encode_profiles(df_mc)
y_mc = dm.score_dnn(model, X_mc, std)

# Compute SHAP values
X_bg = dm.sample_background(std.transform_X(X_train), n_background=200)
shap_result = dm.compute_shap_values(model, std.transform_X(X_mc), X_bg)

# Conditional analysis at tail thresholds
cond = dm.compute_conditional_shap(shap_result.shap_values_11, y_mc)
```

## Quick start (R)

```r
library(rdnnmcshap)

ckpt <- load_checkpoint("model.pth")
marginals <- estimate_marginals(df_raw)
df_mc <- generate_mc_profiles(10000L, marginals)
X_mc <- encode_profiles(df_mc)
y_mc <- score_dnn(ckpt$model, X_mc, ckpt$standardiser)
```

## Full pipeline

For a single-call end-to-end run:

```python
config = dm.PipelineConfig(mc_samples=100_000, max_epochs=200)
result = dm.run_pipeline(X_train, y_train, X_val, y_val, df_raw, config=config)
```

## Package structure

```
src/dnnmcshap/
    _constants.py      Domain constants
    _typing.py         Type aliases
    encoding.py        Reference-category dummy encoding
    standardisation.py Standardiser (fit/transform/inverse)
    model.py           FunnelDNN architecture
    training.py        Training loop and hyperparameter comparison
    metrics.py         R-squared, RMSE, MAE
    montecarlo.py      MC profile generation (Algorithm 3.1)
    sensitivity.py     Residual-augmented MC (Algorithm 3.2)
    shapley.py         SHAP GradientExplainer
    conditional.py     Conditional Shapley at tail thresholds
    plotting.py        Publication-quality visualisations
    io.py              Checkpoint save/load, CSV export
    pipeline.py        End-to-end orchestrator
```

## Testing

```bash
make test        # fast tests only
make test-all    # all tests including slow ones
make lint        # ruff linting
make typecheck   # mypy type checking
```

## Citation

If you use this package, please cite the dissertation:

> Baloyi, T. B. J. (2026). *Understanding Health Insurance Cost
> Predictions Using Deep Neural Networks, Monte Carlo Simulation and
> Shapley Values*. MSc Dissertation, University of the Free State.

## Licence

MIT
