"""DNN training and hyperparameter comparison.

Provides :func:`train_model` for single training runs with
validation-based early stopping, and :func:`run_hyperparameter_comparison`
for the five-phase greedy hyperparameter search.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ._constants import (
    ACTIVATION_CANDIDATES,
    BATCH_SIZE_CANDIDATES,
    DEFAULT_ACTIVATION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETAS,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_VAL_EVERY,
    LR_CANDIDATES,
    MOMENTUM_CANDIDATES,
    OPTIMISER_CANDIDATES,
    SEED,
)
from .metrics import r2_dot
from .model import FunnelDNN, auto_device


@dataclass
class TrainingResult:
    """Stores outputs from a single training run."""

    best_model_state: dict
    best_epoch: int
    best_val_r2: float
    train_losses: list[float]
    val_epochs: list[int]
    val_losses: list[float]
    train_r2s: list[float]
    val_r2s: list[float]
    elapsed: float


@dataclass
class PhaseResult:
    """Stores results for one phase of the hyperparameter comparison."""

    phase_name: str
    results: dict[str, TrainingResult]
    best_config: str
    best_val_r2: float


def _get_optimiser_cls(name: str) -> type:
    """Map optimiser name to torch.optim class."""
    mapping = {
        "Adam": optim.Adam,
        "NAdam": optim.NAdam,
    }
    if name not in mapping:
        raise ValueError(f"Unknown optimiser '{name}'. Choose from {list(mapping)}.")
    return mapping[name]


def train_model(
    model: FunnelDNN,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    optimiser_cls: type | None = None,
    optimiser_name: str = "Adam",
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    val_every: int = DEFAULT_VAL_EVERY,
    betas: tuple[float, float] | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> TrainingResult:
    """Train a FunnelDNN with validation-based best-model selection.

    Validation R-squared is evaluated every *val_every* epochs.
    The best model is checkpointed at the validation epoch with highest
    R-squared. Early stopping patience counts validation checks without
    improvement.

    Parameters
    ----------
    model : FunnelDNN
        Model instance (will be moved to *device*).
    X_train, y_train : torch.Tensor
        Normalised training data.
    X_val, y_val : torch.Tensor
        Normalised validation data.
    optimiser_cls : type, optional
        ``torch.optim`` class. If ``None``, resolved from *optimiser_name*.
    optimiser_name : str
        Fallback optimiser name when *optimiser_cls* is ``None``.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience in validation checks.
    val_every : int
        Validate every N epochs.
    betas : tuple, optional
        ``(beta1, beta2)`` momentum parameters.
    device : torch.device, optional
        Defaults to :func:`auto_device`.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    TrainingResult
    """
    if device is None:
        device = auto_device()
    if optimiser_cls is None:
        optimiser_cls = _get_optimiser_cls(optimiser_name)

    model = model.to(device)
    criterion = nn.MSELoss()

    opt_kwargs: dict = {"lr": lr}
    if betas is not None:
        opt_kwargs["betas"] = betas
    optimizer = optimiser_cls(model.parameters(), **opt_kwargs)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)
    X_tr_d = X_train.to(device)
    y_tr_d = y_train.to(device)

    train_losses: list[float] = []
    val_epochs: list[int] = []
    val_losses: list[float] = []
    train_r2s: list[float] = []
    val_r2s: list[float] = []

    best_val_r2 = -np.inf
    best_epoch = 0
    best_state = None
    checks_no_improve = 0

    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item() * xb.size(0))

        train_loss = sum(batch_losses) / len(X_train)
        train_losses.append(train_loss)

        if epoch % val_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                preds_tr = model(X_tr_d)
                tr_r2 = r2_dot(y_tr_d, preds_tr)

                preds_va = model(X_val_d)
                val_loss = criterion(preds_va, y_val_d).item()
                va_r2 = r2_dot(y_val_d, preds_va)

            val_epochs.append(epoch)
            val_losses.append(val_loss)
            train_r2s.append(tr_r2)
            val_r2s.append(va_r2)

            if va_r2 > best_val_r2:
                best_val_r2 = va_r2
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                checks_no_improve = 0
            else:
                checks_no_improve += 1

            if verbose:
                print(
                    f"Epoch {epoch:>3d}/{max_epochs} [VAL] | "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"Train R2: {tr_r2:.6f} | Val R2: {va_r2:.6f}"
                )

            if checks_no_improve >= patience:
                if verbose:
                    print(
                        f"\nEarly stopping after {checks_no_improve} val checks "
                        f"(epoch {epoch}). Best epoch: {best_epoch}"
                    )
                break

    elapsed = time.time() - start
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    return TrainingResult(
        best_model_state=best_state,
        best_epoch=best_epoch,
        best_val_r2=best_val_r2,
        train_losses=train_losses,
        val_epochs=val_epochs,
        val_losses=val_losses,
        train_r2s=train_r2s,
        val_r2s=val_r2s,
        elapsed=elapsed,
    )


def run_hyperparameter_comparison(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    input_dim: int = 22,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    val_every: int = DEFAULT_VAL_EVERY,
    seed: int = SEED,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[PhaseResult]:
    """Run a five-phase greedy hyperparameter comparison.

    Phases (each carries the best configuration forward):
      1. Activation function
      2. Learning rate
      3. Batch size
      4. Optimiser
      5. Momentum parameters

    Returns
    -------
    list[PhaseResult]
        One :class:`PhaseResult` per phase.
    """
    if device is None:
        device = auto_device()

    phases: list[PhaseResult] = []

    # Phase 1: Activation
    best_act = DEFAULT_ACTIVATION
    p1_results: dict[str, TrainingResult] = {}
    for act in ACTIVATION_CANDIDATES:
        torch.manual_seed(seed)
        m = FunnelDNN(input_dim=input_dim, activation=act)
        res = train_model(
            m, X_train, y_train, X_val, y_val,
            optimiser_name="Adam", lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE,
            max_epochs=max_epochs, patience=patience, val_every=val_every,
            device=device, verbose=verbose,
        )
        p1_results[act] = res
    best_act = max(p1_results, key=lambda k: p1_results[k].best_val_r2)
    phases.append(PhaseResult("Activation", p1_results, best_act, p1_results[best_act].best_val_r2))

    # Phase 2: Learning rate
    best_lr = DEFAULT_LR
    p2_results: dict[str, TrainingResult] = {}
    for lr_val in LR_CANDIDATES:
        label = str(lr_val)
        torch.manual_seed(seed)
        m = FunnelDNN(input_dim=input_dim, activation=best_act)
        res = train_model(
            m, X_train, y_train, X_val, y_val,
            optimiser_name="Adam", lr=lr_val, batch_size=DEFAULT_BATCH_SIZE,
            max_epochs=max_epochs, patience=patience, val_every=val_every,
            device=device, verbose=verbose,
        )
        p2_results[label] = res
    best_lr_label = max(p2_results, key=lambda k: p2_results[k].best_val_r2)
    best_lr = float(best_lr_label)
    phases.append(PhaseResult("Learning Rate", p2_results, best_lr_label, p2_results[best_lr_label].best_val_r2))

    # Phase 3: Batch size
    best_bs = DEFAULT_BATCH_SIZE
    p3_results: dict[str, TrainingResult] = {}
    for bs in BATCH_SIZE_CANDIDATES:
        label = str(bs)
        torch.manual_seed(seed)
        m = FunnelDNN(input_dim=input_dim, activation=best_act)
        res = train_model(
            m, X_train, y_train, X_val, y_val,
            optimiser_name="Adam", lr=best_lr, batch_size=bs,
            max_epochs=max_epochs, patience=patience, val_every=val_every,
            device=device, verbose=verbose,
        )
        p3_results[label] = res
    best_bs_label = max(p3_results, key=lambda k: p3_results[k].best_val_r2)
    best_bs = int(best_bs_label)
    phases.append(PhaseResult("Batch Size", p3_results, best_bs_label, p3_results[best_bs_label].best_val_r2))

    # Phase 4: Optimiser
    best_opt = "Adam"
    p4_results: dict[str, TrainingResult] = {}
    for opt_name in OPTIMISER_CANDIDATES:
        torch.manual_seed(seed)
        m = FunnelDNN(input_dim=input_dim, activation=best_act)
        res = train_model(
            m, X_train, y_train, X_val, y_val,
            optimiser_name=opt_name, lr=best_lr, batch_size=best_bs,
            max_epochs=max_epochs, patience=patience, val_every=val_every,
            device=device, verbose=verbose,
        )
        p4_results[opt_name] = res
    best_opt = max(p4_results, key=lambda k: p4_results[k].best_val_r2)
    phases.append(PhaseResult("Optimiser", p4_results, best_opt, p4_results[best_opt].best_val_r2))

    # Phase 5: Momentum
    p5_results: dict[str, TrainingResult] = {}
    for betas_val in MOMENTUM_CANDIDATES:
        label = f"B1={betas_val[0]}"
        torch.manual_seed(seed)
        m = FunnelDNN(input_dim=input_dim, activation=best_act)
        res = train_model(
            m, X_train, y_train, X_val, y_val,
            optimiser_name=best_opt, lr=best_lr, batch_size=best_bs,
            betas=betas_val,
            max_epochs=max_epochs, patience=patience, val_every=val_every,
            device=device, verbose=verbose,
        )
        p5_results[label] = res
    best_mom_label = max(p5_results, key=lambda k: p5_results[k].best_val_r2)
    phases.append(PhaseResult("Momentum", p5_results, best_mom_label, p5_results[best_mom_label].best_val_r2))

    return phases
