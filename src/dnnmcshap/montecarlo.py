"""Parametric Monte Carlo profile generation (Algorithm 3.1).

Fits marginal distributions from training data and generates synthetic
policyholder profiles for downstream DNN scoring and SHAP attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._constants import (
    AGE_MAX,
    AGE_MIN,
    BMI_MAX,
    BMI_MIN,
    CHILDREN_MAX,
    CHILDREN_MIN,
    COVERAGE_LEVEL_LEVELS,
    DEFAULT_MC_SAMPLES,
    EXERCISE_FREQUENCY_LEVELS,
    FAMILY_MEDICAL_HISTORY_LEVELS,
    MEDICAL_HISTORY_LEVELS,
    OCCUPATION_LEVELS,
    PILOT_SIZES,
    REGION_LEVELS,
    SEED,
)


@dataclass
class MarginalParams:
    """Estimated marginal distribution parameters from training data."""

    p_male: float
    p_smoker: float
    region_probs: np.ndarray
    med_hist_probs: np.ndarray
    fam_hist_probs: np.ndarray
    exercise_probs: np.ndarray
    occ_probs: np.ndarray
    cov_probs: np.ndarray


def estimate_marginals(df_raw: pd.DataFrame) -> MarginalParams:
    """Estimate marginal distribution parameters from raw training data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw training data with original categorical columns.

    Returns
    -------
    MarginalParams
    """
    p_male = (df_raw["gender"] == "male").mean()
    p_smoker = (df_raw["smoker"] == "yes").mean()

    region_probs = np.array(
        [(df_raw["region"] == r).mean() for r in REGION_LEVELS]
    )
    med_hist_probs = np.array(
        [(df_raw["medical_history"] == m).mean() for m in MEDICAL_HISTORY_LEVELS]
    )
    fam_hist_probs = np.array(
        [(df_raw["family_medical_history"] == f).mean()
         for f in FAMILY_MEDICAL_HISTORY_LEVELS]
    )
    exercise_probs = np.array(
        [(df_raw["exercise_frequency"] == e).mean()
         for e in EXERCISE_FREQUENCY_LEVELS]
    )
    occ_probs = np.array(
        [(df_raw["occupation"] == o).mean() for o in OCCUPATION_LEVELS]
    )
    cov_probs = np.array(
        [(df_raw["coverage_level"] == c).mean() for c in COVERAGE_LEVEL_LEVELS]
    )

    return MarginalParams(
        p_male=p_male,
        p_smoker=p_smoker,
        region_probs=region_probs,
        med_hist_probs=med_hist_probs,
        fam_hist_probs=fam_hist_probs,
        exercise_probs=exercise_probs,
        occ_probs=occ_probs,
        cov_probs=cov_probs,
    )


def generate_mc_profiles(
    n: int,
    marginals: MarginalParams,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate *n* raw Monte Carlo profiles from fitted marginals.

    Sampling distributions:
    - age: DiscreteUniform(18, 65)
    - bmi: ContinuousUniform(18, 50)
    - children: DiscreteUniform(0, 5)
    - gender: Bernoulli(p_male)
    - smoker: Bernoulli(p_smoker)
    - Categoricals: Multinomial with estimated probabilities

    Parameters
    ----------
    n : int
        Number of profiles to generate.
    marginals : MarginalParams
        Fitted marginal parameters.
    rng : np.random.Generator, optional
        Random number generator. Uses default seeded generator if ``None``.

    Returns
    -------
    pd.DataFrame
        Raw profiles with 11 columns.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    age = rng.integers(AGE_MIN, AGE_MAX + 1, size=n)
    bmi = rng.uniform(BMI_MIN, BMI_MAX, size=n)
    children = rng.integers(CHILDREN_MIN, CHILDREN_MAX + 1, size=n)

    gender = rng.choice(
        ["male", "female"], size=n, p=[marginals.p_male, 1 - marginals.p_male]
    )
    smoker = rng.choice(
        ["yes", "no"], size=n, p=[marginals.p_smoker, 1 - marginals.p_smoker]
    )
    region = rng.choice(REGION_LEVELS, size=n, p=marginals.region_probs)
    medical_history = rng.choice(
        MEDICAL_HISTORY_LEVELS, size=n, p=marginals.med_hist_probs
    )
    family_medical_history = rng.choice(
        FAMILY_MEDICAL_HISTORY_LEVELS, size=n, p=marginals.fam_hist_probs
    )
    exercise_frequency = rng.choice(
        EXERCISE_FREQUENCY_LEVELS, size=n, p=marginals.exercise_probs
    )
    occupation = rng.choice(OCCUPATION_LEVELS, size=n, p=marginals.occ_probs)
    coverage_level = rng.choice(COVERAGE_LEVEL_LEVELS, size=n, p=marginals.cov_probs)

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "medical_history": medical_history,
        "family_medical_history": family_medical_history,
        "exercise_frequency": exercise_frequency,
        "occupation": occupation,
        "coverage_level": coverage_level,
    })


def run_convergence_diagnostics(
    marginals: MarginalParams,
    score_fn: callable,
    *,
    pilot_sizes: list[int] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run MC convergence diagnostics at increasing sample sizes.

    Parameters
    ----------
    marginals : MarginalParams
        Fitted marginals for profile generation.
    score_fn : callable
        Function that takes a pd.DataFrame of raw profiles and returns
        a 1-D array of predicted costs.
    pilot_sizes : list[int], optional
        Sample sizes to evaluate. Defaults to :data:`PILOT_SIZES`.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Convergence table with columns:
        ``B, Mean, SD, P50, P90, P95, P99, MCSE``.
    """
    if pilot_sizes is None:
        pilot_sizes = list(PILOT_SIZES)

    rng = np.random.default_rng(seed)
    max_b = max(pilot_sizes)
    df_pilot = generate_mc_profiles(max_b, marginals, rng=rng)
    y_pilot = score_fn(df_pilot)

    rows = []
    for b in pilot_sizes:
        y_sub = y_pilot[:b]
        rows.append({
            "B": b,
            "Mean": y_sub.mean(),
            "SD": y_sub.std(),
            "P50": np.percentile(y_sub, 50),
            "P90": np.percentile(y_sub, 90),
            "P95": np.percentile(y_sub, 95),
            "P99": np.percentile(y_sub, 99),
            "MCSE": y_sub.std() / np.sqrt(b),
        })
    return pd.DataFrame(rows)
