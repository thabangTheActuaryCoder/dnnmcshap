"""Reference-category dummy encoding for health insurance features.

Transforms 11 raw variables into a 22-column encoded array using
reference-category (drop-first) dummy encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._constants import ENCODED_FEATURES


def encode_profiles(df: pd.DataFrame) -> np.ndarray:
    """Encode raw MC profiles into the 22-column model input format.

    Parameters
    ----------
    df : pd.DataFrame
        Raw profiles with columns: age, gender, bmi, children, smoker,
        region, medical_history, family_medical_history,
        exercise_frequency, occupation, coverage_level.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 22)`` with float32 dtype.
    """
    n = len(df)
    X = np.zeros((n, 22), dtype=np.float32)

    X[:, 0] = df["age"].values
    X[:, 1] = (df["gender"] == "male").astype(np.float32).values
    X[:, 2] = df["bmi"].values
    X[:, 3] = df["children"].values
    X[:, 4] = (df["smoker"] == "yes").astype(np.float32).values

    # region dummies (ref = northeast)
    X[:, 5] = (df["region"] == "southwest").astype(np.float32).values
    X[:, 6] = (df["region"] == "northwest").astype(np.float32).values
    X[:, 7] = (df["region"] == "southeast").astype(np.float32).values

    # medical_history dummies (ref = None)
    X[:, 8] = (df["medical_history"] == "Heart disease").astype(np.float32).values
    X[:, 9] = (
        (df["medical_history"] == "High blood pressure").astype(np.float32).values
    )
    X[:, 10] = (df["medical_history"] == "Diabetes").astype(np.float32).values

    # family_medical_history dummies (ref = None)
    X[:, 11] = (
        (df["family_medical_history"] == "Heart disease").astype(np.float32).values
    )
    X[:, 12] = (
        (df["family_medical_history"] == "High blood pressure")
        .astype(np.float32)
        .values
    )
    X[:, 13] = (
        (df["family_medical_history"] == "Diabetes").astype(np.float32).values
    )

    # exercise_frequency dummies (ref = Rarely)
    X[:, 14] = (
        (df["exercise_frequency"] == "Occasionally").astype(np.float32).values
    )
    X[:, 15] = (
        (df["exercise_frequency"] == "Frequently").astype(np.float32).values
    )
    X[:, 16] = (df["exercise_frequency"] == "Never").astype(np.float32).values

    # occupation dummies (ref = Unemployed)
    X[:, 17] = (df["occupation"] == "Student").astype(np.float32).values
    X[:, 18] = (df["occupation"] == "Blue collar").astype(np.float32).values
    X[:, 19] = (df["occupation"] == "White collar").astype(np.float32).values

    # coverage_level dummies (ref = Basic)
    X[:, 20] = (df["coverage_level"] == "Standard").astype(np.float32).values
    X[:, 21] = (df["coverage_level"] == "Premium").astype(np.float32).values

    return X


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encode a raw DataFrame into model-ready format with charges column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset including a ``charges`` column.

    Returns
    -------
    pd.DataFrame
        Encoded DataFrame with 22 feature columns plus ``charges``.
    """
    out = pd.DataFrame()
    out["age"] = df["age"].values.astype(np.float32)
    out["gender"] = (df["gender"] == "male").astype(np.float32).values
    out["bmi"] = df["bmi"].values.astype(np.float32)
    out["children"] = df["children"].values.astype(np.float32)
    out["smoker"] = (df["smoker"] == "yes").astype(np.float32).values

    out["region_southwest"] = (df["region"] == "southwest").astype(np.float32).values
    out["region_northwest"] = (df["region"] == "northwest").astype(np.float32).values
    out["region_southeast"] = (df["region"] == "southeast").astype(np.float32).values

    out["medical_history_Heart_disease"] = (
        (df["medical_history"] == "Heart disease").astype(np.float32).values
    )
    out["medical_history_High_blood_pressure"] = (
        (df["medical_history"] == "High blood pressure").astype(np.float32).values
    )
    out["medical_history_Diabetes"] = (
        (df["medical_history"] == "Diabetes").astype(np.float32).values
    )

    out["family_medical_history_Heart_disease"] = (
        (df["family_medical_history"] == "Heart disease").astype(np.float32).values
    )
    out["family_medical_history_High_blood_pressure"] = (
        (df["family_medical_history"] == "High blood pressure")
        .astype(np.float32)
        .values
    )
    out["family_medical_history_Diabetes"] = (
        (df["family_medical_history"] == "Diabetes").astype(np.float32).values
    )

    out["exercise_frequency_Occasionally"] = (
        (df["exercise_frequency"] == "Occasionally").astype(np.float32).values
    )
    out["exercise_frequency_Frequently"] = (
        (df["exercise_frequency"] == "Frequently").astype(np.float32).values
    )
    out["exercise_frequency_Never"] = (
        (df["exercise_frequency"] == "Never").astype(np.float32).values
    )

    out["occupation_Student"] = (
        (df["occupation"] == "Student").astype(np.float32).values
    )
    out["occupation_Blue_collar"] = (
        (df["occupation"] == "Blue collar").astype(np.float32).values
    )
    out["occupation_White_collar"] = (
        (df["occupation"] == "White collar").astype(np.float32).values
    )

    out["coverage_level_Standard"] = (
        (df["coverage_level"] == "Standard").astype(np.float32).values
    )
    out["coverage_level_Premium"] = (
        (df["coverage_level"] == "Premium").astype(np.float32).values
    )

    out["charges"] = df["charges"].values.astype(np.float32)
    return out
