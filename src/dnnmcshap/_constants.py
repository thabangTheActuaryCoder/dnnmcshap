"""Domain constants for the DNN-Monte Carlo-Shapley framework."""

from __future__ import annotations

# Reproducibility
SEED: int = 42

# DNN architecture
INPUT_DIM: int = 22
HIDDEN_LAYERS: list[int] = [256, 128, 64, 32, 16]

# Default hyperparameters
DEFAULT_ACTIVATION: str = "CELU"
DEFAULT_LR: float = 0.001
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_MAX_EPOCHS: int = 200
DEFAULT_PATIENCE: int = 10
DEFAULT_VAL_EVERY: int = 5
DEFAULT_BETAS: tuple[float, float] = (0.9, 0.999)

# 22 encoded feature names (column order after reference-category encoding)
ENCODED_FEATURES: list[str] = [
    "age",
    "gender",
    "bmi",
    "children",
    "smoker",
    "region_southwest",
    "region_northwest",
    "region_southeast",
    "medical_history_Heart_disease",
    "medical_history_High_blood_pressure",
    "medical_history_Diabetes",
    "family_medical_history_Heart_disease",
    "family_medical_history_High_blood_pressure",
    "family_medical_history_Diabetes",
    "exercise_frequency_Occasionally",
    "exercise_frequency_Frequently",
    "exercise_frequency_Never",
    "occupation_Student",
    "occupation_Blue_collar",
    "occupation_White_collar",
    "coverage_level_Standard",
    "coverage_level_Premium",
]

# 11 raw variable names (before encoding)
RAW_VARIABLES: list[str] = [
    "age",
    "gender",
    "bmi",
    "children",
    "smoker",
    "region",
    "medical_history",
    "family_medical_history",
    "exercise_frequency",
    "occupation",
    "coverage_level",
]

# Mapping from raw variable name to column indices in the 22-column encoded array
RAW_TO_ENCODED: dict[str, list[int]] = {
    "age": [0],
    "gender": [1],
    "bmi": [2],
    "children": [3],
    "smoker": [4],
    "region": [5, 6, 7],
    "medical_history": [8, 9, 10],
    "family_medical_history": [11, 12, 13],
    "exercise_frequency": [14, 15, 16],
    "occupation": [17, 18, 19],
    "coverage_level": [20, 21],
}

# Reference categories (omitted dummy for each categorical variable)
REFERENCE_CATEGORIES: dict[str, str] = {
    "region": "northeast",
    "medical_history": "None",
    "family_medical_history": "None",
    "exercise_frequency": "Rarely",
    "occupation": "Unemployed",
    "coverage_level": "Basic",
}

# Categorical level lists (including reference category)
REGION_LEVELS: list[str] = ["northeast", "southwest", "northwest", "southeast"]
MEDICAL_HISTORY_LEVELS: list[str] = [
    "None", "Heart disease", "High blood pressure", "Diabetes",
]
FAMILY_MEDICAL_HISTORY_LEVELS: list[str] = [
    "None", "Heart disease", "High blood pressure", "Diabetes",
]
EXERCISE_FREQUENCY_LEVELS: list[str] = [
    "Rarely", "Occasionally", "Frequently", "Never",
]
OCCUPATION_LEVELS: list[str] = [
    "Unemployed", "Student", "Blue collar", "White collar",
]
COVERAGE_LEVEL_LEVELS: list[str] = ["Basic", "Standard", "Premium"]

# Hyperparameter comparison candidates
ACTIVATION_CANDIDATES: list[str] = ["CELU", "GELU", "Tanh"]
LR_CANDIDATES: list[float] = [0.01, 0.001, 0.0005, 0.0001]
BATCH_SIZE_CANDIDATES: list[int] = [64, 128, 256, 512]
OPTIMISER_CANDIDATES: list[str] = ["Adam", "NAdam"]
MOMENTUM_CANDIDATES: list[tuple[float, float]] = [
    (0.90, 0.999),
    (0.95, 0.999),
    (0.99, 0.999),
]

# Tail thresholds for conditional Shapley analysis
TAIL_THRESHOLDS: dict[str, float] = {
    "tau_90": 0.90,
    "tau_95": 0.95,
    "tau_99": 0.99,
}

# MC simulation defaults
DEFAULT_MC_SAMPLES: int = 100_000
DEFAULT_SHAP_BACKGROUND: int = 200
DEFAULT_SHAP_BATCH: int = 2000
DEFAULT_BOOTSTRAP_R: int = 10_000
DEFAULT_BOOTSTRAP_ALPHA: float = 0.05

# MC convergence pilot sizes
PILOT_SIZES: list[int] = [1000, 2500, 5000, 10000, 25000, 50000, 100000]

# MC sampling bounds
AGE_MIN: int = 18
AGE_MAX: int = 65
BMI_MIN: float = 18.0
BMI_MAX: float = 50.0
CHILDREN_MIN: int = 0
CHILDREN_MAX: int = 5
