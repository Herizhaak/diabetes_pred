"""Shared constants for the diabetes project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "diabetes_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "diabetes"
SEED = 42

# Features available without a blood test (questionnaire + scale + BP cuff).
NO_LAB_NUMERIC = ["age", "bmi"]
NO_LAB_BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]

# Blood tests. HbA1c >= 6.5% and random glucose >= 200 mg/dL are themselves
# diagnostic criteria for diabetes, so these features nearly define the label.
LAB_NUMERIC = ["hbA1c_level", "blood_glucose_level"]

FEATURE_SETS = {
    "with_labs": NO_LAB_NUMERIC + LAB_NUMERIC + NO_LAB_BINARY + CATEGORICAL,
    "no_labs": NO_LAB_NUMERIC + NO_LAB_BINARY + CATEGORICAL,
}
ALL_FEATURES = FEATURE_SETS["with_labs"]

# Clinical rule used as a baseline (American Diabetes Association thresholds).
HBA1C_DIAGNOSTIC = 6.5
GLUCOSE_DIAGNOSTIC = 200
