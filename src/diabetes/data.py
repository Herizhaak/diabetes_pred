"""Loading and validating the processed NHANES table."""
import pandas as pd

from .config import (
    BINARY,
    CATEGORICAL,
    DATA_PATH,
    DESIGN_COLUMNS,
    LAB_COLUMNS,
    NUMERIC,
    RACE_ETHNICITY,
    SEXES,
    TARGET,
)

REQUIRED = NUMERIC + BINARY + CATEGORICAL + LAB_COLUMNS + DESIGN_COLUMNS + [
    "diagnosed_diabetes", TARGET]


def load(path=DATA_PATH):
    """Return the analysis table (one row per adult) after schema checks."""
    df = pd.read_csv(path)
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    if df["seqn"].duplicated().any():
        raise ValueError("Duplicate respondent IDs (seqn)")
    bad = (set(df["sex"]) - set(SEXES)) | (set(df["race_ethnicity"]) - set(RACE_ETHNICITY))
    if bad:
        raise ValueError(f"Unexpected category levels: {sorted(bad)}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be 0/1")
    return df.reset_index(drop=True)


def check_features(X, features):
    """Validate input columns and return them in training order."""
    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return X[features]
