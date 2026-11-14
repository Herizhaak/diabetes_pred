"""Loading and validating the diabetes data."""
import pandas as pd

from .config import ALL_FEATURES, DATA_PATH, TARGET


def load(path=DATA_PATH):
    """Return (X, y). Exact duplicate rows are dropped before any split."""
    df = pd.read_csv(path).drop_duplicates()
    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Data is missing columns: {sorted(missing)}")
    return df[ALL_FEATURES].reset_index(drop=True), df[TARGET].reset_index(drop=True)


def check_features(X, features):
    """Validate input columns and return them in training order."""
    missing = set(features) - set(X.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return X[features]
