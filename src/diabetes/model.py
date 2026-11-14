"""Pipelines. Every preprocessing and resampling step is inside the pipeline,
so it is fit on training folds only. SMOTE uses imblearn's Pipeline, which
resamples during fit and never touches validation or test rows."""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL,
    GLUCOSE_DIAGNOSTIC,
    HBA1C_DIAGNOSTIC,
    LAB_NUMERIC,
    NO_LAB_BINARY,
    NO_LAB_NUMERIC,
    SEED,
)


def preprocessor(features):
    numeric = [c for c in NO_LAB_NUMERIC + LAB_NUMERIC if c in features]
    binary = [c for c in NO_LAB_BINARY if c in features]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("bin", "passthrough", binary),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


class ClinicalRule(ClassifierMixin, BaseEstimator):
    """Baseline with no learning: positive if HbA1c >= 6.5% or glucose >= 200 mg/dL."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        hit = ((X["hbA1c_level"] >= HBA1C_DIAGNOSTIC)
               | (X["blood_glucose_level"] >= GLUCOSE_DIAGNOSTIC)).to_numpy(float)
        return np.column_stack([1 - hit, hit])

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


def candidates(features):
    """Models compared in cross-validation for one feature set."""
    models = {
        "baseline: majority class": DummyClassifier(strategy="most_frequent"),
        "logistic regression (class weights)": Pipeline([
            ("prep", preprocessor(features)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "logistic regression (SMOTE)": ImbPipeline([
            ("prep", preprocessor(features)),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "gradient boosting": Pipeline([
            ("prep", preprocessor(features)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.1, max_iter=300, early_stopping=True, random_state=SEED)),
        ]),
    }
    if "hbA1c_level" in features:
        models = {"baseline: clinical rule (HbA1c/glucose)": ClinicalRule(), **models}
    return models
