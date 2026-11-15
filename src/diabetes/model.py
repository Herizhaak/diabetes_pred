"""The ADA risk-test baseline and the learned pipelines.

Every preprocessing step (imputation, scaling, one-hot encoding) is inside a
scikit-learn Pipeline, so it is fit on training folds only.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY, CATEGORICAL, LAB_COLUMNS, NUMERIC, RACE_ETHNICITY, SEED, SEXES

LEVELS = {"sex": list(SEXES), "race_ethnicity": list(RACE_ETHNICITY)}


class ADARiskScore(ClassifierMixin, BaseEstimator):
    """American Diabetes Association diabetes risk test, a fixed points score with
    no learned parameters (Bang et al., Ann Intern Med 2009;151:775-83).

    Points: age 40-49 = 1, 50-59 = 2, 60+ = 3; male = 1; gestational diabetes = 1;
    parent or sibling with diabetes = 1; told high blood pressure = 1; not physically
    active = 1; BMI 25-29.9 = 1, 30-39.9 = 2, 40+ = 3. A total of 5 or more is
    "high risk". ``variant="bang2009"`` gives the original paper's version (no
    gestational-diabetes item; physically active = -1 instead of inactive = +1).

    An unanswered item scores 0 points, as it would on the paper form.
    """

    def __init__(self, variant="ada", threshold=5):
        self.variant = variant
        self.threshold = threshold

    def fit(self, X, y=None):
        if self.variant not in ("ada", "bang2009"):
            raise ValueError(f"Unknown variant: {self.variant}")
        self.classes_ = np.array([0, 1])
        return self

    def _range(self):
        return (0, 11) if self.variant == "ada" else (-1, 9)

    def points(self, X):
        age = X["age"].to_numpy(float)
        bmi = X["bmi"].to_numpy(float)

        def flag(col):
            return (X[col].to_numpy(float) == 1).astype(int)

        score = (np.digitize(age, [40, 50, 60]) + np.digitize(np.nan_to_num(bmi), [25, 30, 40])
                 + (X["sex"].to_numpy() == "Male").astype(int)
                 + flag("family_history") + flag("hypertension"))
        if self.variant == "ada":
            inactive = (X["physically_active"].to_numpy(float) == 0).astype(int)
            score = score + flag("gestational_diabetes") + inactive
        else:
            score = score - flag("physically_active")
        return score

    def decision_function(self, X):
        return self.points(X).astype(float)

    def predict_proba(self, X):
        lo, hi = self._range()
        p = (self.points(X) - lo) / (hi - lo)  # monotone rescaling, not a probability
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.points(X) >= self.threshold).astype(int)


def preprocessor(features, scale=True):
    # Lab columns are only accepted for the fasting-glucose reference analysis.
    numeric = [c for c in NUMERIC + LAB_COLUMNS if c in features]
    binary = [c for c in BINARY if c in features]
    categorical = [c for c in CATEGORICAL if c in features]
    if scale:  # linear model: impute (with missingness flags) and standardise
        num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                            StandardScaler())
    else:  # gradient boosting routes missing values natively
        num = "passthrough"
    return ColumnTransformer([
        ("num", num, numeric),
        ("bin", SimpleImputer(strategy="most_frequent", add_indicator=scale), binary),
        ("cat", OneHotEncoder(categories=[LEVELS[c] for c in categorical],
                              handle_unknown="ignore", sparse_output=False), categorical),
    ])


def logistic(features):
    return Pipeline([
        ("prep", preprocessor(features, scale=True)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


def boosting(features):
    return Pipeline([
        ("prep", preprocessor(features, scale=False)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=SEED)),
    ])


def candidates(feature_sets):
    """Models compared in cross-validation: name -> (feature set, estimator)."""
    return {
        "baseline: majority class": ("ada_inputs", DummyClassifier(strategy="prior")),
        "baseline: ADA risk test": ("ada_inputs", ADARiskScore()),
        "logistic regression, ADA inputs": ("ada_inputs", logistic(feature_sets["ada_inputs"])),
        "logistic regression, all no-lab": ("no_labs", logistic(feature_sets["no_labs"])),
        "gradient boosting, all no-lab": ("no_labs", boosting(feature_sets["no_labs"])),
    }
