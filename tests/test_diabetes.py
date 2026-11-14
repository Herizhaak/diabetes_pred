import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes.config import FEATURE_SETS  # noqa: E402
from diabetes.data import check_features, load  # noqa: E402
from diabetes.model import ClinicalRule, candidates  # noqa: E402
from diabetes.predict import predict  # noqa: E402
from diabetes.train import pick_threshold  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    X, y = load()
    idx = y.sample(4000, random_state=0).index
    return X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)


def test_exact_duplicates_dropped_and_binary_target():
    from diabetes.config import DATA_PATH
    raw = pd.read_csv(DATA_PATH)
    X, y = load()
    assert len(y) == len(raw.drop_duplicates()) < len(raw)
    assert set(y.unique()) == {0, 1}


def test_no_lab_set_excludes_blood_tests():
    assert "hbA1c_level" not in FEATURE_SETS["no_labs"]
    assert "blood_glucose_level" not in FEATURE_SETS["no_labs"]


def test_clinical_rule():
    X = pd.DataFrame({"hbA1c_level": [5.0, 6.5, 5.0], "blood_glucose_level": [100, 100, 200]})
    assert ClinicalRule().fit(X, None).predict(X).tolist() == [0, 1, 1]


def test_no_labs_threshold_meets_recall_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    prob = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    t = pick_threshold(y, prob, "no_labs")
    assert (prob[y == 1] >= t).mean() >= 0.80


def test_column_order_does_not_change_predictions(sample, tmp_path):
    X, y = sample
    features = FEATURE_SETS["with_labs"]
    model = candidates(features)["logistic regression (class weights)"].fit(X[features], y)
    joblib.dump({"model": model, "features": features, "threshold": 0.5},
                tmp_path / "diabetes_with_labs.joblib")
    a = predict(X, "with_labs", tmp_path)["risk_probability"]
    b = predict(X[X.columns[::-1]], "with_labs", tmp_path)["risk_probability"]
    pd.testing.assert_series_equal(a, b)


def test_missing_column_rejected(sample):
    X, _ = sample
    with pytest.raises(ValueError, match="missing"):
        check_features(X.drop(columns=["bmi"]), FEATURE_SETS["no_labs"])
