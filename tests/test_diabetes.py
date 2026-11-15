import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes import nhanes  # noqa: E402
from diabetes.config import FEATURE_SETS, LAB_COLUMNS, TARGET  # noqa: E402
from diabetes.data import check_features, load  # noqa: E402
from diabetes.model import ADARiskScore, boosting, candidates, logistic  # noqa: E402
from diabetes.predict import predict  # noqa: E402
from diabetes.survey import weighted_prevalence  # noqa: E402
from diabetes.train import (  # noqa: E402
    bootstrap,
    operating_point,
    threshold_for_sensitivity,
)

nan = np.nan


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def sample(data):
    return data.sample(2000, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------- data / outcome
def test_processed_data_schema_and_population(data):
    assert data["seqn"].is_unique
    assert data["age"].min() >= 20
    assert set(data[TARGET].unique()) == {0, 1}
    assert 0.10 < data[TARGET].mean() < 0.30
    assert (data["mec_weight"] > 0).all()


def test_outcome_is_diagnosis_or_hba1c(data):
    expected = (data["diagnosed_diabetes"] == 1) | (data["hba1c"] >= 6.5)
    assert (data[TARGET] == expected.astype(int)).all()
    # Outcome is determinable for everyone kept.
    assert (data["hba1c"].notna() | (data["diagnosed_diabetes"] == 1)).all()


def test_no_lab_feature_sets_exclude_blood_tests():
    for features in FEATURE_SETS.values():
        assert not set(features) & set(LAB_COLUMNS)
        assert "diagnosed_diabetes" not in features and TARGET not in features


def _frames():
    idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8], name="SEQN")
    one = lambda v: pd.DataFrame(v, index=idx)  # noqa: E731
    return {
        "P_DEMO": one({"RIDAGEYR": [45, 30, 50, 19, 30, 40, 40, 40],
                       "RIAGENDR": [1, 2, 2, 1, 2, 1, 1, 1],
                       "RIDRETH3": [3, 6, 1, 3, 3, 3, 3, 3],
                       "INDFMPIR": [1.5, nan, 5.0, 1, 1, 1, 1, 1],
                       "RIDSTATR": [2, 2, 2, 2, 2, 1, 2, 2],
                       "RIDEXPRG": [nan, 2, 2, nan, 1, nan, nan, nan],
                       "WTMECPRP": [1000.0] * 8, "SDMVPSU": [1] * 8, "SDMVSTRA": [150] * 8}),
        "P_BMX": one({"BMXBMI": [31, 22, 41, 20, 20, 20, 20, 20],
                      "BMXWAIST": [100, 80, nan, 70, 70, 70, 70, 70]}),
        "P_BPQ": one({"BPQ020": [9, 2, 1, 2, 2, 2, 2, 2]}),
        "P_MCQ": one({"MCQ300C": [1, 7, 2, 2, 2, 2, 2, 2]}),
        "P_RHQ": one({"RHQ162": [nan, nan, 1, nan, nan, nan, nan, nan],
                      "RHQ131": [nan, 2, 1, nan, 1, nan, nan, nan]}),
        "P_PAQ": one({"PAQ650": [2, 1, 2, 2, 2, 2, 2, 2], "PAQ665": [2, 9, nan, 2, 2, 2, 2, 2],
                      "PAD680": [9999, 300, 600, 60, 60, 60, 60, 60]}),
        "P_GHB": one({"LBXGH": [nan, 6.7, 5.5, 5, 5, 5, nan, 5]}),
        "P_GLU": one({"LBXGLU": [nan, 130, nan, nan, nan, nan, nan, nan],
                      "WTSAFPRP": [nan, 2000, nan, nan, nan, nan, nan, nan]}),
        "P_DIQ": one({"DIQ010": [1, 2, 3, 2, 2, 2, 2, 9]}),
    }


def test_build_recodes_and_applies_inclusion_rules():
    df = nhanes.build(_frames()).set_index("seqn")
    # 4: age 19, 5: pregnant, 6: not examined, 7: no HbA1c and not diagnosed, 8: DIQ "don't know"
    assert list(df.index) == [1, 2, 3]
    assert df[TARGET].tolist() == [1, 1, 0]  # diagnosed / HbA1c 6.7 / borderline + 5.5
    assert df["diagnosed_diabetes"].tolist() == [1, 0, 0]
    assert df["sex"].tolist() == ["Male", "Female", "Female"]
    assert df.loc[2, "race_ethnicity"] == "Non-Hispanic Asian"
    assert df["gestational_diabetes"].tolist() == [0, 0, 1]  # man / never pregnant / yes
    assert pd.isna(df.loc[1, "hypertension"]) and pd.isna(df.loc[2, "family_history"])
    # No/no = inactive; any yes = active; no + missing = unknown.
    assert df.loc[1, "physically_active"] == 0 and df.loc[2, "physically_active"] == 1
    assert pd.isna(df.loc[3, "physically_active"])
    assert pd.isna(df.loc[1, "sedentary_minutes"])
    assert df.loc[2, "fasting_glucose"] == 130


def test_download_uses_cache_and_checks_hash(tmp_path, monkeypatch):
    f = tmp_path / "X.xpt"
    f.write_bytes(b"cached")
    monkeypatch.setattr(nhanes, "NHANES_FILES", {"X": (nhanes.sha256(f), "test")})
    assert nhanes.download(tmp_path) == {"X": f}  # no network: file is cached
    monkeypatch.setattr(nhanes, "NHANES_FILES", {"X": ("0" * 64, "test")})
    with pytest.raises(RuntimeError, match="SHA-256"):
        nhanes.download(tmp_path)


# ------------------------------------------------------------------- ADA score
def _person(**kw):
    base = {"age": 35, "sex": "Female", "gestational_diabetes": 0, "family_history": 0,
            "hypertension": 0, "physically_active": 1, "bmi": 22.0}
    return {**base, **kw}


@pytest.mark.parametrize("kw,points", [
    ({}, 0),
    ({"age": 40}, 1), ({"age": 55}, 2), ({"age": 80}, 3),
    ({"sex": "Male"}, 1), ({"gestational_diabetes": 1}, 1), ({"family_history": 1}, 1),
    ({"hypertension": 1}, 1), ({"physically_active": 0}, 1),
    ({"bmi": 25.0}, 1), ({"bmi": 35.0}, 2), ({"bmi": 40.0}, 3),
    ({"age": 62, "sex": "Male", "family_history": 1, "hypertension": 1,
      "physically_active": 0, "bmi": 42.0}, 10),
    ({"family_history": nan, "bmi": nan, "physically_active": nan}, 0),  # unanswered = 0
])
def test_ada_points(kw, points):
    assert ADARiskScore().fit(None).points(pd.DataFrame([_person(**kw)]))[0] == points


def test_ada_threshold_and_bang_variant():
    X = pd.DataFrame([_person(age=62, sex="Male", bmi=27.0),  # 3 + 1 + 1 = 5
                      _person(age=62, sex="Male")])           # 4
    assert ADARiskScore().fit(X).predict(X).tolist() == [1, 0]
    bang = ADARiskScore(variant="bang2009").fit(X)
    # Bang 2009: physically active scores -1 and there is no GDM item.
    assert bang.points(X).tolist() == [4, 3]
    proba = ADARiskScore().fit(X).predict_proba(X)
    assert ((proba >= 0) & (proba <= 1)).all() and np.allclose(proba.sum(axis=1), 1)
    with pytest.raises(ValueError):
        ADARiskScore(variant="nope").fit(X)


def test_ada_is_a_sklearn_classifier(sample):
    ada = ADARiskScore()
    assert is_classifier(ada) and clone(ada).get_params() == {"threshold": 5, "variant": "ada"}
    auc = cross_val_score(ada, sample, sample[TARGET], cv=3, scoring="roc_auc")
    assert (auc > 0.6).all()


# ------------------------------------------------------------------- pipelines
def test_pipelines_handle_missing_values(sample):
    feats = FEATURE_SETS["no_labs"]
    X = sample[feats].copy()
    X.loc[:50, ["bmi", "waist_cm", "hypertension", "income_poverty_ratio"]] = nan
    for make in (logistic, boosting):
        prob = make(feats).fit(X, sample[TARGET]).predict_proba(X)[:, 1]
        assert np.isfinite(prob).all()


def test_candidates_cover_baselines_and_models():
    names = set(candidates(FEATURE_SETS))
    assert "baseline: ADA risk test" in names
    assert {"logistic regression, all no-lab", "gradient boosting, all no-lab"} <= names


def test_threshold_matches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    prob = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    t = threshold_for_sensitivity(y, prob, 0.85)
    assert operating_point(y, prob >= t)["sensitivity"] >= 0.85
    assert operating_point(y, prob >= t + 1e-3)["sensitivity"] < 0.86


def test_operating_point_weighted():
    y, pred = np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])
    assert operating_point(y, pred)["sensitivity"] == 0.5
    op = operating_point(y, pred, w=[3, 1, 1, 1])
    assert op["sensitivity"] == 0.75 and op["specificity"] == 0.5 and op["ppv"] == 0.75


def test_bootstrap_cis_bracket_point_estimates():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p1, p2 = y * 0.5 + rng.random(500) * 0.5, rng.random(500)
    cis, diffs = bootstrap(y, {"a": (p1, p1 > 0.5), "b": (p2, p2 > 0.5)}, n=200)
    lo, hi = cis["a"]["roc_auc"]
    assert lo <= round(roc_auc_score(y, p1), 3) <= hi
    assert diffs["a minus b"]["roc_auc"][0] > 0  # a is clearly better than random b


# ---------------------------------------------------------------------- survey
def test_weighted_prevalence():
    y = np.array([1, 0, 0, 0] * 50)
    strata, psu = np.repeat([1, 2], 100), np.tile(np.repeat([1, 2], 50), 2)
    p, lo, hi = weighted_prevalence(y, np.ones(200), strata, psu)
    assert p == pytest.approx(0.25) and lo <= p <= hi
    w = np.where(y == 1, 3.0, 1.0)  # up-weighting cases raises the estimate
    assert weighted_prevalence(y, w, strata, psu)[0] == pytest.approx(0.5)


# --------------------------------------------------------------------- predict
def test_column_order_does_not_change_predictions(sample, tmp_path):
    feats = FEATURE_SETS["no_labs"]
    model = logistic(feats).fit(sample[feats], sample[TARGET])
    joblib.dump({"model": model, "features": feats, "threshold": 0.2},
                tmp_path / "diabetes_no_labs.joblib")
    a = predict(sample, "no_labs", tmp_path)
    b = predict(sample[sample.columns[::-1]], "no_labs", tmp_path)
    pd.testing.assert_series_equal(a["risk_probability"], b["risk_probability"])
    assert a["ada_points"].between(0, 11).all()


def test_missing_column_rejected(sample):
    with pytest.raises(ValueError, match="missing"):
        check_features(sample.drop(columns=["bmi"]), FEATURE_SETS["no_labs"])
