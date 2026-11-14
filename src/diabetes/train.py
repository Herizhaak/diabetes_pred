"""Train and evaluate two diabetes models:

* with_labs - uses HbA1c and blood glucose (near-diagnostic; compared with a
  simple clinical rule to show what the model adds)
* no_labs   - screening without a blood test (age, BMI, blood pressure,
  heart disease, smoking, gender)

Run from the repo root:  python -m diabetes.train   (with src/ on PYTHONPATH)
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from .config import FEATURE_SETS, FIGURES, MODEL_DIR, REPORTS, SEED  # noqa: E402
from .data import load  # noqa: E402
from .model import ClinicalRule, candidates  # noqa: E402

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCREENING_RECALL = 0.80  # no-lab model: catch at least 80% of diabetics


def pick_threshold(y, prob, feature_set):
    """Choose the decision threshold on out-of-fold training predictions."""
    precision, recall, thresholds = precision_recall_curve(y, prob)
    precision, recall = precision[:-1], recall[:-1]
    if feature_set == "no_labs":
        ok = recall >= SCREENING_RECALL
        return float(thresholds[ok][np.argmax(precision[ok])])
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def scores(y, prob, pred):
    return {
        "pr_auc": round(average_precision_score(y, prob), 3),
        "roc_auc": round(roc_auc_score(y, prob), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred), 3),
        "f1": round(f1_score(y, pred), 3),
        "flagged_share": round(float(np.mean(pred)), 3),
    }


def bootstrap_ci(y, prob, pred, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, prob, pred = map(np.asarray, (y, prob, pred))
    out = {"pr_auc": [], "roc_auc": [], "precision": [], "recall": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out["pr_auc"].append(average_precision_score(y[i], prob[i]))
        out["roc_auc"].append(roc_auc_score(y[i], prob[i]))
        out["precision"].append(precision_score(y[i], pred[i], zero_division=0))
        out["recall"].append(recall_score(y[i], pred[i]))
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
            for k, v in out.items()}


def subgroups(X, y, pred):
    bands = pd.cut(X["age"], [0, 40, 60, 120], right=False, labels=["<40", "40-59", "60+"])
    rows = []
    for col, groups in [("gender", X["gender"]), ("age band", bands)]:
        for g in groups.dropna().unique():
            m = (groups == g).to_numpy()
            if y[m].sum() < 20:  # too few positives for a stable estimate
                continue
            rows.append({"group": col, "value": str(g), "n": int(m.sum()),
                         "prevalence": round(float(y[m].mean()), 3),
                         "recall": round(recall_score(y[m], pred[m]), 3),
                         "precision": round(precision_score(y[m], pred[m], zero_division=0), 3)})
    return pd.DataFrame(rows)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    y_test_np = y_test.to_numpy()

    cv_rows, results, curves = [], {}, {}
    for fs, features in FEATURE_SETS.items():
        Xtr, Xte = X_train[features], X_test[features]

        # 1. Cross-validated comparison on the training set.
        for name, model in candidates(features).items():
            res = cross_validate(model, Xtr, y_train, cv=CV,
                                 scoring=["average_precision", "roc_auc"])
            cv_rows.append({
                "feature_set": fs, "model": name,
                "pr_auc_mean": round(res["test_average_precision"].mean(), 3),
                "pr_auc_std": round(res["test_average_precision"].std(), 3),
                "roc_auc_mean": round(res["test_roc_auc"].mean(), 3),
                "roc_auc_std": round(res["test_roc_auc"].std(), 3),
            })
            print(cv_rows[-1])
        learned = [r for r in cv_rows if r["feature_set"] == fs and "baseline" not in r["model"]]
        chosen = max(learned, key=lambda r: r["pr_auc_mean"])["model"]

        # 2. Threshold from out-of-fold predictions (training data only).
        model = candidates(features)[chosen]
        oof = cross_val_predict(model, Xtr, y_train, cv=CV, method="predict_proba")[:, 1]
        threshold = pick_threshold(y_train.to_numpy(), oof, fs)

        # 3. Fit on all training data, evaluate once on the test set.
        model.fit(Xtr, y_train)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= threshold).astype(int)
        results[fs] = {
            "model": chosen,
            "threshold": round(threshold, 3),
            "threshold_rule": ("max precision with OOF recall >= 0.80" if fs == "no_labs"
                               else "max F1 on out-of-fold predictions"),
            "test": scores(y_test_np, prob, pred),
            "test_95ci_bootstrap": bootstrap_ci(y_test_np, prob, pred),
            "brier": round(brier_score_loss(y_test_np, prob), 4),
        }
        subgroups(Xte, y_test_np, pred).to_csv(REPORTS / f"subgroups_{fs}.csv", index=False)
        curves[fs] = (prob, model, Xte)
        joblib.dump({"model": model, "features": features, "threshold": threshold},
                    MODEL_DIR / f"diabetes_{fs}.joblib")

    rule_prob = ClinicalRule().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    results["clinical_rule"] = {"test": scores(y_test_np, rule_prob, rule_prob.astype(int))}

    pd.DataFrame(cv_rows).to_csv(REPORTS / "cv_results.csv", index=False)
    summary = {
        "n_rows_after_dedup": int(len(y)), "n_train": int(len(y_train)),
        "n_test": int(len(y_test)), "test_prevalence": round(float(y_test.mean()), 3),
        "results": results, "versions": {"scikit-learn": sklearn.__version__},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    make_figures(y_test_np, curves, rule_prob, results)


def make_figures(y, curves, rule_prob, results):
    labels = {"with_labs": "With HbA1c + glucose", "no_labs": "No blood test"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for fs, (prob, _, _) in curves.items():
        p, r, _ = precision_recall_curve(y, prob)
        ax.plot(r, p, label=f"{labels[fs]} (PR-AUC {results[fs]['test']['pr_auc']})")
        t = results[fs]["test"]
        ax.scatter(t["recall"], t["precision"], zorder=3)
    rule = results["clinical_rule"]["test"]
    ax.scatter(rule["recall"], rule["precision"], marker="x", s=80, color="black",
               label="Clinical rule: HbA1c >= 6.5 or glucose >= 200", zorder=3)
    ax.axhline(y.mean(), color="grey", ls="--", label=f"Prevalence ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, test set (dots = chosen thresholds)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for fs, (prob, _, _) in curves.items():
        frac, mean = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=labels[fs])
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed share with diabetes")
    ax.set_title("Calibration, test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for ax, (fs, (_, model, Xte)) in zip(axes, curves.items()):
        sample = Xte.sample(5000, random_state=SEED)
        imp = permutation_importance(model, sample, y[sample.index.map(Xte.index.get_loc)],
                                     scoring="average_precision", n_repeats=5,
                                     random_state=SEED)
        s = pd.Series(imp.importances_mean, index=sample.columns).sort_values()
        rows += [{"feature_set": fs, "feature": k, "pr_auc_drop": round(v, 4)}
                 for k, v in s.items()]
        ax.barh(s.index, s.values)
        ax.set_title(labels[fs])
        ax.set_xlabel("Drop in PR-AUC when shuffled")
    fig.suptitle("Permutation importance (5,000 test rows)")
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_importance.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORTS / "permutation_importance.csv", index=False)


if __name__ == "__main__":
    main()
