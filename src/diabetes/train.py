"""Can a no-blood-test model beat the ADA diabetes risk test? (NHANES 2017-March 2020)

Steps:
1. Survey-weighted prevalence and the undiagnosed share (MEC weights, design-based CIs).
2. Stratified 80/20 split. RepeatedStratifiedKFold (5 x 3) on the training set
   compares the ADA risk test with logistic regression and gradient boosting.
3. The ML decision threshold is set on out-of-fold training predictions to match the
   ADA test's sensitivity (score >= 5), then everything is evaluated once on the test set.
4. Secondary analyses: screening for *undiagnosed* diabetes among people without a
   diagnosis, and how much one fasting glucose test adds (fasting subsample).

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
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from .config import (  # noqa: E402
    FEATURE_SETS,
    FIGURES,
    FPG_DIAGNOSTIC,
    MODEL_DIR,
    REPORTS,
    SEED,
    TARGET,
)
from .data import load  # noqa: E402
from .model import ADARiskScore, candidates, logistic  # noqa: E402
from .survey import weighted_prevalence  # noqa: E402

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
OOF_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
N_BOOT = 1000
ADA_CUTOFF = 5
AGE_BANDS = ([20, 40, 60, 200], ["20-39", "40-59", "60+"])

# Chart styling: categorical slots 1-3 of a CVD-validated palette, recessive ink.
COLORS = {"ml": "#2a78d6", "lr_ada": "#eb6834", "ada": "#1baf7a"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"


# --------------------------------------------------------------------- metrics
def threshold_for_sensitivity(y, prob, target):
    """Highest threshold whose sensitivity on (y, prob) is at least `target`."""
    fpr, tpr, thresholds = roc_curve(y, prob)
    ok = tpr >= target - 1e-12
    return float(thresholds[ok][0])


def operating_point(y, pred, w=None):
    """Sensitivity, specificity, PPV and share flagged (optionally weighted)."""
    y, pred = np.asarray(y), np.asarray(pred).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pos, neg = y == 1, y == 0
    return {
        "sensitivity": float(w[pos & pred].sum() / w[pos].sum()),
        "specificity": float(w[neg & ~pred].sum() / w[neg].sum()),
        "ppv": float(w[pos & pred].sum() / max(w[pred].sum(), 1e-12)),
        "flagged_share": float(w[pred].sum() / w.sum()),
    }


def curve_scores(y, prob, w=None):
    return {"roc_auc": float(roc_auc_score(y, prob, sample_weight=w)),
            "pr_auc": float(average_precision_score(y, prob, sample_weight=w))}


def all_scores(y, prob, pred):
    return {**curve_scores(y, prob), **operating_point(y, pred)}


def bootstrap(y, systems, n=N_BOOT, seed=SEED):
    """Percentile 95% CIs for each system's metrics and for (system - reference)
    differences, from paired bootstrap resamples of the evaluation rows.

    `systems` maps name -> (prob, pred); the last entry is the reference.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    names = list(systems)
    ref = names[-1]
    draws = {k: [] for k in names}
    diffs = {k: [] for k in names[:-1]}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        s = {k: all_scores(y[i], p[i], d[i]) for k, (p, d) in systems.items()}
        for k in names:
            draws[k].append(s[k])
        for k in names[:-1]:
            diffs[k].append({m: s[k][m] - s[ref][m] for m in s[k]})

    def ci(rows):
        df = pd.DataFrame(rows)
        return {m: [round(float(df[m].quantile(0.025)), 3),
                    round(float(df[m].quantile(0.975)), 3)] for m in df}

    return ({k: ci(v) for k, v in draws.items()},
            {f"{k} minus {ref}": ci(v) for k, v in diffs.items()})


def rounded(d, nd=3):
    return {k: round(v, nd) if isinstance(v, float) else v for k, v in d.items()}


# -------------------------------------------------------------------- analyses
def population_summary(df):
    w, h, j = df["mec_weight"], df["stratum"], df["psu"]
    prev = weighted_prevalence(df[TARGET], w, h, j)
    cases = df[df[TARGET] == 1]
    undiag = weighted_prevalence(1 - cases["diagnosed_diabetes"], cases["mec_weight"],
                                 cases["stratum"], cases["psu"])
    fast = df[(df["fasting_weight"] > 0) & df["fasting_glucose"].notna()]
    with_fpg = (fast[TARGET] == 1) | (fast["fasting_glucose"] >= FPG_DIAGNOSTIC)
    fw = (fast["fasting_weight"], fast["stratum"], fast["psu"])
    return {
        "n_adults": int(len(df)),
        "n_cases": int(df[TARGET].sum()),
        "prevalence_unweighted": round(float(df[TARGET].mean()), 3),
        "prevalence_weighted_95ci": [round(x, 3) for x in prev],
        "undiagnosed_share_of_cases_unweighted": round(
            float(1 - cases["diagnosed_diabetes"].mean()), 3),
        "undiagnosed_share_of_cases_weighted_95ci": [round(x, 3) for x in undiag],
        "fasting_subsample": {
            "n": int(len(fast)),
            "prevalence_weighted_label": round(weighted_prevalence(fast[TARGET], *fw)[0], 3),
            "prevalence_weighted_if_fpg_ge_126_added": round(
                weighted_prevalence(with_fpg.astype(int), *fw)[0], 3),
        },
    }


def cross_validation(X_train, y_train):
    rows = []
    for name, (fs, model) in candidates(FEATURE_SETS).items():
        res = cross_validate(model, X_train[FEATURE_SETS[fs]], y_train, cv=CV,
                             scoring={"pr": "average_precision", "roc": "roc_auc",
                                      "brier": "neg_brier_score"})
        is_prob = not isinstance(model, ADARiskScore)  # ADA points are not probabilities
        rows.append({
            "model": name, "features": fs,
            "pr_auc_mean": round(res["test_pr"].mean(), 3),
            "pr_auc_std": round(res["test_pr"].std(), 3),
            "roc_auc_mean": round(res["test_roc"].mean(), 3),
            "roc_auc_std": round(res["test_roc"].std(), 3),
            "brier_mean": round(-res["test_brier"].mean(), 4) if is_prob else None,
        })
        print(rows[-1])
    return pd.DataFrame(rows)


def oof_comparison(X, y, model, features, n_boot=N_BOOT):
    """Out-of-fold comparison of `model` with the ADA test on (X, y)."""
    prob = cross_val_predict(model, X[features], y, cv=OOF_CV, method="predict_proba")[:, 1]
    ada = ADARiskScore().fit(X, y)
    ada_prob, ada_pred = ada.predict_proba(X)[:, 1], ada.predict(X)
    sens = operating_point(y, ada_pred)["sensitivity"]
    pred = prob >= threshold_for_sensitivity(y, prob, sens)
    cis, diffs = bootstrap(y, {"ml": (prob, pred), "ada": (ada_prob, ada_pred)}, n=n_boot)
    return {
        "n": int(len(y)), "n_cases": int(y.sum()), "prevalence": round(float(y.mean()), 3),
        "ml": rounded(all_scores(y, prob, pred)), "ml_95ci": cis["ml"],
        "ada_score_ge_5": rounded(all_scores(y, ada_prob, ada_pred)), "ada_95ci": cis["ada"],
        "difference_95ci": diffs["ml minus ada"],
        "note": "Out-of-fold predictions (5-fold CV). The ML operating point is read off "
                "its OOF ROC curve at the ADA test's sensitivity.",
    }


def fasting_glucose_reference(df, n_boot=N_BOOT):
    """How much does one blood test add? Fasting glucose is not part of the label."""
    fast = df[df["fasting_glucose"].notna()].reset_index(drop=True)
    y = fast[TARGET].to_numpy()
    feats = FEATURE_SETS["no_labs"]
    out = {"n": int(len(fast)), "n_cases": int(y.sum())}
    probs = {}
    for name, fs in [("no_labs", feats),
                     ("no_labs_plus_fasting_glucose", ["fasting_glucose"] + feats)]:
        probs[name] = cross_val_predict(logistic(fs), fast[fs], y, cv=OOF_CV,
                                        method="predict_proba")[:, 1]
        out[name] = rounded(curve_scores(y, probs[name]))
    systems = {k: (p, p >= 0.5) for k, p in probs.items()}
    systems = {"no_labs_plus_fasting_glucose": systems["no_labs_plus_fasting_glucose"],
               "no_labs": systems["no_labs"]}
    cis, diffs = bootstrap(y, systems, n=n_boot)
    out["95ci"] = {k: {m: cis[k][m] for m in ("roc_auc", "pr_auc")} for k in cis}
    out["difference_95ci"] = {k: {m: v[m] for m in ("roc_auc", "pr_auc")}
                              for k, v in diffs.items()}
    out["note"] = ("Logistic regression, 5-fold out-of-fold predictions on the fasting "
                   "subsample. HbA1c is never used as a feature because it defines the label.")
    return out


def subgroups(X, y, systems):
    bands = pd.cut(X["age"], AGE_BANDS[0], right=False, labels=AGE_BANDS[1]).astype(str)
    rows = []
    for col, groups in [("age band", bands), ("sex", X["sex"]),
                        ("race/ethnicity", X["race_ethnicity"])]:
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy()
            ym = y[m]
            row = {"group": col, "value": g, "n": int(m.sum()), "n_cases": int(ym.sum()),
                   "prevalence": round(float(ym.mean()), 3)}
            for name, (prob, pred) in systems.items():
                enough = ym.sum() >= 10 and (1 - ym).sum() >= 10
                row[f"{name}_roc_auc"] = (round(roc_auc_score(ym, prob[m]), 3)
                                          if enough else None)
                op = operating_point(ym, pred[m]) if ym.sum() and (1 - ym).sum() else {}
                row[f"{name}_sensitivity"] = round(op.get("sensitivity", np.nan), 3)
                row[f"{name}_specificity"] = round(op.get("specificity", np.nan), 3)
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ main
def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    population = population_summary(df)
    print(json.dumps(population, indent=2))

    y = df[TARGET].to_numpy()
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, stratify=y, random_state=SEED)
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # 1. Cross-validated comparison on the training set.
    cv = cross_validation(train_df, y_train)
    cv.to_csv(REPORTS / "cv_results.csv", index=False)
    learned = cv[cv["features"] == "no_labs"]
    chosen = learned.loc[learned["pr_auc_mean"].idxmax(), "model"]
    print(f"selected: {chosen}")
    all_models = candidates(FEATURE_SETS)
    final_fs, final_model = all_models[chosen]
    lr_ada = all_models["logistic regression, ADA inputs"][1]

    # 2. Thresholds on out-of-fold training predictions, matched to the ADA sensitivity.
    ada = ADARiskScore().fit(train_df, y_train)
    ada_train_sens = operating_point(y_train, ada.predict(train_df))["sensitivity"]
    fitted, thresholds = {}, {}
    for key, fs, model in [("ml", final_fs, final_model), ("lr_ada", "ada_inputs", lr_ada)]:
        feats = FEATURE_SETS[fs]
        oof = cross_val_predict(model, train_df[feats], y_train, cv=OOF_CV,
                                method="predict_proba")[:, 1]
        thresholds[key] = threshold_for_sensitivity(y_train, oof, ada_train_sens)
        fitted[key] = (fs, clone(model).fit(train_df[feats], y_train))

    # 3. One evaluation on the test set.
    systems = {}
    for key, (fs, model) in fitted.items():
        prob = model.predict_proba(test_df[FEATURE_SETS[fs]])[:, 1]
        systems[key] = (prob, prob >= thresholds[key])
    systems["ada"] = (ada.predict_proba(test_df)[:, 1], ada.predict(test_df).astype(bool))
    cis, diffs = bootstrap(y_test, systems)
    w_test = test_df["mec_weight"].to_numpy()
    test_results = {}
    labels = {"ml": f"{chosen} (matched sensitivity)",
              "lr_ada": "logistic regression, ADA inputs (matched sensitivity)",
              "ada": "ADA risk test, score >= 5"}
    for key, (prob, pred) in systems.items():
        test_results[key] = {
            "label": labels[key],
            "threshold": round(thresholds.get(key, ADA_CUTOFF), 4),
            "unweighted": rounded(all_scores(y_test, prob, pred)),
            "unweighted_95ci": cis[key],
            "weighted": rounded({**curve_scores(y_test, prob, w_test),
                                 **operating_point(y_test, pred, w_test)}),
        }
        if key != "ada":
            test_results[key]["brier"] = round(brier_score_loss(y_test, prob), 4)
    bang = ADARiskScore(variant="bang2009").fit(train_df, y_train)
    test_results["bang2009_original"] = {
        "label": "Bang 2009 original scoring (active = -1, no GDM item), score >= 5",
        "unweighted": rounded(all_scores(y_test, bang.predict_proba(test_df)[:, 1],
                                         bang.predict(test_df))),
    }

    # 4. Secondary analyses.
    nodx = df[df["diagnosed_diabetes"] == 0].reset_index(drop=True)
    undiagnosed = oof_comparison(nodx, nodx[TARGET].to_numpy(), clone(final_model),
                                 FEATURE_SETS[final_fs])
    undiagnosed["population"] = ("adults without a diagnosis; outcome = HbA1c >= 6.5% "
                                 "(undiagnosed diabetes)")
    fpg = fasting_glucose_reference(df)

    sub = subgroups(test_df, y_test, {"ml": systems["ml"], "ada": systems["ada"]})
    sub.to_csv(REPORTS / "subgroups.csv", index=False)

    summary = {
        "data": "NHANES 2017-March 2020 pre-pandemic (CDC/NCHS), adults 20+",
        "outcome": "diabetes = doctor-diagnosed (DIQ010 = 1) OR HbA1c >= 6.5%",
        "population": population,
        "split": {"n_train": int(len(y_train)), "n_test": int(len(y_test)),
                  "test_cases": int(y_test.sum()),
                  "test_prevalence": round(float(y_test.mean()), 3)},
        "selected_model": chosen,
        "matched_sensitivity_target": round(ada_train_sens, 3),
        "test": test_results,
        "test_differences_95ci": diffs,
        "undiagnosed_screening_oof": undiagnosed,
        "fasting_glucose_reference": fpg,
        "notes": {
            "weights": "'unweighted' = sample estimates; 'weighted' uses MEC exam weights "
                       "(point estimates only). Bootstrap CIs resample respondents and "
                       "ignore the clustered design, so they are somewhat too narrow.",
            "ada_auc": "ADA ROC/PR-AUC use the 0-11 points total as the score.",
        },
        "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__},
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("test", "test_differences_95ci")}, indent=2))

    for key, name in [("ml", "no_labs"), ("lr_ada", "ada_inputs")]:
        fs, model = fitted[key]
        joblib.dump({"model": model, "features": FEATURE_SETS[fs],
                     "threshold": thresholds[key]}, MODEL_DIR / f"diabetes_{name}.joblib")

    importance = permutation_table(fitted["ml"][1], test_df[FEATURE_SETS[final_fs]], y_test)
    importance.to_csv(REPORTS / "permutation_importance.csv", index=False)
    make_figures(y_test, systems, test_results, importance, sub, chosen)


def permutation_table(model, X, y):
    imp = permutation_importance(model, X, y, scoring="average_precision", n_repeats=30,
                                 random_state=SEED)
    return (pd.DataFrame({"feature": X.columns,
                          "pr_auc_drop_mean": imp.importances_mean.round(4),
                          "pr_auc_drop_std": imp.importances_std.round(4)})
            .sort_values("pr_auc_drop_mean", ascending=False))


# --------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def make_figures(y, systems, results, importance, sub, chosen):
    names = {"ml": chosen.capitalize(), "lr_ada": "Logistic regression, ADA inputs",
             "ada": "ADA risk test (points)"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for key, (prob, pred) in systems.items():
        r = results[key]["unweighted"]
        fpr, tpr, _ = roc_curve(y, prob)
        ax1.plot(fpr, tpr, color=COLORS[key], lw=2,
                 label=f"{names[key]} (AUC {r['roc_auc']:.3f})")
        ax1.scatter(1 - r["specificity"], r["sensitivity"], s=64, color=COLORS[key],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
        p, rc, _ = precision_recall_curve(y, prob)
        ax2.plot(rc, p, color=COLORS[key], lw=2,
                 label=f"{names[key]} (AP {r['pr_auc']:.3f})")
        ax2.scatter(r["sensitivity"], r["ppv"], s=64, color=COLORS[key],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
    ax1.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax2.axhline(y.mean(), ls="--", color=MUTED, lw=1)
    ax2.text(0.01, y.mean() - 0.04, f"prevalence {y.mean():.2f}", ha="left",
             color=MUTED, fontsize=8)
    ax1.set(xlabel="1 - specificity", ylabel="Sensitivity", title="ROC")
    ax2.set(xlabel="Sensitivity (recall)", ylabel="Precision (PPV)", title="Precision-recall")
    for ax in (ax1, ax2):
        _style(ax)
        ax.legend(fontsize=8, frameon=False, loc="lower right" if ax is ax1 else "upper right")
    fig.suptitle("Test set. Dots: ADA score >= 5, and ML thresholds matched to its sensitivity",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_pr_curves.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=SURFACE)
    for key in ("ml", "lr_ada"):
        frac, mean = calibration_curve(y, systems[key][0], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=5, lw=2, color=COLORS[key],
                label=f"{names[key]} (Brier {results[key]['brier']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    ax.set(xlabel="Predicted probability (decile bins)", ylabel="Observed share with diabetes",
           title="Calibration, test set")
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    imp = importance.sort_values("pr_auc_drop_mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    ax.barh(imp["feature"], imp["pr_auc_drop_mean"], xerr=imp["pr_auc_drop_std"],
            color=COLORS["ml"], height=0.6, error_kw={"ecolor": MUTED, "lw": 1})
    ax.set(xlabel="Drop in PR-AUC when the feature is shuffled (30 repeats)",
           title=f"Permutation importance, {chosen}, test set")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_importance.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE, sharey=True)
    ylab = [f"{r.group}: {r.value} (n={r.n})" for r in sub.itertuples()]
    pos = np.arange(len(sub))
    for ax, metric, title in [(axes[0], "sensitivity", "Sensitivity"),
                              (axes[1], "specificity", "Specificity")]:
        for k, off in (("ml", -0.17), ("ada", 0.17)):
            ax.barh(pos + off, sub[f"{k}_{metric}"], height=0.32, color=COLORS[k],
                    label="ADA risk test, score >= 5" if k == "ada" else names["ml"])
        ax.set(title=title, xlim=(0, 1))
        _style(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(pos, ylab)
    axes[0].invert_yaxis()
    fig.suptitle("Subgroups, test set: ML at matched sensitivity vs ADA score >= 5",
                 fontsize=10, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "subgroups.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
