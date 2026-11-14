"""Score patients with a saved model.

Run from the repo root:
    python -m diabetes.predict patients.csv --model no_labs     (or with_labs)
"""
import argparse

import joblib
import pandas as pd

from .config import MODEL_DIR
from .data import check_features


def predict(df, model_name="with_labs", model_dir=MODEL_DIR):
    bundle = joblib.load(model_dir / f"diabetes_{model_name}.joblib")
    prob = bundle["model"].predict_proba(check_features(df, bundle["features"]))[:, 1]
    out = df.copy()
    out["risk_probability"] = prob.round(3)
    out["flag"] = prob >= bundle["threshold"]
    return out


def main():
    parser = argparse.ArgumentParser(description="Score patients with a diabetes model.")
    parser.add_argument("csv")
    parser.add_argument("--model", choices=["with_labs", "no_labs"], default="with_labs")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.csv), args.model)
    print(result[["risk_probability", "flag"]].to_string())


if __name__ == "__main__":
    main()
