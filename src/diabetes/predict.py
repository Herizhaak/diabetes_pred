"""Score people with a saved model and, alongside, the ADA risk test.

Run from the repo root:
    python -m diabetes.predict people.csv --model no_labs     (or ada_inputs)

The CSV needs the model's feature columns (see config.FEATURE_SETS); missing
values are allowed and imputed inside the pipeline.
"""
import argparse

import joblib
import pandas as pd

from .config import ADA_INPUTS, MODEL_DIR
from .data import check_features
from .model import ADARiskScore


def predict(df, model_name="no_labs", model_dir=MODEL_DIR):
    bundle = joblib.load(model_dir / f"diabetes_{model_name}.joblib")
    prob = bundle["model"].predict_proba(check_features(df, bundle["features"]))[:, 1]
    out = df.copy()
    out["risk_probability"] = prob.round(3)
    out["flag"] = prob >= bundle["threshold"]
    out["ada_points"] = ADARiskScore().fit(None).points(check_features(df, ADA_INPUTS))
    return out


def main():
    parser = argparse.ArgumentParser(description="Score people with a diabetes model.")
    parser.add_argument("csv")
    parser.add_argument("--model", choices=["no_labs", "ada_inputs"], default="no_labs")
    args = parser.parse_args()
    result = predict(pd.read_csv(args.csv), args.model)
    print(result[["risk_probability", "flag", "ada_points"]].to_string())


if __name__ == "__main__":
    main()
