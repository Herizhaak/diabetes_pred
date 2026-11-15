"""Download NHANES 2017-March 2020 files and build the analysis table.

Run from the repo root (src/ on PYTHONPATH):
    python -m diabetes.nhanes            # download to data/raw/ (cached) and build the CSV

Raw XPT files are cached in data/raw/ (git-ignored) and checked against pinned
SHA-256 hashes. The processed CSV (about 1 MB) is committed so tests and CI run
without contacting the CDC server.

Source: CDC National Center for Health Statistics, NHANES 2017-March 2020
pre-pandemic data files. US government work, public domain.
"""
import argparse
import hashlib
import urllib.request

import numpy as np
import pandas as pd

from .config import (
    DATA_PATH,
    HBA1C_DIAGNOSTIC,
    NHANES_BASE_URL,
    NHANES_FILES,
    RACE_ETHNICITY,
    RAW_DIR,
)

RACE_CODES = {  # RIDRETH3
    1: RACE_ETHNICITY[0], 2: RACE_ETHNICITY[1], 3: RACE_ETHNICITY[2],
    4: RACE_ETHNICITY[3], 6: RACE_ETHNICITY[4], 7: RACE_ETHNICITY[5],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(raw_dir=RAW_DIR, verify=True):
    """Fetch every file once into raw_dir; return {name: path}."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (expected, _) in NHANES_FILES.items():
        path = raw_dir / f"{name}.xpt"
        if not path.exists():
            url = NHANES_BASE_URL.format(name=name)
            print(f"downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "nhanes-download/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                path.write_bytes(r.read())
        if verify and sha256(path) != expected:
            raise RuntimeError(
                f"{path.name}: SHA-256 mismatch. CDC may have re-released the file; "
                "delete it and re-run, or pass --no-verify to accept the new version.")
        paths[name] = path
    return paths


def read_xpt(path):
    df = pd.read_sas(path, format="xport", encoding="latin-1")
    num = df.select_dtypes("number").columns
    # XPORT stores exact zeros as ~5.4e-79; snap them back to 0.
    df[num] = df[num].mask(df[num].abs() < 1e-10, 0.0)
    return df.set_index("SEQN")


def yes_no(s):
    """NHANES 1 = yes, 2 = no; refused (7), don't know (9) and skips become NaN."""
    return s.map({1.0: 1.0, 2.0: 0.0})


def build(frames):
    """Join the component files into one row per adult. `frames` maps file name to a
    DataFrame indexed by SEQN (see read_xpt)."""
    demo = frames["P_DEMO"]
    df = pd.DataFrame(index=demo.index)
    df["age"] = demo["RIDAGEYR"]  # top-coded at 80
    df["sex"] = demo["RIAGENDR"].map({1.0: "Male", 2.0: "Female"})
    df["race_ethnicity"] = demo["RIDRETH3"].map(RACE_CODES)
    df["income_poverty_ratio"] = demo["INDFMPIR"]  # top-coded at 5

    bmx = frames["P_BMX"].reindex(df.index)
    df["bmi"] = bmx["BMXBMI"]
    df["waist_cm"] = bmx["BMXWAIST"]

    df["hypertension"] = yes_no(frames["P_BPQ"]["BPQ020"]).reindex(df.index)
    df["family_history"] = yes_no(frames["P_MCQ"]["MCQ300C"]).reindex(df.index)

    # Gestational diabetes: asked of women who have been pregnant (RHQ162). Men and
    # women never pregnant (RHQ131 = 2) score 0, as on the ADA form.
    rhq = frames["P_RHQ"].reindex(df.index)
    gdm = yes_no(rhq["RHQ162"])
    gdm = gdm.mask(gdm.isna() & (rhq["RHQ131"] == 2), 0.0)
    df["gestational_diabetes"] = gdm.mask(df["sex"] == "Male", 0.0)

    # "Physically active": any moderate or vigorous leisure-time activity in a typical
    # week (PAQ665 / PAQ650). Work and transport activity are not counted.
    paq = frames["P_PAQ"].reindex(df.index)
    vig, mod = yes_no(paq["PAQ650"]), yes_no(paq["PAQ665"])
    active = pd.Series(np.nan, index=df.index)
    active[(vig == 0) & (mod == 0)] = 0.0
    active[(vig == 1) | (mod == 1)] = 1.0
    df["physically_active"] = active
    df["sedentary_minutes"] = paq["PAD680"].where(paq["PAD680"] <= 1440)

    df["hba1c"] = frames["P_GHB"]["LBXGH"].reindex(df.index)
    glu = frames["P_GLU"].reindex(df.index)
    df["fasting_glucose"] = glu["LBXGLU"]

    diq = frames["P_DIQ"]["DIQ010"].reindex(df.index)
    df["diagnosed_diabetes"] = (diq == 1).astype(float).where(diq.isin([1, 2, 3]))

    df["mec_weight"] = demo["WTMECPRP"]
    df["fasting_weight"] = glu["WTSAFPRP"]
    df["psu"] = demo["SDMVPSU"]
    df["stratum"] = demo["SDMVSTRA"]

    # Inclusion: adults 20+, examined at the MEC, not pregnant, diabetes question
    # answered, and outcome determinable (diagnosed, or HbA1c measured).
    keep = (
        (df["age"] >= 20)
        & (demo["RIDSTATR"] == 2)
        & (demo["RIDEXPRG"] != 1)
        & df["diagnosed_diabetes"].notna()
        & ((df["diagnosed_diabetes"] == 1) | df["hba1c"].notna())
    )
    df = df[keep].copy()
    df["diabetes"] = ((df["diagnosed_diabetes"] == 1)
                      | (df["hba1c"] >= HBA1C_DIAGNOSTIC)).astype(int)
    df["diagnosed_diabetes"] = df["diagnosed_diabetes"].astype(int)
    for col in ["hypertension", "family_history", "gestational_diabetes", "physically_active"]:
        df[col] = df[col].astype("Int64")
    for col in ["age", "psu", "stratum"]:
        df[col] = df[col].astype(int)
    df.index.name = "seqn"
    return df.reset_index()


def main():
    parser = argparse.ArgumentParser(description="Download NHANES files and build the CSV.")
    parser.add_argument("--no-verify", action="store_true", help="skip SHA-256 checks")
    args = parser.parse_args()
    paths = download(verify=not args.no_verify)
    df = build({name: read_xpt(p) for name, p in paths.items()})
    df.to_csv(DATA_PATH, index=False, float_format="%.8g", lineterminator="\n")
    print(f"wrote {DATA_PATH} ({len(df)} rows, {df['diabetes'].mean():.3f} with diabetes)")


if __name__ == "__main__":
    main()
