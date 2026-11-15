"""Shared constants for the diabetes project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_PATH = DATA_DIR / "nhanes_2017_2020_adults.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
TARGET = "diabetes"

# NHANES 2017-March 2020 pre-pandemic cycle (file prefix P_). The 2021-2023 cycle
# (suffix _L) was checked first but drops the family-history question (MCQ300C) and
# the gestational-diabetes question (RHQ162), both inputs of the ADA risk test.
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{name}.xpt"
# SHA-256 of each file as downloaded on 2026-11-15. CDC occasionally re-releases
# files; a mismatch stops the build so results never change silently.
NHANES_FILES = {  # name: (sha256, contents)
    "P_DEMO": ("2e46c6c26bf77cd8989f64011ace12cbf42c0f3e03414eb59acc5328c8f87913",
               "demographics, MEC exam weights, survey design"),
    "P_BMX": ("7038d0da3169420a4a3cbaec09ac586a701b38e1c7b925431b60e13ac24fed66",
              "body measures (BMI, waist)"),
    "P_BPQ": ("88af3557dd185cee2533c94c5302b361e89a54eb50754704f35d4e8120868c02",
              "blood pressure questionnaire (told high blood pressure)"),
    "P_DIQ": ("79153f799fc2171792771c4aa250029c62807f14915f1f81405b03233b1e5ae3",
              "diabetes questionnaire (doctor-told diabetes)"),
    "P_MCQ": ("1de8e4e184ed375fddc053dbb5b82db13ee55e4d4d0ae12da9e407537230125c",
              "medical conditions (family history of diabetes)"),
    "P_PAQ": ("d0e13cebf96181949e983ab58d14c58689e801e8711d366d2b5daab3c7b9ee65",
              "physical activity questionnaire"),
    "P_RHQ": ("1344cd314b9144d88ae2cdb0aaacf09fa29a996ac00ad3e7c861c3d64b7bb116",
              "reproductive health (gestational diabetes)"),
    "P_GHB": ("dac9e423f56041c2ec46486cb16be16d3347e0a81a575985882ffad5a60e1195",
              "glycohemoglobin (HbA1c)"),
    "P_GLU": ("91b83c46fb4707c431838159135e93693a0572bce54074f5a890f0ee104732b3",
              "fasting plasma glucose (fasting subsample)"),
}

# Outcome: doctor-diagnosed diabetes OR HbA1c >= 6.5% (ADA diagnostic cut-off).
# Fasting glucose is measured only in a morning subsample, so it is NOT part of the
# primary label (that would make the outcome definition differ between people).
HBA1C_DIAGNOSTIC = 6.5
FPG_DIAGNOSTIC = 126

RACE_ETHNICITY = (
    "Mexican American",
    "Other Hispanic",
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Non-Hispanic Asian",
    "Other/Multiracial",
)
SEXES = ("Female", "Male")

# Features available without a blood test (questionnaire + scale + tape measure).
ADA_INPUTS = [
    "age", "sex", "gestational_diabetes", "family_history",
    "hypertension", "physically_active", "bmi",
]
NUMERIC = ["age", "bmi", "waist_cm", "income_poverty_ratio", "sedentary_minutes"]
BINARY = ["hypertension", "family_history", "gestational_diabetes", "physically_active"]
CATEGORICAL = ["sex", "race_ethnicity"]

FEATURE_SETS = {
    # Same seven questions the ADA test asks, but age and BMI kept continuous.
    "ada_inputs": ADA_INPUTS,
    # Everything available without a blood test.
    "no_labs": NUMERIC + BINARY + CATEGORICAL,
}
# Lab measurements kept in the data for the outcome and a reference analysis only.
# HbA1c defines the label, so it is never used as a model feature.
LAB_COLUMNS = ["hba1c", "fasting_glucose"]
DESIGN_COLUMNS = ["mec_weight", "fasting_weight", "psu", "stratum"]
