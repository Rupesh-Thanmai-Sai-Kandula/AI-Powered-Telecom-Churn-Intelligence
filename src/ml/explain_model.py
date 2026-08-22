import sys
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = MODEL_DIR / "tuned_random_forest.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
THRESHOLD_PATH = MODEL_DIR / "best_threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL AND PREPROCESSOR")
print("=" * 70)

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

best_threshold = joblib.load(
    THRESHOLD_PATH
)

print("\nModel loaded:")
print(MODEL_PATH)

print("\nPreprocessor loaded:")
print(PREPROCESSOR_PATH)

print(
    f"\nBest threshold: {best_threshold:.2f}"
)


# ============================================================
# GET REAL FEATURE NAMES
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING TRANSFORMED FEATURE NAMES")
print("=" * 70)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_importances = (
    model.feature_importances_
)


print(
    f"\nNumber of transformed features: "
    f"{len(feature_names)}"
)

print(
    f"Number of model importances: "
    f"{len(feature_importances)}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(feature_names) != len(feature_importances):

    raise ValueError(
        "Mismatch between preprocessing features "
        "and model feature importances."
    )


# ============================================================
# FEATURE-LEVEL IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": feature_importances
    }
)


importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# CLEAN FEATURE NAMES
# ============================================================

importance_df["Original_Feature"] = (
    importance_df["Feature"]
    .str.split("__", n=1)
    .str[-1]
)


# ============================================================
# SAVE DETAILED IMPORTANCE
# ============================================================

detailed_path = (
    REPORT_DIR /
    "feature_importance_detailed.csv"
)

importance_df.to_csv(
    detailed_path,
    index=False
)


# ============================================================
# DISPLAY TOP TRANSFORMED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 TRANSFORMED FEATURES")
print("=" * 70)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# AGGREGATE BY ORIGINAL FEATURE
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATING IMPORTANCE BY ORIGINAL FEATURE")
print("=" * 70)


aggregated_df = (
    importance_df
    .groupby(
        "Original_Feature",
        as_index=False
    )["Importance"]
    .sum()
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE AGGREGATED IMPORTANCE
# ============================================================

aggregated_path = (
    REPORT_DIR /
    "feature_importance_by_original_feature.csv"
)

aggregated_df.to_csv(
    aggregated_path,
    index=False
)


# ============================================================
# DISPLAY ORIGINAL FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL FEATURE IMPORTANCE")
print("=" * 70)

print(
    aggregated_df.to_string(
        index=False
    )
)


# ============================================================
# ZERO-IMPORTANCE FEATURES
# ============================================================

zero_importance = (
    importance_df[
        importance_df["Importance"] == 0
    ]
)


print("\n" + "=" * 70)
print("ZERO-IMPORTANCE FEATURES")
print("=" * 70)

print(
    f"\nNumber of zero-importance "
    f"transformed features: "
    f"{len(zero_importance)}"
)


# ============================================================
# SAVE ZERO-IMPORTANCE FEATURES
# ============================================================

zero_path = (
    REPORT_DIR /
    "zero_importance_features.csv"
)

zero_importance.to_csv(
    zero_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINABILITY ANALYSIS COMPLETED")
print("=" * 70)

print(
    "\nGenerated files:"
)

print(
    f" - {detailed_path.name}"
)

print(
    f" - {aggregated_path.name}"
)

print(
    f" - {zero_path.name}"
)

print(
    "\nReports saved to:"
)

print(REPORT_DIR)

print(
    "\nExplainability analysis completed successfully."
)