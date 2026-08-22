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
# DATABASE CONNECTION
# ============================================================

from database.database import get_engine


# ============================================================
# CONFIGURATION
# ============================================================

NUMERICAL_FEATURES = [
    "montant",
    "frequence_rech",
    "revenue",
    "arpu_segment",
    "frequence",
    "data_volume",
    "on_net",
    "orange",
    "tigo",
    "zone1",
    "zone2",
    "regularity",
    "freq_top_pack"
]

CATEGORICAL_FEATURES = [
    "region",
    "tenure",
    "mrg",
    "top_pack"
]

ALL_FEATURES = (
    NUMERICAL_FEATURES +
    CATEGORICAL_FEATURES
)


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
# MODEL FILES
# ============================================================

MODEL_PATH = (
    MODEL_DIR /
    "tuned_random_forest.pkl"
)

PREPROCESSOR_PATH = (
    MODEL_DIR /
    "preprocessor.pkl"
)

THRESHOLD_PATH = (
    MODEL_DIR /
    "best_threshold.pkl"
)


# ============================================================
# OUTPUT FILE
# ============================================================

OUTPUT_PATH = (
    REPORT_DIR /
    "customer_churn_predictions.csv"
)


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

print("=" * 70)
print("LOADING MODEL COMPONENTS")
print("=" * 70)


model = joblib.load(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

best_threshold = joblib.load(
    THRESHOLD_PATH
)


print(
    f"\nModel loaded from:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nPreprocessor loaded from:"
    f"\n{PREPROCESSOR_PATH}"
)

print(
    f"\nClassification threshold:"
    f" {best_threshold:.2f}"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST CUSTOMERS")
print("=" * 70)


engine = get_engine()


query = f"""
SELECT
    user_id,
    {", ".join(ALL_FEATURES)}
FROM test_customers;
"""


df = pd.read_sql(
    query,
    engine
)


print(
    f"\nLoaded test customers: "
    f"{len(df):,}"
)

print(
    f"Number of columns loaded: "
    f"{len(df.columns)}"
)


# ============================================================
# PRESERVE USER IDS
# ============================================================

user_ids = df["user_id"].copy()


# ============================================================
# PREPARE FEATURES
# ============================================================

X_test_customers = (
    df[ALL_FEATURES]
    .copy()
)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("TRANSFORMING TEST DATA")
print("=" * 70)


X_test_processed = (
    preprocessor.transform(
        X_test_customers
    )
)


print(
    f"\nProcessed test matrix shape:"
    f" {X_test_processed.shape}"
)


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

expected_features = (
    len(
        preprocessor
        .get_feature_names_out()
    )
)

actual_features = (
    X_test_processed.shape[1]
)


print(
    f"\nExpected features : "
    f"{expected_features}"
)

print(
    f"Actual features   : "
    f"{actual_features}"
)


if expected_features != actual_features:

    raise ValueError(
        "Feature count mismatch between "
        "preprocessor and transformed test data."
    )


# ============================================================
# GENERATE CHURN PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING CHURN PROBABILITIES")
print("=" * 70)


probabilities = (
    model
    .predict_proba(
        X_test_processed
    )[:, 1]
)


print(
    "\nChurn probabilities generated."
)


# ============================================================
# APPLY OPTIMIZED THRESHOLD
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)


# ============================================================
# CREATE RISK LEVEL
# ============================================================

def assign_risk_level(probability):

    if probability >= 0.80:
        return "Very High"

    elif probability >= 0.60:
        return "High"

    elif probability >= 0.40:
        return "Medium"

    elif probability >= 0.20:
        return "Low"

    else:
        return "Very Low"


risk_levels = [
    assign_risk_level(probability)
    for probability in probabilities
]


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

predictions_df = pd.DataFrame(
    {
        "user_id": user_ids,
        "churn_probability": probabilities,
        "churn_prediction": predictions,
        "risk_level": risk_levels
    }
)


# ============================================================
# SORT BY CHURN PROBABILITY
# ============================================================

predictions_df = (
    predictions_df
    .sort_values(
        by="churn_probability",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)


print(
    f"\nTotal customers:"
    f" {len(predictions_df):,}"
)


print(
    f"\nPredicted No Churn:"
    f" {(predictions_df['churn_prediction'] == 0).sum():,}"
)


print(
    f"Predicted Churn:"
    f" {(predictions_df['churn_prediction'] == 1).sum():,}"
)


print("\nRisk distribution:")

print(
    predictions_df[
        "risk_level"
    ]
    .value_counts()
    .reindex(
        [
            "Very High",
            "High",
            "Medium",
            "Low",
            "Very Low"
        ],
        fill_value=0
    )
)


# ============================================================
# TOP HIGH-RISK CUSTOMERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 HIGHEST-RISK CUSTOMERS")
print("=" * 70)


print(
    predictions_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CUSTOMER PREDICTION COMPLETED")
print("=" * 70)


print(
    f"\nPredictions saved to:"
    f"\n{OUTPUT_PATH}"
)


print(
    "\nCustomer churn prediction "
    "completed successfully."
)