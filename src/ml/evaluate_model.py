import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# DIRECTORIES
# ============================================================

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT / "models"
)

REPORT_DIR = (
    PROJECT_ROOT / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

X_TEST_PATH = (
    PROCESSED_DATA_DIR / "X_test.npz"
)

Y_TEST_PATH = (
    PROCESSED_DATA_DIR / "y_test.pkl"
)

MODEL_PATH = (
    MODEL_DIR / "tuned_random_forest.pkl"
)

THRESHOLD_PATH = (
    MODEL_DIR / "best_threshold.pkl"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("=" * 70)
print("LOADING VALIDATION DATA")
print("=" * 70)

X_test = sparse.load_npz(
    X_TEST_PATH
)

y_test = pd.read_pickle(
    Y_TEST_PATH
)

print(
    f"\nValidation features : "
    f"{X_test.shape}"
)

print(
    f"Validation samples  : "
    f"{len(y_test):,}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING TUNED RANDOM FOREST")
print("=" * 70)

model = joblib.load(
    MODEL_PATH
)

best_threshold = joblib.load(
    THRESHOLD_PATH
)

print(
    f"\nModel loaded from:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nClassification threshold:"
    f" {best_threshold:.2f}"
)


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# APPLY OPTIMIZED THRESHOLD
# ============================================================

predictions = (
    probabilities >= best_threshold
).astype(int)


print(
    "\nPredictions generated successfully."
)


# ============================================================
# MODEL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# ============================================================
# PRINT PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    "\n"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    predictions
)

print(
    "\n"
)

print(
    "                 Predicted"
)

print(
    "              No Churn  Churn"
)

print(
    f"Actual No Churn   "
    f"{cm[0, 0]:>8,} "
    f"{cm[0, 1]:>7,}"
)

print(
    f"Actual Churn      "
    f"{cm[1, 0]:>8,} "
    f"{cm[1, 1]:>7,}"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "Confusion Matrix - Telecom Churn Model"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    [0, 1],
    ["No Churn", "Churn"]
)

plt.yticks(
    [0, 1],
    ["No Churn", "Churn"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_matrix_path = (
    REPORT_DIR / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    probabilities
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Telecom Churn Model"
)

plt.legend()

plt.tight_layout()

roc_curve_path = (
    REPORT_DIR / "roc_curve.png"
)

plt.savefig(
    roc_curve_path,
    dpi=150
)

plt.close()


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        probabilities
    )
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Telecom Churn Model"
)

plt.tight_layout()

pr_curve_path = (
    REPORT_DIR / "precision_recall_curve.png"
)

plt.savefig(
    pr_curve_path,
    dpi=150
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


if hasattr(
    model,
    "feature_importances_"
):

    feature_importances = (
        model.feature_importances_
    )

    feature_count = (
        len(feature_importances)
    )

    feature_names = [
        f"Feature_{i}"
        for i in range(
            feature_count
        )
    ]

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
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop 20 features:"
    )

    print(
        importance_df.head(
            20
        ).to_string(
            index=False
        )
    )

    feature_importance_path = (
        REPORT_DIR /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        feature_importance_path,
        index=False
    )

else:

    print(
        "\nThis model does not provide "
        "feature_importances_."
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

probability_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Churn_Probability": probabilities,
        "Prediction": predictions
    }
)

probability_path = (
    REPORT_DIR /
    "prediction_probabilities.csv"
)

probability_df.to_csv(
    probability_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nModel:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nThreshold:"
    f" {best_threshold:.2f}"
)

print(
    f"\nF1 Score:"
    f" {f1:.4f}"
)

print(
    f"\nROC-AUC:"
    f" {roc_auc:.4f}"
)

print(
    f"\nReports saved to:"
    f"\n{REPORT_DIR}"
)

print(
    "\nGenerated files:"
)

print(
    f" - {confusion_matrix_path.name}"
)

print(
    f" - {roc_curve_path.name}"
)

print(
    f" - {pr_curve_path.name}"
)

print(
    " - feature_importance.csv"
)

print(
    " - prediction_probabilities.csv"
)

print(
    "\nModel evaluation completed successfully."
)