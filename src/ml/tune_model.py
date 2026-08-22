import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

X_TRAIN_PATH = (
    PROCESSED_DATA_DIR / "X_train.npz"
)

X_TEST_PATH = (
    PROCESSED_DATA_DIR / "X_test.npz"
)

Y_TRAIN_PATH = (
    PROCESSED_DATA_DIR / "y_train.pkl"
)

Y_TEST_PATH = (
    PROCESSED_DATA_DIR / "y_test.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

X_train = sparse.load_npz(
    X_TRAIN_PATH
)

X_test = sparse.load_npz(
    X_TEST_PATH
)

y_train = pd.read_pickle(
    Y_TRAIN_PATH
)

y_test = pd.read_pickle(
    Y_TEST_PATH
)

print(
    f"\nTraining data : {X_train.shape}"
)

print(
    f"Validation data : {X_test.shape}"
)

print(
    f"Training samples : {len(y_train):,}"
)

print(
    f"Validation samples : {len(y_test):,}"
)


# ============================================================
# RANDOM FOREST CONFIGURATIONS
# ============================================================

configs = [

    {
        "name": "RF_1",
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 10
    },

    {
        "name": "RF_2",
        "n_estimators": 150,
        "max_depth": 20,
        "min_samples_leaf": 10
    },

    {
        "name": "RF_3",
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_leaf": 20
    },

    {
        "name": "RF_4",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 20
    }
]


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

results = []

best_model = None
best_config = None
best_f1 = -1


for config in configs:

    print("\n" + "=" * 70)
    print(
        f"TRAINING {config['name']}"
    )
    print("=" * 70)

    print(
        f"\nTrees           : "
        f"{config['n_estimators']}"
    )

    print(
        f"Max depth       : "
        f"{config['max_depth']}"
    )

    print(
        f"Min samples leaf: "
        f"{config['min_samples_leaf']}"
    )

    model = RandomForestClassifier(

        n_estimators=config[
            "n_estimators"
        ],

        max_depth=config[
            "max_depth"
        ],

        min_samples_leaf=config[
            "min_samples_leaf"
        ],

        class_weight="balanced",

        n_jobs=-1,

        random_state=42
    )

    print("\nFitting model...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # METRICS
    # ========================================================

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


    # ========================================================
    # PRINT RESULTS
    # ========================================================

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
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results.append(
        {
            "Configuration":
                config["name"],

            "Trees":
                config["n_estimators"],

            "Max Depth":
                config["max_depth"],

            "Min Samples Leaf":
                config["min_samples_leaf"],

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC-AUC":
                roc_auc
        }
    )


    # ========================================================
    # TRACK BEST MODEL
    # ========================================================

    if f1 > best_f1:

        best_f1 = f1

        best_model = model

        best_config = config


# ============================================================
# HYPERPARAMETER RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("HYPERPARAMETER COMPARISON")
print("=" * 70)

print(
    "\n"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE TUNING RESULTS
# ============================================================

tuning_results_path = (
    MODEL_DIR / "tuning_results.csv"
)

results_df.to_csv(
    tuning_results_path,
    index=False
)


# ============================================================
# BEST MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BEST TUNED MODEL")
print("=" * 70)

print(
    f"\nConfiguration : "
    f"{best_config['name']}"
)

print(
    f"Trees         : "
    f"{best_config['n_estimators']}"
)

print(
    f"Max Depth     : "
    f"{best_config['max_depth']}"
)

print(
    f"Min Leaf      : "
    f"{best_config['min_samples_leaf']}"
)

print(
    f"\nBest F1       : "
    f"{best_f1:.4f}"
)


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

best_probabilities = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


threshold_results = []

thresholds = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


for threshold in thresholds:

    threshold_predictions = (
        best_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        threshold_predictions
    )

    precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "Threshold":
                threshold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1
        }
    )


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)

threshold_df = threshold_df.sort_values(
    by="F1",
    ascending=False
).reset_index(
    drop=True
)


print(
    "\n"
)

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

best_threshold = (
    threshold_df.iloc[0]["Threshold"]
)

best_threshold_f1 = (
    threshold_df.iloc[0]["F1"]
)

best_threshold_precision = (
    threshold_df.iloc[0]["Precision"]
)

best_threshold_recall = (
    threshold_df.iloc[0]["Recall"]
)

best_threshold_accuracy = (
    threshold_df.iloc[0]["Accuracy"]
)


# ============================================================
# SAVE THRESHOLD RESULTS
# ============================================================

threshold_results_path = (
    MODEL_DIR / "threshold_results.csv"
)

threshold_df.to_csv(
    threshold_results_path,
    index=False
)


# ============================================================
# SAVE TUNED MODEL
# ============================================================

tuned_model_path = (
    MODEL_DIR / "tuned_random_forest.pkl"
)

joblib.dump(
    best_model,
    tuned_model_path
)


# ============================================================
# SAVE THRESHOLD
# ============================================================

threshold_path = (
    MODEL_DIR / "best_threshold.pkl"
)

joblib.dump(
    best_threshold,
    threshold_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETED")
print("=" * 70)

print(
    f"\nBest Random Forest F1 : "
    f"{best_f1:.4f}"
)

print(
    f"Best Threshold        : "
    f"{best_threshold:.2f}"
)

print(
    f"Threshold Accuracy    : "
    f"{best_threshold_accuracy:.4f}"
)

print(
    f"Threshold Precision   : "
    f"{best_threshold_precision:.4f}"
)

print(
    f"Threshold Recall      : "
    f"{best_threshold_recall:.4f}"
)

print(
    f"Threshold F1          : "
    f"{best_threshold_f1:.4f}"
)

print(
    f"\nTuned model saved to:"
    f"\n{tuned_model_path}"
)

print(
    f"\nBest threshold saved to:"
    f"\n{threshold_path}"
)

print(
    f"\nTuning results saved to:"
    f"\n{tuning_results_path}"
)

print(
    f"\nThreshold results saved to:"
    f"\n{threshold_results_path}"
)

print(
    "\nModel optimization completed successfully."
)