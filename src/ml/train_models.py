import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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
# LOAD PROCESSED DATA
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
    f"\nTraining feature matrix : "
    f"{X_train.shape}"
)

print(
    f"Validation feature matrix : "
    f"{X_test.shape}"
)

print(
    f"Training target samples : "
    f"{len(y_train):,}"
)

print(
    f"Validation target samples : "
    f"{len(y_test):,}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(
    "\nTraining target distribution:"
)

print(
    y_train.value_counts(
        normalize=False
    ).sort_index()
)

print(
    "\nTraining target percentage:"
)

print(
    (
        y_train.value_counts(
            normalize=True
        )
        .sort_index()
        .mul(100)
        .round(2)
    )
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
}


# ============================================================
# TRAINING
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    print("\nFitting model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

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

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        }
    )

    trained_models[name] = model


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
).reset_index(
    drop=True
)

print(
    "\n"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)

best_model = (
    trained_models[best_model_name]
)

best_f1 = (
    results_df.iloc[0]["F1"]
)

best_roc_auc = (
    results_df.iloc[0]["ROC-AUC"]
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model_path = (
    MODEL_DIR / "best_model.pkl"
)

joblib.dump(
    best_model,
    best_model_path
)


# ============================================================
# SAVE MODEL RESULTS
# ============================================================

results_path = (
    MODEL_DIR / "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"\nBest Model : "
    f"{best_model_name}"
)

print(
    f"Best F1    : "
    f"{best_f1:.4f}"
)

print(
    f"Best ROC-AUC: "
    f"{best_roc_auc:.4f}"
)

print(
    f"\nModel saved to:"
    f"\n{best_model_path}"
)

print(
    f"\nModel comparison saved to:"
    f"\n{results_path}"
)

print(
    "\nModel training completed successfully."
)