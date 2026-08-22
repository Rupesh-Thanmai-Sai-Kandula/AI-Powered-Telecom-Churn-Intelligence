import sys
from pathlib import Path

import joblib
import pandas as pd
import scipy.sparse as sparse

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

TARGET = "churn"

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

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# LOAD DATA FROM POSTGRESQL
# ============================================================

print("=" * 70)
print("LOADING TRAINING DATA FROM POSTGRESQL")
print("=" * 70)

engine = get_engine()

query = f"""
SELECT
    {", ".join(ALL_FEATURES)},
    {TARGET}
FROM train_customers;
"""

df = pd.read_sql(query, engine)

print(f"\nLoaded {len(df):,} rows.")
print(f"Loaded {len(df.columns)} columns.")


# ============================================================
# DATA VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATA VALIDATION")
print("=" * 70)

print("\nTarget distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)

print("\nMissing values:")

missing_values = df.isnull().sum()

print(
    missing_values[
        missing_values > 0
    ].sort_values(ascending=False)
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = df[ALL_FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"\nTraining samples   : {len(X_train):,}"
)

print(
    f"Validation samples : {len(X_test):,}"
)


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)


# ============================================================
# COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# FIT PREPROCESSOR
# ============================================================

print("\n" + "=" * 70)
print("FITTING PREPROCESSOR")
print("=" * 70)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# PROCESSED FEATURE INFORMATION
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    f"\nProcessed feature count : "
    f"{len(feature_names)}"
)

print(
    f"Training matrix shape   : "
    f"{X_train_processed.shape}"
)

print(
    f"Validation matrix shape : "
    f"{X_test_processed.shape}"
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

preprocessor_path = (
    MODEL_DIR / "preprocessor.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

processed_data_dir = (
    PROJECT_ROOT / "data" / "processed"
)

processed_data_dir.mkdir(
    parents=True,
    exist_ok=True
)


X_train_path = (
    processed_data_dir / "X_train.npz"
)

X_test_path = (
    processed_data_dir / "X_test.npz"
)

y_train_path = (
    processed_data_dir / "y_train.pkl"
)

y_test_path = (
    processed_data_dir / "y_test.pkl"
)


# ============================================================
# SAVE SPARSE FEATURE MATRICES
# ============================================================

sparse.save_npz(
    X_train_path,
    X_train_processed
)

sparse.save_npz(
    X_test_path,
    X_test_processed
)


# ============================================================
# SAVE TARGET VARIABLES
# ============================================================

y_train.to_pickle(
    y_train_path
)

y_test.to_pickle(
    y_test_path
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print(
    f"\nOriginal features     : "
    f"{len(ALL_FEATURES)}"
)

print(
    f"Processed features    : "
    f"{len(feature_names)}"
)

print(
    f"Training samples      : "
    f"{X_train_processed.shape[0]:,}"
)

print(
    f"Validation samples    : "
    f"{X_test_processed.shape[0]:,}"
)

print(
    f"\nPreprocessor saved to:"
    f"\n{preprocessor_path}"
)

print(
    f"\nProcessed training data:"
    f"\n{X_train_path}"
)

print(
    f"\nProcessed validation data:"
    f"\n{X_test_path}"
)

print(
    f"\nTraining target saved to:"
    f"\n{y_train_path}"
)

print(
    f"\nValidation target saved to:"
    f"\n{y_test_path}"
)

print(
    "\nFeature engineering pipeline "
    "completed successfully."
)