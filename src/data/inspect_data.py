import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ==========================================
# FIND CSV FILES
# ==========================================

csv_files = list(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {RAW_DATA_DIR}"
    )


print("=" * 80)
print("EXPRESSO TELECOM DATASET INSPECTION")
print("=" * 80)

print(f"\nData directory:")
print(RAW_DATA_DIR)

print(f"\nCSV files found: {len(csv_files)}")

for file in csv_files:
    print(f" - {file.name}")


# ==========================================
# INSPECT EACH DATASET
# ==========================================

for file in csv_files:

    print("\n")
    print("=" * 80)
    print(f"FILE: {file.name}")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv(file)

    # --------------------------------------
    # BASIC INFORMATION
    # --------------------------------------

    print("\nShape:")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    print("\nColumns:")
    for column in df.columns:
        print(f" - {column}")

    # --------------------------------------
    # DATA TYPES
    # --------------------------------------

    print("\nData Types:")
    print(df.dtypes)

    # --------------------------------------
    # MISSING VALUES
    # --------------------------------------

    print("\nMissing Values:")

    missing = df.isnull().sum()

    missing_percentage = (
        df.isnull().mean() * 100
    )

    missing_report = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": missing_percentage.round(2)
    })

    missing_report = missing_report[
        missing_report["Missing_Count"] > 0
    ].sort_values(
        "Missing_Count",
        ascending=False
    )

    if missing_report.empty:

        print("No missing values found.")

    else:

        print(missing_report)

    # --------------------------------------
    # DUPLICATES
    # --------------------------------------

    print("\nDuplicate Rows:")

    duplicates = df.duplicated().sum()

    print(f"{duplicates:,}")

    # --------------------------------------
    # FIRST FIVE ROWS
    # --------------------------------------

    print("\nFirst 5 Rows:")

    print(df.head())

    # --------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------

    print("\nNumerical Summary:")

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numerical_columns) > 0:

        print(
            df[numerical_columns]
            .describe()
            .transpose()
        )

    else:

        print("No numerical columns.")

    # --------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------

    print("\nCategorical Columns:")

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    if len(categorical_columns) > 0:

        for column in categorical_columns:

            print(f"\n{column}")

            print(
                df[column]
                .value_counts(dropna=False)
                .head(10)
            )

    else:

        print("No categorical columns.")


# ==========================================
# TARGET ANALYSIS
# ==========================================

print("\n")
print("=" * 80)
print("TARGET VARIABLE ANALYSIS")
print("=" * 80)

train_files = [
    file for file in csv_files
    if "train" in file.name.lower()
]

if train_files:

    train_df = pd.read_csv(train_files[0])

    if "CHURN" in train_df.columns:

        print("\nTarget variable: CHURN")

        print("\nTarget Distribution:")

        print(
            train_df["CHURN"]
            .value_counts(dropna=False)
        )

        print("\nTarget Proportion:")

        print(
            train_df["CHURN"]
            .value_counts(
                normalize=True,
                dropna=False
            ).round(4)
        )

    else:

        print(
            "\nCHURN column was not found in the training dataset."
        )

else:

    print(
        "\nTraining CSV could not be identified automatically."
    )


print("\n")
print("=" * 80)
print("DATA INSPECTION COMPLETED")
print("=" * 80)