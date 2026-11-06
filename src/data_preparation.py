"""Step 1: load, validate, and inspect the original shopping dataset.

Run from the project root with:

    python -m src.data_preparation

This script intentionally does not train or modify any data.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers_intention.csv"
TARGET_COLUMN = "Revenue"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = set(FEATURE_COLUMNS + [TARGET_COLUMN])


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the original CSV and ensure its schema matches the UCI dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    dataset = pd.read_csv(path)

    missing_columns = EXPECTED_COLUMNS - set(dataset.columns)
    unexpected_columns = set(dataset.columns) - EXPECTED_COLUMNS
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Dataset schema mismatch. "
            f"Missing columns: {sorted(missing_columns)}; "
            f"unexpected columns: {sorted(unexpected_columns)}"
        )

    if len(dataset) < 500 or len(FEATURE_COLUMNS) < 12:
        raise ValueError("Dataset does not satisfy the assignment minimums.")

    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError("Target column Revenue contains missing values.")

    if set(dataset[TARGET_COLUMN].unique()) != {False, True}:
        raise ValueError("Revenue should contain both True and False labels.")

    return dataset


def inspect_dataset(dataset: pd.DataFrame) -> None:
    """Print reproducible checks for the first step of the assignment."""
    print("ONLINE SHOPPING DATASET - INITIAL INSPECTION")
    print("=" * 50)
    print(f"Rows: {len(dataset):,}")
    print(f"Input features: {len(FEATURE_COLUMNS)}")
    print(f"Total columns including target: {dataset.shape[1]}")
    print(f"Missing cells: {int(dataset.isna().sum().sum()):,}")
    print(f"Exact duplicate rows: {int(dataset.duplicated().sum()):,}")
    print("\nTarget distribution (Revenue):")

    counts = dataset[TARGET_COLUMN].value_counts().sort_index()
    for label, count in counts.items():
        percentage = count / len(dataset) * 100
        print(f"  {label}: {count:,} ({percentage:.2f}%)")

    print("\nColumn names:")
    for name in dataset.columns:
        print(f"  - {name}")

    print("\nNo rows or columns were changed by this inspection.")

# ---------------------------------------------------------
# Step 2: Data Cleaning
# ---------------------------------------------------------

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing exact duplicate records.

    The original downloaded dataset is preserved.
    The cleaned dataset is saved separately for model training.

    Returns:
        pd.DataFrame: Dataset after duplicate removal.
    """

    print("\nDATA CLEANING")
    print("=" * 50)

    # Record the number of observations before cleaning.
    initial_rows = len(dataset)

    # Identify duplicate rows across all columns,
    # including the classification target.
    duplicate_mask = dataset.duplicated()

    duplicate_count = duplicate_mask.sum()

    print(f"Original records: {initial_rows:,}")
    print(f"Duplicate records identified: {duplicate_count:,}")

    # Examine which target classes contain duplicate records.
    print("\nDuplicate records by target class:")
    print(dataset.loc[duplicate_mask, "Revenue"].value_counts())

    # Remove exact duplicates, retaining the first occurrence.
    # Reset the index to maintain sequential row numbering.
    cleaned_dataset = (
        dataset.drop_duplicates()
        .reset_index(drop=True)
    )

    # Calculate the number of removed records.
    removed_rows = initial_rows - len(cleaned_dataset)

    print(f"\nRecords removed: {removed_rows:,}")
    print(f"Records after cleaning: {len(cleaned_dataset):,}")

    # Save the cleaned dataset separately.
    # This preserves the original UCI dataset for reproducibility.
    cleaned_path = PROJECT_ROOT / "data" / "online_shoppers_clean.csv"

    cleaned_dataset.to_csv(cleaned_path, index=False)

    print(f"\nCleaned dataset saved: {cleaned_path}")

    # Verify that no exact duplicate records remain.
    remaining_duplicates = cleaned_dataset.duplicated().sum()

    print(f"Remaining duplicates: {remaining_duplicates:,}")

    return cleaned_dataset


if __name__ == "__main__":

    # Step 1: Load the original downloaded dataset.
    dataset = load_dataset()

    # Step 2: Inspect the original dataset.
    inspect_dataset(dataset)

    # Step 3: Remove exact duplicates and save cleaned data.
    cleaned_dataset = clean_dataset(dataset)

