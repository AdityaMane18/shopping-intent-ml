
"""
Machine Learning Assignment 2
Step 3: Train-Test Data Splitting

Purpose:
    Split the cleaned dataset into training and testing sets
    using stratified random sampling.

    The testing dataset will be used for evaluating all five
    classification models and for the Streamlit application.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input: cleaned dataset from the previous step.
CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "online_shoppers_clean.csv"
)

# Output paths for training and testing datasets.
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train_data.csv"

# The assignment requires test_data.csv in the repository.
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

TARGET_COLUMN = "Revenue"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# 2. Split dataset
# ---------------------------------------------------------

def split_dataset():
    """
    Perform a stratified 80:20 train-test split.

    Stratification maintains approximately the same
    proportion of purchase and non-purchase sessions
    in both datasets.
    """

    # Load the previously cleaned dataset.
    df = pd.read_csv(CLEANED_DATA_PATH)

    print("TRAIN-TEST DATA SPLITTING")
    print("=" * 50)

    print(f"Total cleaned records: {len(df):,}")

    # Verify that the target column exists.
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column Revenue is missing.")

    # Ensure data cleaning was performed successfully.
    if df.duplicated().any():
        raise ValueError(
            "Cleaned dataset still contains duplicate records."
        )

    # -----------------------------------------------------
    # 3. Separate input features and target variable
    # -----------------------------------------------------

    # X contains the 17 independent input features.
    X = df.drop(columns=[TARGET_COLUMN])

    # y contains the dependent variable Revenue.
    y = df[TARGET_COLUMN]

    # -----------------------------------------------------
    # 4. Perform stratified train-test splitting
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # -----------------------------------------------------
    # 5. Reconstruct datasets including target column
    # -----------------------------------------------------

    # Reset indexes before combining features and labels.
    train_df = X_train.reset_index(drop=True).copy()
    train_df[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_df = X_test.reset_index(drop=True).copy()
    test_df[TARGET_COLUMN] = y_test.reset_index(drop=True)

    # -----------------------------------------------------
    # 6. Save training and testing data
    # -----------------------------------------------------

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    # -----------------------------------------------------
    # 7. Validate the data split
    # -----------------------------------------------------

    print(f"\nTraining records: {len(train_df):,}")
    print(f"Testing records: {len(test_df):,}")

    print("\nTraining target distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nTesting target distribution:")
    print(test_df[TARGET_COLUMN].value_counts())

    # Confirm that no records were lost during splitting.
    assert len(train_df) + len(test_df) == len(df)

    # Confirm that the training and testing sets do not
    # share original row indexes.
    assert set(X_train.index).isdisjoint(set(X_test.index))

    print("\nVALIDATION SUCCESSFUL")
    print("All records accounted for.")
    print("Training and testing indexes are disjoint.")

    print(f"\nTraining dataset saved: {TRAIN_DATA_PATH}")
    print(f"Testing dataset saved: {TEST_DATA_PATH}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 8. Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    split_dataset()
