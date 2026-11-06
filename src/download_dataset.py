
"""
Machine Learning Assignment 2
Dataset: Online Shoppers Purchasing Intention
Source: UCI Machine Learning Repository

Purpose:
    Download the classification dataset using the UCI API,
    validate its structure, and save it as a local CSV file.

UCI Dataset ID: 468
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------
# 1. Define dataset source and local storage path
# ---------------------------------------------------------

# Unique dataset identifier assigned by UCI.
DATASET_ID = 468

# Identify the root directory of our project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the directory where the dataset will be stored.
DATA_DIR = PROJECT_ROOT / "data"

# Define the CSV filename.
DATASET_PATH = DATA_DIR / "online_shoppers_intention.csv"


# ---------------------------------------------------------
# 2. Download and save the dataset
# ---------------------------------------------------------

def download_dataset():
    """
    Download the dataset from UCI and save it locally.

    Returns:
        pd.DataFrame: Dataset containing input features
        and the target variable (Revenue).
    """

    print("Connecting to UCI Machine Learning Repository...")

    # Fetch the dataset using the official UCI Python API.
    # Dataset ID 468 corresponds to Online Shoppers
    # Purchasing Intention.
    try:
        dataset = fetch_ucirepo(id=DATASET_ID)

    except Exception as error:
        raise RuntimeError(
            "Unable to download the dataset from UCI. "
            "Please check your internet connection."
        ) from error

    print("Dataset downloaded successfully.")

    # Extract independent variables (input features).
    # X contains 17 features describing shopping sessions.
    X = dataset.data.features

    # Extract the dependent variable (target).
    # y contains Revenue: True (purchase) or False (no purchase).
    y = dataset.data.targets

    # Ensure both components were retrieved successfully.
    if X is None or y is None:
        raise ValueError(
            "Dataset features or target could not be retrieved."
        )

    # Combine features and target into one DataFrame.
    # Resetting indexes ensures that rows align correctly.
    df = pd.concat(
        [
            X.reset_index(drop=True),
            y.reset_index(drop=True)
        ],
        axis=1
    )

    # -----------------------------------------------------
    # 3. Validate dataset structure
    # -----------------------------------------------------

    # Verify that the dataset meets assignment requirements.
    if len(df) < 500:
        raise ValueError("Dataset contains fewer than 500 rows.")

    if X.shape[1] < 12:
        raise ValueError("Dataset contains fewer than 12 features.")

    # Revenue must exist because this is our classification target.
    if "Revenue" not in df.columns:
        raise ValueError("Target column 'Revenue' is missing.")

    # -----------------------------------------------------
    # 4. Save dataset locally
    # -----------------------------------------------------

    # Create the data directory if it does not exist.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the retrieved dataset as CSV.
    # index=False prevents pandas from adding an unnecessary
    # index column to the file.
    #
    # This replaces the existing local CSV if one is present.
    # No cleaning or preprocessing is performed at this stage.
    df.to_csv(DATASET_PATH, index=False)

    print("\nDataset saved successfully.")
    print(f"File location: {DATASET_PATH}")

    # -----------------------------------------------------
    # 5. Display basic dataset information
    # -----------------------------------------------------

    print("\nDATASET SUMMARY")
    print("-" * 50)

    print(f"Total observations: {df.shape[0]:,}")
    print(f"Total input features: {X.shape[1]}")
    print(f"Total columns: {df.shape[1]}")

    print("\nTarget distribution:")
    print(df["Revenue"].value_counts())

    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Duplicate records: {df.duplicated().sum()}")

    return df


# ---------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------

# This block executes only when this file is run directly
# as a Python module. It does not execute when imported
# into another Python file.
if __name__ == "__main__":

    download_dataset()
