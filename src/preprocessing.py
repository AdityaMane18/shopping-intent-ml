
"""
Machine Learning Assignment 2
Step 4: Feature Preprocessing

Purpose:
    Apply appropriate transformations to numerical
    and categorical input features.

    Numerical features: StandardScaler
    Categorical features: OneHotEncoder

Important:
    Preprocessing is fitted using training data only
    to prevent data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.data_preparation import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    PROJECT_ROOT
)


# ---------------------------------------------------------
# 1. Dataset paths
# ---------------------------------------------------------

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "train_data.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"


# ---------------------------------------------------------
# 2. Create preprocessing transformer
# ---------------------------------------------------------

def build_preprocessor():
    """
    Create an unfitted preprocessing transformer.

    Numerical columns:
        Standardization using mean and standard deviation.

    Categorical columns:
        One-hot encoding to convert categories into
        numerical indicator variables.

    Returns:
        ColumnTransformer: Unfitted preprocessing object.
    """

    # StandardScaler standardizes numerical variables.
    numerical_transformer = StandardScaler()

    # OneHotEncoder converts categorical values into
    # binary indicator columns.
    #
    # handle_unknown="ignore":
    # Allows unseen categories during testing.
    #
    # sparse_output=False:
    # Produces a dense array, compatible with GaussianNB.
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Apply each transformation only to its assigned columns.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ---------------------------------------------------------
# 3. Validate preprocessing
# ---------------------------------------------------------

def validate_preprocessing():
    """
    Validate preprocessing using the saved train/test split.

    This function only verifies transformations.
    It does not train or save classification models.
    """

    print("FEATURE PREPROCESSING VALIDATION")
    print("=" * 50)

    # Load the training and testing datasets.
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # Separate input features from the target.
    # Revenue must not be included in preprocessing.
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    X_test = test_df.drop(columns=[TARGET_COLUMN])

    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")

    print(f"\nNumerical features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")

    # Create a new unfitted preprocessor.
    preprocessor = build_preprocessor()

    # IMPORTANT:
    # Fit preprocessing on training data only.
    #
    # This learns:
    # 1. Mean and standard deviation of numerical columns.
    # 2. Categories appearing in training data.
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Apply the SAME learned transformations to testing data.
    #
    # We do not call fit() or fit_transform() on X_test.
    X_test_transformed = preprocessor.transform(X_test)

    # -----------------------------------------------------
    # 4. Validate transformed feature matrices
    # -----------------------------------------------------

    print("\nTRANSFORMATION RESULTS")
    print("-" * 50)

    print(f"Original training shape: {X_train.shape}")
    print(f"Transformed training shape: {X_train_transformed.shape}")

    print(f"\nOriginal testing shape: {X_test.shape}")
    print(f"Transformed testing shape: {X_test_transformed.shape}")

    # Both transformed datasets must have the same
    # number of input columns.
    assert (
        X_train_transformed.shape[1]
        == X_test_transformed.shape[1]
    ), "Training and testing feature counts do not match."

    # Ensure that preprocessing has not changed
    # the number of records.
    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)

    # Verify all transformed values are finite.
    assert np.isfinite(X_train_transformed).all()
    assert np.isfinite(X_test_transformed).all()

    # Display the names of the generated features.
    feature_names = preprocessor.get_feature_names_out()

    print(f"\nTotal transformed features: {len(feature_names)}")

    print("\nFirst 10 transformed feature names:")
    for feature in feature_names[:10]:
        print(f"  {feature}")

    print("\nPREPROCESSING VALIDATION SUCCESSFUL")
    print("Training and testing data have compatible feature shapes.")
    print("No invalid numerical values detected.")
    print("Testing data was transformed without fitting.")


# ---------------------------------------------------------
# 5. Main execution
# ---------------------------------------------------------

if __name__ == "__main__":
    validate_preprocessing()
