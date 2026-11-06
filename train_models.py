"""
Machine Learning Model Training Pipeline
----------------------------------------

This script supports training and evaluating multiple classifiers.

Currently supported:
1. Logistic Regression
2. Decision Tree

Workflow:
1. Load training and testing datasets.
2. Separate input features and target variable.
3. Build the preprocessing and classification pipeline.
4. Train using training data only.
5. Evaluate using testing data.
6. Save the trained pipeline and evaluation metrics.
7. Reload the saved model to verify prediction consistency.

Every classifier uses the same training/testing split and
evaluation methodology to support a fair comparison.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.evaluation import evaluate_classifier

from src.models.logistic_regression import (
    build_estimator as build_logistic_regression
)

from src.models.decision_tree import (
    build_estimator as build_decision_tree
)

from src.models.knn import (
    build_estimator as build_knn
)


# -------------------------------------------------------
# PROJECT PATHS
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"
TEST_PATH = PROJECT_ROOT / "test_data.csv"

MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"


# -------------------------------------------------------
# MODEL REGISTRY
# -------------------------------------------------------

# Maps model names to the corresponding model-builder functions.
# Additional models will be registered here as we implement them.

MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "decision_tree": build_decision_tree,
    "knn": build_knn
}


def train_and_evaluate(model_name):
    """
    Train, evaluate and save the selected classification model.

    Parameters:
        model_name: Name of the classifier to train.
    """

    # Use readable names when displaying and saving model results.
    MODEL_DISPLAY_NAMES = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "KNN"
    }

    display_name = MODEL_DISPLAY_NAMES[model_name]

    print(f"\n{display_name.upper()} MODEL TRAINING")
    print("=" * 50)

    # -------------------------------------------------------
    # STEP 1: Load prepared datasets.
    # -------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training records: {len(train_df):,}")
    print(f"Testing records: {len(test_df):,}")

    # -------------------------------------------------------
    # STEP 2: Separate features and target.
    # -------------------------------------------------------

    # X contains the original 17 input features.
    # y contains the Revenue target.

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    # Validate that the training and testing features match.

    assert X_train.columns.tolist() == X_test.columns.tolist(), (
        "Training and testing feature columns do not match."
    )

    # -------------------------------------------------------
    # STEP 3: Create the machine learning pipeline.
    # -------------------------------------------------------

    # Preprocessing and classification are combined so that
    # transformations are learned from training data only.
    #
    # During prediction, the saved pipeline automatically
    # applies the same transformations to new input records.

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", MODEL_BUILDERS[model_name]())
        ]
    )

    # -------------------------------------------------------
    # STEP 4: Train the selected classifier.
    # -------------------------------------------------------

    print(f"\nTraining {display_name}...")

    pipeline.fit(X_train, y_train)

    print("Model training completed successfully.")

    # -------------------------------------------------------
    # STEP 5: Display Decision Tree characteristics.
    # -------------------------------------------------------

    if model_name == "decision_tree":

        tree_model = pipeline.named_steps["classifier"]

        print("\nDECISION TREE STRUCTURE")
        print("-" * 50)

        print(f"Tree depth: {tree_model.get_depth()}")
        print(f"Number of leaf nodes: {tree_model.get_n_leaves()}")

        # Tree depth measures the longest decision path.
        # Leaf nodes represent terminal prediction regions.
        #
        # Very deep trees may memorize training patterns,
        # potentially resulting in overfitting.

    # -------------------------------------------------------
    # STEP 6: Evaluate on unseen testing data.
    # -------------------------------------------------------

    metrics = evaluate_classifier(
        model=pipeline,
        X_test=X_test,
        y_test=y_test
    )

    # -------------------------------------------------------
    # STEP 7: Save model and metrics.
    # -------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{model_name}.joblib"

    results_path = RESULTS_DIR / f"{model_name}_metrics.csv"

    # Save the complete fitted pipeline.
    # The saved object contains both preprocessing and model.

    joblib.dump(pipeline, model_path)

    print(f"\nTrained model saved: {model_path}")

    # Store the six evaluation metrics in CSV format.

    metrics_df = pd.DataFrame([
        {
            "Model": display_name,
            **metrics
        }
    ])

    metrics_df.to_csv(results_path, index=False)

    print(f"Evaluation metrics saved: {results_path}")

    # -------------------------------------------------------
    # STEP 8: Verify model persistence.
    # -------------------------------------------------------

    # Reload only the trusted model created by this script.

    loaded_pipeline = joblib.load(model_path)

    original_predictions = pipeline.predict(X_test)

    loaded_predictions = loaded_pipeline.predict(X_test)

    assert (original_predictions == loaded_predictions).all(), (
        "Saved model validation failed."
    )

    print("\nMODEL VALIDATION SUCCESSFUL")
    print("Saved model was reloaded successfully.")
    print("Original and reloaded predictions match.")


def main():
    """
    Accept the model name from the command line.
    """

    parser = argparse.ArgumentParser(
        description="Train and evaluate a classification model."
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_BUILDERS.keys()),
        default="logistic_regression",
        help="Name of the classifier to train."
    )

    args = parser.parse_args()

    train_and_evaluate(args.model)


if __name__ == "__main__":
    main()