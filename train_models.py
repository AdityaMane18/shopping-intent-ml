"""
Logistic Regression Training Pipeline
-------------------------------------
This script performs the following operations:

1. Loads training and testing datasets.
2. Separates input features from the target.
3. Creates preprocessing and classification pipeline.
4. Fits the pipeline using training data only.
5. Evaluates the trained model on testing data.
6. Saves the trained pipeline and evaluation metrics.
7. Verifies that the saved model can be loaded.

The testing dataset is never used for model training.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor
from src.models.logistic_regression import build_estimator
from src.evaluation import evaluate_classifier


# Resolve paths relative to this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"
TEST_PATH = PROJECT_ROOT / "test_data.csv"

MODEL_PATH = PROJECT_ROOT / "model" / "logistic_regression.joblib"
RESULT_PATH = PROJECT_ROOT / "results" / "logistic_regression_metrics.csv"


def main():

    print("LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 50)

    # -------------------------------------------------------
    # STEP 1: Load the previously prepared datasets.
    # -------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training records: {len(train_df):,}")
    print(f"Testing records: {len(test_df):,}")

    # -------------------------------------------------------
    # STEP 2: Separate input features and target variable.
    # -------------------------------------------------------

    # X contains the 17 original input features.
    # y contains the Revenue target.
    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    # Confirm that both datasets use the same feature columns.
    assert X_train.columns.tolist() == X_test.columns.tolist(), (
        "Training and testing feature columns do not match."
    )

    # -------------------------------------------------------
    # STEP 3: Build the machine learning pipeline.
    # -------------------------------------------------------

    # The pipeline combines preprocessing and classification.
    # During fit(), preprocessing learns its parameters only
    # from the training dataset.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_estimator())
        ]
    )

    # -------------------------------------------------------
    # STEP 4: Train the Logistic Regression model.
    # -------------------------------------------------------

    print("\nTraining Logistic Regression...")

    pipeline.fit(X_train, y_train)

    print("Model training completed successfully.")

    # -------------------------------------------------------
    # STEP 5: Evaluate using unseen testing records.
    # -------------------------------------------------------

    metrics = evaluate_classifier(
        model=pipeline,
        X_test=X_test,
        y_test=y_test
    )

    # -------------------------------------------------------
    # STEP 6: Save trained model and evaluation results.
    # -------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the entire fitted pipeline, including preprocessing.
    # This allows Streamlit to predict directly from raw CSV inputs.
    joblib.dump(pipeline, MODEL_PATH)

    print(f"\nTrained model saved: {MODEL_PATH}")

    # Save the metrics in tabular form for later comparison
    # with Decision Tree, KNN, Naive Bayes and SVM.
    metrics_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            **metrics
        }
    ])

    metrics_df.to_csv(RESULT_PATH, index=False)

    print(f"Evaluation metrics saved: {RESULT_PATH}")

    # -------------------------------------------------------
    # STEP 7: Verify the saved model.
    # -------------------------------------------------------

    # Only load a model file that we created and trust.
    loaded_pipeline = joblib.load(MODEL_PATH)

    original_predictions = pipeline.predict(X_test)
    loaded_predictions = loaded_pipeline.predict(X_test)

    assert (original_predictions == loaded_predictions).all(), (
        "Loaded model predictions do not match."
    )

    print("\nMODEL VALIDATION SUCCESSFUL")
    print("Saved model was reloaded successfully.")
    print("Original and reloaded model predictions match.")


if __name__ == "__main__":
    main()