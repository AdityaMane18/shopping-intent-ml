"""
Machine Learning Model Comparison
---------------------------------

This script compares the performance of all five classification
models using their previously saved evaluation metrics.

Models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

Evaluation metrics:
1. Accuracy
2. ROC AUC
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

The comparison uses existing evaluation results.
Models are NOT retrained in this script.

The resulting table is saved for reporting and visualization.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------
# PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"

OUTPUT_PATH = RESULTS_DIR / "model_comparison.csv"


# List of models evaluated in this assignment.
MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "svm"
]


# Six metrics required by the assignment.
METRIC_COLUMNS = [
    "Accuracy",
    "ROC AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]


def compare_models():
    """
    Load saved metrics, compare model performance,
    and generate a consolidated comparison report.
    """

    print("CLASSIFICATION MODEL COMPARISON")
    print("=" * 65)

    model_results = []

    # -------------------------------------------------------
    # STEP 1: Load metrics from each model.
    # -------------------------------------------------------

    for model_name in MODEL_NAMES:

        metrics_path = RESULTS_DIR / f"{model_name}_metrics.csv"

        # Ensure evaluation results exist before comparison.
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}"
            )

        metrics_df = pd.read_csv(metrics_path)

        # Validate that each file contains the expected columns.
        required_columns = ["Model"] + METRIC_COLUMNS

        missing_columns = [
            column
            for column in required_columns
            if column not in metrics_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{metrics_path.name} is missing: {missing_columns}"
            )

        if len(metrics_df) != 1:
            raise ValueError(
                f"Expected one result row in {metrics_path.name}"
            )

        model_results.append(metrics_df[required_columns])

    # -------------------------------------------------------
    # STEP 2: Combine all evaluation results.
    # -------------------------------------------------------

    comparison_df = pd.concat(
        model_results,
        ignore_index=True
    )

    # -------------------------------------------------------
    # STEP 3: Rank models by MCC.
    # -------------------------------------------------------

    # MCC is useful for evaluating imbalanced classification
    # because it considers all four confusion matrix outcomes.

    comparison_df = comparison_df.sort_values(
        by="MCC",
        ascending=False
    ).reset_index(drop=True)

    # -------------------------------------------------------
    # STEP 4: Display the comparison table.
    # -------------------------------------------------------

    print("\nMODEL PERFORMANCE COMPARISON")
    print("-" * 65)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # -------------------------------------------------------
    # STEP 5: Identify the leading model for key metrics.
    # -------------------------------------------------------

    print("\nBEST MODEL BY METRIC")
    print("-" * 65)

    for metric in METRIC_COLUMNS:

        best_index = comparison_df[metric].idxmax()

        best_model = comparison_df.loc[best_index, "Model"]
        best_value = comparison_df.loc[best_index, metric]

        print(
            f"{metric:<15}: {best_model} ({best_value:.4f})"
        )

    # -------------------------------------------------------
    # STEP 6: Save the consolidated report.
    # -------------------------------------------------------

    comparison_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nComparison report saved: {OUTPUT_PATH}")

    # -------------------------------------------------------
    # STEP 7: Validate the results.
    # -------------------------------------------------------

    assert len(comparison_df) == 5, (
        "Expected evaluation results for five models."
    )

    assert comparison_df["Model"].is_unique, (
        "Duplicate model names detected."
    )

    assert comparison_df[METRIC_COLUMNS].notna().all().all(), (
        "Missing metric values detected."
    )

    print("\nCOMPARISON VALIDATION SUCCESSFUL")
    print("All five models and six metrics are available.")


if __name__ == "__main__":
    compare_models()