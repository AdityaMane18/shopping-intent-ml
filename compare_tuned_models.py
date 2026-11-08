"""
Consolidated Model Cross-Validation Comparison
----------------------------------------------

BITS Pilani - Machine Learning Assignment 2

Objective:
    Compare baseline and optimized classification models
    using training-only cross-validation results.

Methodology:
    1. Read saved cross-validation summaries.
    2. Compare baseline and tuned MCC.
    3. Use the predefined model-selection decisions.
    4. Consolidate all six classification metrics.
    5. Rank selected models by cross-validation MCC.
    6. Save the comparison for final evaluation.

IMPORTANT:
    This script does not load the test dataset.
    No models are retrained or modified.
"""

from pathlib import Path

import pandas as pd


# -------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results" / "tuned"

MODEL_DIR = PROJECT_ROOT / "model"

OUTPUT_PATH = (
    RESULTS_DIR / "model_cv_comparison.csv"
)


# -------------------------------------------------------
# STEP 2: MODEL CONFIGURATION
# -------------------------------------------------------

# These decisions are based only on training CV results.
# They must be finalized before test-set evaluation.

MODEL_CONFIG = {
    "Logistic Regression": {
        "filename": "logistic_regression",
        "selected_version": "tuned"
    },

    "Decision Tree": {
        "filename": "decision_tree",
        "selected_version": "tuned"
    },

    "KNN": {
        "filename": "knn",
        "selected_version": "baseline"
    },

    "Gaussian Naive Bayes": {
        "filename": "naive_bayes",
        "selected_version": "tuned"
    },

    "Support Vector Machine": {
        "filename": "svm",
        "selected_version": "baseline"
    }
}


METRICS = [
    "Accuracy",
    "ROC_AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]


# -------------------------------------------------------
# STEP 3: CONSOLIDATE RESULTS
# -------------------------------------------------------

def compare_models():

    print("FINAL TRAINING CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 75)

    comparison_rows = []

    for model_name, config in MODEL_CONFIG.items():

        filename = config["filename"]
        selected_version = config["selected_version"]

        summary_path = (
            RESULTS_DIR / f"{filename}_cv_summary.csv"
        )

        # Ensure optimization results are available.
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing CV summary: {summary_path}"
            )

        summary_df = pd.read_csv(summary_path)

        # Every evaluation metric must appear once.
        if summary_df["Metric"].duplicated().any():
            raise ValueError(
                f"Duplicate metrics found: {summary_path}"
            )

        summary_df = summary_df.set_index("Metric")

        missing_metrics = [
            metric
            for metric in METRICS
            if metric not in summary_df.index
        ]

        if missing_metrics:
            raise ValueError(
                f"{model_name} is missing: {missing_metrics}"
            )

        # Choose the appropriate saved model.
        if selected_version == "tuned":

            model_path = (
                MODEL_DIR / "tuned" / f"{filename}.joblib"
            )

            score_column = "Selected_CV"

        else:

            model_path = (
            MODEL_DIR / "baseline" / f"{filename}.joblib"
            )

            score_column = "Baseline_CV"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing selected model: {model_path}"
            )

        # Extract baseline and tuned MCC.
        baseline_mcc = float(
            summary_df.loc["MCC", "Baseline_CV"]
        )

        tuned_mcc = float(
            summary_df.loc["MCC", "Selected_CV"]
        )

        row = {
            "Model": model_name,
            "Selected_Version": selected_version,
            "Model_Path": str(
                model_path.relative_to(PROJECT_ROOT)
            ),
            "Baseline_CV_MCC": baseline_mcc,
            "Tuned_CV_MCC": tuned_mcc,
            "MCC_Improvement": tuned_mcc - baseline_mcc
        }

        # Extract the six metrics for the selected version.
        for metric_name in METRICS:

            row[metric_name] = float(
                summary_df.loc[metric_name, score_column]
            )

        comparison_rows.append(row)

    # ---------------------------------------------------
    # STEP 4: RANK MODELS BY MCC
    # ---------------------------------------------------

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df = comparison_df.sort_values(
        by="MCC",
        ascending=False
    ).reset_index(drop=True)

    # ---------------------------------------------------
    # STEP 5: DISPLAY PERFORMANCE COMPARISON
    # ---------------------------------------------------

    print("\nSELECTED MODEL PERFORMANCE")
    print("-" * 75)

    display_columns = [
        "Model",
        "Selected_Version",
        "Accuracy",
        "ROC_AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    print(
        comparison_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # ---------------------------------------------------
    # STEP 6: DISPLAY OPTIMIZATION IMPROVEMENTS
    # ---------------------------------------------------

    print("\nBASELINE VS TUNED MCC")
    print("-" * 75)

    improvement_columns = [
        "Model",
        "Baseline_CV_MCC",
        "Tuned_CV_MCC",
        "MCC_Improvement"
    ]

    print(
        comparison_df[improvement_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # ---------------------------------------------------
    # STEP 7: DISPLAY LEADING CV MODEL
    # ---------------------------------------------------

    leading_model = comparison_df.iloc[0]

    print("\nLEADING MODEL BASED ON CROSS-VALIDATION")
    print("-" * 75)

    print(f"Model: {leading_model['Model']}")
    print(f"Version: {leading_model['Selected_Version']}")
    print(f"CV MCC: {leading_model['MCC']:.4f}")

    # ---------------------------------------------------
    # STEP 8: SAVE COMPARISON
    # ---------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nCOMPARISON REPORT SAVED")
    print(OUTPUT_PATH)

    # ---------------------------------------------------
    # STEP 9: VALIDATION
    # ---------------------------------------------------

    assert len(comparison_df) == 5, (
        "Expected five classification models."
    )

    assert comparison_df["MCC"].notna().all(), (
        "Missing MCC values."
    )

    print("\nCOMPARISON VALIDATION SUCCESSFUL")
    print("All five models are available.")
    print("All six evaluation metrics are available.")
    print("The testing dataset was not accessed.")


if __name__ == "__main__":
    compare_models()