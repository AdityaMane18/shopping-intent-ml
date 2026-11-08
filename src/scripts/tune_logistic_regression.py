"""
Logistic Regression Hyperparameter Optimization
------------------------------------------------

BITS Pilani - Machine Learning Assignment 2

Objective:
    Optimize Logistic Regression using training data only.

Methodology:
    1. Load the training dataset.
    2. Define stratified five-fold cross-validation.
    3. Evaluate the original model using cross-validation.
    4. Search predefined hyperparameter combinations.
    5. Select the configuration with the highest mean CV MCC.
    6. Record all six evaluation metrics.
    7. Save the optimized pipeline separately.
    8. Save detailed cross-validation results.

IMPORTANT:
    The testing dataset is never loaded in this script.

    All preprocessing is included inside the Pipeline so that
    scaling and encoding are fitted separately within each
    training fold.

    Existing baseline models and results are not overwritten.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from src.preprocessing import build_preprocessor

from src.models.logistic_regression import build_estimator


# -------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"

TUNED_MODEL_PATH = (
    PROJECT_ROOT / "model" / "tuned" / "logistic_regression.joblib"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tuned"

CV_RESULTS_PATH = (
    RESULTS_DIR / "logistic_regression_cv_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "logistic_regression_cv_summary.csv"
)


# -------------------------------------------------------
# STEP 2: DEFINE EVALUATION METRICS
# -------------------------------------------------------

# MCC is the primary optimization criterion.
# The remaining five metrics are recorded for interpretation.

SCORING = {
    "Accuracy": "accuracy",

    "ROC_AUC": "roc_auc",

    "Precision": make_scorer(
        precision_score,
        pos_label=True,
        zero_division=0
    ),

    "Recall": make_scorer(
        recall_score,
        pos_label=True,
        zero_division=0
    ),

    "F1": make_scorer(
        f1_score,
        pos_label=True,
        zero_division=0
    ),

    "MCC": make_scorer(matthews_corrcoef)
}


# -------------------------------------------------------
# STEP 3: DEFINE THE MODEL PIPELINE
# -------------------------------------------------------

def create_pipeline():
    """
    Create an unfitted Logistic Regression pipeline.

    Keeping preprocessing inside the pipeline prevents
    information leakage between cross-validation folds.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_estimator())
        ]
    )


# -------------------------------------------------------
# STEP 4: OPTIMIZATION FUNCTION
# -------------------------------------------------------

def optimize_logistic_regression():

    print("LOGISTIC REGRESSION HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    # ---------------------------------------------------
    # Load training data only.
    # ---------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    print(f"Training observations: {len(train_df):,}")
    print(f"Input features: {X_train.shape[1]}")

    print("\nTarget distribution:")
    print(y_train.value_counts())

    # ---------------------------------------------------
    # Define stratified five-fold cross-validation.
    # ---------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\nCross-validation: Stratified 5-Fold")
    print("Primary optimization metric: MCC")

    # ---------------------------------------------------
    # Evaluate original configuration using CV.
    # ---------------------------------------------------

    print("\nBASELINE CROSS-VALIDATION")
    print("-" * 65)

    baseline_pipeline = create_pipeline()

    baseline_cv = cross_validate(
        estimator=baseline_pipeline,
        X=X_train,
        y=y_train,
        scoring=SCORING,
        cv=cv,
        n_jobs=2,
        error_score="raise"
    )

    for metric_name in SCORING:

        values = baseline_cv[f"test_{metric_name}"]

        print(
            f"{metric_name:<12}: "
            f"{values.mean():.4f} "
            f"(+/- {values.std():.4f})"
        )

    # ---------------------------------------------------
    # Define hyperparameter search space.
    # ---------------------------------------------------

    # Parameter names use classifier__ because the
    # classifier is a step inside the sklearn Pipeline.

    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0
        ],

        "classifier__class_weight": [
            None,
            "balanced"
        ]
    }

    # ---------------------------------------------------
    # Run Grid Search.
    # ---------------------------------------------------

    print("\nHYPERPARAMETER GRID SEARCH")
    print("-" * 65)

    grid_search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid=param_grid,
        scoring=SCORING,

        # Select the best candidate using mean CV MCC.
        refit="MCC",

        cv=cv,
        n_jobs=2,

        # Also record training scores to inspect overfitting.
        return_train_score=True,

        error_score="raise",
        verbose=1
    )

    # Every candidate is evaluated using training-only folds.
    # GridSearchCV then refits the selected configuration on
    # the complete 9,764-record training dataset.

    grid_search.fit(X_train, y_train)

    # ---------------------------------------------------
    # Display best configuration.
    # ---------------------------------------------------

    print("\nBEST HYPERPARAMETERS")
    print("-" * 65)

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest cross-validation MCC: "
        f"{grid_search.best_score_:.4f}"
    )

    # ---------------------------------------------------
    # Compare baseline CV and selected candidate CV.
    # ---------------------------------------------------

    best_index = grid_search.best_index_

    comparison_rows = []

    print("\nTRAINING-ONLY CROSS-VALIDATION COMPARISON")
    print("-" * 65)

    print(
        f"{'Metric':<12}"
        f"{'Baseline CV':>15}"
        f"{'Selected CV':>15}"
    )

    for metric_name in SCORING:

        baseline_mean = baseline_cv[
            f"test_{metric_name}"
        ].mean()

        tuned_mean = grid_search.cv_results_[
            f"mean_test_{metric_name}"
        ][best_index]

        comparison_rows.append({
            "Metric": metric_name,
            "Baseline_CV": baseline_mean,
            "Selected_CV": tuned_mean,
            "Difference": tuned_mean - baseline_mean
        })

        print(
            f"{metric_name:<12}"
            f"{baseline_mean:>15.4f}"
            f"{tuned_mean:>15.4f}"
        )

    # ---------------------------------------------------
    # Save detailed cross-validation results.
    # ---------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.DataFrame(
        grid_search.cv_results_
    )

    cv_results_df = cv_results_df.sort_values(
        by="mean_test_MCC",
        ascending=False
    )

    cv_results_df.to_csv(
        CV_RESULTS_PATH,
        index=False
    )

    summary_df = pd.DataFrame(comparison_rows)

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    # ---------------------------------------------------
    # Save the selected fitted pipeline.
    # ---------------------------------------------------

    TUNED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_pipeline = grid_search.best_estimator_

    joblib.dump(
        best_pipeline,
        TUNED_MODEL_PATH
    )

    print("\nOPTIMIZED MODEL SAVED")
    print(TUNED_MODEL_PATH)

    print("\nCROSS-VALIDATION RESULTS SAVED")
    print(CV_RESULTS_PATH)
    print(SUMMARY_PATH)

    # ---------------------------------------------------
    # Validate saved model.
    # ---------------------------------------------------

    loaded_pipeline = joblib.load(TUNED_MODEL_PATH)

    # Use a small training sample only for serialization
    # verification. No testing records are accessed.
    sample_features = X_train.head(10)

    original_predictions = best_pipeline.predict(
        sample_features
    )

    loaded_predictions = loaded_pipeline.predict(
        sample_features
    )

    assert (
        original_predictions == loaded_predictions
    ).all(), "Saved model validation failed."

    print("\nOPTIMIZATION VALIDATION SUCCESSFUL")
    print("Saved model predictions match after reloading.")
    print("The held-out testing dataset was not accessed.")


if __name__ == "__main__":
    optimize_logistic_regression()