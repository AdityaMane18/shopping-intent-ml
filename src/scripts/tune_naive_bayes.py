"""
Gaussian Naive Bayes Hyperparameter Optimization
------------------------------------------------

BITS Pilani - Machine Learning Assignment 2

Objective:
    Optimize Gaussian Naive Bayes using training data only.

Methodology:
    1. Load the training dataset.
    2. Evaluate the original model using stratified CV.
    3. Search variance smoothing and class prior values.
    4. Select the configuration with the highest CV MCC.
    5. Compare all six evaluation metrics.
    6. Save the optimized pipeline and CV results.
    7. Validate model serialization.

IMPORTANT:
    Testing data is never accessed.

    Preprocessing is included inside the pipeline to
    prevent data leakage during cross-validation.
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
from src.models.naive_bayes import build_estimator


# -------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"

TUNED_MODEL_PATH = (
    PROJECT_ROOT / "model" / "tuned" / "naive_bayes.joblib"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tuned"

CV_RESULTS_PATH = (
    RESULTS_DIR / "naive_bayes_cv_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "naive_bayes_cv_summary.csv"
)


# -------------------------------------------------------
# STEP 2: DEFINE EVALUATION METRICS
# -------------------------------------------------------

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
# STEP 3: CREATE MODEL PIPELINE
# -------------------------------------------------------

def create_pipeline():
    """
    Create an unfitted Gaussian Naive Bayes pipeline.

    Preprocessing is fitted independently within each
    cross-validation training fold.
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

def optimize_naive_bayes():

    print("GAUSSIAN NAIVE BAYES HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    print(f"Training observations: {len(train_df):,}")
    print(f"Input features: {X_train.shape[1]}")

    print("\nTarget distribution:")
    print(y_train.value_counts())

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\nCross-validation: Stratified 5-Fold")
    print("Primary optimization metric: MCC")

    # ---------------------------------------------------
    # STEP 5: BASELINE CROSS-VALIDATION
    # ---------------------------------------------------

    print("\nBASELINE CROSS-VALIDATION")
    print("-" * 65)

    baseline_cv = cross_validate(
        estimator=create_pipeline(),
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
    # STEP 6: HYPERPARAMETER SEARCH SPACE
    # ---------------------------------------------------

    # Variance smoothing adds a proportion of the
    # largest feature variance to estimated variances.
    #
    # This helps avoid extreme probability estimates
    # caused by very small feature variances.

    param_grid = {
        "classifier__var_smoothing": [
            1e-9,
            1e-8,
            1e-7,
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-2,
            1e-1,
            1.0
        ],

        # None uses class proportions learned from data.
        # [0.5, 0.5] assumes equal prior probabilities.
        "classifier__priors": [
            None,
            [0.5, 0.5]
        ]
    }

    # ---------------------------------------------------
    # STEP 7: GRID SEARCH
    # ---------------------------------------------------

    print("\nHYPERPARAMETER GRID SEARCH")
    print("-" * 65)

    grid_search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid=param_grid,
        scoring=SCORING,

        # Select the configuration with highest CV MCC.
        refit="MCC",

        cv=cv,
        n_jobs=2,
        return_train_score=True,
        error_score="raise",
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # ---------------------------------------------------
    # STEP 8: DISPLAY BEST HYPERPARAMETERS
    # ---------------------------------------------------

    print("\nBEST HYPERPARAMETERS")
    print("-" * 65)

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest cross-validation MCC: "
        f"{grid_search.best_score_:.4f}"
    )

    best_index = grid_search.best_index_

    # ---------------------------------------------------
    # STEP 9: CHECK OVERFITTING
    # ---------------------------------------------------

    train_mcc = grid_search.cv_results_[
        "mean_train_MCC"
    ][best_index]

    validation_mcc = grid_search.cv_results_[
        "mean_test_MCC"
    ][best_index]

    print("\nOVERFITTING CHECK")
    print("-" * 65)

    print(f"Mean training MCC   : {train_mcc:.4f}")
    print(f"Mean validation MCC : {validation_mcc:.4f}")
    print(f"MCC difference      : {train_mcc - validation_mcc:.4f}")

    # ---------------------------------------------------
    # STEP 10: COMPARE BASELINE AND OPTIMIZED RESULTS
    # ---------------------------------------------------

    print("\nTRAINING-ONLY CROSS-VALIDATION COMPARISON")
    print("-" * 65)

    print(
        f"{'Metric':<12}"
        f"{'Baseline CV':>15}"
        f"{'Selected CV':>15}"
        f"{'Difference':>15}"
    )

    comparison_rows = []

    for metric_name in SCORING:

        baseline_mean = baseline_cv[
            f"test_{metric_name}"
        ].mean()

        baseline_std = baseline_cv[
            f"test_{metric_name}"
        ].std()

        tuned_mean = grid_search.cv_results_[
            f"mean_test_{metric_name}"
        ][best_index]

        tuned_std = grid_search.cv_results_[
            f"std_test_{metric_name}"
        ][best_index]

        difference = tuned_mean - baseline_mean

        comparison_rows.append({
            "Metric": metric_name,
            "Baseline_CV": baseline_mean,
            "Baseline_CV_STD": baseline_std,
            "Selected_CV": tuned_mean,
            "Selected_CV_STD": tuned_std,
            "Difference": difference
        })

        print(
            f"{metric_name:<12}"
            f"{baseline_mean:>15.4f}"
            f"{tuned_mean:>15.4f}"
            f"{difference:>15.4f}"
        )

    # ---------------------------------------------------
    # STEP 11: SAVE CROSS-VALIDATION RESULTS
    # ---------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

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
    # STEP 12: SAVE OPTIMIZED MODEL
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
    # STEP 13: VALIDATE SAVED MODEL
    # ---------------------------------------------------

    loaded_pipeline = joblib.load(TUNED_MODEL_PATH)

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

    # ---------------------------------------------------
    # STEP 14: MODEL SELECTION RECOMMENDATION
    # ---------------------------------------------------

    baseline_mcc = baseline_cv["test_MCC"].mean()
    tuned_mcc = grid_search.best_score_

    print("\nMODEL SELECTION RECOMMENDATION")
    print("-" * 65)

    if tuned_mcc > baseline_mcc:
        print("Optimized model improved cross-validation MCC.")
        print("Candidate accepted for final model comparison.")
    else:
        print("No improvement in cross-validation MCC.")
        print("Retain the baseline Naive Bayes model.")

    print("\nThe held-out testing dataset was not accessed.")


if __name__ == "__main__":
    optimize_naive_bayes()