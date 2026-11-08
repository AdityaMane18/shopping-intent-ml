"""
Decision Tree Hyperparameter Optimization
----------------------------------------

BITS Pilani - Machine Learning Assignment 2

Objective:
    Optimize the Decision Tree classifier using training data only.

Methodology:
    1. Load training data.
    2. Evaluate the baseline Decision Tree with stratified CV.
    3. Search predefined hyperparameter combinations.
    4. Select the configuration with the highest mean CV MCC.
    5. Compare baseline and optimized performance.
    6. Save optimization results and the selected pipeline.
    7. Validate model serialization.

IMPORTANT:
    The testing dataset is never loaded.

    Preprocessing is fitted independently inside each CV fold.

    Original baseline files are never overwritten.
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
from src.models.decision_tree import build_estimator


# -------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "train_data.csv"

TUNED_MODEL_PATH = (
    PROJECT_ROOT / "model" / "tuned" / "decision_tree.joblib"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "tuned"

CV_RESULTS_PATH = (
    RESULTS_DIR / "decision_tree_cv_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR / "decision_tree_cv_summary.csv"
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
    Create an unfitted Decision Tree pipeline.

    The preprocessing step remains inside the pipeline
    to prevent leakage between cross-validation folds.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_estimator())
        ]
    )


# -------------------------------------------------------
# STEP 4: OPTIMIZE DECISION TREE
# -------------------------------------------------------

def optimize_decision_tree():

    print("DECISION TREE HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    # Load only the training dataset.
    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df.drop(columns=["Revenue"])
    y_train = train_df["Revenue"]

    print(f"Training observations: {len(train_df):,}")
    print(f"Input features: {X_train.shape[1]}")

    print("\nTarget distribution:")
    print(y_train.value_counts())

    # Preserve class proportions in each validation fold.
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
    # STEP 6: DEFINE HYPERPARAMETER SEARCH SPACE
    # ---------------------------------------------------

    param_grid = {

        # Controls splitting criterion.
        "classifier__criterion": [
            "gini",
            "entropy"
        ],

        # Limits tree complexity.
        "classifier__max_depth": [
            5,
            10,
            None
        ],

        # Controls the minimum samples required to split.
        "classifier__min_samples_split": [
            2,
            10
        ],

        # Prevents excessively small leaf nodes.
        "classifier__min_samples_leaf": [
            1,
            10,
            30
        ],

        # Tests whether class balancing helps.
        "classifier__class_weight": [
            None,
            "balanced"
        ],

        # Tests cost-complexity pruning.
        "classifier__ccp_alpha": [
            0.0,
            0.001
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

        # Select the candidate with the highest mean CV MCC.
        refit="MCC",

        cv=cv,
        n_jobs=2,
        return_train_score=True,
        error_score="raise",
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # ---------------------------------------------------
    # STEP 8: BEST HYPERPARAMETERS
    # ---------------------------------------------------

    print("\nBEST HYPERPARAMETERS")
    print("-" * 65)

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest cross-validation MCC: "
        f"{grid_search.best_score_:.4f}"
    )

    # Inspect the structure of the selected Decision Tree.
    best_pipeline = grid_search.best_estimator_

    best_tree = best_pipeline.named_steps["classifier"]

    print("\nOPTIMIZED DECISION TREE STRUCTURE")
    print("-" * 65)

    print(f"Tree depth: {best_tree.get_depth()}")
    print(f"Number of leaf nodes: {best_tree.get_n_leaves()}")

    # ---------------------------------------------------
    # STEP 9: COMPARE BASELINE AND OPTIMIZED CV
    # ---------------------------------------------------

    print("\nTRAINING-ONLY CROSS-VALIDATION COMPARISON")
    print("-" * 65)

    best_index = grid_search.best_index_

    comparison_rows = []

    print(
        f"{'Metric':<12}"
        f"{'Baseline CV':>15}"
        f"{'Selected CV':>15}"
        f"{'Difference':>15}"
    )

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
    # STEP 10: SAVE CROSS-VALIDATION RESULTS
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
    # STEP 11: SAVE THE OPTIMIZED PIPELINE
    # ---------------------------------------------------

    TUNED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
    # STEP 12: VALIDATE SAVED MODEL
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

    # Decide whether the tuned model is a candidate
    # for replacing the baseline model.
    baseline_mcc = baseline_cv["test_MCC"].mean()
    tuned_mcc = grid_search.best_score_

    print("\nMODEL SELECTION RECOMMENDATION")
    print("-" * 65)

    if tuned_mcc > baseline_mcc:
        print("Optimized model improved cross-validation MCC.")
        print("Candidate accepted for final model comparison.")
    else:
        print("No improvement in cross-validation MCC.")
        print("Retain the original model as the preferred baseline.")

    print("\nThe held-out testing dataset was not accessed.")


if __name__ == "__main__":
    optimize_decision_tree()