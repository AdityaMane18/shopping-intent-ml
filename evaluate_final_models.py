"""
Final Classification Model Evaluation
-------------------------------------

BITS Pilani - Machine Learning Assignment 2

Objective:
    Evaluate the five preselected classification models
    using the reserved testing dataset.

Methodology:
    1. Read the training-only CV model selection.
    2. Load the testing dataset.
    3. Load each selected, already-fitted model.
    4. Generate predictions without fitting.
    5. Calculate six classification metrics.
    6. Generate confusion matrices and reports.
    7. Compare CV MCC with testing MCC.
    8. Save final evaluation results.

IMPORTANT:
    No model fitting or hyperparameter tuning occurs here.

    Model versions were selected before this evaluation.

    The test dataset must not be used for further tuning.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# -------------------------------------------------------
# STEP 1: PROJECT CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_PATH = PROJECT_ROOT / "test_data.csv"

CV_COMPARISON_PATH = (
    PROJECT_ROOT
    / "results"
    / "tuned"
    / "model_cv_comparison.csv"
)

FINAL_RESULTS_DIR = PROJECT_ROOT / "results" / "final"

FINAL_COMPARISON_PATH = (
    FINAL_RESULTS_DIR / "final_model_comparison.csv"
)


# -------------------------------------------------------
# STEP 2: FINAL EVALUATION FUNCTION
# -------------------------------------------------------

def evaluate_final_models():

    print("FINAL CLASSIFICATION MODEL EVALUATION")
    print("=" * 75)

    # ---------------------------------------------------
    # Load preselected model configuration.
    # ---------------------------------------------------

    cv_comparison = pd.read_csv(CV_COMPARISON_PATH)

    if len(cv_comparison) != 5:
        raise ValueError(
            "Expected five models in CV comparison."
        )

    print("\nMODEL SELECTION FROM TRAINING CROSS-VALIDATION")
    print("-" * 75)

    print(
        cv_comparison[
            ["Model", "Selected_Version", "MCC"]
        ].to_string(index=False)
    )

    # ---------------------------------------------------
    # Load the reserved testing dataset.
    # ---------------------------------------------------

    test_df = pd.read_csv(TEST_PATH)

    if "Revenue" not in test_df.columns:
        raise ValueError(
            "Testing dataset must contain Revenue."
        )

    X_test = test_df.drop(columns=["Revenue"])
    y_test = test_df["Revenue"]

    print("\nTEST DATASET INFORMATION")
    print("-" * 75)

    print(f"Testing observations: {len(test_df):,}")
    print(f"Input features: {X_test.shape[1]}")

    print("\nTarget distribution:")
    print(y_test.value_counts())

    FINAL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_rows = []

    # ---------------------------------------------------
    # STEP 3: EVALUATE EACH SELECTED MODEL
    # ---------------------------------------------------

    for _, record in cv_comparison.iterrows():

        model_name = record["Model"]
        version = record["Selected_Version"]

        model_path = (
        PROJECT_ROOT / str(record["Model_Path"]).replace("\\", "/")
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        print(f"\nEVALUATING: {model_name}")
        print("-" * 75)

        # Load the previously fitted model.
        model = joblib.load(model_path)

        # Verify expected input columns when available.
        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

            if set(expected_features) != set(X_test.columns):
                raise ValueError(
                    f"Feature mismatch for {model_name}"
                )

            X_model = X_test[expected_features]

        else:
            X_model = X_test

        # Generate predictions only.
        # No model.fit() is called.
        y_pred = model.predict(X_model)

        # Locate the probability column for Revenue=True.
        class_labels = np.asarray(model.classes_)

        positive_indices = np.flatnonzero(
            class_labels == True
        )

        if len(positive_indices) != 1:
            raise ValueError(
                f"Positive class not found: {model_name}"
            )

        positive_index = int(positive_indices[0])

        y_probability = model.predict_proba(
            X_model
        )[:, positive_index]

        # ------------------------------------------------
        # STEP 4: COMPUTE EVALUATION METRICS
        # ------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        roc_auc = roc_auc_score(
            y_test,
            y_probability
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=True,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=True,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=True,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        cv_mcc = float(record["MCC"])

        # ------------------------------------------------
        # STEP 5: GENERATE CONFUSION MATRIX
        # ------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[False, True]
        )

        tn, fp, fn, tp = cm.ravel()

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"ROC AUC   : {roc_auc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"MCC       : {mcc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print(
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )

        # Save confusion matrix with readable labels.
        confusion_df = pd.DataFrame(
            cm,
            index=[
                "Actual_No_Purchase",
                "Actual_Purchase"
            ],
            columns=[
                "Predicted_No_Purchase",
                "Predicted_Purchase"
            ]
        )

        model_filename = model_path.stem

        confusion_df.to_csv(
            FINAL_RESULTS_DIR
            / f"{model_filename}_confusion_matrix.csv"
        )

        # ------------------------------------------------
        # STEP 6: CLASSIFICATION REPORT
        # ------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            labels=[False, True],
            target_names=[
                "No Purchase",
                "Purchase"
            ],
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        report_path = (
            FINAL_RESULTS_DIR
            / f"{model_filename}_classification_report.txt"
        )

        report_path.write_text(
            report,
            encoding="utf-8"
        )

        # ------------------------------------------------
        # STEP 7: SAVE PREDICTIONS
        # ------------------------------------------------

        predictions_df = pd.DataFrame({
            "Actual_Revenue": y_test.to_numpy(),
            "Predicted_Revenue": y_pred,
            "Purchase_Probability": y_probability
        })

        predictions_df.to_csv(
            FINAL_RESULTS_DIR
            / f"{model_filename}_predictions.csv",
            index=False
        )

        # ------------------------------------------------
        # STEP 8: COLLECT COMPARISON RESULTS
        # ------------------------------------------------

        comparison_rows.append({
            "Model": model_name,
            "Version": version,
            "CV_MCC": cv_mcc,
            "Test_Accuracy": accuracy,
            "Test_ROC_AUC": roc_auc,
            "Test_Precision": precision,
            "Test_Recall": recall,
            "Test_F1": f1,
            "Test_MCC": mcc,
            "Test_Minus_CV_MCC": mcc - cv_mcc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    # ---------------------------------------------------
    # STEP 9: CONSOLIDATE FINAL RESULTS
    # ---------------------------------------------------

    final_df = pd.DataFrame(comparison_rows)

    # Preserve the training-CV ranking.
    # Do not select a new winner based on test performance.
    final_df = final_df.sort_values(
        by="CV_MCC",
        ascending=False
    ).reset_index(drop=True)

    final_df.to_csv(
        FINAL_COMPARISON_PATH,
        index=False
    )

    print("\nFINAL MODEL PERFORMANCE COMPARISON")
    print("=" * 75)

    display_columns = [
        "Model",
        "Version",
        "CV_MCC",
        "Test_Accuracy",
        "Test_ROC_AUC",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
        "Test_MCC",
        "Test_Minus_CV_MCC"
    ]

    print(
        final_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # ---------------------------------------------------
    # STEP 10: REPORT THE CV-SELECTED LEADING MODEL
    # ---------------------------------------------------

    leading_model = final_df.iloc[0]

    print("\nCROSS-VALIDATION SELECTED MODEL")
    print("-" * 75)

    print(f"Model: {leading_model['Model']}")
    print(f"Version: {leading_model['Version']}")
    print(f"CV MCC: {leading_model['CV_MCC']:.4f}")
    print(f"Test MCC: {leading_model['Test_MCC']:.4f}")

    print("\nFINAL RESULTS SAVED")
    print(FINAL_COMPARISON_PATH)

    # ---------------------------------------------------
    # STEP 11: VALIDATION
    # ---------------------------------------------------

    assert len(final_df) == 5, (
        "Final comparison must contain five models."
    )

    assert final_df["Test_MCC"].notna().all(), (
        "Missing evaluation results."
    )

    print("\nFINAL EVALUATION COMPLETED SUCCESSFULLY")
    print("All five selected models were evaluated.")
    print("No model was retrained or tuned.")


if __name__ == "__main__":
    evaluate_final_models()