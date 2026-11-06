"""
Model Evaluation
----------------
This module provides a common evaluation function for all
classification models in the assignment.

The six evaluation metrics are:
1. Accuracy
2. ROC AUC
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)
"""

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


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a trained classification model on unseen testing data.

    Parameters:
        model: Trained classification pipeline
        X_test: Testing input features
        y_test: Actual target values

    Returns:
        Dictionary containing six evaluation metrics.
    """

    # Predict the purchase outcome for each testing record.
    y_pred = model.predict(X_test)

    # Obtain predicted probabilities for the positive class (Revenue=True).
    # ROC AUC evaluates how well the model ranks purchasing sessions.
    positive_class_index = list(model.classes_).index(True)

    y_prob = model.predict_proba(X_test)[:, positive_class_index]

    # Calculate the six metrics required by the assignment.
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(
            y_test, y_pred, pos_label=True, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, pos_label=True, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, pos_label=True, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    # Display metrics.
    print("\nMODEL EVALUATION RESULTS")
    print("=" * 50)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<15}: {metric_value:.4f}")

    # Confusion matrix:
    # Rows represent actual classes.
    # Columns represent predicted classes.
    print("\nCONFUSION MATRIX")
    print("=" * 50)

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[False, True]
        )
    )

    print("\nCLASSIFICATION REPORT")
    print("=" * 50)

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[False, True],
            zero_division=0
        )
    )

    return metrics