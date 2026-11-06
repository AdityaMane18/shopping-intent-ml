"""
Support Vector Machine Classification
-------------------------------------

This module implements Support Vector Classification (SVC)
with probability calibration.

Target:
    Revenue = True  -> Purchase
    Revenue = False -> No Purchase

SVM learns a decision boundary between the two target classes.

The RBF kernel enables the model to learn nonlinear patterns.

CalibratedClassifierCV converts the SVM decision scores into
probability estimates, allowing ROC AUC evaluation and
probability-based predictions in our Streamlit application.
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def build_estimator():
    """
    Create an SVM classifier with calibrated probabilities.

    SVM parameters:
        C=1.0:
            Regularization parameter.

        kernel='rbf':
            Radial Basis Function kernel for nonlinear classification.

        gamma='scale':
            Automatically determines the RBF kernel coefficient.

        random_state=42:
            Reproducibility parameter.

    Calibration parameters:
        method='sigmoid':
            Uses sigmoid-based probability calibration.

        cv=5:
            Uses five-fold cross-validation on training data
            to estimate the probability calibration function.

        ensemble=False:
            Fits the final underlying SVM using all training
            observations after obtaining calibration predictions.

    Returns:
        SVM classifier with calibrated probability estimates.
    """

    # Create the underlying SVM classifier.
    svm_model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        random_state=42
    )

    # Add probability calibration.
    model = CalibratedClassifierCV(
        estimator=svm_model,
        method="sigmoid",
        cv=5,
        ensemble=False
    )

    return model