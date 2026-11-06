"""
Logistic Regression Classifier
------------------------------
This module defines the Logistic Regression model used to predict
whether an online shopping session will result in a purchase.

Target variable:
    Revenue = True  -> Customer made a purchase
    Revenue = False -> Customer did not make a purchase
"""

from sklearn.linear_model import LogisticRegression


def build_estimator():
    """
    Create a Logistic Regression classifier.

    Parameters:
        C=1.0:
            Controls regularization strength.
            Smaller C means stronger regularization.

        max_iter=1000:
            Maximum iterations allowed for model convergence.

        solver='lbfgs':
            Optimization algorithm used to estimate model parameters.

        random_state=42:
            Helps maintain reproducibility.

    Returns:
        Configured Logistic Regression classifier.
    """

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )

    return model