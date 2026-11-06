"""
Gaussian Naive Bayes Classification
----------------------------------

This module implements Gaussian Naive Bayes for predicting
whether an online shopping session results in a purchase.

Target:
    Revenue = True  -> Purchase
    Revenue = False -> No Purchase

Naive Bayes is a probabilistic classification algorithm
based on Bayes' theorem.

The algorithm assumes that input features are conditionally
independent given the target class.

Gaussian Naive Bayes assumes that feature values within each
class follow a Gaussian (normal) distribution.

Although this assumption may not hold perfectly for our data,
the algorithm provides a useful classification baseline.
"""

from sklearn.naive_bayes import GaussianNB


def build_estimator():
    """
    Create the Gaussian Naive Bayes classifier.

    Parameters:
        priors=None:
            Class probabilities are estimated automatically
            from the training dataset.

        var_smoothing=1e-9:
            Adds a small value to feature variances to improve
            numerical stability and avoid division by zero.

    Returns:
        Configured Gaussian Naive Bayes classifier.
    """

    model = GaussianNB(
        priors=None,
        var_smoothing=1e-9
    )

    return model