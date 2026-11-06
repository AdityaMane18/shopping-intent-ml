"""
K-Nearest Neighbors (KNN) Classification
----------------------------------------

This module defines the KNN classifier for predicting
whether an online shopping session results in a purchase.

KNN is a distance-based classification algorithm.

For a new observation:
1. Calculate its distance from training observations.
2. Identify the K nearest observations.
3. Examine their target classes.
4. Predict the majority class.

Example:
    Suppose K = 5.

    Among the five nearest observations:
        Revenue=True  -> 3 observations
        Revenue=False -> 2 observations

    Predicted Revenue = True

Feature scaling is important because KNN uses distances.
Our preprocessing pipeline already applies StandardScaler
to numerical features.
"""

from sklearn.neighbors import KNeighborsClassifier


def build_estimator():
    """
    Create the K-Nearest Neighbors classifier.

    Parameters:
        n_neighbors=5:
            Uses the five nearest training observations
            to predict the class.

        weights='uniform':
            Every neighbor contributes equally to the prediction.

        metric='minkowski':
            Defines the distance measurement.

        p=2:
            Makes Minkowski distance equivalent to
            Euclidean distance.

        algorithm='auto':
            Lets scikit-learn select an appropriate
            nearest-neighbor search algorithm.

    Returns:
        Configured KNN classifier.
    """

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        metric="minkowski",
        p=2,
        algorithm="auto"
    )

    return model