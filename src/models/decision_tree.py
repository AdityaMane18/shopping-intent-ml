"""
Decision Tree Classification Model
----------------------------------
This module defines a Decision Tree classifier to predict
whether an online shopping session results in a purchase.

Target:
    Revenue = True  -> Purchase
    Revenue = False -> No Purchase

A Decision Tree makes predictions through a sequence of
feature-based decisions.

For example:
    Is PageValues greater than a particular value?
        |
        |-- Yes -> Evaluate another condition
        |
        |-- No  -> Evaluate a different condition

The algorithm automatically learns these decision rules
from the training dataset.
"""

from sklearn.tree import DecisionTreeClassifier


def build_estimator():
    """
    Create the Decision Tree classification model.

    Parameters:
        criterion='gini':
            Measures the impurity of a node.
            The algorithm searches for splits that reduce impurity.

        splitter='best':
            Selects the best available split at each node.

        max_depth=None:
            Allows the tree to grow without a predefined depth limit.

        min_samples_split=2:
            At least two samples are required to split a node.

        min_samples_leaf=1:
            A leaf node must contain at least one sample.

        random_state=42:
            Helps ensure reproducibility.

    Returns:
        Configured Decision Tree classifier.
    """

    model = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    return model