"""
Online Shopping Purchase Intention Prediction
---------------------------------------------

BITS Pilani - Machine Learning Assignment 2

This Streamlit application evaluates five trained classifiers:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

Application workflow:
    1. User selects a classification model.
    2. User uploads a CSV containing the input features and Revenue.
    3. Application validates the uploaded dataset.
    4. Saved preprocessing and classification pipeline is loaded.
    5. Model predicts the Revenue target.
    6. Six classification metrics are calculated.
    7. Confusion matrix and classification report are displayed.
    8. User can download the prediction results.

IMPORTANT:
    The application only performs inference.
    Models are NOT retrained inside Streamlit.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

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
# STEP 1: APPLICATION CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="Online Shopping Intent Prediction",
    page_icon="🛒",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "model" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"


# Map display names to saved model filenames.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}


# -------------------------------------------------------
# STEP 2: LOAD THE SELECTED TRAINED MODEL
# -------------------------------------------------------

@st.cache_resource
def load_saved_model(model_name):
    """
    Load a previously trained model pipeline.

    The pipeline includes:
        - Numerical feature scaling
        - Categorical feature encoding
        - Trained classification model

    Caching avoids reloading the model on every interaction.

    No training is performed in this function.
    """

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model file not found: {model_path}"
        )

    # Load only the trusted model files bundled with this project.
    return joblib.load(model_path)


# -------------------------------------------------------
# STEP 3: VALIDATE THE TARGET COLUMN
# -------------------------------------------------------

def validate_target(dataframe):
    """
    Validate and convert the Revenue column into boolean values.

    Accepted values:
        True, False
        true, false
        1, 0

    Returns:
        Boolean target Series.
    """

    if "Revenue" not in dataframe.columns:
        raise ValueError(
            "The uploaded CSV must contain the 'Revenue' column."
        )

    # Convert values into a consistent string representation.
    target_text = (
        dataframe["Revenue"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    value_mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False
    }

    y_true = target_text.map(value_mapping)

    if y_true.isna().any():
        raise ValueError(
            "Revenue must contain only True/False or 1/0 values."
        )

    # ROC AUC requires observations from both target classes.
    if y_true.nunique() < 2:
        raise ValueError(
            "The uploaded dataset must contain both Revenue classes "
            "to calculate all six evaluation metrics."
        )

    return y_true.astype(bool)


# -------------------------------------------------------
# STEP 4: CALCULATE CLASSIFICATION METRICS
# -------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_probability):
    """
    Calculate the six evaluation metrics required by the assignment.

    Revenue=True is treated as the positive class.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "ROC AUC": roc_auc_score(y_true, y_probability),

        "Precision": precision_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            pos_label=True,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(y_true, y_pred)
    }

    return metrics


# -------------------------------------------------------
# STEP 5: DISPLAY APPLICATION HEADER
# -------------------------------------------------------

st.title("🛒 Online Shopping Purchase Intention Prediction")

st.write(
    "Evaluate trained machine learning classifiers using "
    "online shopping session data."
)

st.caption(
    "Dataset: UCI Online Shoppers Purchasing Intention | "
    "Classification target: Revenue"
)


# -------------------------------------------------------
# STEP 6: MODEL SELECTION
# -------------------------------------------------------

st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    options=list(MODEL_FILES.keys())
)

st.sidebar.info(
    "The selected classifier is loaded from a saved model file. "
    "No retraining is performed."
)


# -------------------------------------------------------
# STEP 7: DISPLAY EXISTING MODEL COMPARISON
# -------------------------------------------------------

comparison_path = RESULTS_DIR / "model_comparison.csv"

with st.expander("View Five-Model Baseline Comparison"):

    if comparison_path.exists():

        comparison_df = pd.read_csv(comparison_path)

        st.dataframe(
            comparison_df.round(4),
            hide_index=True
        )

        st.caption(
            "These results were calculated previously on the "
            "same held-out testing dataset."
        )

    else:

        st.warning(
            "The consolidated model comparison file is unavailable."
        )


# -------------------------------------------------------
# STEP 8: UPLOAD TESTING DATA
# -------------------------------------------------------

st.subheader("Upload Testing Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV containing the input features and Revenue target",
    type=["csv"]
)

st.caption(
    "For testing, upload the test_data.csv file provided "
    "in the GitHub repository."
)


if uploaded_file is None:

    st.info(
        "Select a model and upload a CSV file to begin evaluation."
    )

    st.stop()


# -------------------------------------------------------
# STEP 9: READ AND VALIDATE UPLOADED DATA
# -------------------------------------------------------

try:

    test_df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(f"Unable to read the uploaded CSV: {error}")
    st.stop()


if test_df.empty:

    st.error("The uploaded CSV contains no observations.")
    st.stop()


st.success("CSV uploaded successfully.")

st.write(f"**Total testing records:** {len(test_df):,}")

with st.expander("Preview Uploaded Dataset"):

    st.dataframe(test_df.head(10))


# Validate the target before running predictions.
try:

    y_test = validate_target(test_df)

except ValueError as error:

    st.error(str(error))
    st.stop()


# -------------------------------------------------------
# STEP 10: LOAD SELECTED MODEL
# -------------------------------------------------------

try:

    pipeline = load_saved_model(selected_model)

except Exception as error:

    st.error(f"Unable to load the selected model: {error}")
    st.stop()


# -------------------------------------------------------
# STEP 11: VALIDATE INPUT FEATURE COLUMNS
# -------------------------------------------------------

# Retrieve the original input columns used during training.
expected_features = list(
    pipeline.named_steps["preprocessor"].feature_names_in_
)

missing_features = [
    column
    for column in expected_features
    if column not in test_df.columns
]

if missing_features:

    st.error(
        f"Missing required feature columns: {missing_features}"
    )

    st.stop()


# Select features in the same order used during training.
# Revenue is excluded because it is the prediction target.
X_test = test_df[expected_features]


# -------------------------------------------------------
# STEP 12: GENERATE PREDICTIONS
# -------------------------------------------------------

try:

    # Generate predicted class labels.
    y_pred = pipeline.predict(X_test)

    # Locate the probability column corresponding to Revenue=True.
    positive_class_index = list(pipeline.classes_).index(True)

    # Generate the predicted purchase probabilities.
    y_probability = pipeline.predict_proba(X_test)[
        :, positive_class_index
    ]

except Exception as error:

    st.error(f"Prediction failed: {error}")
    st.stop()


st.success(
    f"Predictions generated using {selected_model}."
)


# -------------------------------------------------------
# STEP 13: DISPLAY SIX EVALUATION METRICS
# -------------------------------------------------------

metrics = calculate_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_probability=y_probability
)

st.subheader("Classification Evaluation Metrics")

metric_columns = st.columns(3)

for index, (metric_name, metric_value) in enumerate(metrics.items()):

    metric_columns[index % 3].metric(
        label=metric_name,
        value=f"{metric_value:.4f}"
    )


# -------------------------------------------------------
# STEP 14: DISPLAY CONFUSION MATRIX
# -------------------------------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[False, True]
)

cm_df = pd.DataFrame(
    cm,

    index=[
        "Actual: No Purchase",
        "Actual: Purchase"
    ],

    columns=[
        "Predicted: No Purchase",
        "Predicted: Purchase"
    ]
)

st.dataframe(cm_df)

st.caption(
    "Rows represent actual classes; columns represent "
    "predicted classes."
)


# -------------------------------------------------------
# STEP 15: DISPLAY CLASSIFICATION REPORT
# -------------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    labels=[False, True],
    target_names=["No Purchase", "Purchase"],
    output_dict=True,
    zero_division=0
)

# Convert the classification report into a readable table.
report_df = pd.DataFrame(report).transpose()

# Accuracy is already displayed in the evaluation metrics section.
# Remove its redundant row from the classification report.
report_df = report_df.drop(index="accuracy")

# Support represents the number of observations, so use integers.
report_df["support"] = report_df["support"].astype(int)

# Display Precision, Recall, F1 Score and Support.
st.dataframe(report_df.round(4))


# -------------------------------------------------------
# STEP 16: DOWNLOAD PREDICTION RESULTS
# -------------------------------------------------------

st.subheader("Download Predictions")

# Preserve the uploaded features and actual Revenue target.
predictions_df = test_df.copy()

# Add model predictions and corresponding probabilities.
predictions_df["Predicted_Revenue"] = y_pred

predictions_df["Purchase_Probability"] = y_probability

csv_output = predictions_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction Results",
    data=csv_output,
    file_name="shopping_intent_predictions.csv",
    mime="text/csv"
)


# -------------------------------------------------------
# STEP 17: APPLICATION COMPLETION MESSAGE
# -------------------------------------------------------

st.success(
    "Evaluation completed successfully. "
    "The selected saved model was used without retraining."
)