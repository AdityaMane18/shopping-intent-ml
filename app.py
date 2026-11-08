"""Streamlit homepage for BITS Pilani Machine Learning Assignment 2.

The app evaluates already-trained classifiers on an uploaded, labeled test CSV.
No model fitting, hyperparameter tuning, or saving takes place here.

Run from the project root:
    python -m streamlit run app.py
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_TEST_ROWS = 2_441
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB: the official test CSV is much smaller.

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# These versions were selected using training-only, five-fold cross-validation.
# Paths are deliberately explicit: the app never picks models using test scores.
MODEL_OPTIONS = {
    "Decision Tree · Tuned": "model/tuned/decision_tree.joblib",
    "Logistic Regression · Tuned": "model/tuned/logistic_regression.joblib",
    "Support Vector Machine · Baseline": "model/baseline/svm.joblib",
    "KNN · Baseline": "model/baseline/knn.joblib",
    "Gaussian Naive Bayes · Tuned": "model/tuned/naive_bayes.joblib",
}


st.set_page_config(
    page_title="Shopping Intent | BITS ML Assignment 2",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    """Theme-aware styling for the dashboard and persistent sidebar."""
    # Streamlit themes are selectable independently from the user's OS theme.
    # Adjust custom heading accents to keep contrast readable in both modes.
    active_theme = st.context.theme.type
    accent = "#0F766E" if active_theme == "light" else "#2DD4BF"
    st.markdown(
        """
        <style>
        :root {
            --intent-accent: #14b8a6;
            --intent-border: rgba(148, 163, 184, 0.24);
            --intent-surface: rgba(148, 163, 184, 0.07);
        }
        [data-testid="stAppViewContainer"] {
            background-image:
                radial-gradient(circle at 97% 4%, rgba(20,184,166,.095), transparent 38%),
                radial-gradient(circle at 0% 93%, rgba(20,184,166,.045), transparent 35%);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--intent-border);
        }
        .block-container { max-width: 1430px; padding-top: 1.7rem; }
        .sidebar-brand {
            padding: 0.8rem 0 1.25rem;
            border-bottom: 1px solid var(--intent-border);
            margin-bottom: 1.35rem;
        }
        .sidebar-brand .brand-mark {
            display: inline-flex;
            width: 42px;
            height: 42px;
            align-items: center;
            justify-content: center;
            border-radius: 13px;
            background: rgba(20, 184, 166, 0.14);
            font-size: 1.45rem;
            margin-bottom: 0.85rem;
        }
        .sidebar-brand h3 { font-size: 1.15rem; margin: 0 0 0.25rem; }
        .sidebar-brand p { opacity: .65; font-size: .84rem; margin: 0; }
        .sidebar-step {
            color: var(--intent-accent);
            text-transform: uppercase;
            font-weight: 750;
            letter-spacing: .10em;
            font-size: .76rem;
            margin: 0.8rem 0 .6rem;
        }
        .assignment-hero {
            padding: 2rem 2.2rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 184, 166, 0.27);
            background: linear-gradient(120deg, rgba(20, 184, 166, 0.12), rgba(148, 163, 184, 0.055));
            box-shadow: 0 20px 60px rgba(2, 6, 23, 0.07);
            margin-bottom: 1.15rem;
        }
        .hero-layout {
            display: grid;
            grid-template-columns: minmax(0, 1.7fr) minmax(220px, .7fr);
            gap: 2.2rem;
            align-items: center;
        }
        .assignment-eyebrow {
            color: var(--intent-accent);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 0.65rem;
        }
        .assignment-hero h1 {
            font-size: clamp(2rem, 3.45vw, 3.15rem);
            letter-spacing: -0.045em;
            line-height: 1.1;
            margin: 0 0 0.7rem;
            color: inherit;
        }
        .assignment-hero h1 span { color: var(--intent-accent); }
        .assignment-hero p {
            color: inherit;
            opacity: 0.78;
            margin: 0;
            line-height: 1.55;
            max-width: 690px;
            font-size: .98rem;
        }
        .assignment-identity {
            margin-top: 1.05rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        .assignment-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            background: rgba(148, 163, 184, 0.10);
            color: inherit;
            border: 1px solid var(--intent-border);
            border-radius: 999px;
            font-size: 0.81rem;
        }
        .hero-legend {
            padding: 1.25rem;
            background: rgba(148, 163, 184, 0.075);
            border: 1px solid var(--intent-border);
            border-radius: 19px;
        }
        .hero-legend .legend-title {
            text-transform: uppercase;
            letter-spacing: .13em;
            font-size: .67rem;
            opacity: .65;
            font-weight: 800;
            margin-bottom: .9rem;
        }
        .legend-outcome {
            display: flex;
            align-items: center;
            gap: .65rem;
            margin: .75rem 0;
            color: inherit;
            font-size: .91rem;
            font-weight: 650;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }
        .legend-dot.green { background: #10b981; }
        .legend-dot.blue { background: #64748b; }
        .legend-footer {
            margin-top: .85rem;
            padding-top: .85rem;
            border-top: 1px solid var(--intent-border);
            opacity: .68;
            font-size: .75rem;
        }
        .problem-panel {
            display: grid;
            grid-template-columns: minmax(0, 1.6fr) minmax(220px, .85fr);
            gap: 1.25rem;
            padding: 1.25rem 1.45rem;
            border: 1px solid var(--intent-border);
            border-radius: 20px;
            background: var(--intent-surface);
            margin-bottom: 1.25rem;
        }
        .problem-kicker {
            color: var(--intent-accent);
            font-size: .71rem;
            letter-spacing: .12em;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: .4rem;
        }
        .problem-panel h3 { margin: 0 0 .5rem; font-size: 1.15rem; }
        .problem-panel p {
            font-size: .9rem;
            line-height: 1.6;
            color: inherit;
            opacity: .79;
            margin: 0;
        }
        .problem-facts {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .6rem;
        }
        .problem-fact {
            padding: .72rem .85rem;
            border-radius: 13px;
            background: rgba(148, 163, 184, 0.09);
            border: 1px solid var(--intent-border);
        }
        .problem-fact strong {
            display: block;
            color: var(--intent-accent);
            font-size: 1.12rem;
            margin-bottom: .1rem;
        }
        .problem-fact span { display: block; font-size: .74rem; opacity: .8; }
        .section-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.2rem 0 .8rem;
        }
        .section-heading h2 { margin: 0; font-size: 1.45rem; letter-spacing: -.025em; }
        .section-heading p { margin: .18rem 0 0; opacity: .68; font-size: .89rem; }
        .section-index {
            color: var(--intent-accent);
            font-size: .8rem;
            font-weight: 800;
            letter-spacing: .06em;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: .25rem;
            padding: .35rem .3rem;
            border-radius: 13px;
            background: rgba(148, 163, 184, 0.075);
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            font-weight: 670;
            padding: .65rem 1rem;
            border-radius: 9px;
        }
        .ready-panel {
            border: 1px dashed rgba(20, 184, 166, 0.42);
            border-radius: 18px;
            background: rgba(20, 184, 166, 0.065);
            padding: 1.45rem 1.6rem;
            color: inherit;
        }
        .ready-panel h3 { margin: 0 0 .35rem; font-size: 1.1rem; }
        .ready-panel p { margin: 0; opacity: .72; font-size: .91rem; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .9rem;
            margin: 1rem 0 1.2rem;
        }
        .metric-card {
            position: relative;
            overflow: hidden;
            border-radius: 19px;
            padding: 1.25rem 1.35rem;
            min-height: 157px;
            border: 1px solid var(--intent-border);
            background: var(--intent-surface);
            box-shadow: 0 10px 28px rgba(2, 6, 23, 0.05);
        }
        .metric-card:before {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: var(--metric-accent);
            content: "";
        }
        .metric-label {
            display: flex;
            justify-content: space-between;
            font-size: .85rem;
            font-weight: 730;
            color: inherit;
            opacity: .83;
        }
        .metric-signal { color: var(--metric-accent); opacity: 1; font-size: .9rem; }
        .metric-number {
            display: block;
            font-size: clamp(1.85rem, 2.9vw, 2.55rem);
            font-weight: 780;
            letter-spacing: -.04em;
            font-variant-numeric: tabular-nums;
            color: inherit;
            line-height: 1.25;
            margin: .55rem 0 .4rem;
        }
        .metric-description { font-size: .78rem; color: inherit; opacity: .66; }
        .metric-card.mcc {
            background: rgba(20, 184, 166, 0.105);
            border-color: rgba(20, 184, 166, 0.42);
        }
        .cm-grid {
            display: grid;
            grid-template-columns: minmax(92px, 0.65fr) repeat(2, minmax(110px, 1fr));
            gap: 0.55rem;
            align-items: stretch;
            margin-top: 0.75rem;
        }
        .cm-heading, .cm-row-label {
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 0.82rem;
            font-weight: 650;
            opacity: 0.88;
            padding: 0.45rem;
        }
        .cm-cell {
            border-radius: 14px;
            padding: 0.8rem;
            min-height: 98px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            border: 1px solid var(--intent-border);
            background: rgba(148, 163, 184, 0.065);
        }
        .cm-cell.correct {
            background: rgba(16, 185, 129, 0.19);
            border: 1px solid rgba(16, 185, 129, 0.52);
            box-shadow: inset 4px 0 0 #10b981;
        }
        .cm-cell.error {
            background: rgba(245, 158, 11, 0.18);
            border: 1px solid rgba(245, 158, 11, 0.55);
            box-shadow: inset 4px 0 0 #f59e0b;
        }
        .cm-cell .cm-count {
            font-weight: 750;
            font-size: clamp(1.35rem, 2vw, 1.85rem);
            color: inherit;
        }
        .cm-cell .cm-description {
            color: inherit;
            opacity: 0.76;
            font-size: 0.76rem;
            text-align: center;
        }
        @media (max-width: 1100px) {
            .hero-layout, .problem-panel { grid-template-columns: 1fr; }
            .hero-legend { display: none; }
        }
        @media (max-width: 800px) {
            .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 580px) {
            .assignment-hero { padding: 1.35rem; }
            .problem-panel { padding: 1.1rem; }
            .metrics-grid { grid-template-columns: 1fr; }
            .cm-grid { grid-template-columns: minmax(68px, .6fr) repeat(2, 1fr); }
            .cm-cell { padding: .5rem; min-height: 82px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<style>:root {{ --intent-accent: {accent}; }}</style>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_saved_model(model_path: str):
    """Read a trusted, fitted joblib pipeline; never retrain in the app."""
    return joblib.load(model_path)


def read_and_validate_test_csv(raw_csv: bytes) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the uploaded labeled test sample and parse its target safely."""
    if len(raw_csv) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the 2 MB upload limit for this demonstration.")

    try:
        frame = pd.read_csv(BytesIO(raw_csv))
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise ValueError("The uploaded file is not a readable CSV.") from exc

    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    if len(frame) > MAX_TEST_ROWS:
        raise ValueError(
            f"Upload test data only (maximum {MAX_TEST_ROWS:,} rows). "
            "Do not upload the training dataset."
        )

    missing = [column for column in (*FEATURE_COLUMNS, "Revenue") if column not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Evaluating Accuracy/F1/MCC requires known labels. Only True/False or 1/0
    # are accepted so accidental strings cannot silently become purchase=True.
    target_text = frame["Revenue"].astype("string").str.strip().str.lower()
    target_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }
    if not target_text.isin(target_map).all():
        raise ValueError("Revenue must contain only True/False or 1/0 values.")
    target = target_text.map(target_map).astype(bool)

    if frame[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("The CSV contains missing feature values. Clean it before upload.")

    return frame, target


def evaluate_saved_pipeline(pipeline, data: pd.DataFrame, y_true: pd.Series) -> dict:
    """Evaluate the selected saved pipeline against uploaded labels."""
    expected = list(getattr(pipeline, "feature_names_in_", FEATURE_COLUMNS))
    missing = [name for name in expected if name not in data.columns]
    if missing:
        raise ValueError("Model input is missing: " + ", ".join(missing))

    # Match the exact feature order expected by the saved preprocessing pipeline.
    x_test = data.loc[:, expected]
    y_pred = np.asarray(pipeline.predict(x_test), dtype=bool)

    # Every model in this project was trained with Revenue=True as positive.
    classes = list(pipeline.classes_)
    positive_index = next((i for i, cls in enumerate(classes) if cls == True), None)
    if positive_index is None:
        raise ValueError("The saved model has no positive (Revenue=True) class.")
    probabilities = pipeline.predict_proba(x_test)[:, positive_index]

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC AUC": (
            roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else None
        ),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    report = classification_report(
        y_true,
        y_pred,
        labels=[False, True],
        target_names=["No Purchase", "Purchase"],
        output_dict=True,
        zero_division=0,
    )
    report_table = pd.DataFrame(report).transpose()
    # Match sklearn's complete text classification report: accuracy is shown
    # once as a summary row under the F1-score column.
    report_table.loc["accuracy", "precision"] = np.nan
    report_table.loc["accuracy", "recall"] = np.nan
    report_table.loc["accuracy", "f1-score"] = metrics["Accuracy"]
    report_table.loc["accuracy", "support"] = len(y_true)
    report_table = report_table.loc[
        ["No Purchase", "Purchase", "accuracy", "macro avg", "weighted avg"],
        ["precision", "recall", "f1-score", "support"],
    ]

    predictions = data.copy()
    predictions["Predicted_Revenue"] = y_pred
    predictions["Purchase_Probability"] = probabilities
    predictions["Prediction_Correct"] = y_true.to_numpy() == y_pred

    return {
        "metrics": metrics,
        "matrix": matrix,
        "report": report_table,
        "predictions_csv": predictions.to_csv(index=False).encode("utf-8"),
        "rows": len(data),
        "purchase_count": int(y_true.sum()),
    }


def render_confusion_matrix(matrix: np.ndarray) -> None:
    """Render a compact, labeled 2x2 confusion matrix without chart dependencies."""
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    st.markdown(
        f"""
        <div class="cm-grid" role="table" aria-label="Confusion matrix">
          <div></div>
          <div class="cm-heading">Predicted: No Purchase</div>
          <div class="cm-heading">Predicted: Purchase</div>
          <div class="cm-row-label">Actual:<br>No Purchase</div>
          <div class="cm-cell correct"><span class="cm-count">{tn:,}</span>
            <span class="cm-description">True Negative</span></div>
          <div class="cm-cell error"><span class="cm-count">{fp:,}</span>
            <span class="cm-description">False Positive</span></div>
          <div class="cm-row-label">Actual:<br>Purchase</div>
          <div class="cm-cell error"><span class="cm-count">{fn:,}</span>
            <span class="cm-description">False Negative</span></div>
          <div class="cm-cell correct"><span class="cm-count">{tp:,}</span>
            <span class="cm-description">True Positive</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: dict) -> None:
    """Render six theme-aware metric cards with explanatory labels."""
    descriptions = {
        "Accuracy": "Correct predictions across all sessions",
        "ROC AUC": "Ability to rank purchases above non-purchases",
        "Precision": "Reliability of predicted purchases",
        "Recall": "Share of actual purchases identified",
        "F1 Score": "Balance between precision and recall",
        "MCC": "Primary balanced classification metric",
    }
    accents = {
        "Accuracy": "#3B82F6",
        "ROC AUC": "#8B5CF6",
        "Precision": "#06B6D4",
        "Recall": "#F59E0B",
        "F1 Score": "#10B981",
        "MCC": "#14B8A6",
    }
    cards = []
    for title, value in metrics.items():
        display_value = f"{value:.4f}" if value is not None else "N/A"
        highlight_class = " mcc" if title == "MCC" else ""
        cards.append(
            f'<article class="metric-card{highlight_class}" '
            f'style="--metric-accent: {accents[title]};">'
            f'<div class="metric-label"><span>{title}</span>'
            '<span class="metric-signal">●</span></div>'
            f'<strong class="metric-number">{display_value}</strong>'
            f'<div class="metric-description">{descriptions[title]}</div>'
            '</article>'
        )
    st.markdown('<div class="metrics-grid">' + ''.join(cards) + '</div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    """Load training data for EDA only; test labels are never used in EDA."""
    return pd.read_csv(PROJECT_ROOT / "data" / "train_data.csv")


def render_evaluation_tab(uploaded_file, selected_model: str, evaluate_clicked: bool) -> None:
    """The four assignment-marking requirements live together on the first tab."""
    st.subheader("Evaluate a saved classifier")
    st.caption("Choose a model and upload labeled test_data.csv in the left sidebar.")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="ready-panel">
              <h3>Evaluation workspace</h3>
              <p>After uploading the test CSV, evaluate one of the five saved
              models to see its six metrics, confusion matrix, and full
              classification report.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    raw_csv = uploaded_file.getvalue()
    evaluation_key = (selected_model, sha256(raw_csv).hexdigest())
    try:
        data, labels = read_and_validate_test_csv(raw_csv)
    except ValueError as exc:
        st.error(str(exc))
        return

    with st.expander(f"Test data preview · {len(data):,} records · 17 features"):
        st.dataframe(data.head(5), use_container_width=True, hide_index=True)

    if evaluate_clicked:
        model_file = PROJECT_ROOT / MODEL_OPTIONS[selected_model]
        if not model_file.is_file():
            st.error(f"Saved model not found: `{model_file.relative_to(PROJECT_ROOT)}`")
            return
        try:
            with st.spinner("Evaluating the selected saved model..."):
                model = load_saved_model(str(model_file))
                result = evaluate_saved_pipeline(model, data, labels)
        except Exception as exc:
            st.error(f"Evaluation could not be completed: {exc}")
            return
        st.session_state["evaluation_key"] = evaluation_key
        st.session_state["evaluation_result"] = result

    if st.session_state.get("evaluation_key") != evaluation_key:
        st.info("Test data validated. Click **Evaluate model** in the sidebar to continue.")
        return

    result = st.session_state["evaluation_result"]
    st.caption(
        f"**{selected_model}** · {result['rows']:,} test sessions · "
        f"{result['purchase_count']:,} actual purchases"
    )
    render_metric_cards(result["metrics"])

    st.markdown(
        '<div class="section-heading"><div><span class="section-index">DIAGNOSTICS</span>'
        '<h2>Confusion matrix and classification report</h2>'
        '<p>Actual versus predicted purchases, with class-level scores.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    matrix_col, report_col = st.columns([1.05, 1.2], gap="large")
    with matrix_col:
        with st.container(border=True):
            st.markdown("**Confusion Matrix**")
            render_confusion_matrix(result["matrix"])
            st.caption(
                f"{result['rows']:,} uploaded test sessions only. "
                "Green = correct; amber = incorrect."
            )

    with report_col:
        with st.container(border=True):
            st.markdown("**Classification Report**")
            report = result["report"].copy()
            report["support"] = report["support"].astype(int)
            st.dataframe(
                report,
                use_container_width=True,
                column_config={
                    "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                    "f1-score": st.column_config.NumberColumn("F1 Score", format="%.4f"),
                    "support": st.column_config.NumberColumn("Support", format="%d"),
                },
            )
            st.caption(
                "Purchase (Revenue=True) is positive. The accuracy summary row "
                "uses the F1 Score column, as in scikit-learn's text report."
            )

    st.download_button(
        "Download test predictions (CSV)",
        data=result["predictions_csv"],
        file_name="shopping_intent_test_predictions.csv",
        mime="text/csv",
    )
    st.caption("The app loads fitted models and does not retrain or tune them.")


def render_problem_tab() -> None:
    st.subheader("Problem statement and dataset")
    st.markdown(
        """
        <section class="problem-panel">
          <div>
            <div class="problem-kicker">Binary classification problem</div>
            <h3>Did a completed online browsing session result in a purchase?</h3>
            <p>Online visitors interact with administrative, informational, and
            product pages before leaving or purchasing. Using 17 session-level
            attributes, we classify the resulting <b>Revenue</b> outcome:
            <b>True</b> for Purchase and <b>False</b> for No Purchase.
            The classes are imbalanced, so overall accuracy alone cannot
            describe model quality. We train five classifiers, compare them by
            five-fold cross-validation, and report six performance metrics on
            a held-out test set.</p>
          </div>
          <div class="problem-facts">
            <div class="problem-fact"><strong>12,205</strong><span>Cleaned sessions</span></div>
            <div class="problem-fact"><strong>9,764</strong><span>Training sessions</span></div>
            <div class="problem-fact"><strong>2,441</strong><span>Testing sessions</span></div>
            <div class="problem-fact"><strong>17</strong><span>Input features</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Feature groups**")
    numerical = (
        "Administrative, Administrative_Duration, Informational, "
        "Informational_Duration, ProductRelated, ProductRelated_Duration, "
        "BounceRates, ExitRates, PageValues, SpecialDay"
    )
    categorical = (
        "Month, OperatingSystems, Browser, Region, TrafficType, "
        "VisitorType, Weekend"
    )
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Numerical/session behaviour (10)**")
            st.write(numerical)
    with right:
        with st.container(border=True):
            st.markdown("**Categorical/context features (7)**")
            st.write(categorical)
    st.info(
        "This project classifies completed sessions. It is not presented "
        "as a real-time prediction before a visitor finishes browsing."
    )


def render_eda_tab() -> None:
    st.subheader("Exploratory data analysis")
    st.caption("All analyses below use the training split only (9,764 sessions).")
    with st.expander("EDA workflow and checks", expanded=False):
        st.markdown(
            """
            1. Inspect dataset shape, feature types, and the first training records.
            2. Check missing values and duplicate sessions.
            3. Study the Revenue class distribution and purchase imbalance.
            4. Compare monthly purchase rates and browsing behaviour by outcome.
            5. Explore numerical correlations to identify candidate associations.

            These checks describe the data. Feature selection and tuning remain
            separate model-development steps.
            """
        )
    train_path = PROJECT_ROOT / "data" / "train_data.csv"
    if not train_path.is_file():
        st.warning("EDA requires `data/train_data.csv` in the project directory.")
        return

    data = load_training_data().copy()
    if "Revenue" not in data.columns:
        st.error("The training CSV has no Revenue target column.")
        return
    labels = data["Revenue"].astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    if labels.isna().any():
        st.error("Revenue labels in training data are not valid binary values.")
        return
    data["RevenueFlag"] = labels.astype(int)

    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.drop(columns="RevenueFlag").duplicated().sum())
    positive_count = int(labels.sum())
    purchase_rate = 100 * positive_count / len(data)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training sessions", f"{len(data):,}")
    col2.metric("Purchase rate", f"{purchase_rate:.1f}%")
    col3.metric("Missing values", f"{missing_count:,}")
    col4.metric("Duplicate rows", f"{duplicate_count:,}")

    first, second = st.columns(2, gap="large")
    with first:
        st.markdown("**Target distribution**")
        distribution = pd.DataFrame(
            {"Outcome": ["No Purchase", "Purchase"],
             "Sessions": [len(data) - positive_count, positive_count]}
        )
        st.bar_chart(distribution, x="Outcome", y="Sessions", color="#14B8A6")
        st.caption("The positive class is less common; this motivates MCC.")
    with second:
        st.markdown("**Purchase rate by month**")
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = (data.groupby("Month")["RevenueFlag"].mean() * 100).reindex(month_order).dropna()
        monthly.name = "Purchase rate (%)"
        st.bar_chart(monthly, color="#14B8A6")
        st.caption("These rates describe the training dataset, not future months.")

    st.markdown("**Browsing behaviour by outcome**")
    behavioural = ["PageValues", "BounceRates", "ExitRates", "ProductRelated_Duration"]
    medians = data.groupby("RevenueFlag")[behavioural].median().T
    medians.columns = ["No Purchase", "Purchase"]
    st.dataframe(medians.round(4), use_container_width=True)
    st.caption("Median feature values by class. Associations do not imply causation.")

    with st.expander("Numeric feature correlations with Revenue"):
        continuous = [
            "Administrative", "Administrative_Duration", "Informational",
            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
            "BounceRates", "ExitRates", "PageValues", "SpecialDay",
        ]
        correlations = data[continuous].corrwith(data["RevenueFlag"]).dropna()
        correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
        st.bar_chart(correlations.rename("Correlation"), color="#14B8A6")
        st.caption("Univariate Pearson correlations; not model feature-importance scores.")

    with st.expander("First 10 training rows"):
        st.dataframe(data.drop(columns="RevenueFlag").head(10),
                     use_container_width=True, hide_index=True)


def render_methodology_tab() -> None:
    st.subheader("Model development and optimization")
    st.markdown(
        """
        1. Clean the dataset and separate the **Revenue** target from the 17 input features.
        2. Make an **80/20 stratified train–test split** (9,764 training and 2,441 test rows).
        3. Build five classifier pipelines with preprocessing inside each pipeline.
        4. Evaluate and tune candidates with **stratified five-fold cross-validation on training data only**.
        5. Select each version by cross-validation **MCC**, then evaluate those frozen versions on the held-out test set.
        """
    )
    st.markdown("**Chosen model versions**")
    choices = pd.DataFrame(
        [(name, path.split("/")[1]) for name, path in MODEL_OPTIONS.items()],
        columns=["Model", "Selected version"],
    )
    st.dataframe(choices, use_container_width=True, hide_index=True)

    cv_path = PROJECT_ROOT / "results" / "tuned" / "model_cv_comparison.csv"
    if cv_path.is_file():
        st.markdown("**Training cross-validation comparison**")
        cv_results = pd.read_csv(cv_path)
        columns = [
            name for name in ["Model", "Selected_Version", "Baseline_CV_MCC",
                              "Tuned_CV_MCC", "MCC_Improvement", "MCC"]
            if name in cv_results.columns
        ]
        if columns:
            st.dataframe(cv_results[columns], use_container_width=True, hide_index=True)
        else:
            st.dataframe(cv_results, use_container_width=True, hide_index=True)
    else:
        st.info("Add `results/tuned/model_cv_comparison.csv` to display the CV comparison.")
    st.caption("Hyperparameter search and preprocessing were fitted only within training folds.")


def render_comparison_tab() -> None:
    st.subheader("Overall comparison: five classifiers")
    st.caption("Final held-out test scores for the versions selected by training CV MCC.")
    report_path = PROJECT_ROOT / "results" / "final" / "final_model_comparison.csv"
    if not report_path.is_file():
        st.warning(
            "Overall results require `results/final/final_model_comparison.csv`. "
            "Generate it with `python -m evaluate_final_models`."
        )
        return

    report = pd.read_csv(report_path)
    required = {"Model", "CV_MCC", "Test_MCC", "Test_F1"}
    if not required.issubset(report.columns):
        st.error("The final comparison CSV does not contain the expected columns.")
        return
    report = report.sort_values("CV_MCC", ascending=False)
    winner = report.iloc[0]
    st.success(
        f"CV-selected leading model: **{winner['Model']}** "
        f"(CV MCC {winner['CV_MCC']:.4f}; test MCC {winner['Test_MCC']:.4f})."
    )
    chart_data = report.set_index("Model")[["Test_MCC", "Test_F1"]]
    st.markdown("**MCC and F1 Score on the test set**")
    st.bar_chart(chart_data)

    columns = [
        name for name in ["Model", "Version", "CV_MCC", "Test_Accuracy", "Test_ROC_AUC",
                          "Test_Precision", "Test_Recall", "Test_F1", "Test_MCC"]
        if name in report.columns
    ]
    st.markdown("**Full model performance table**")
    st.dataframe(report[columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Download overall model report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="final_model_comparison.csv",
        mime="text/csv",
    )
    st.caption(
        "The overall report is read from the saved final evaluation; it does not "
        "retrain models or use the uploaded CSV to change model selection."
    )


def main() -> None:
    apply_styles()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="brand-mark">🛍️</div>
              <h3>Shopping Intent Lab</h3>
              <p>BITS Pilani · ML Assignment 02</p>
            </div>
            <div class="sidebar-step">Step 01 / Choose model</div>
            """,
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Classification model",
            options=list(MODEL_OPTIONS),
            index=0,
            help="The version was selected by training-only five-fold CV MCC.",
        )
        st.caption("Decision Tree · Tuned has the highest cross-validation MCC.")
        st.markdown('<div class="sidebar-step">Step 02 / Upload test dataset</div>',
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload labeled CSV", type=["csv"],
            help="Use test_data.csv with the Revenue target column.",
        )
        st.caption("test_data.csv only · Up to 2,441 rows · 2 MB")
        evaluate_clicked = st.button(
            "Evaluate model", type="primary", use_container_width=True,
            disabled=uploaded_file is None,
        )

    st.markdown(
        """
        <section class="assignment-hero">
          <div class="hero-layout">
            <div>
              <div class="assignment-eyebrow">BITS Pilani / M.Tech AI & ML / Assignment 02</div>
              <h1>Online Shopping<br><span>Intent Prediction</span></h1>
              <p>Classification of completed shopping sessions using browsing
              behaviour, interpretable evaluation metrics, and five ML models.</p>
              <div class="assignment-identity">
                <span class="assignment-pill">Aditya Mane</span>
                <span class="assignment-pill">BITS ID · 2026AJ05604</span>
                <span class="assignment-pill">12,205 cleaned sessions</span>
              </div>
            </div>
            <div class="hero-legend">
              <div class="legend-title">Dataset overview</div>
              <div class="legend-outcome">9,764 training sessions</div>
              <div class="legend-outcome">2,441 testing sessions</div>
              <div class="legend-footer">17 features · Revenue=True / False</div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    evaluation, problem, eda, methods, comparison = st.tabs([
        "Model Evaluation", "Problem & Dataset", "EDA", "Model Development", "Overall Report"
    ])
    with evaluation:
        render_evaluation_tab(uploaded_file, selected_model, evaluate_clicked)
    with problem:
        render_problem_tab()
    with eda:
        render_eda_tab()
    with methods:
        render_methodology_tab()
    with comparison:
        render_comparison_tab()


if __name__ == "__main__":
    main()
