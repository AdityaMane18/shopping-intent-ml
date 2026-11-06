# Online Shopping Purchase Intention Classification

BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2

**Project status:** Step 1 starter. Dataset loading and inspection are implemented. Model training, evaluation, Streamlit, and submission documentation are not yet complete.

## Problem statement

Predict whether an online shopping session will result in a purchase using visitor activity and session-context features. The target is `Revenue` (`True` = purchase, `False` = no purchase).

## Dataset description

- Source: UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset (ID 468)
- Original observations: 12,330
- Input features: 17
- Target: `Revenue` (binary classification)
- Source dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Dataset citation: Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Local setup (Windows 11 / VS Code)

1. Open this folder in VS Code.
2. Check Python: `py --version`.
3. Create a virtual environment: `py -m venv .venv`.
4. In VS Code, select the Python interpreter at `.venv\\Scripts\\python.exe`.
5. Install dependencies without activating PowerShell scripts:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

6. Run the first data inspection:

   ```powershell
   .\.venv\Scripts\python.exe -m src.data_preparation
   ```

This prints the dataset shape, feature count, missing values, duplicate rows, and target distribution. It does **not** train a model or change the original CSV.

## Planned module organization

- `src/data_preparation.py`: validate and inspect the source CSV (implemented)
- `src/models/`: one module per required classifier (baseline estimator factories implemented; not trained)
- `src/preprocessing.py`: shared categorical encoding and numerical transformation (next stage)
- `src/evaluation.py`: shared Accuracy, AUC, Precision, Recall, F1, and MCC calculations (next stage)
- `train_models.py`: one reproducible training/evaluation orchestrator (next stage)
- `model/`: saved trained pipelines, one for each classifier (generated later)
- `test_data.csv`: held-out test data with all input features and `Revenue` (generated later)
- `app.py`: Streamlit interface that loads saved pipelines without retraining (later stage)

All five models will use the **same** processed dataset, reproducible split, and metric definitions. Preprocessing must be fitted on training data only.

## Submission items still to complete

- Full metric comparison table for five models
- Written observations for each model and overall winner
- GitHub repository link
- Interactive Streamlit Cloud link and verified CSV upload
- BITS Virtual Lab execution screenshot
- One final PDF with links, screenshot, and required README content

