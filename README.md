# Shopping Intent Lab — Online Purchase Intention Classification

**BITS Pilani | M.Tech in Artificial Intelligence and Machine Learning | Machine Learning — Assignment 2**  
**Student:** Aditya Mane  
**BITS ID:** 2026AJ05604  
**Application:** Interactive Streamlit dashboard for classification and evaluation

## 1. Problem Statement

An online store receives many browsing sessions, but only a fraction result in a completed purchase. The objective of this project is to predict whether a visitor's current session will generate revenue, based on page activity and session characteristics.

This is a **binary classification** problem:

- `Revenue = True`: the session ended in a purchase.
- `Revenue = False`: the session did not end in a purchase.

Five supervised classification algorithms are trained on a common dataset and assessed using the same evaluation measures. The project also demonstrates a complete workflow: dataset inspection and preparation, a reproducible train/test split, preprocessing, baseline training, training-only hyperparameter tuning, held-out evaluation, saving model pipelines, and an interactive Streamlit interface.

Because completed purchases are less frequent than non-purchases, **Matthews Correlation Coefficient (MCC)** is used as the primary model-selection metric. Accuracy, ROC AUC, precision, recall, and F1 are also reported to make the trade-offs visible.

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset  
**Source:** [UCI Machine Learning Repository — Dataset 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
**Task:** Binary classification of online browsing sessions

| Attribute | Value |
|---|---|
| Original records | 12,330 sessions |
| Original input features | 17 |
| Target column | `Revenue` |
| Rows after removing duplicate records | 12,205 |
| Duplicate records removed | 125 |
| Training records | 9,764 (80%) |
| Held-out testing records | 2,441 (20%) |
| Missing values in the prepared train/test CSVs | 0 |
| Duplicate rows in the prepared train/test CSVs | 0 |
| Exact overlapping rows between train and test | 0 |

The input features describe three broad aspects of a browsing session:

1. **Browsing engagement:** counts and durations for administrative, informational, and product-related pages.
2. **Page and transaction signals:** bounce rate, exit rate, page value, and proximity to a special day.
3. **Visitor and session context:** month, operating system, browser, region, traffic type, visitor type, and weekend indicator.

The source contains **10 numerical predictors and 7 categorical/binary predictors**; `Revenue` is the additional target column.

### Class distribution after preparation

| Target | Training | Testing | Combined |
|---|---:|---:|---:|
| `False` — No Purchase | 8,238 | 2,059 | 10,297 |
| `True` — Purchase | 1,526 | 382 | 1,908 |
| **Total** | **9,764** | **2,441** | **12,205** |

Only approximately **15.6%** of the prepared sessions ended in a purchase. This class imbalance is why high accuracy alone should not be treated as proof of a good purchase-intention classifier.

### Dataset files

- `data/online_shoppers_intention.csv` — original source CSV.
- `data/online_shoppers_clean.csv` — cleaned dataset.
- `data/train_data.csv` — training observations used for model fitting and cross-validation.
- `test_data.csv` — held-out labeled test dataset used for final evaluation and the Streamlit demonstration.
- `data/SOURCE.md` — dataset provenance information.

**Dataset citation:** Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Data set]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

The source dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/AdityaMane18/shopping-intent-ml

**Live Streamlit Application:** https://shopping-intent-ml.streamlit.app/

## 4. Models Used and Evaluation Metrics

All five classifiers use the same dataset and preprocessing workflow. Baseline and tuned candidates were assessed using **stratified five-fold cross-validation on the training set**, with MCC as the primary selection criterion. After selecting the model version, its saved fitted pipeline was evaluated on the **2,441-row held-out test set**.

The table below presents the **final selected version** for each algorithm, using test-set metrics rather than training metrics.

| ML Model | Selected version | Accuracy | AUC (ROC) | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Tuned | 0.8587 | 0.9099 | 0.5325 | 0.7932 | 0.6372 | 0.5706 |
| Decision Tree Classifier | Tuned | **0.9017** | **0.9271** | 0.7113 | 0.6257 | **0.6657** | **0.6101** |
| K-Nearest Neighbors (KNN) | Baseline | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Gaussian Naive Bayes | Tuned | 0.8161 | 0.8418 | 0.4419 | 0.6675 | 0.5318 | 0.4371 |
| Support Vector Machine (SVM) | Baseline | 0.8939 | 0.8729 | **0.7450** | 0.4895 | 0.5908 | 0.5484 |

**Metric interpretation:**

- **Accuracy:** fraction of all sessions classified correctly.
- **ROC AUC:** ability of predicted scores/probabilities to rank purchases above non-purchases over classification thresholds.
- **Precision:** fraction of predicted purchases that were actual purchases.
- **Recall:** fraction of actual purchases detected by the classifier.
- **F1 score:** harmonic mean of precision and recall.
- **MCC:** correlation-style measure using true positives, true negatives, false positives, and false negatives; useful when the classes are imbalanced.

The positive class for precision, recall, and F1 is **`Revenue = True` (Purchase)**.

## 5. Observations on Model Performance

| ML Model | Observation on the held-out test set |
|---|---|
| **Logistic Regression** | Its tuned version has the **highest purchase recall (0.7932)**, detecting 303 of 382 purchases. This comes with more false purchase alerts (266 false positives) and a lower precision of 0.5325. It is useful when missing a potential purchaser is more costly than contacting a non-purchaser. |
| **Decision Tree** | The tuned tree achieves the **best MCC (0.6101), F1 (0.6657), accuracy (0.9017), and ROC AUC (0.9271)**. A depth-5 tree balances purchase detection and false positives more effectively than the original unpruned tree. |
| **KNN** | The selected baseline KNN performs reasonably on overall accuracy (0.8734) but identifies only **39.01% of actual purchases**. The tuned KNN provides only a small cross-validation MCC gain and reduces recall, so the baseline was retained. |
| **Gaussian Naive Bayes** | Tuning the variance smoothing parameter improves performance substantially over its baseline. Its recall of 0.6675 is useful, but precision remains relatively low (0.4419), producing 322 false positives. |
| **SVM** | The baseline calibrated SVM has the **highest precision (0.7450)**, meaning its positive predictions are comparatively trustworthy. However, recall is only 0.4895, and tuning provided only a small cross-validation MCC improvement. |
| **Overall winner for this dataset** | **Tuned Decision Tree**, selected using training-only cross-validation and confirmed on the held-out test set with **MCC = 0.6101**. It provides the strongest overall balance among the evaluated models. |

The winning model's confusion matrix on the original held-out test set is:

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| No Purchase | 1,962 (TN) | 97 (FP) |
| Purchase | 143 (FN) | 239 (TP) |

These four values sum to **2,441 test observations**.

---

## 6. Methodology

1. **Source inspection and cleaning:** inspect the UCI data, remove duplicate rows, and check column types, missing values, and target distribution.
2. **Data partitioning:** create the 80:20 training and held-out testing CSVs and verify matching feature columns.
3. **Preprocessing:** place numerical transformations and categorical encoding inside scikit-learn `Pipeline` objects so that transformations are learned from training data (and, in cross-validation, from the training portion of each fold).
4. **Baseline training:** fit the five classifier pipelines and save them under `model/baseline/`.
5. **Hyperparameter optimization:** run model-specific searches with **stratified five-fold cross-validation**, optimizing mean validation MCC and recording all six metrics. Save the resulting fitted pipelines separately under `model/tuned/`.
6. **Model-version selection:** compare baseline versus tuned cross-validation performance. Retain tuned Decision Tree, Logistic Regression, and Gaussian Naive Bayes; retain baseline SVM and KNN.
7. **Final evaluation:** evaluate each selected saved pipeline on the held-out test CSV without fitting or tuning again. Generate metrics, confusion matrices, classification reports, predictions, and the final comparison CSV.
8. **Application:** load the saved pipelines into Streamlit for user-uploaded labeled test-data evaluation. Model fitting is **not** performed by the app.

**Evaluation discipline:** Hyperparameter tuning and model-version selection use the training cross-validation results. The baseline training script also produces test metrics for reporting, but those test metrics are not the stated optimization or version-selection criterion.

### Training cross-validation comparison

| Model | Baseline CV MCC | Tuned CV MCC | Selected version |
|---|---:|---:|---|
| Logistic Regression | 0.4709 | 0.5466 | Tuned |
| Decision Tree | 0.4809 | 0.5928 | Tuned |
| KNN | 0.4309 | 0.4348 | Baseline |
| Gaussian Naive Bayes | 0.1362 | 0.4180 | Tuned |
| SVM | 0.5335 | 0.5365 | Baseline |

Some tuned candidates improve CV MCC by only about 0.003–0.004. The chosen comparison policy does not replace the KNN or SVM baseline for such marginal gains.

### Selected hyperparameters

- **Logistic Regression:** `C=0.01`, `class_weight='balanced'`.
- **Decision Tree:** `criterion='gini'`, `max_depth=5`, `min_samples_leaf=30`, `min_samples_split=2`, `ccp_alpha=0`, `class_weight=None`. The selected tree has **30 leaf nodes**.
- **Gaussian Naive Bayes:** `var_smoothing=0.1`, default class priors.
- **KNN:** original baseline configuration retained.
- **SVM:** original RBF-kernel SVC with sigmoid probability calibration retained.

## 7. Project Structure

```text
shopping-intent-ml/
├── app.py                          # Streamlit frontend
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── test_data.csv                   # Labeled held-out test set
├── train_models.py                 # Train/evaluate one baseline classifier
├── compare_models.py              # Baseline comparison
├── compare_tuned_models.py        # CV-based model-version selection
├── evaluate_final_models.py       # Final held-out evaluation
├── .streamlit/
│   └── config.toml                 # Streamlit theme/configuration
├── data/
│   ├── SOURCE.md
│   ├── online_shoppers_intention.csv
│   ├── online_shoppers_clean.csv
│   └── train_data.csv
├── src/
│   ├── data_preparation.py
│   ├── download_dataset.py
│   ├── split_data.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── models/                    # Five classifier definitions
│   └── scripts/
│       ├── tune_logistic_regression.py
│       ├── tune_decision_tree.py
│       ├── tune_knn.py
│       ├── tune_naive_bayes.py
│       └── tune_svm.py
├── model/
│   ├── baseline/                  # Five fitted baseline .joblib files
│   └── tuned/                     # Five fitted tuned .joblib files
└── results/
    ├── baseline/                 # Baseline metrics and comparison
    ├── tuned/                    # CV details, summaries and selection
    └── final/                    # Final metrics, confusion matrices,
                                  # classification reports and predictions
```

Each `.joblib` file contains the fitted preprocessing and classifier pipeline needed for prediction. The tuning scripts are **Python programs that create models**, while `.joblib` files are the **saved fitted models**.

## 8. Local Setup and Reproduction (Windows / PowerShell)

### Prerequisites

- Python with a compatible version of the packages in `requirements.txt`.
- The repository's committed training CSV, labeled test CSV, and source files.
- PowerShell or a VS Code terminal opened in the project root.

Create an environment, install the dependencies, and check consistency:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

### Recreate baseline models and comparison

The `--model` option is required to select any baseline other than the default Logistic Regression:

```powershell
.\.venv\Scripts\python.exe -m train_models --model logistic_regression
.\.venv\Scripts\python.exe -m train_models --model decision_tree
.\.venv\Scripts\python.exe -m train_models --model knn
.\.venv\Scripts\python.exe -m train_models --model naive_bayes
.\.venv\Scripts\python.exe -m train_models --model svm
.\.venv\Scripts\python.exe -m compare_models
```

### Recreate tuned models and CV-based selection

```powershell
.\.venv\Scripts\python.exe -m src.scripts.tune_logistic_regression
.\.venv\Scripts\python.exe -m src.scripts.tune_decision_tree
.\.venv\Scripts\python.exe -m src.scripts.tune_knn
.\.venv\Scripts\python.exe -m src.scripts.tune_naive_bayes
.\.venv\Scripts\python.exe -m src.scripts.tune_svm
.\.venv\Scripts\python.exe -m compare_tuned_models
```

The SVM tuning script can take longer due to repeated fitting and probability calibration.

### Recreate final results

```powershell
.\.venv\Scripts\python.exe -m evaluate_final_models
```

The final evaluation script loads the selected saved models, evaluates them on `test_data.csv`, and writes `results/final/final_model_comparison.csv` together with model-specific predictions, confusion matrices, and reports. It does not retrain the classifiers.

### Launch the dashboard locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost address shown in the terminal (commonly `http://localhost:8501`).

## 9. Streamlit Application Features

The homepage supports the assignment's four mandatory interactive elements:

1. **CSV upload:** supply a labeled test CSV with the 17 expected input columns and `Revenue`.
2. **Model selection:** choose one of the five saved classifier versions from the sidebar.
3. **Evaluation cards:** display accuracy, ROC AUC, precision, recall, F1, and MCC calculated from the selected model's predictions on the uploaded data.
4. **Diagnostics:** show the confusion matrix and classification report; download prediction results where offered.

Additional pages/tabs document the problem and dataset, exploratory analysis, model-development workflow, and the overall report.

**Current distinction:** The selected-model evaluation cards and diagnostics are based on the **currently uploaded CSV**. The **Overall Report tab** displays the saved official 2,441-row benchmark from `results/final/final_model_comparison.csv`; it does **not** recalculate the five-model comparison for each upload. This design keeps the official experimental results separate from ad-hoc examiner uploads.

**Current upload restrictions:** use a CSV named `test_data.csv`, with at most 2,441 rows and a maximum file size of 2 MB, as indicated in the app's upload guidance. ROC AUC requires both target classes to be present in the uploaded labeled data.
