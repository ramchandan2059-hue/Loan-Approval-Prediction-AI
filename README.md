# Loan Approval Prediction AI

A machine learning system that predicts whether a loan application is likely
to be **approved** or **rejected**, based on applicant demographics, income,
and credit history. Built from the original EDA/modeling notebook and
reorganized into a clean, reusable project structure.

## Project structure

```
Loan_Approval_Prediction_AI/
├── data/
│   └── loan.csv                     # Raw applicant/loan dataset (614 records)
├── docs/
│   └── project_report_outline.md    # Outline for a written project report
├── models/
│   ├── loan_approval_model.joblib   # Trained sklearn pipeline (preprocessing + classifier)
│   ├── feature_importance.csv       # Ranked feature importances from the model
│   └── metrics.json                 # Accuracy / precision / recall / F1 on the test set
├── app.py                           # Streamlit app for interactive predictions
├── train_model.py                   # Trains the model and writes everything in models/
├── requirements.txt                 # Python dependencies
└── README.md
```

## How it works

1. **`train_model.py`** loads `data/loan.csv`, cleans missing values, engineers
   a `TotalIncome_log` feature, fits a `RandomForestClassifier` inside an
   sklearn `Pipeline` (one-hot encoding + scaling baked in), evaluates it on a
   held-out test split, and saves the model + metrics + feature importances
   into `models/`.
2. **`app.py`** loads the saved pipeline and exposes a simple form where a
   user enters applicant details and gets an instant prediction with a
   confidence score, plus a panel showing model performance and the most
   influential features.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train (or retrain) the model:**

```bash
python train_model.py
```

This regenerates `models/loan_approval_model.joblib`, `feature_importance.csv`,
and `metrics.json`.

**2. Launch the prediction app:**

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your browser.

## Model performance (current run)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 87.8% |
| Precision | 86.5% |
| Recall    | 97.7% |
| F1 Score  | 91.7% |

The most predictive feature is **Credit_History**, followed by income and
loan amount — consistent with the EDA in the original notebook.

## Dataset

`data/loan.csv` contains 614 historical loan applications with the following
columns:

| Column | Description |
|---|---|
| Loan_ID | Unique loan application ID |
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Graduate / Not Graduate |
| Self_Employed | Self-employment status |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Requested loan amount (in thousands) |
| Loan_Amount_Term | Loan repayment term (in days) |
| Credit_History | 1 = good credit history, 0 = bad |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: Y = approved, N = rejected |

## Notes

- This project simplifies the original notebook by collapsing the EDA and
  multiple model comparisons (SVM, Logistic Regression, Random Forest, Naive
  Bayes, Decision Tree) down to a single, well-performing Random Forest model
  wrapped in a proper preprocessing pipeline, so it can be trained and served
  reliably outside of a notebook.
- See `docs/project_report_outline.md` for a suggested outline if you want to
  write up the EDA and modeling findings as a formal report.
