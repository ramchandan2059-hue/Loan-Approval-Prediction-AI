# Project Report Outline — Loan Approval Prediction AI

## 1. Introduction
- Problem statement: predicting loan approval outcomes for applicants
- Business motivation (faster, more consistent loan decisions)
- Dataset overview (614 applications, 12 features + target)

## 2. Data Understanding & Cleaning
- Column descriptions and data types
- Missing value summary (Gender, Married, Dependents, Self_Employed,
  LoanAmount, Loan_Amount_Term, Credit_History)
- Imputation strategy (mode for categorical, mean/median for numeric)

## 3. Exploratory Data Analysis
- Distribution of Loan_Status (class balance: ~69% approved / 31% rejected)
- Gender, Marital status, Dependents, Education, Self-Employment breakdowns
- Income and loan amount distributions (right-skewed → log transform)
- Credit History vs. Loan_Status relationship

## 4. Feature Engineering
- `TotalIncome` = ApplicantIncome + CoapplicantIncome
- `TotalIncome_log` to reduce skew
- One-hot encoding of categorical variables
- Scaling of numeric variables

## 5. Modeling
- Train/test split (80/20, stratified)
- Model selection: Random Forest Classifier (chosen for robustness and
  interpretability via feature importances)
- Pipeline design: preprocessing + model combined into a single
  `sklearn.Pipeline` for reproducibility

## 6. Evaluation
- Accuracy, Precision, Recall, F1 Score (see `models/metrics.json`)
- Feature importance ranking (see `models/feature_importance.csv`)
- Discussion: Credit_History dominates predictive power

## 7. Deployment
- `train_model.py` — reproducible training script
- `app.py` — Streamlit interface for real-time predictions
- Saved artifacts in `models/` for serving without retraining

## 8. Limitations & Future Work
- Class imbalance (handled via `class_weight="balanced"`, could try SMOTE)
- Limited feature set (no employment history, loan purpose, etc.)
- Could compare against other models (Logistic Regression, Gradient
  Boosting, XGBoost) and add cross-validation / hyperparameter tuning

## 9. Conclusion
- Summary of model performance and key drivers of loan approval
