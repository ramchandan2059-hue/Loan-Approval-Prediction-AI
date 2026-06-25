"""
app.py
------
Streamlit app for the Loan Approval Prediction system.
Loads the trained models from models/ and lets a
user enter applicant details to get a live prediction + confidence score.

Run:
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")

@st.cache_resource
def load_models():
    models = {}
    for model_name, filename in [
        ("Logistic Regression", "logistic_regression.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("Naive Bayes", "naive_bayes.joblib")
    ]:
        try:
            models[model_name] = joblib.load(MODEL_DIR / filename)
        except Exception:
            pass
    return models


@st.cache_data
def load_metrics():
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def main():
    st.title("💰 Loan Approval Prediction")
    st.write(
        "Enter the applicant's details below to predict whether the loan "
        "is likely to be **approved** or **rejected**."
    )

    models = load_models()
    
    if not models:
        st.error("Models not found. Please run the training notebook first.")
        return

    selected_model_name = st.selectbox("Select Model for Prediction", list(models.keys()))
    model = models[selected_model_name]

    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=5)
            loan_term = st.selectbox(
                "Loan Term (days)", [360, 180, 120, 84, 60, 36, 12, 300, 480], index=0
            )
            credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        total_income = applicant_income + coapplicant_income
        total_income_log = np.log(total_income) if total_income > 0 else 0.0

        input_df = pd.DataFrame([{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": 1.0 if credit_history.startswith("Good") else 0.0,
            "TotalIncome_log": total_income_log,
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.success(f"✅ Loan likely **APPROVED** (confidence: {probability:.1%})")
        else:
            st.error(f"❌ Loan likely **REJECTED** (confidence: {1 - probability:.1%})")

    st.divider()

    with st.expander(f"📊 {selected_model_name} Performance"):
        all_metrics = load_metrics()
        metrics = all_metrics.get(selected_model_name, {})
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            m2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
            m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
            m4.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")
            st.caption(f"Trained on {metrics.get('n_train', 0)} rows, tested on {metrics.get('n_test', 0)} rows.")
        else:
            st.warning("Metrics not available.")

    with st.expander("🔍 Models Comparison & Visualizations"):
        st.subheader("Model Comparison (Accuracy vs F1)")
        comparison_img = MODEL_DIR / "model_comparison.png"
        if comparison_img.exists():
            st.image(Image.open(comparison_img), width="stretch")
            
        st.subheader("Confusion Matrices")
        cm_img = MODEL_DIR / "confusion_matrices.png"
        if cm_img.exists():
            st.image(Image.open(cm_img), width="stretch")
            
        st.subheader("Top Feature Importances (Random Forest)")
        fi_img = MODEL_DIR / "feature_importance.png"
        if fi_img.exists():
            st.image(Image.open(fi_img), width="stretch")

if __name__ == "__main__":
    main()
