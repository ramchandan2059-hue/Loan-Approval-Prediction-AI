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

st.set_page_config(page_title="Loan Predictor AI", page_icon="🏦", layout="wide")

def apply_modern_css():
    # Removed the background color overrides so Streamlit's native Dark/Light mode handles text correctly.
    st.markdown("""
        <style>
        /* Button styling */
        button[kind="primary"] {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(124, 58, 237, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

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

def render_home(models):
    st.title("🏦 AI Loan Approval Prediction")
    st.markdown("Enter applicant details below to instantly predict loan approval status with our advanced ML models.")
    st.markdown("---")
    
    if not models:
        st.error("Models not found. Please run the training notebook first.")
        return

    selected_model_name = st.selectbox("🤖 Select Prediction Model", list(models.keys()))
    model = models[selected_model_name]

    with st.form("loan_form"):
        st.subheader("Applicant Information")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        with col2:
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=100)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120, step=5)
            loan_term = st.selectbox(
                "Loan Term (days)", [360, 180, 120, 84, 60, 36, 12, 300, 480], index=0
            )
            credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Prediction Engine", use_container_width=True)

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

        # Safely predict and handle model versioning errors with predict_proba
        prediction = model.predict(input_df)[0]
        try:
            probability = model.predict_proba(input_df)[0][1]
        except AttributeError:
            # Fallback if the saved model throws an AttributeError due to scikit-learn version differences
            probability = 1.0 if prediction == 1 else 0.0

        st.markdown("---")
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if prediction == 1:
                st.success("✅ APPROVED")
            else:
                st.error("❌ REJECTED")
        with res_col2:
            conf = probability if prediction == 1 else (1 - probability)
            st.info(f"**Confidence Score:** {conf:.1%} likelihood.")

def render_performance(models):
    st.title("📊 Model Performance Metrics")
    st.markdown("Compare how accurate each machine learning model is based on our test data.")
    st.markdown("---")
    
    all_metrics = load_metrics()
    selected_model_name = st.selectbox("Select Model to View Metrics", list(models.keys()))
    
    metrics = all_metrics.get(selected_model_name, {})
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        col4.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")
        
        st.markdown(f"**Trained on:** {metrics.get('n_train', 0)} rows | **Tested on:** {metrics.get('n_test', 0)} rows.")
        
        st.markdown("### Metrics Bar Chart")
        # Creating a dynamic bar chart for the selected model
        chart_data = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Score": [
                metrics.get('accuracy', 0), 
                metrics.get('precision', 0), 
                metrics.get('recall', 0), 
                metrics.get('f1_score', 0)
            ]
        })
        chart_data.set_index("Metric", inplace=True)
        st.bar_chart(chart_data)
        
    else:
        st.warning("Metrics not available.")

def render_visualizations():
    st.title("📈 Visualizations & Insights")
    st.markdown("Deep dive into how the models work and which features are most important.")
    st.markdown("---")
    
    st.subheader("Model Comparison")
    comparison_img = MODEL_DIR / "model_comparison.png"
    if comparison_img.exists():
        st.image(Image.open(comparison_img), use_column_width=True, caption="Accuracy vs F1-Score across all models")
        
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrices")
        cm_img = MODEL_DIR / "confusion_matrices.png"
        if cm_img.exists():
            st.image(Image.open(cm_img), use_column_width=True)
            
    with col2:
        st.subheader("Feature Importances")
        fi_img = MODEL_DIR / "feature_importance.png"
        if fi_img.exists():
            st.image(Image.open(fi_img), use_column_width=True, caption="Top drivers for the Random Forest model")

def render_about():
    st.title("ℹ️ About the Project")
    st.markdown("---")
    st.markdown("""
    ### Loan Approval Prediction System
    This application is an end-to-end Machine Learning tool designed to automate and assist with loan approval decisions based on historical data.
    
    **Features:**
    - **Multiple ML Models:** Logistic Regression, Random Forest, Naive Bayes.
    - **Live Inference:** Instant predictions with confidence scores.
    - **Modern UI:** Built with Streamlit, providing a sleek, reactive single-page application experience.
    
    **Dataset:**
    - Real-world Kaggle credit risk dataset.
    - Key signals include Credit History, Applicant Income, and Loan Amount.
    """)

def main():
    apply_modern_css()
    
    # Sidebar Navigation
    try:
        st.sidebar.image(Image.open(BASE_DIR / "logo.png"), width=200)
    except Exception:
        pass
    st.sidebar.title("Loan Approval Prediction AI")
    st.sidebar.markdown("---")
    
    # Removed emojis from the sidebar keys
    nav_options = {
        "Home": render_home,
        "Performance": render_performance,
        "Visualizations": render_visualizations,
        "About": render_about
    }
    
    selection = st.sidebar.radio("Go to", list(nav_options.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 AI Loan Predictor")
    
    # Load models once
    models = load_models()
    
    # Execute the selected page function
    if selection in ["Home", "Performance"]:
        nav_options[selection](models)
    else:
        nav_options[selection]()

if __name__ == "__main__":
    main()
