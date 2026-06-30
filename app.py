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

st.set_page_config(page_title="Loan Predictor AI", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

def apply_modern_css():
    st.markdown("""
        <style>
        /* General styling for dark sleek theme */
        
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
        
        /* Style for the logo */
        .logo-img {
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
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
            model = joblib.load(MODEL_DIR / filename)
            if model_name == "Logistic Regression":
                estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
                if not hasattr(estimator, 'multi_class'):
                    estimator.multi_class = 'ovr'
            models[model_name] = model
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

def render_home(models, filters):
    # Header area
    st.markdown("<h2>🏦 AI Loan Approval Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #bbb;'>Enter applicant details below to instantly predict loan approval status with our advanced ML models.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not models:
        st.error("Models not found. Please run the training notebook first.")
        return

    selected_model_name = st.selectbox("🤖 Select Prediction Model", list(models.keys()))
    model = models[selected_model_name]
    
    # Batch Prediction Section
    st.markdown("### 📄 Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload your dataset (.csv) Limit 200MB per file", type="csv")
    
    if uploaded_file is not None:
        try:
            st.success("File uploaded successfully!")
            df = pd.read_csv(uploaded_file)
            
            # Apply filters from sidebar to the preview (optional feature for Data Explorer aspect)
            filtered_df = df.copy()
            if filters and filters.get("credit_history") != "All":
                val = 1 if filters["credit_history"] == "Good (1.0)" else 0
                if "Credit_History" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["Credit_History"] == val]
            if filters and filters.get("property_area") != "All":
                if "Property_Area" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["Property_Area"] == filters["property_area"]]
            
            st.markdown("**Data Preview:**")
            st.dataframe(filtered_df.head(10))
            
            if st.button("🔍 Run Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    pred_df = df.copy()
                    
                    # Preprocess for model (calculate TotalIncome_log if not present)
                    if "TotalIncome_log" not in pred_df.columns and "ApplicantIncome" in pred_df.columns and "CoapplicantIncome" in pred_df.columns:
                        total_income = pred_df["ApplicantIncome"] + pred_df["CoapplicantIncome"]
                        pred_df["TotalIncome_log"] = np.log(total_income.replace(0, np.nan)).fillna(0)
                        
                    # Drop extra columns that model doesn't expect (like Loan_ID, Loan_Status) if they exist
                    # We just pass the exact 12 features the model expects
                    expected_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", 
                                         "Property_Area", "ApplicantIncome", "CoapplicantIncome", 
                                         "LoanAmount", "Loan_Amount_Term", "Credit_History", "TotalIncome_log"]
                    
                    try:
                        X_batch = pred_df[expected_features].copy()
                        predictions = model.predict(X_batch)
                        
                        # Calculate probabilities
                        try:
                            probabilities = model.predict_proba(X_batch)[:, 1]
                        except Exception:
                            probabilities = np.where(predictions == 1, 1.0, 0.0)
                            
                        pred_df["Predicted_Status"] = np.where(predictions == 1, "Approved", "Rejected")
                        pred_df["Confidence"] = np.round(probabilities * 100, 2).astype(str) + "%"
                        
                        st.markdown("### Batch Prediction Results")
                        st.dataframe(pred_df)
                        
                        # Save to session state for dynamic visualization
                        st.session_state["uploaded_data"] = pred_df
                        
                        # Download link
                        csv = pred_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name='loan_batch_predictions.csv',
                            mime='text/csv',
                        )
                    except KeyError as e:
                        st.error(f"Missing required columns in CSV for prediction: {e}")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Manual Prediction Section
    st.markdown("### Applicant Information")
    
    with st.form("loan_form"):
        # 2 columns layout
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
            loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 84, 60, 36, 12, 300, 480], index=0)
            credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        # Center the submit button
        _, sub_col, _ = st.columns([1, 2, 1])
        with sub_col:
            submitted = st.form_submit_button("Run Prediction Engine", type="primary", use_container_width=True)

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
        try:
            probability = model.predict_proba(input_df)[0][1]
        except AttributeError:
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

def render_performance(models, filters):
    st.subheader("📊 Model Performance Metrics")
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

def render_visualizations(models, filters):
    st.subheader("📈 Visualizations & Insights")
    st.markdown("Deep dive into the dataset distributions and model performance metrics.")
    st.markdown("---")
    
    # 1. Dataset Explorer
    st.subheader("Dataset Explorer")
    df = None
    is_default = False
    
    if "uploaded_data" in st.session_state:
        df = st.session_state["uploaded_data"]
        st.info("Currently viewing: **Uploaded Batch Dataset**")
    else:
        data_path = BASE_DIR / "data" / "loan.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            is_default = True
            st.info("Currently viewing: **Default Training Dataset** (Upload a CSV in 'Home' to see custom data)")
            
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            target_col = "Loan_Status" if is_default and "Loan_Status" in df.columns else "Predicted_Status" if "Predicted_Status" in df.columns else None
            if target_col:
                st.markdown(f"**{target_col} Distribution**")
                status_counts = df[target_col].value_counts()
                st.bar_chart(status_counts)
            else:
                st.warning("Target column not found for distribution.")
                
        with col2:
            if "ApplicantIncome" in df.columns:
                st.markdown("**Applicant Income Histogram**")
                try:
                    # Pure streamlit histogram
                    hist, edges = np.histogram(df["ApplicantIncome"].dropna(), bins=20)
                    hist_df = pd.DataFrame({"Count": hist}, index=np.round(edges[:-1], 2))
                    st.bar_chart(hist_df)
                except Exception:
                    st.line_chart(df["ApplicantIncome"].reset_index(drop=True))
    
    st.divider()

    # 2. Dynamic Model Comparison
    st.subheader("Model Comparison (Accuracy vs F1-Score)")
    all_metrics = load_metrics()
    if all_metrics:
        comp_data = []
        for m_name, m_vals in all_metrics.items():
            comp_data.append({
                "Model": m_name,
                "Accuracy": m_vals.get("accuracy", 0),
                "F1 Score": m_vals.get("f1_score", 0)
            })
        comp_df = pd.DataFrame(comp_data).set_index("Model")
        st.bar_chart(comp_df)
    else:
        comparison_img = MODEL_DIR / "model_comparison.png"
        if comparison_img.exists():
            st.image(Image.open(comparison_img), caption="Accuracy vs F1-Score across all models")
            
    st.divider()
    
    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        st.subheader("Confusion Matrices")
        cm_img = MODEL_DIR / "confusion_matrices.png"
        if cm_img.exists():
            st.image(Image.open(cm_img))
            
    st.divider()
        
    st.subheader("Feature Importances")
    if "Random Forest" in models:
        rf = models["Random Forest"]
        try:
            # Extract importances from pipeline or direct model
            if hasattr(rf, 'steps'):
                importances = rf.steps[-1][1].feature_importances_
                # Standard feature names used during training
                feature_names = ["Gender", "Married", "Dependents", "Education", "Self_Employed", 
                                 "Property_Area", "ApplicantIncome", "CoapplicantIncome", 
                                 "LoanAmount", "Loan_Amount_Term", "Credit_History", "TotalIncome_log"]
            else:
                importances = rf.feature_importances_
                feature_names = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else [f"Feature {i}" for i in range(len(importances))]
            
            imp_df = pd.DataFrame({"Importance": importances}, index=feature_names)
            imp_df = imp_df.sort_values(by="Importance", ascending=True).tail(10)
            st.bar_chart(imp_df)
        except Exception:
            fi_img = MODEL_DIR / "feature_importance.png"
            if fi_img.exists():
                st.image(Image.open(fi_img), caption="Top drivers for the Random Forest model")
    else:
        fi_img = MODEL_DIR / "feature_importance.png"
        if fi_img.exists():
            st.image(Image.open(fi_img), caption="Top drivers for the Random Forest model")

def render_about(models, filters):
    st.subheader("ℹ️ About the Project")
    st.markdown("---")
    st.markdown("""
    ### Loan Approval Prediction System
    This application is an end-to-end Machine Learning tool designed to automate and assist with loan approval decisions based on historical data.
    
    **Features:**
    - **Multiple ML Models:** Logistic Regression, Random Forest, Naive Bayes.
    - **Live Inference:** Instant predictions with confidence scores.
    - **Batch Processing:** Upload a CSV for bulk applicant scoring.
    - **Modern UI:** Built with Streamlit, providing a sleek, reactive single-page application experience.
    
    **Dataset:**
    - Real-world Kaggle credit risk dataset.
    - Key signals include Credit History, Applicant Income, and Loan Amount.
    """)

def main():
    apply_modern_css()
    
    try:
        st.sidebar.image(Image.open(BASE_DIR / "logo.png"), width=150)
    except Exception:
        pass
    st.sidebar.markdown("### Loan Approval<br>Prediction AI", unsafe_allow_html=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    nav_options = {
        "Home": render_home,
        "Performance": render_performance,
        "Visualizations": render_visualizations,
        "About": render_about
    }
    
    st.sidebar.markdown("**Go to**")
    selection = st.sidebar.radio(
        "Navigation", 
        list(nav_options.keys()), 
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.caption("© 2026 AI Loan Predictor")
    
    # Execute the selected page function
    models = load_models()
    nav_options[selection](models, {})

if __name__ == "__main__":
    main()
