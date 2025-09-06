# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from utils.preprocessing import single_input_to_df, clean_data
from utils.evaluation import load_metrics, load_test_data, plot_roc_from_metrics, plot_feature_importance

MODEL_DIR = Path("models") / "saved"

st.set_page_config(
    page_title="MetaHealth: Smart Diabetes Predictor",
    page_icon="🩺",  # you can change emoji if you like
    layout="wide"
)

st.title("MetaHealth: Smart Diabetes Predictor")
st.markdown("### AI-powered solution for early diabetes risk detection 🚀")


st.markdown(
    """
    This app uses Logistic Regression and Random Forest models trained on the Pima Indians Diabetes dataset.
    - Enter patient details in the left sidebar (single prediction), or
    - Upload a CSV with the same feature columns to make batch predictions.
    """
)

# Ensure models exist; if not offer to train
def models_exist():
    return (MODEL_DIR / "logistic_model.pkl").exists() and (MODEL_DIR / "random_forest_model.pkl").exists() and (MODEL_DIR / "scaler.pkl").exists()

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Model actions")
    if not models_exist():
        if st.button("Train models (will download dataset & train)"):
            with st.spinner("Training models — this may take ~10-30s"):
                import models.train_model as trainer
                trainer.train_and_save_models()
                st.success("Training complete! Models saved.")
    else:
        st.success("Models found in models/saved/ ✓")
        if st.button("Retrain models (overwrite)"):
            with st.spinner("Retraining models..."):
                import models.train_model as trainer
                trainer.train_and_save_models()
                st.success("Retraining complete!")

# Load models & scaler if present
log_reg = rf = scaler = None
if models_exist():
    with open(MODEL_DIR / "logistic_model.pkl", "rb") as f:
        log_reg = pickle.load(f)
    with open(MODEL_DIR / "random_forest_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Sidebar: single-input form
st.sidebar.title("MetaHealth")
st.sidebar.markdown("Smart Diabetes Predictor")

st.sidebar.header("Single patient input")
with st.sidebar.form("input_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=1, max_value=300, value=120)
    bloodpressure = st.number_input("BloodPressure", min_value=1, max_value=200, value=70)
    skin = st.number_input("SkinThickness", min_value=1, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=1, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, format="%.1f")
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=33)
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    try:
        input_df = single_input_to_df(input_dict)
    except Exception as e:
        st.error(f"Input error: {e}")
        st.stop()

    if not models_exist():
        st.warning("No trained models found. Click 'Train models' on the right to train and save models.")
    else:
        X_scaled = scaler.transform(input_df)
        log_prob = log_reg.predict_proba(X_scaled)[0, 1]
        rf_prob = rf.predict_proba(X_scaled)[0, 1]
        log_pred = int(log_reg.predict(X_scaled)[0])
        rf_pred = int(rf.predict(X_scaled)[0])

        st.subheader("Predictions")
        st.write("Logistic Regression -> Probability of diabetes: **{:.2%}**, Predicted class: **{}**".format(log_prob, log_pred))
        st.write("Random Forest      -> Probability of diabetes: **{:.2%}**, Predicted class: **{}**".format(rf_prob, rf_pred))

        st.subheader("Model agreement")
        if log_pred == rf_pred:
            st.success(f"Both models agree on class {log_pred}")
        else:
            st.warning(f"Models disagree (Logistic: {log_pred}, RF: {rf_pred})")

# Batch upload
st.header("Batch predict (CSV)")
st.markdown("Upload a CSV with the same 8 feature columns (no 'Outcome' column required). Column names must match exactly:")
st.code("Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age")

uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded is not None:
    df_in = pd.read_csv(uploaded)
    # Validate columns
    expected = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age"]
    if not all(col in df_in.columns for col in expected):
        st.error("CSV missing required columns. Make sure the file contains these columns exactly:\n" + ", ".join(expected))
    else:
        df_clean = clean_data(df_in[expected])
        if not models_exist():
            st.error("Models not found. Train models first before batch predictions.")
        else:
            X_scaled = scaler.transform(df_clean)
            log_probs = log_reg.predict_proba(X_scaled)[:, 1]
            rf_probs = rf.predict_proba(X_scaled)[:, 1]
            df_out = df_clean.copy()
            df_out["Logistic_Prob"] = log_probs
            df_out["RandomForest_Prob"] = rf_probs
            df_out["Logistic_Pred"] = (log_probs >= 0.5).astype(int)
            df_out["RandomForest_Pred"] = (rf_probs >= 0.5).astype(int)
            st.dataframe(df_out.head(50))
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

# Model evaluation plots
st.header("Model evaluation")
if models_exist():
    metrics = load_metrics()
    fig_roc = plot_roc_from_metrics(metrics)
    st.pyplot(fig_roc)

    # Load RF model for feature importance
    with open(MODEL_DIR / "random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    feat_fig = plot_feature_importance(rf_model, metrics.get("feature_names"))
    st.pyplot(feat_fig)

    st.markdown(f"**AUC (Logistic):** {metrics['auc_log']:.3f}  \n**AUC (RandomForest):** {metrics['auc_rf']:.3f}")
else:
    st.info("Train the models to view evaluation plots (click the button at the top-right).")

# Footer (non-sticky, only at the bottom after scrolling)
st.markdown(
    """
    <style>
    .footer {
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: gray;
        text-align: center;
        padding: 15px;
        font-size: 14px;
        margin-top: 50px;
    }
    </style>
    <div class="footer">
        <b>MetaHealth: Smart Diabetes Predictor</b><br><br>
        Developed by Team MetaMorphs<br>
        Vishnu V<br>
        Vijay V<br>
        Yukthi Reddy D S<br>
        Tanusrii S
    </div>
    """,
    unsafe_allow_html=True
)
