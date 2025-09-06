# models/train_model.py
import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

from utils.preprocessing import clean_data, get_X_y

MODEL_DIR = Path(__file__).resolve().parents[0] / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMN_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def train_and_save_models(data_path: str = None, random_state: int = 42):
    """
    Train logistic regression and random forest on the Pima dataset and save:
    - logistic_model.pkl
    - random_forest_model.pkl
    - scaler.pkl
    - train_test_data.pkl (X_test, y_test) for plotting ROC in the app
    - metrics.pkl (roc curves + AUCs)
    """
    # Load dataset
    if data_path:
        df = pd.read_csv(data_path, names=COLUMN_NAMES)
    else:
        df = pd.read_csv(DATA_URL, names=COLUMN_NAMES)

    # Clean
    df = clean_data(df)

    # Split X,y
    X, y = get_X_y(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    log_reg = LogisticRegression(max_iter=500, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    # Fit
    log_reg.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    # Predictions & metrics
    log_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
    rf_probs = rf.predict_proba(X_test_scaled)[:, 1]

    fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

    auc_log = roc_auc_score(y_test, log_probs)
    auc_rf = roc_auc_score(y_test, rf_probs)

    # Save models and scaler
    with open(MODEL_DIR / "logistic_model.pkl", "wb") as f:
        pickle.dump(log_reg, f)
    with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    # Save test split and metrics for app plotting
    with open(MODEL_DIR / "test_data.pkl", "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test, "X_test_scaled": X_test_scaled}, f)
    with open(MODEL_DIR / "metrics.pkl", "wb") as f:
        pickle.dump({
            "fpr_log": fpr_log, "tpr_log": tpr_log, "auc_log": auc_log,
            "fpr_rf": fpr_rf, "tpr_rf": tpr_rf, "auc_rf": auc_rf,
            "feature_names": list(X.columns)
        }, f)

    # Save a short README about models
    with open(MODEL_DIR / "README.txt", "w") as f:
        f.write("Models saved: logistic_model.pkl, random_forest_model.pkl, scaler.pkl\n")
        f.write(f"AUC - Logistic: {auc_log:.4f}, RandomForest: {auc_rf:.4f}\n")

    # Optionally return details
    report_log = classification_report(y_test, log_reg.predict(X_test_scaled), output_dict=True)
    report_rf = classification_report(y_test, rf.predict(X_test_scaled), output_dict=True)

    return {
        "logistic": {"model": log_reg, "auc": auc_log, "report": report_log},
        "random_forest": {"model": rf, "auc": auc_rf, "report": report_rf},
    }


if __name__ == "__main__":
    print("Training models and saving to models/saved/ ...")
    results = train_and_save_models()
    print("Done.")
    print("Logistic AUC:", results["logistic"]["auc"])
    print("RF AUC:", results["random_forest"]["auc"])
