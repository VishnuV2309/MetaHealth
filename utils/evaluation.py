# utils/evaluation.py
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "saved"

def load_metrics():
    with open(MODEL_DIR / "metrics.pkl", "rb") as f:
        return pickle.load(f)

def load_test_data():
    with open(MODEL_DIR / "test_data.pkl", "rb") as f:
        return pickle.load(f)

def plot_roc_from_metrics(metrics):
    """
    Returns a matplotlib figure with ROC curves for both models.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(metrics["fpr_log"], metrics["tpr_log"], label=f"Logistic (AUC={metrics['auc_log']:.2f})")
    ax.plot(metrics["fpr_rf"], metrics["tpr_rf"], label=f"RandomForest (AUC={metrics['auc_rf']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    return fig

def plot_feature_importance(rf_model, feature_names):
    """
    Return matplotlib figure showing horizontal bar chart of feature importances.
    """
    importances = rf_model.feature_importances_
    df = pd.Series(importances, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    df.plot(kind="barh", ax=ax)
    ax.set_title("Random Forest - Feature Importance")
    ax.set_xlabel("Importance")
    return fig

def pretty_classification_report(y_true, y_pred):
    """
    Returns sklearn classification_report text and confusion matrix
    """
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
