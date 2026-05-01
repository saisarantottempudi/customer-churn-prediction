"""
Model evaluation module for the Customer Churn Prediction system.

Generates comprehensive evaluation metrics, plots, and a model report.
"""

import logging
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
    }


def plot_confusion_matrix(y_true, y_pred, save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"], ax=ax
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = save_path or f"{FIGURES_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {path}")


def plot_roc_curve(y_true, y_prob, save_path: str = None) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", color="#2563EB", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = save_path or f"{FIGURES_DIR}/roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved ROC curve to {path}")


def plot_precision_recall_curve(y_true, y_prob, save_path: str = None) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#16A34A", lw=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=y_true.mean(), color="gray", linestyle="--", label="Baseline (prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = save_path or f"{FIGURES_DIR}/precision_recall_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved Precision-Recall curve to {path}")


def plot_feature_importance(model, feature_names: list, top_n: int = 20, save_path: str = None) -> None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model does not have feature importances")
        return

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("Blues_r", len(fi_df))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = save_path or f"{FIGURES_DIR}/feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Saved feature importance chart to {path}")


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None) -> None:
    metrics = ["Recall", "Precision", "F1", "ROC_AUC"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        df_sorted = results_df.sort_values(metric, ascending=False)
        colors = ["#2563EB" if i == 0 else "#93C5FD" for i in range(len(df_sorted))]
        ax.barh(df_sorted["Model"], df_sorted[metric], color=colors)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_path or f"{FIGURES_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved model comparison to {path}")


def generate_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def run_full_evaluation(model, X_test, y_test, feature_names: list, results_df: pd.DataFrame = None) -> dict:
    """Run all evaluations and save all plots."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    logger.info(f"Evaluation metrics: {metrics}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_precision_recall_curve(y_test, y_prob)
    plot_feature_importance(model, feature_names)

    if results_df is not None:
        plot_model_comparison(results_df)

    clf_report = generate_classification_report(y_test, y_pred)
    print("\n===== Classification Report =====")
    print(clf_report)
    print(f"Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, ".")
    from src.train_model import run_training_pipeline

    model, results_df, X_test, y_test, feature_names = run_training_pipeline()
    metrics = run_full_evaluation(model, X_test, y_test, feature_names, results_df)
    print(f"\nAll evaluation plots saved to {FIGURES_DIR}/")
