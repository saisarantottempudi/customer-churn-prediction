"""
Model training module for the Customer Churn Prediction system.

Trains and compares multiple classifiers, handles class imbalance,
selects the best model, and persists it for inference.
"""

import logging
import os
import json
import joblib
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_zoo() -> dict:
    """Return all candidate models with default hyperparameters."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=8, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", max_depth=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            scale_pos_weight=3, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=42, verbose=-1
        ),
    }


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """Oversample minority class (churners) to balance training data."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE: {X_train.shape[0]} -> {X_res.shape[0]} samples | Churn rate: {y_res.mean():.2%}")
    return X_res, y_res


def train_and_evaluate_models(
    X_train, y_train, X_test, y_test, cv_folds: int = 5
) -> pd.DataFrame:
    """
    Train every model, run cross-validation, and return a comparison DataFrame.
    Sorted by recall (churn recall is the primary business metric).
    """
    from sklearn.metrics import recall_score, roc_auc_score, f1_score, precision_score

    models = get_model_zoo()
    results = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=StratifiedKFold(cv_folds), scoring="recall"
        )

        results.append({
            "Model": name,
            "Recall": recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            "CV_Recall_Mean": cv_scores.mean(),
            "CV_Recall_Std": cv_scores.std(),
        })
        logger.info(f"  Recall={results[-1]['Recall']:.4f} | AUC={results[-1]['ROC_AUC']:.4f}")

    df_results = pd.DataFrame(results).sort_values("Recall", ascending=False)
    return df_results, models


def select_best_model(results_df: pd.DataFrame, models: dict, strategy: str = "recall"):
    """Select best model based on primary metric."""
    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    logger.info(f"Best model: {best_name} (Recall={results_df.iloc[0]['Recall']:.4f})")
    return best_name, best_model


def save_model(model, path: str = "models/churn_model.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")


def load_model(path: str = "models/churn_model.pkl"):
    model = joblib.load(path)
    logger.info(f"Loaded model from {path}")
    return model


def save_training_metadata(metadata: dict, path: str = "models/training_metadata.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to {path}")


def run_training_pipeline(config_path: str = "config.yaml"):
    """Full training pipeline: load data -> engineer features -> train -> save."""
    import sys
    sys.path.insert(0, ".")
    from src.feature_engineering import prepare_features, save_pipeline

    config = load_config(config_path)
    random_state = config["data"]["random_state"]
    test_size = config["data"]["test_size"]
    cv_folds = config["model"]["cv_folds"]

    # Load processed data
    df = pd.read_csv(config["data"]["processed_path"])
    logger.info(f"Loaded processed data: {df.shape}")

    # Feature engineering
    X, y, preprocessor, feature_names = prepare_features(df, config, fit=True)
    save_pipeline(preprocessor, config["model"]["pipeline_path"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Apply SMOTE to training set
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)

    # Train and compare all models
    results_df, models = train_and_evaluate_models(
        X_train_res, y_train_res, X_test, y_test, cv_folds=cv_folds
    )
    print("\n===== Model Comparison =====")
    print(results_df.to_string(index=False))

    # Refit best model on the full resampled training set for final use
    best_name, best_model = select_best_model(results_df, models)
    best_model.fit(X_train_res, y_train_res)
    save_model(best_model, config["model"]["path"])

    # Save metadata
    metadata = {
        "model_type": best_name,
        "model_version": config["api"]["model_version"],
        "training_date": datetime.now().isoformat(),
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "training_samples": int(X_train_res.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "churn_rate": float(y.mean()),
        "metrics": results_df[results_df["Model"] == best_name].to_dict("records")[0],
    }
    save_training_metadata(metadata)

    return best_model, results_df, X_test, y_test, feature_names


if __name__ == "__main__":
    model, results, X_test, y_test, feat_names = run_training_pipeline()
    print("\nTraining complete. Model saved to models/churn_model.pkl")
