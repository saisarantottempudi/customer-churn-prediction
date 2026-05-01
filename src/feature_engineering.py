"""
Feature engineering module for the Customer Churn Prediction system.

Creates domain-meaningful features and builds a reusable scikit-learn
preprocessing pipeline (ColumnTransformer + Pipeline).
"""

import logging
import os
import joblib
import yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Domain Feature Creation
# ---------------------------------------------------------------------------

def create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into business-meaningful groups."""
    df = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    return df


def create_monthly_charge_band(df: pd.DataFrame) -> pd.DataFrame:
    """Segment customers by monthly spend tier."""
    df = df.copy()
    df["monthly_charge_band"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, float("inf")],
        labels=["Low", "Medium", "High", "Premium"],
        include_lowest=True,
    )
    return df


def create_total_spend_category(df: pd.DataFrame) -> pd.DataFrame:
    """Classify total lifetime spend."""
    df = df.copy()
    df["total_spend_category"] = pd.cut(
        df["TotalCharges"],
        bins=[0, 500, 2000, 5000, float("inf")],
        labels=["Low", "Medium", "High", "VIP"],
        include_lowest=True,
    )
    return df


def create_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Month-to-month contracts carry the highest churn risk.
    Assign numeric risk: Month-to-month=2, One year=1, Two year=0.
    """
    df = df.copy()
    risk_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk_score"] = df["Contract"].map(risk_map).fillna(1)
    return df


def create_payment_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Electronic check is associated with higher churn probability.
    Assign numeric risk accordingly.
    """
    df = df.copy()
    risk_map = {
        "Electronic check": 2,
        "Mailed check": 1,
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 0,
    }
    df["payment_risk_score"] = df["PaymentMethod"].map(risk_map).fillna(1)
    return df


def create_customer_value_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine tenure and monthly charges into a composite value score.
    High-value = long tenure + high spend.
    """
    df = df.copy()
    tenure_norm = df["tenure"] / df["tenure"].max()
    charge_norm = df["MonthlyCharges"] / df["MonthlyCharges"].max()
    score = (tenure_norm * 0.6 + charge_norm * 0.4)
    df["customer_value_segment"] = pd.cut(
        score,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low-Value", "Mid-Value", "High-Value"],
        include_lowest=True,
    )
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    df = create_tenure_group(df)
    df = create_monthly_charge_band(df)
    df = create_total_spend_category(df)
    df = create_contract_risk_score(df)
    df = create_payment_risk_score(df)
    df = create_customer_value_segment(df)
    logger.info("Added 6 engineered features")
    return df


# ---------------------------------------------------------------------------
# Scikit-learn Preprocessing Pipeline
# ---------------------------------------------------------------------------

def get_feature_lists(config: dict) -> tuple:
    numerical = config["features"]["numerical"]
    categorical = config["features"]["categorical"]
    engineered_num = ["contract_risk_score", "payment_risk_score"]
    engineered_cat = ["tenure_group", "monthly_charge_band", "total_spend_category", "customer_value_segment"]
    all_numerical = numerical + engineered_num
    all_categorical = categorical + engineered_cat
    return all_numerical, all_categorical


def build_preprocessing_pipeline(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer with:
    - Numerical: impute median -> StandardScaler
    - Categorical: impute most_frequent -> OneHotEncoder
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_features(df: pd.DataFrame, config: dict, fit: bool = True, pipeline=None):
    """
    Add engineered features, separate X/y, apply preprocessing pipeline.

    Args:
        df: Cleaned DataFrame (must contain 'Churn' column for training).
        config: Project config dict.
        fit: If True, fit the pipeline; if False, transform only (inference).
        pipeline: Pre-fitted pipeline (required when fit=False).

    Returns:
        X_transformed (np.ndarray), y (pd.Series or None), fitted_pipeline, feature_names
    """
    target_col = config["data"]["target_column"]
    id_col = config["data"]["customer_id_column"]

    df = df.copy()

    # Drop ID column if present
    df = df.drop(columns=[id_col], errors="ignore")

    # Add engineered features
    df = add_engineered_features(df)

    # Separate target
    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")

    numerical_features, categorical_features = get_feature_lists(config)
    # Keep only columns that exist in X
    numerical_features = [c for c in numerical_features if c in X.columns]
    categorical_features = [c for c in categorical_features if c in X.columns]

    if fit:
        pipeline = build_preprocessing_pipeline(numerical_features, categorical_features)
        X_transformed = pipeline.fit_transform(X)
    else:
        X_transformed = pipeline.transform(X)

    # Retrieve feature names for explainability
    try:
        cat_encoder = pipeline.named_transformers_["cat"]["encoder"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    except Exception:
        cat_feature_names = []
    feature_names = numerical_features + cat_feature_names

    logger.info(f"Transformed to {X_transformed.shape[1]} features")
    return X_transformed, y, pipeline, feature_names


def save_pipeline(pipeline, path: str = "models/preprocessing_pipeline.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"Saved preprocessing pipeline to {path}")


def load_pipeline(path: str = "models/preprocessing_pipeline.pkl"):
    pipeline = joblib.load(path)
    logger.info(f"Loaded preprocessing pipeline from {path}")
    return pipeline


if __name__ == "__main__":
    import yaml
    from src.data_preprocessing import load_raw_data, run_preprocessing_pipeline

    config = load_config()
    df = pd.read_csv(config["data"]["processed_path"])
    X, y, pipe, feat_names = prepare_features(df, config, fit=True)
    save_pipeline(pipe)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"First 5 feature names: {feat_names[:5]}")
