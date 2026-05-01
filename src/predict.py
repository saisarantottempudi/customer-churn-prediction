"""
Prediction module for the Customer Churn Prediction system.

Loads trained model + pipeline, accepts raw customer data,
returns churn probability, risk segment, and top contributing features.
"""

import logging
import json
import joblib
import yaml
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_artifacts(config: dict):
    """Load model, pipeline, and metadata."""
    model = joblib.load(config["model"]["path"])
    pipeline = joblib.load(config["model"]["pipeline_path"])
    with open("models/training_metadata.json") as f:
        metadata = json.load(f)
    return model, pipeline, metadata


def assign_risk_level(probability: float, config: dict) -> str:
    seg = config["risk_segments"]
    if probability < seg["low"]["max_probability"]:
        return "Low Risk"
    elif probability < seg["medium"]["max_probability"]:
        return "Medium Risk"
    else:
        return "High Risk"


def get_top_churn_reasons(
    customer_df: pd.DataFrame,
    model,
    pipeline,
    feature_names: list,
    top_n: int = 3,
) -> list:
    """
    Use model coefficients / feature importances + customer values to explain
    the top reasons this individual customer is at churn risk.
    """
    import sys
    sys.path.insert(0, ".")
    from src.feature_engineering import add_engineered_features
    df_eng = add_engineered_features(customer_df)
    X_transformed = pipeline.transform(df_eng)

    if hasattr(model, "coef_"):
        # Logistic Regression: coef * feature_value gives contribution
        coefs = model.coef_[0]
        contributions = coefs * X_transformed[0]
    elif hasattr(model, "feature_importances_"):
        contributions = model.feature_importances_ * X_transformed[0]
    else:
        return ["Insufficient data for explanation"]

    # Only positive contributions push toward churn
    positive_idx = np.argsort(contributions)[::-1][:top_n]
    reasons = []
    for idx in positive_idx:
        if idx < len(feature_names) and contributions[idx] > 0:
            reasons.append(feature_names[idx])

    # Fall back to human-readable business logic if ML reasons are sparse
    if not reasons:
        reasons = _heuristic_reasons(customer_df)

    return reasons[:top_n]


def _heuristic_reasons(df: pd.DataFrame) -> list:
    reasons = []
    row = df.iloc[0]
    if "Contract" in df.columns and row["Contract"] == "Month-to-month":
        reasons.append("Month-to-month contract (no commitment)")
    if "tenure" in df.columns and row["tenure"] < 12:
        reasons.append("Low tenure (new customer, high early churn risk)")
    if "MonthlyCharges" in df.columns and row["MonthlyCharges"] > 70:
        reasons.append("High monthly charges")
    if "PaymentMethod" in df.columns and row["PaymentMethod"] == "Electronic check":
        reasons.append("Electronic check payment (associated with higher churn)")
    if "InternetService" in df.columns and row["InternetService"] == "Fiber optic":
        reasons.append("Fiber optic internet (competitive alternatives available)")
    if "OnlineSecurity" in df.columns and row["OnlineSecurity"] == "No":
        reasons.append("No online security add-on")
    return reasons


def preprocess_single_customer(customer_dict: dict, config: dict) -> pd.DataFrame:
    """Convert a single customer dict to a DataFrame ready for pipeline transform."""
    df = pd.DataFrame([customer_dict])
    # Standardize
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    # Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype in [int, "int64"]:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


def predict_single(customer_dict: dict, config_path: str = "config.yaml") -> dict:
    """
    Full prediction pipeline for a single customer.

    Args:
        customer_dict: dict with raw customer fields (same format as Telco dataset)
        config_path: path to config.yaml

    Returns:
        dict with customer_id, churn_prediction, churn_probability, risk_level,
              top_reasons, recommended_action
    """
    import sys
    sys.path.insert(0, ".")
    from src.feature_engineering import add_engineered_features

    config = load_config(config_path)
    model, pipeline, metadata = load_artifacts(config)
    feature_names = metadata["feature_names"]

    customer_id = customer_dict.get(config["data"]["customer_id_column"], "UNKNOWN")

    # Preprocess
    df = preprocess_single_customer(customer_dict, config)
    df_no_id = df.drop(columns=[config["data"]["customer_id_column"]], errors="ignore")
    df_no_id = df_no_id.drop(columns=[config["data"]["target_column"]], errors="ignore")

    # Engineer features
    df_engineered = add_engineered_features(df_no_id)

    # Transform
    X = pipeline.transform(df_engineered)

    # Predict
    prob = float(model.predict_proba(X)[0][1])
    prediction = "Yes" if prob >= config["model"]["threshold"] else "No"
    risk_level = assign_risk_level(prob, config)

    # Explain
    top_reasons = get_top_churn_reasons(df_no_id, model, pipeline, feature_names)
    if not top_reasons:
        top_reasons = _heuristic_reasons(df_no_id)

    return {
        "customer_id": customer_id,
        "churn_prediction": prediction,
        "churn_probability": round(prob, 4),
        "risk_level": risk_level,
        "top_reasons": top_reasons,
    }


def predict_batch(df_raw: pd.DataFrame, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Args:
        df_raw: DataFrame with raw customer data
        config_path: path to config.yaml

    Returns:
        DataFrame with predictions appended
    """
    import sys
    sys.path.insert(0, ".")
    from src.feature_engineering import add_engineered_features

    config = load_config(config_path)
    model, pipeline, metadata = load_artifacts(config)

    df = df_raw.copy()
    id_col = config["data"]["customer_id_column"]
    target_col = config["data"]["target_column"]

    ids = df[id_col].values if id_col in df.columns else [f"CUST_{i}" for i in range(len(df))]

    # Fix types
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])

    df = df.drop(columns=[id_col, target_col], errors="ignore")
    df = add_engineered_features(df)

    X = pipeline.transform(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= config["model"]["threshold"]).astype(int)

    results = pd.DataFrame({
        "customer_id": ids,
        "churn_prediction": ["Yes" if p == 1 else "No" for p in preds],
        "churn_probability": probs.round(4),
        "risk_level": [assign_risk_level(p, config) for p in probs],
    })
    return results


if __name__ == "__main__":
    sample = {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.10,
        "TotalCharges": "267.30",
    }
    result = predict_single(sample)
    import json
    print(json.dumps(result, indent=2))
