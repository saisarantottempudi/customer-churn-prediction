"""
FastAPI application for the Customer Churn Prediction system.

Endpoints:
  POST /predict          - Predict churn for a single customer
  POST /predict/batch    - Predict churn for multiple customers (JSON list)
  GET  /health           - API health check
  GET  /model-info       - Model metadata and performance metrics
  GET  /predictions      - Retrieve stored prediction history
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI, HTTPException, Depends, status  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from api.schemas import (  # noqa: E402
    CustomerInput, PredictionResponse, HealthResponse,
    ModelInfoResponse, BatchPredictionResponse, BatchPredictionRow,
)
from api.database import init_db, get_db, log_prediction  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Global model state
_model_state: dict = {}


def _load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_models():
    import joblib
    config = _load_config()
    model = joblib.load(config["model"]["path"])
    pipeline = joblib.load(config["model"]["pipeline_path"])
    with open("models/training_metadata.json") as f:
        metadata = json.load(f)
    return model, pipeline, metadata, config


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    try:
        model, pipeline, metadata, config = _load_models()
        _model_state["model"] = model
        _model_state["pipeline"] = pipeline
        _model_state["metadata"] = metadata
        _model_state["config"] = config
        logger.info(f"Model loaded: {metadata['model_type']} v{metadata['model_version']}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model_state["model"] = None
    yield
    # Shutdown
    _model_state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability and recommends personalized retention strategies.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_model():
    if _model_state.get("model") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Ensure training has been completed.",
        )
    return _model_state["model"], _model_state["pipeline"], _model_state["metadata"], _model_state["config"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API health status and model availability."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model_state.get("model") is not None,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
def model_info():
    """Returns model metadata, version, training date, and performance metrics."""
    _, _, metadata, _ = _get_model()
    return ModelInfoResponse(
        model_type=metadata["model_type"],
        model_version=metadata["model_version"],
        training_date=metadata["training_date"],
        n_features=metadata["n_features"],
        metrics=metadata["metrics"],
        churn_rate_in_training=metadata["churn_rate"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(customer: CustomerInput, db: Session = Depends(get_db)):
    """
    Predict churn for a single customer.

    Returns churn probability, risk segment, top churn reasons,
    and a personalised retention recommendation.
    """
    model, pipeline, metadata, config = _get_model()

    from src.predict import predict_single as _predict
    from src.recommendation_engine import build_full_recommendation_output

    customer_dict = customer.model_dump()
    customer_dict["TotalCharges"] = str(customer_dict["TotalCharges"])

    try:
        prediction = _predict(customer_dict)
        full_output = build_full_recommendation_output(prediction, customer_dict)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Persist to DB
    try:
        log_prediction(db, full_output)
    except Exception as e:
        logger.warning(f"DB logging failed: {e}")

    return PredictionResponse(**full_output)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(customers: List[CustomerInput], db: Session = Depends(get_db)):
    """
    Predict churn for a list of customers.

    Returns risk distribution and per-customer predictions.
    """
    _get_model()

    from src.predict import predict_batch as _batch_predict

    records = [c.model_dump() for c in customers]
    for r in records:
        r["TotalCharges"] = str(r["TotalCharges"])

    df = pd.DataFrame(records)
    # Rename to match dataset convention
    df = df.rename(columns={"customerID": "customerID"})

    try:
        results = _batch_predict(df)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    rows = [
        BatchPredictionRow(
            customer_id=row["customer_id"],
            churn_prediction=row["churn_prediction"],
            churn_probability=row["churn_probability"],
            risk_level=row["risk_level"],
        )
        for _, row in results.iterrows()
    ]

    return BatchPredictionResponse(
        total_customers=len(rows),
        high_risk_count=sum(1 for r in rows if r.risk_level == "High Risk"),
        medium_risk_count=sum(1 for r in rows if r.risk_level == "Medium Risk"),
        low_risk_count=sum(1 for r in rows if r.risk_level == "Low Risk"),
        predictions=rows,
    )


@app.get("/predictions", tags=["History"])
def get_prediction_history(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve the most recent prediction records from the database."""
    from api.database import PredictionLog
    records = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "customer_id": r.customer_id,
            "churn_prediction": r.churn_prediction,
            "churn_probability": r.churn_probability,
            "risk_level": r.risk_level,
            "recommended_action": r.recommended_action,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
