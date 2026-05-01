"""
Pydantic schemas for the Customer Churn Prediction API.

Defines request/response models with validation and documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    customerID: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 = No, 1 = Yes")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Months as a customer")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(...)
    OnlineBackup: str = Field(...)
    DeviceProtection: str = Field(...)
    TechSupport: str = Field(...)
    StreamingTV: str = Field(...)
    StreamingMovies: str = Field(...)
    Contract: str = Field(..., description="Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ..., description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)"
    )
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: str = Field(..., description="String as in original dataset; blank for new customers")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    customer_id: str
    churn_prediction: str = Field(..., description="Yes or No")
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., description="Low Risk, Medium Risk, or High Risk")
    urgency: str = Field(..., description="Normal, High, or Critical")
    top_reasons: List[str]
    recommended_action: str
    additional_actions: List[str]
    expected_impact: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_version: str
    training_date: str
    n_features: int
    metrics: dict
    churn_rate_in_training: float


class BatchPredictionRow(BaseModel):
    customer_id: str
    churn_prediction: str
    churn_probability: float
    risk_level: str


class BatchPredictionResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[BatchPredictionRow]


class PredictionRecord(BaseModel):
    id: Optional[int] = None
    customer_id: str
    churn_prediction: str
    churn_probability: float
    risk_level: str
    recommended_action: str
    created_at: Optional[str] = None
