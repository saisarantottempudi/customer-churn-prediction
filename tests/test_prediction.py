"""
Unit and integration tests for the prediction and recommendation modules.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_CUSTOMER = {
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

LOW_RISK_CUSTOMER = {
    **SAMPLE_CUSTOMER,
    "customerID": "TEST-002",
    "tenure": 60,
    "Contract": "Two year",
    "MonthlyCharges": 25.00,
    "TotalCharges": "1500.00",
    "PaymentMethod": "Bank transfer (automatic)",
    "InternetService": "DSL",
}


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPredictSingle:
    def test_returns_expected_keys(self):
        from src.predict import predict_single
        result = predict_single(SAMPLE_CUSTOMER)
        expected_keys = {"customer_id", "churn_prediction", "churn_probability", "risk_level", "top_reasons"}
        assert expected_keys.issubset(set(result.keys()))

    def test_probability_in_valid_range(self):
        from src.predict import predict_single
        result = predict_single(SAMPLE_CUSTOMER)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_high_risk_customer_predicts_yes(self):
        from src.predict import predict_single
        result = predict_single(SAMPLE_CUSTOMER)
        # Month-to-month + fiber optic + low tenure should be high risk
        assert result["churn_probability"] > 0.5
        assert result["churn_prediction"] == "Yes"

    def test_risk_level_is_valid(self):
        from src.predict import predict_single
        result = predict_single(SAMPLE_CUSTOMER)
        assert result["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}

    def test_top_reasons_is_list(self):
        from src.predict import predict_single
        result = predict_single(SAMPLE_CUSTOMER)
        assert isinstance(result["top_reasons"], list)
        assert len(result["top_reasons"]) > 0


class TestPredictBatch:
    def test_batch_returns_correct_number_of_rows(self):
        from src.predict import predict_batch
        df = pd.DataFrame([SAMPLE_CUSTOMER, LOW_RISK_CUSTOMER])
        results = predict_batch(df)
        assert len(results) == 2

    def test_batch_has_required_columns(self):
        from src.predict import predict_batch
        df = pd.DataFrame([SAMPLE_CUSTOMER])
        results = predict_batch(df)
        assert set(["customer_id", "churn_prediction", "churn_probability", "risk_level"]).issubset(results.columns)

    def test_batch_probabilities_in_range(self):
        from src.predict import predict_batch
        df = pd.DataFrame([SAMPLE_CUSTOMER, LOW_RISK_CUSTOMER])
        results = predict_batch(df)
        assert results["churn_probability"].between(0, 1).all()


class TestRiskAssignment:
    def test_high_probability_gets_high_risk(self):
        from src.predict import assign_risk_level, load_config
        config = load_config()
        assert assign_risk_level(0.80, config) == "High Risk"

    def test_low_probability_gets_low_risk(self):
        from src.predict import assign_risk_level, load_config
        config = load_config()
        assert assign_risk_level(0.20, config) == "Low Risk"

    def test_medium_probability_gets_medium_risk(self):
        from src.predict import assign_risk_level, load_config
        config = load_config()
        assert assign_risk_level(0.50, config) == "Medium Risk"


# ---------------------------------------------------------------------------
# Recommendation engine tests
# ---------------------------------------------------------------------------

class TestRecommendationEngine:
    def test_recommendation_has_required_keys(self):
        from src.predict import predict_single
        from src.recommendation_engine import build_full_recommendation_output
        prediction = predict_single(SAMPLE_CUSTOMER)
        result = build_full_recommendation_output(prediction, SAMPLE_CUSTOMER)
        required = {
            "customer_id", "churn_prediction", "churn_probability",
            "risk_level", "urgency", "top_reasons",
            "recommended_action", "additional_actions", "expected_impact"
        }
        assert required.issubset(set(result.keys()))

    def test_month_to_month_gets_contract_recommendation(self):
        from src.recommendation_engine import generate_recommendation
        rec = generate_recommendation(
            customer_id="TEST",
            customer_data=SAMPLE_CUSTOMER,
            churn_probability=0.82,
            risk_level="High Risk",
            top_reasons=["contract"],
        )
        all_actions = [rec.primary_action] + rec.secondary_actions
        contract_recs = [a for a in all_actions if "contract" in a.lower() or "annual" in a.lower()]
        assert len(contract_recs) > 0

    def test_high_risk_urgency_is_critical(self):
        from src.recommendation_engine import generate_recommendation
        rec = generate_recommendation(
            customer_id="TEST",
            customer_data=SAMPLE_CUSTOMER,
            churn_probability=0.85,
            risk_level="High Risk",
            top_reasons=["tenure"],
        )
        assert rec.urgency == "Critical"

    def test_low_risk_urgency_is_normal(self):
        from src.recommendation_engine import generate_recommendation
        rec = generate_recommendation(
            customer_id="TEST-LOW",
            customer_data=LOW_RISK_CUSTOMER,
            churn_probability=0.15,
            risk_level="Low Risk",
            top_reasons=[],
        )
        assert rec.urgency == "Normal"

    def test_new_customer_gets_onboarding_recommendation(self):
        from src.recommendation_engine import generate_recommendation
        rec = generate_recommendation(
            customer_id="NEW",
            customer_data={**SAMPLE_CUSTOMER, "tenure": 2},
            churn_probability=0.70,
            risk_level="High Risk",
            top_reasons=["tenure"],
        )
        all_actions = [rec.primary_action] + rec.secondary_actions
        onboarding_recs = [a for a in all_actions if "onboard" in a.lower() or "welcome" in a.lower()]
        assert len(onboarding_recs) > 0
