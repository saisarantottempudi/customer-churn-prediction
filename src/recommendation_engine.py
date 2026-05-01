"""
Retention Recommendation Engine for the Customer Churn Prediction system.

Uses rule-based logic driven by churn probability and key churn features
to produce actionable, personalized retention strategies for each customer.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class RetentionRecommendation:
    customer_id: str
    churn_probability: float
    risk_level: str
    top_reasons: List[str]
    primary_action: str
    secondary_actions: List[str] = field(default_factory=list)
    urgency: str = "Normal"  # Low | Normal | High | Critical
    expected_impact: str = ""


# ---------------------------------------------------------------------------
# Individual recommendation rules
# ---------------------------------------------------------------------------

def _recommend_for_contract(contract: str, prob: float, actions: list) -> None:
    if contract == "Month-to-month" and prob >= 0.5:
        actions.append("Offer 20% discount to switch to an annual contract")
    elif contract == "Month-to-month":
        actions.append("Highlight benefits of annual contract with a loyalty incentive")
    elif contract == "One year" and prob >= 0.65:
        actions.append("Offer two-year contract with priority support upgrade")


def _recommend_for_tenure(tenure: float, actions: list) -> None:
    if tenure < 6:
        actions.append("Assign dedicated onboarding specialist for first 90 days")
        actions.append("Send personalised welcome series with product tips")
    elif tenure < 12:
        actions.append("Schedule proactive success check-in call")
    elif tenure > 48:
        actions.append("Recognise loyalty with a thank-you reward or exclusive perk")


def _recommend_for_monthly_charges(monthly_charges: float, prob: float, actions: list) -> None:
    if monthly_charges > 80 and prob >= 0.6:
        actions.append("Offer personalised pricing review and bundle discount")
    elif monthly_charges > 65:
        actions.append("Suggest a lower-cost bundle that meets core needs")


def _recommend_for_payment_method(payment_method: str, prob: float, actions: list) -> None:
    if payment_method == "Electronic check" and prob >= 0.5:
        actions.append("Offer $5/month discount for switching to auto-pay (bank/card)")
    elif payment_method == "Mailed check" and prob >= 0.6:
        actions.append("Encourage paperless billing with auto-pay incentive")


def _recommend_for_internet_service(internet_service: str, online_security: str, tech_support: str, actions: list) -> None:
    if internet_service == "Fiber optic":
        if online_security == "No":
            actions.append("Offer free 3-month trial of Online Security add-on")
        if tech_support == "No":
            actions.append("Offer complimentary Tech Support upgrade for 1 month")


def _recommend_for_senior(senior: str, actions: list) -> None:
    if senior == "Yes":
        actions.append("Assign senior-dedicated customer support line")


def _recommend_for_no_services(row: dict, actions: list) -> None:
    add_ons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    no_count = sum(1 for a in add_ons if row.get(a) in ("No", "No internet service"))
    if no_count >= 4:
        actions.append("Bundle 3+ add-ons at a discounted rate to increase stickiness")


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def generate_recommendation(
    customer_id: str,
    customer_data: dict,
    churn_probability: float,
    risk_level: str,
    top_reasons: list,
) -> RetentionRecommendation:
    """
    Generate a personalised retention recommendation for a single customer.

    Args:
        customer_id: Unique customer identifier
        customer_data: Raw customer feature dict
        churn_probability: Model predicted probability of churn (0-1)
        risk_level: "Low Risk", "Medium Risk", or "High Risk"
        top_reasons: Top ML-derived churn reasons

    Returns:
        RetentionRecommendation dataclass
    """
    actions = []

    contract = customer_data.get("Contract", "")
    tenure = float(customer_data.get("tenure", 0))
    monthly_charges = float(customer_data.get("MonthlyCharges", 0))
    payment_method = customer_data.get("PaymentMethod", "")
    internet_service = customer_data.get("InternetService", "")
    online_security = customer_data.get("OnlineSecurity", "")
    tech_support = customer_data.get("TechSupport", "")
    senior = str(customer_data.get("SeniorCitizen", "No"))

    _recommend_for_contract(contract, churn_probability, actions)
    _recommend_for_tenure(tenure, actions)
    _recommend_for_monthly_charges(monthly_charges, churn_probability, actions)
    _recommend_for_payment_method(payment_method, churn_probability, actions)
    _recommend_for_internet_service(internet_service, online_security, tech_support, actions)
    _recommend_for_senior(senior, actions)
    _recommend_for_no_services(customer_data, actions)

    # Fallback generic action if no specific rules fired
    if not actions:
        if churn_probability >= 0.65:
            actions.append("Escalate to retention specialist for personalised outreach")
        elif churn_probability >= 0.35:
            actions.append("Send proactive NPS survey to identify pain points")
        else:
            actions.append("Include in standard loyalty rewards programme")

    # Deduplicate while preserving order
    seen = set()
    unique_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)

    primary = unique_actions[0]
    secondary = unique_actions[1:4]

    # Urgency based on risk
    urgency_map = {"High Risk": "Critical", "Medium Risk": "High", "Low Risk": "Normal"}
    urgency = urgency_map.get(risk_level, "Normal")

    # Expected impact statement
    if risk_level == "High Risk":
        expected_impact = "Estimated 30-40% reduction in churn probability if acted upon within 48 hours"
    elif risk_level == "Medium Risk":
        expected_impact = "Estimated 15-25% reduction in churn probability"
    else:
        expected_impact = "Maintain engagement; low immediate risk"

    return RetentionRecommendation(
        customer_id=customer_id,
        churn_probability=churn_probability,
        risk_level=risk_level,
        top_reasons=top_reasons,
        primary_action=primary,
        secondary_actions=secondary,
        urgency=urgency,
        expected_impact=expected_impact,
    )


def build_full_recommendation_output(prediction_result: dict, customer_data: dict) -> dict:
    """
    Combine prediction output and recommendation into a single response dict.

    Args:
        prediction_result: Output of predict_single() or predict_batch() row
        customer_data: Original raw customer feature dict

    Returns:
        Complete response dict for API / dashboard
    """
    rec = generate_recommendation(
        customer_id=prediction_result["customer_id"],
        customer_data=customer_data,
        churn_probability=prediction_result["churn_probability"],
        risk_level=prediction_result["risk_level"],
        top_reasons=prediction_result["top_reasons"],
    )
    return {
        "customer_id": rec.customer_id,
        "churn_prediction": prediction_result["churn_prediction"],
        "churn_probability": rec.churn_probability,
        "risk_level": rec.risk_level,
        "urgency": rec.urgency,
        "top_reasons": rec.top_reasons,
        "recommended_action": rec.primary_action,
        "additional_actions": rec.secondary_actions,
        "expected_impact": rec.expected_impact,
    }


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, ".")
    from src.predict import predict_single

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

    prediction = predict_single(sample)
    full_output = build_full_recommendation_output(prediction, sample)
    print(json.dumps(full_output, indent=2))
