"""
Streamlit Dashboard for Customer Churn Prediction System.

Features:
- Upload customer CSV for batch prediction
- Single customer risk assessment
- Risk distribution charts
- Feature importance visualization
- Retention recommendations display
- Download results as CSV
"""

import os
import sys
import json
import io
import joblib
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS for a clean professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border-left: 4px solid #2563EB;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .risk-high { color: #DC2626; font-weight: bold; }
    .risk-medium { color: #D97706; font-weight: bold; }
    .risk-low { color: #16A34A; font-weight: bold; }
    .recommendation-box {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
    }
    h1 { color: #1E3A5F; }
    .stButton > button { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model = joblib.load(config["model"]["path"])
    pipeline = joblib.load(config["model"]["pipeline_path"])
    with open("models/training_metadata.json") as f:
        metadata = json.load(f)
    return model, pipeline, metadata, config


def get_risk_color(risk: str) -> str:
    return {"High Risk": "#DC2626", "Medium Risk": "#D97706", "Low Risk": "#16A34A"}.get(risk, "#6B7280")


def format_probability_gauge(prob: float, risk: str):
    color = get_risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Churn Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35], "color": "#DCFCE7"},
                {"range": [35, 65], "color": "#FEF9C3"},
                {"range": [65, 100], "color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": prob * 100},
        },
        number={"suffix": "%", "font": {"size": 36}},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/line-chart.png", width=60)
st.sidebar.title("Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔍 Single Customer", "📁 Batch Analysis", "📈 Model Insights"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Telco Customer Churn | ML-Powered Retention System")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview(model, pipeline, metadata, config):
    st.title("📊 Customer Churn Prediction System")
    st.markdown("**Identify at-risk customers, understand churn drivers, and act before customers leave.**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    metrics = metadata.get("metrics", {})
    with col1:
        st.metric("Model Type", metadata.get("model_type", "N/A"))
    with col2:
        st.metric("Recall (Churn)", f"{metrics.get('Recall', 0):.1%}")
    with col3:
        st.metric("ROC-AUC", f"{metrics.get('ROC_AUC', 0):.3f}")
    with col4:
        st.metric("Training Churn Rate", f"{metadata.get('churn_rate', 0):.1%}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Business Problem")
        st.markdown("""
        Customer churn is one of the most costly business challenges in subscription-based services.
        Acquiring a new customer costs **5-7x more** than retaining an existing one.

        This system helps your team:
        - **Predict** which customers are likely to leave
        - **Understand** the key reasons driving churn
        - **Act** with personalised retention strategies before it's too late
        """)

    with col_b:
        st.subheader("How to Use This Dashboard")
        st.markdown("""
        | Page | Use Case |
        |------|----------|
        | 🔍 Single Customer | Assess one customer's churn risk in real-time |
        | 📁 Batch Analysis | Upload a CSV, predict risk for all customers |
        | 📈 Model Insights | Understand feature importance & model performance |

        **Quick Start:** Upload a customer CSV on the Batch Analysis page or
        enter a single customer profile to get instant predictions.
        """)

    st.markdown("---")
    st.subheader("Risk Segments")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("🔴 High Risk (>65%)\nImmediate retention action required. Critical priority.")
    with c2:
        st.warning("🟡 Medium Risk (35-65%)\nProactive outreach recommended. High priority.")
    with c3:
        st.success("🟢 Low Risk (<35%)\nMonitor regularly. Include in loyalty programme.")


# ---------------------------------------------------------------------------
# Page: Single Customer Prediction
# ---------------------------------------------------------------------------
def page_single_customer(model, pipeline, metadata, config):
    st.title("🔍 Single Customer Risk Assessment")
    st.markdown("Enter a customer's details to get an instant churn risk assessment and retention recommendation.")
    st.markdown("---")

    with st.form("customer_form"):
        st.subheader("Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            customer_id = st.text_input("Customer ID", value="CUST-2024-001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        col_c, col_d = st.columns(2)
        with col_c:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with col_d:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        customer_dict = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges),
        }

        with st.spinner("Analysing customer profile..."):
            from src.predict import predict_single
            from src.recommendation_engine import build_full_recommendation_output
            prediction = predict_single(customer_dict)
            result = build_full_recommendation_output(prediction, customer_dict)

        st.markdown("---")
        st.subheader("Prediction Results")

        col_gauge, col_details = st.columns([1, 2])

        with col_gauge:
            fig = format_probability_gauge(result["churn_probability"], result["risk_level"])
            st.plotly_chart(fig, use_container_width=True)
            risk_color = get_risk_color(result["risk_level"])
            st.markdown(
                f"<div style='text-align:center; font-size:20px; font-weight:bold; color:{risk_color}'>"
                f"{result['risk_level']}</div>",
                unsafe_allow_html=True
            )

        with col_details:
            st.markdown(f"**Customer ID:** `{result['customer_id']}`")
            st.markdown(f"**Churn Prediction:** {'⚠️ Yes — At Risk' if result['churn_prediction'] == 'Yes' else '✅ No — Likely to Stay'}")
            st.markdown(f"**Churn Probability:** `{result['churn_probability']:.1%}`")
            st.markdown(f"**Urgency:** `{result['urgency']}`")

            st.markdown("**Top Churn Drivers:**")
            for reason in result["top_reasons"]:
                st.markdown(f"- {reason}")

            st.markdown(
                f"<div class='recommendation-box'>"
                f"<b>🎯 Primary Retention Action</b><br>{result['recommended_action']}"
                f"</div>",
                unsafe_allow_html=True
            )

            if result["additional_actions"]:
                st.markdown("**Additional Actions:**")
                for action in result["additional_actions"]:
                    st.markdown(f"- {action}")

            st.info(f"📈 Expected Impact: {result['expected_impact']}")


# ---------------------------------------------------------------------------
# Page: Batch Analysis
# ---------------------------------------------------------------------------
def page_batch_analysis(model, pipeline, metadata, config):
    st.title("📁 Batch Customer Analysis")
    st.markdown("Upload a CSV file with customer data to predict churn for all customers at once.")
    st.markdown("---")

    st.info("📎 Upload a CSV with the same columns as the Telco Customer Churn dataset. The `customerID` and `Churn` columns are optional.")

    uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df_raw)} customers")

        with st.expander("Preview uploaded data"):
            st.dataframe(df_raw.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            with st.spinner(f"Predicting churn risk for {len(df_raw)} customers..."):
                from src.predict import predict_batch
                results = predict_batch(df_raw)

            st.markdown("---")
            st.subheader("Prediction Summary")

            total = len(results)
            high = (results["risk_level"] == "High Risk").sum()
            medium = (results["risk_level"] == "Medium Risk").sum()
            low = (results["risk_level"] == "Low Risk").sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Customers", total)
            col2.metric("🔴 High Risk", high, delta=f"{high/total:.1%}")
            col3.metric("🟡 Medium Risk", medium, delta=f"{medium/total:.1%}")
            col4.metric("🟢 Low Risk", low, delta=f"{low/total:.1%}")

            # Risk distribution pie
            col_pie, col_hist = st.columns(2)
            with col_pie:
                fig_pie = px.pie(
                    values=[high, medium, low],
                    names=["High Risk", "Medium Risk", "Low Risk"],
                    color=["High Risk", "Medium Risk", "Low Risk"],
                    color_discrete_map={"High Risk": "#DC2626", "Medium Risk": "#D97706", "Low Risk": "#16A34A"},
                    title="Customer Risk Distribution",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_hist:
                fig_hist = px.histogram(
                    results, x="churn_probability", nbins=30,
                    color_discrete_sequence=["#2563EB"],
                    title="Churn Probability Distribution",
                    labels={"churn_probability": "Churn Probability", "count": "Customers"},
                )
                fig_hist.update_layout(bargap=0.05)
                st.plotly_chart(fig_hist, use_container_width=True)

            # High-risk customers table
            st.subheader("🔴 High-Risk Customers (Immediate Action Required)")
            high_risk_df = results[results["risk_level"] == "High Risk"].sort_values(
                "churn_probability", ascending=False
            )
            st.dataframe(
                high_risk_df.style.background_gradient(subset=["churn_probability"], cmap="Reds"),
                use_container_width=True,
            )

            # Full results
            with st.expander("View all predictions"):
                st.dataframe(results.sort_values("churn_probability", ascending=False), use_container_width=True)

            # Download button
            csv_buffer = io.StringIO()
            results.to_csv(csv_buffer, index=False)
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page: Model Insights
# ---------------------------------------------------------------------------
def page_model_insights(model, pipeline, metadata, config):
    st.title("📈 Model Insights & Performance")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Information")
        metrics = metadata.get("metrics", {})
        info_data = {
            "Model Type": metadata.get("model_type", "N/A"),
            "Model Version": metadata.get("model_version", "N/A"),
            "Training Date": metadata.get("training_date", "N/A")[:10],
            "Training Samples": f"{metadata.get('training_samples', 0):,}",
            "Test Samples": f"{metadata.get('test_samples', 0):,}",
            "Number of Features": metadata.get("n_features", 0),
        }
        for k, v in info_data.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.subheader("Performance Metrics")
        perf_metrics = {
            "Recall": metrics.get("Recall", 0),
            "Precision": metrics.get("Precision", 0),
            "F1 Score": metrics.get("F1", 0),
            "ROC-AUC": metrics.get("ROC_AUC", 0),
        }
        fig_metrics = go.Figure(go.Bar(
            x=list(perf_metrics.values()),
            y=list(perf_metrics.keys()),
            orientation="h",
            marker_color=["#2563EB", "#7C3AED", "#059669", "#DC2626"],
        ))
        fig_metrics.update_layout(
            xaxis={"range": [0, 1], "title": "Score"},
            title="Model Metrics",
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("---")

    # Feature importance
    st.subheader("Top 15 Churn Drivers (Feature Importance)")
    feature_names = metadata.get("feature_names", [])
    if feature_names and hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False).head(15)

        fig_fi = px.bar(
            fi_df.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="Feature Importance (Logistic Regression Coefficients)",
            labels={"importance": "Absolute Coefficient", "feature": "Feature"},
        )
        fig_fi.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_fi, use_container_width=True)
    elif os.path.exists("reports/figures/feature_importance.png"):
        st.image("reports/figures/feature_importance.png")

    st.markdown("---")
    col_roc, col_pr = st.columns(2)
    with col_roc:
        if os.path.exists("reports/figures/roc_curve.png"):
            st.subheader("ROC Curve")
            st.image("reports/figures/roc_curve.png")
    with col_pr:
        if os.path.exists("reports/figures/precision_recall_curve.png"):
            st.subheader("Precision-Recall Curve")
            st.image("reports/figures/precision_recall_curve.png")

    if os.path.exists("reports/figures/confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("reports/figures/confusion_matrix.png", width=450)


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------
def main():
    try:
        model, pipeline, metadata, config = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.info("Please run `python3 src/train_model.py` first to train the model.")
        return

    if page == "🏠 Overview":
        page_overview(model, pipeline, metadata, config)
    elif page == "🔍 Single Customer":
        page_single_customer(model, pipeline, metadata, config)
    elif page == "📁 Batch Analysis":
        page_batch_analysis(model, pipeline, metadata, config)
    elif page == "📈 Model Insights":
        page_model_insights(model, pipeline, metadata, config)


if __name__ == "__main__":
    main()
