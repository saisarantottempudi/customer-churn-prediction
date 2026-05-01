# Customer Churn Prediction & Retention Recommendation System

> A production-ready, end-to-end machine learning system that predicts customer churn probability, explains the key drivers, segments customers by risk level, and recommends personalised retention strategies.

[![CI Pipeline](https://github.com/saisarantottempudi/customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/saisarantottempudi/customer-churn-prediction/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io/)

---

## Business Problem

Customer churn is one of the most expensive challenges in subscription businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This system helps businesses:

- **Predict** which customers are likely to leave before they do
- **Understand** the specific reasons driving churn for each customer
- **Act** with personalised, data-driven retention strategies

---

## Demo

| Component | Screenshot Description |
|-----------|----------------------|
| Dashboard Overview | Model KPIs, business context, risk segment guide |
| Single Customer | Probability gauge, top reasons, retention recommendation |
| Batch Analysis | CSV upload, risk distribution charts, high-risk customer table |
| Model Insights | Feature importance, ROC curve, confusion matrix |

---

## Project Architecture

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Original Telco dataset
│   └── processed/                # Cleaned, type-corrected dataset
│
├── notebooks/
│   ├── 01_data_understanding.ipynb   # Business context + data profiling
│   ├── 02_eda.ipynb                  # 8 churn-driver visualisations
│   ├── 03_feature_engineering.ipynb  # 6 engineered features + pipeline
│   ├── 04_model_training.ipynb       # 6-model comparison, SMOTE
│   └── 05_model_explainability.ipynb # SHAP values, local/global explanations
│
├── src/
│   ├── data_preprocessing.py      # Cleaning pipeline
│   ├── feature_engineering.py     # Feature creation + sklearn pipeline
│   ├── train_model.py             # Model training and selection
│   ├── evaluate_model.py          # Metrics, plots, classification report
│   ├── predict.py                 # Single + batch inference
│   └── recommendation_engine.py  # Rule-based retention actions
│
├── api/
│   ├── main.py                    # FastAPI application
│   ├── schemas.py                 # Pydantic request/response models
│   └── database.py               # SQLAlchemy prediction logging
│
├── dashboard/
│   └── app.py                    # Streamlit 4-page dashboard
│
├── models/
│   ├── churn_model.pkl           # Trained Logistic Regression model
│   ├── preprocessing_pipeline.pkl # Fitted sklearn ColumnTransformer
│   └── training_metadata.json    # Version, metrics, feature names
│
├── reports/
│   ├── figures/                  # ROC, PR curve, confusion matrix, feature importance
│   └── model_report.md          # Full model documentation
│
├── tests/
│   ├── test_preprocessing.py     # 10 unit tests for data cleaning
│   └── test_prediction.py        # 16 unit + integration tests
│
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── config.yaml
```

---

## Dataset

**Source:** [IBM Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Features | 20 |
| Target | Churn (Yes/No) |
| Churn Rate | 26.54% |
| Missing Values | 11 (TotalCharges for new customers) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualisation | Matplotlib, Seaborn, Plotly |
| API | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite + SQLAlchemy |
| Testing | Pytest, pytest-cov |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Model Performance

| Metric | Score |
|--------|-------|
| **Recall (Churn)** | **81.02%** |
| Precision | 51.79% |
| F1 Score | 63.19% |
| ROC-AUC | **0.842** |
| Accuracy | 74.95% |

> **Model:** Logistic Regression with SMOTE  
> **Selection rationale:** Highest recall (catches 81% of churners) + lowest cross-validation variance (±0.018) = most reliable in production

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/saisarantottempudi/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Download dataset

```bash
mkdir -p data/raw
curl -o data/raw/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

### 4. Train the model

```bash
python src/data_preprocessing.py   # Clean data -> data/processed/
python src/train_model.py          # Train + save model -> models/
```

### 5. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 6. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard available at: `http://localhost:8501`

---

## API Reference

### `GET /health`
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

### `GET /model-info`
Returns model type, version, training date, feature count, and performance metrics.

### `POST /predict`

**Request:**
```json
{
  "customerID": "CUST-001",
  "gender": "Female",
  "SeniorCitizen": 0,
  "tenure": 3,
  "Contract": "Month-to-month",
  "MonthlyCharges": 89.10,
  "TotalCharges": "267.30",
  "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic",
  ...
}
```

**Response:**
```json
{
  "customer_id": "CUST-001",
  "churn_prediction": "Yes",
  "churn_probability": 0.8147,
  "risk_level": "High Risk",
  "urgency": "Critical",
  "top_reasons": ["tenure", "InternetService_Fiber optic", "contract_risk_score"],
  "recommended_action": "Offer 20% discount to switch to an annual contract",
  "additional_actions": [
    "Assign dedicated onboarding specialist for first 90 days",
    "Offer personalised pricing review and bundle discount"
  ],
  "expected_impact": "Estimated 30-40% reduction in churn probability if acted upon within 48 hours"
}
```

### `POST /predict/batch`
Upload a list of customers, get risk distribution and per-customer predictions.

### `GET /predictions`
Retrieve prediction history from the database.

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

Or build and run individually:

```bash
# API only
docker build -t churn-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models churn-api

# Dashboard only
docker run -p 8501:8501 -v $(pwd)/models:/app/models churn-api \
  streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_prediction.py -v
```

**Test Coverage:** 26 tests (10 preprocessing + 16 prediction/recommendation)

---

## Retention Recommendation Logic

| Churn Driver | Recommended Action |
|-------------|-------------------|
| Month-to-month contract + high risk | 20% discount for annual contract upgrade |
| Tenure < 6 months | Dedicated onboarding specialist + welcome series |
| Monthly charges > $80 | Personalised pricing review + bundle discount |
| Electronic check payment | $5/month discount for auto-pay switch |
| Fiber optic + no security | Free 3-month Online Security trial |
| 4+ missing add-ons | Bundle discount to increase product stickiness |
| Senior citizen | Dedicated senior support line |

---

## MLOps Features

- **Model versioning** via `training_metadata.json`
- **Structured logging** in all modules
- **Database persistence** for every prediction (SQLite/PostgreSQL-ready)
- **Health check** and **model info** endpoints for monitoring
- **Unit + integration tests** with pytest (26 tests, 100% pass)
- **CI/CD pipeline** via GitHub Actions (lint → test → API smoke test)
- **Config-driven** hyperparameters and paths (`config.yaml`)
- **Retraining placeholder** via `run_training_pipeline()` — call it on schedule

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_understanding.ipynb` | Business problem, data loading, missing values, class imbalance |
| `02_eda.ipynb` | 8 churn-driver visualisations with business insights |
| `03_feature_engineering.ipynb` | 6 engineered features, encoding, sklearn pipeline |
| `04_model_training.ipynb` | 6-model comparison, SMOTE, recall-focused selection |
| `05_model_explainability.ipynb` | SHAP global/local explanations, recommendation demo |

---

## Future Improvements

- Hyperparameter optimisation (Optuna)
- Model drift monitoring with Evidently
- Feature store integration
- A/B testing for retention offer effectiveness
- Kafka integration for real-time churn scoring
- Uplift modelling for incremental retention impact

---

## License

MIT License — see [LICENSE](LICENSE) for details.
