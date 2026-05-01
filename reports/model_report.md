# Model Report — Customer Churn Prediction System

**Version:** 1.0.0  
**Training Date:** 2026-05-01  
**Dataset:** Telco Customer Churn (IBM)  
**Author:** Data Science Team

---

## 1. Business Problem

A subscription-based telecommunications company is losing customers every month. The business objective is to:

1. **Predict** which customers are likely to cancel their subscription in the next billing cycle
2. **Quantify** the churn risk for each customer (probability score)
3. **Explain** the key factors driving each customer's churn risk
4. **Recommend** personalised retention actions before the customer leaves

**Why Recall is the Primary Metric:**  
Missing a churning customer (False Negative) is more costly than incorrectly flagging a loyal customer (False Positive). A retention offer sent to a loyal customer costs ~$10; losing a churning customer can mean $500–$2,000 in lost lifetime value. We therefore optimise for **Recall** over Precision.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Source | IBM Telco Customer Churn Dataset |
| Rows | 7,043 |
| Columns | 21 |
| Target Column | Churn (Yes/No) |
| Churn Rate | 26.54% |
| Missing Values | 11 (TotalCharges for new customers with tenure=0) |
| Duplicates | 0 |

### Feature Categories

**Numerical (3):** tenure, MonthlyCharges, TotalCharges  
**Categorical (17):** gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod  

---

## 3. Data Preprocessing

| Step | Action |
|------|--------|
| Type Correction | TotalCharges: object → float; SeniorCitizen: 0/1 → No/Yes |
| Missing Values | TotalCharges NaN → 0.0 (new customers, zero billing history) |
| Duplicates | None removed (0 found) |
| Outlier Treatment | Clipped numerical features at 1st–99th percentile |
| Target Encoding | Churn: Yes→1, No→0 |

---

## 4. Feature Engineering

6 domain-driven features were created to improve model performance:

| Feature | Description | Motivation |
|---------|-------------|------------|
| `tenure_group` | Binned tenure: 0-1yr, 1-2yr, 2-4yr, 4-6yr | Customer lifecycle stage |
| `monthly_charge_band` | Low/Medium/High/Premium spend tier | Price sensitivity |
| `total_spend_category` | Lifetime spend: Low/Medium/High/VIP | Customer value |
| `contract_risk_score` | Month-to-month=2, One year=1, Two year=0 | Commitment level |
| `payment_risk_score` | Electronic check=2, auto-pay=0 | Payment friction |
| `customer_value_segment` | Composite tenure + charge score | Retention priority |

**Final Feature Matrix:** 7,043 rows × 63 features after one-hot encoding

---

## 5. Class Imbalance Handling

The dataset has a 73.5% / 26.5% class split. We applied **SMOTE** (Synthetic Minority Oversampling Technique) to the training set only:

- Before SMOTE: 5,634 training samples (73.5% No, 26.5% Yes)
- After SMOTE: 8,278 training samples (50% / 50%)

Models with `class_weight="balanced"` were also used as a secondary strategy.

---

## 6. Model Comparison

All models were trained on the SMOTE-resampled training set and evaluated on the held-out test set (20% = 1,409 samples).

| Model | Recall | Precision | F1 | ROC-AUC | CV Recall (mean ± std) |
|-------|--------|-----------|-----|---------|------------------------|
| **Logistic Regression** | **0.8102** | 0.5179 | 0.6319 | 0.8420 | 0.808 ± 0.018 |
| XGBoost | 0.7968 | 0.5138 | 0.6247 | 0.8364 | 0.908 ± 0.112 |
| Random Forest | 0.7086 | 0.5556 | 0.6228 | 0.8420 | 0.858 ± 0.095 |
| Decision Tree | 0.6524 | 0.5148 | 0.5755 | 0.8080 | 0.827 ± 0.098 |
| Gradient Boosting | 0.6497 | 0.5927 | 0.6199 | 0.8423 | 0.840 ± 0.181 |
| LightGBM | 0.6123 | 0.5933 | 0.6026 | 0.8372 | 0.837 ± 0.192 |

---

## 7. Selected Model: Logistic Regression

**Rationale for selection:**

1. **Highest Recall (0.8102):** Catches 81% of all actual churners — minimizes costly false negatives
2. **Lowest CV Std (±0.018):** Most stable across cross-validation folds — reliable in production
3. **Interpretability:** Linear coefficients directly explain feature contributions — supports SHAP analysis
4. **Speed:** Near-instant inference — suitable for real-time API calls
5. **Competitive AUC (0.842):** Ranks customers well for prioritised outreach

Although Gradient Boosting has a marginally higher AUC (0.8423), its recall is significantly lower (0.6497) and CV instability (±0.181) makes it a worse production choice.

---

## 8. Final Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 74.95% |
| Precision | 51.79% |
| **Recall** | **81.02%** |
| F1 Score | 63.19% |
| ROC-AUC | 0.8420 |
| Avg Precision | 0.6419 |

**Confusion Matrix (test set, n=1,409):**

|  | Predicted No | Predicted Yes |
|--|--------------|---------------|
| **Actual No** | 757 | 278 |
| **Actual Yes** | 71 | 303 |

- True Positives: 303 churners correctly identified
- False Negatives: 71 churners missed (at retention cost)
- False Positives: 278 loyal customers flagged (at offer cost)

---

## 9. Top Churn Drivers

Based on Logistic Regression coefficients and SHAP analysis:

1. **Contract Type (Month-to-month)** — Strongest churn predictor; no long-term commitment
2. **Tenure** — Short-tenure customers churn at much higher rates
3. **Internet Service (Fiber optic)** — Higher charges + competitive alternatives
4. **Monthly Charges** — Higher bills increase churn sensitivity
5. **Payment Method (Electronic check)** — Associated with lower engagement and higher churn
6. **Online Security (No)** — Customers without security add-ons are less sticky
7. **Tech Support (No)** — Lack of support increases frustration-driven churn

---

## 10. Retention Recommendation System

The rule-based engine maps churn drivers to retention actions:

| Driver | Recommendation |
|--------|---------------|
| Month-to-month contract + High risk | Offer 20% discount for annual upgrade |
| Tenure < 6 months | Assign onboarding specialist + welcome series |
| Monthly charges > $80 | Personalised pricing review + bundle discount |
| Electronic check payment | $5/month discount for auto-pay switch |
| No online security (Fiber optic) | Free 3-month Online Security trial |
| 4+ missing add-ons | Bundle discount to increase stickiness |

---

## 11. MLOps & Production Readiness

| Component | Implementation |
|-----------|---------------|
| Model Versioning | `models/training_metadata.json` with version, date, metrics |
| Logging | Python `logging` module in all modules |
| API | FastAPI with health check, model info, and prediction history |
| Database | SQLAlchemy + SQLite for prediction logging |
| Containerisation | Dockerfile + docker-compose.yml |
| Tests | 26 pytest unit + integration tests (100% pass) |
| CI/CD | GitHub Actions pipeline (lint, test, API smoke test) |
| Config Management | `config.yaml` for all hyperparameters and paths |

---

## 12. Future Improvements

1. **Hyperparameter tuning** — Grid search or Optuna for model optimisation
2. **Model monitoring** — Track prediction drift and trigger retraining alerts
3. **Feature store** — Centralise feature computation for consistency
4. **A/B testing** — Measure actual retention lift from recommendations
5. **Deep learning** — TabNet or NODE for potential uplift in AUC
6. **Real-time streaming** — Kafka integration for live churn scoring
7. **Uplift modelling** — Predict incremental retention impact, not just churn probability
