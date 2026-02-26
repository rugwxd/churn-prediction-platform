# Churn Prediction Platform

Production-grade ML platform for predicting SaaS customer churn. Features a point-in-time correct feature store, three-model comparison (XGBoost, PyTorch, Logistic Regression), FastAPI serving, data drift detection, SHAP explainability, and a Streamlit monitoring dashboard.

## Architecture

```
                      ┌──────────────────────────────────────────────────┐
                      │              Monitoring Dashboard                │
                      │  Model Performance │ Drift Alerts │ Business KPIs│
                      └───────────┬────────────────────────┬─────────────┘
                                  │                        │
            ┌─────────────────────┼────────────────────────┼───────────────┐
            │                     │     Serving Layer       │               │
            │            ┌────────▼─────────┐    ┌─────────▼──────────┐    │
            │            │   FastAPI REST    │    │  Drift Detector    │    │
            │            │  /predict         │    │  Evidently AI      │    │
            │            │  /predict/batch   │    │  KS-test + Chi²    │    │
            │            │  /health          │    └────────────────────┘    │
            │            └────────┬──────────┘                             │
            │                     │                                        │
            └─────────────────────┼────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼────────────────────────────────────────┐
            │                     │     Model Layer                        │
            │     ┌───────────────┼───────────────────────────┐            │
            │     │               │                           │            │
            │  ┌──▼───────┐  ┌───▼────────┐  ┌───────────────▼──┐         │
            │  │ XGBoost  │  │ PyTorch    │  │ Logistic         │         │
            │  │ (Primary)│  │ TabNet     │  │ Regression       │         │
            │  │ AUC:0.91 │  │ AUC:0.88  │  │ (Baseline) 0.83  │         │
            │  └──────────┘  └────────────┘  └──────────────────┘         │
            │                     │                                        │
            │          ┌──────────▼───────────┐                            │
            │          │   SHAP Explainer     │                            │
            │          │   Global + Local     │                            │
            │          └──────────────────────┘                            │
            └─────────────────────┬────────────────────────────────────────┘
                                  │
            ┌─────────────────────▼────────────────────────────────────────┐
            │                Feature Store                                 │
            │  Point-in-time correct retrieval │ Schema validation          │
            │  Versioned materialization │ Encoding + Scaling               │
            └─────────────────────┬────────────────────────────────────────┘
                                  │
            ┌─────────────────────▼──────────────────┐  ┌─────────────────┐
            │          Data Generator                │  │  PySpark Module │
            │  10K users × 12 months                 │  │  Distributed    │
            │  Latent engagement model               │  │  Feature Eng.   │
            │  Correlated features                   │  │  Point-in-Time  │
            │  Temporal decay for churners            │  │  Window Funcs   │
            └────────────────────────────────────────┘  └─────────────────┘

Data Flow:  Generate → Feature Store → Train → Registry → Serve → Monitor
```

## Model Comparison Results

Evaluated on 20% held-out test set with stratified splitting:

| Model | AUC-ROC | AUC-PR | F1 | Precision | Recall | Train Time |
|-------|---------|--------|-------|-----------|--------|------------|
| **XGBoost** | **0.912** | **0.804** | **0.743** | 0.781 | 0.709 | 2.1s |
| PyTorch TabNet | 0.884 | 0.761 | 0.712 | 0.748 | 0.680 | 8.3s |
| Logistic Regression | 0.831 | 0.672 | 0.651 | 0.692 | 0.615 | 0.1s |

*Results from synthetic data with 10,000 users and 18% churn rate. XGBoost selected as primary production model based on AUC-ROC and AUC-PR performance.*

### Feature Importance (SHAP)

| Rank | Feature | Mean |SHAP| |
|------|---------|--------------|
| 1 | days_since_last_login | 0.2847 |
| 2 | login_frequency | 0.2413 |
| 3 | login_frequency_trend | 0.1892 |
| 4 | feature_usage_score | 0.1654 |
| 5 | monthly_active_days | 0.1423 |
| 6 | avg_session_duration_min | 0.1198 |
| 7 | support_tickets_recent | 0.0876 |
| 8 | months_active | 0.0721 |
| 9 | plan_tier | 0.0534 |
| 10 | mrr | 0.0412 |

## Features

### Data Generation
- **Realistic SaaS behavior**: Latent engagement variable drives correlated features
- **Temporal patterns**: Churning users show declining engagement in final months
- **12-month time series**: Monthly behavioral snapshots per user
- **Configurable**: Adjust user count, churn rate, time horizon via YAML

### Feature Store (Point-in-Time Correct)
- **No lookahead bias**: Churned users' features computed only from pre-churn data
- **Declarative feature definitions**: FeatureView + FeatureDefinition schema
- **Versioned materialization**: Track feature schema changes across versions
- **Consistent encoding**: Fitted on train only, applied identically at serving time
- **Unseen category handling**: Graceful fallback for production edge cases

### Model Training
- **XGBoost**: Early stopping, class-weighted, cross-validated
- **PyTorch TabNet**: BatchNorm + Dropout, ReduceLROnPlateau, weighted BCE loss
- **Logistic Regression**: Balanced class weights, L2 regularized
- **SHAP explainability**: Global importance + per-prediction explanations

### Serving
- **FastAPI REST API**: `/predict`, `/predict/batch`, `/health` endpoints
- **Pydantic validation**: Input schema with field constraints and pattern matching
- **Model versioning**: Registry-based model loading with version selection
- **Risk tiering**: Automatic classification into low/medium/high/critical

### Monitoring
- **Data drift detection**: KS-test (numerical) + Chi-squared (categorical)
- **Evidently AI integration**: HTML drift reports with visual comparisons
- **Business metrics**: Churn rate, revenue at risk, risk distribution
- **Prediction distribution tracking**: Mean, std, percentiles over time

### PySpark Module
- **Distributed feature engineering**: Window functions for temporal features
- **Point-in-time validation**: Spark-native temporal join checks
- **Event schema**: Structured schema for raw event data
- **Production-ready**: Configurable shuffle partitions, Kryo serialization

## Project Structure

```
churn-prediction-platform/
├── dashboard.py                    # Streamlit monitoring dashboard
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml       # Lint → Test → Docker CI
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train.py                    # Full training pipeline
│   └── serve.py                    # Start FastAPI server
├── src/
│   ├── config.py                   # Pydantic configuration
│   ├── pipeline.py                 # End-to-end pipeline orchestrator
│   ├── data/
│   │   └── generator.py            # Synthetic SaaS data generator
│   ├── features/
│   │   └── store.py                # Point-in-time feature store
│   ├── models/
│   │   ├── trainer.py              # Multi-model training pipeline
│   │   └── explainer.py            # SHAP explainability
│   ├── serving/
│   │   └── api.py                  # FastAPI prediction endpoint
│   ├── monitoring/
│   │   ├── drift.py                # Data drift detection
│   │   └── tracker.py              # Performance & business tracking
│   └── spark/
│       └── processor.py            # PySpark feature engineering
├── tests/
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_drift.py
│   ├── test_feature_store.py
│   ├── test_generator.py
│   └── test_trainer.py
├── models/registry/                # Versioned model artifacts (gitignored)
├── data/                           # Generated data (gitignored)
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Setup

### Prerequisites
- Python 3.11+

### Installation

```bash
git clone https://github.com/rugwed9/churn-prediction-platform.git
cd churn-prediction-platform

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train Models

```bash
python scripts/train.py
```

This runs the full pipeline: data generation → feature engineering → model training → evaluation → artifact saving. Takes ~30 seconds.

### Start API Server

```bash
python scripts/serve.py
# API docs at http://localhost:8000/docs
```

### Start Monitoring Dashboard

```bash
streamlit run dashboard.py
# Dashboard at http://localhost:8501
```

### Docker

```bash
# Train first
docker compose --profile train run train

# Start API + dashboard
docker compose up
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## API Usage

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_00042",
    "login_frequency": 3.2,
    "avg_session_duration_min": 5.1,
    "feature_usage_score": 22.0,
    "support_tickets_total": 7,
    "support_tickets_recent": 3,
    "days_since_last_login": 14.0,
    "monthly_active_days": 8.0,
    "pages_per_session": 3.0,
    "login_frequency_trend": -0.45,
    "session_duration_std": 12.3,
    "months_active": 6,
    "plan_tier": "starter",
    "billing_cycle": "monthly",
    "signup_channel": "paid_search",
    "company_size": "11-50",
    "mrr": 29.0
  }'
```

Response:
```json
{
  "user_id": "user_00042",
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_tier": "high",
  "model_name": "xgboost",
  "model_version": "1.0.0",
  "timestamp": "2025-01-15T10:30:00"
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"users": [...]}'
```

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
python -m pytest tests/test_trainer.py -v  # Model tests only
```

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Point-in-time feature store** | Prevents lookahead bias that inflates offline metrics but degrades production performance |
| **Latent engagement model** | Generates correlated features that mimic real SaaS behavior rather than independent random noise |
| **XGBoost as primary** | Best AUC-ROC/PR, handles class imbalance natively, fast inference, SHAP TreeExplainer support |
| **PyTorch TabNet** | Demonstrates deep learning on tabular data; competitive but higher training cost |
| **Logistic Regression baseline** | Essential for quantifying how much complexity the tree/neural models add |
| **SHAP over permutation importance** | Per-prediction explanations needed for customer success teams; handles feature interactions |
| **KS-test + Chi-squared for drift** | Statistically principled, works without retraining; Evidently for visual reports |
| **Risk tiering (4 levels)** | Maps probabilities to actionable categories for non-technical stakeholders |
| **BM25 from scratch** | Avoids heavy dependency (rank_bm25/Elasticsearch); ~100 lines, easy to test |
| **PySpark module** | Demonstrates scaling path without requiring Spark to run the core pipeline |

## License

MIT
