"""FastAPI REST endpoint for real-time churn prediction inference.

Provides /predict and /predict/batch endpoints with input validation,
model versioning, feature transformation, and prediction logging.
"""

import logging
import time
from datetime import datetime
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, ServingConfig, load_config
from src.features.store import FeatureStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time SaaS churn prediction with model versioning",
    version="1.0.0",
)


# --- Request / Response models ---


class UserFeatures(BaseModel):
    """Input features for a single user prediction."""

    user_id: str
    login_frequency: float = Field(..., ge=0, description="Avg logins per month")
    avg_session_duration_min: float = Field(..., ge=0)
    feature_usage_score: float = Field(..., ge=0, le=100)
    support_tickets_total: int = Field(..., ge=0)
    support_tickets_recent: int = Field(0, ge=0)
    days_since_last_login: float = Field(..., ge=0)
    monthly_active_days: float = Field(..., ge=0, le=31)
    pages_per_session: float = Field(..., ge=0)
    login_frequency_trend: float = Field(0.0)
    session_duration_std: float = Field(0.0, ge=0)
    months_active: int = Field(1, ge=1)
    plan_tier: str = Field(..., pattern="^(free|starter|professional|enterprise)$")
    billing_cycle: str = Field(..., pattern="^(monthly|annual)$")
    signup_channel: str = Field(..., pattern="^(organic|paid_search|referral|partner)$")
    company_size: str = Field(default="11-50")
    mrr: float = Field(0, ge=0)


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    user_id: str
    churn_probability: float
    churn_prediction: bool
    risk_tier: str  # low, medium, high, critical
    model_name: str
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    users: list[UserFeatures]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


# --- Application state ---


class ModelServer:
    """Manages model loading and prediction serving."""

    def __init__(self) -> None:
        self.model: Any = None
        self.feature_store: FeatureStore | None = None
        self.model_name: str = ""
        self.model_version: str = ""
        self.start_time: float = 0
        self._prediction_count: int = 0
        self._serving_config: ServingConfig | None = None

    def load(self, config: ServingConfig | None = None) -> None:
        """Load model and feature store from registry."""
        if config is None:
            config = load_config().serving

        self._serving_config = config
        self.model_name = config.model_name
        self.model_version = config.model_version
        self.start_time = time.time()

        # Load model
        model_dir = PROJECT_ROOT / "models" / "registry" / config.model_name
        if config.model_version == "latest":
            versions = sorted(model_dir.iterdir()) if model_dir.exists() else []
            if not versions:
                raise FileNotFoundError(f"No model versions found in {model_dir}")
            version_dir = versions[-1]
            self.model_version = version_dir.name
        else:
            version_dir = model_dir / config.model_version

        model_path = version_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info("Loaded model: %s v%s", self.model_name, self.model_version)

        # Load feature store
        self.feature_store = FeatureStore(load_config().features)
        self.feature_store.load()
        logger.info("Feature store loaded")

    def predict(self, features: UserFeatures) -> PredictionResponse:
        """Generate prediction for a single user."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        df = pd.DataFrame([features.model_dump()])
        X = self.feature_store.transform(df)
        prob = float(self.model.predict_proba(X)[0, 1])
        self._prediction_count += 1

        return PredictionResponse(
            user_id=features.user_id,
            churn_probability=round(prob, 4),
            churn_prediction=prob >= 0.5,
            risk_tier=self._risk_tier(prob),
            model_name=self.model_name,
            model_version=self.model_version,
            timestamp=datetime.now().isoformat(),
        )

    def _risk_tier(self, prob: float) -> str:
        cfg = self._serving_config
        if cfg:
            t_crit, t_high, t_med = (
                cfg.risk_threshold_critical,
                cfg.risk_threshold_high,
                cfg.risk_threshold_medium,
            )
        else:
            t_crit, t_high, t_med = 0.8, 0.6, 0.3

        if prob >= t_crit:
            return "critical"
        elif prob >= t_high:
            return "high"
        elif prob >= t_med:
            return "medium"
        return "low"


server = ModelServer()


# --- Endpoints ---


@app.on_event("startup")
async def startup():
    """Load model on application startup."""
    try:
        server.load()
    except FileNotFoundError as e:
        logger.warning("Model not loaded on startup: %s", e)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if server.model is not None else "degraded",
        model_loaded=server.model is not None,
        model_name=server.model_name,
        model_version=server.model_version,
        uptime_seconds=time.time() - server.start_time if server.start_time else 0,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: UserFeatures):
    """Predict churn probability for a single user."""
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return server.predict(features)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn probability for multiple users."""
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    for user in request.users:
        try:
            pred = server.predict(user)
            predictions.append(pred)
        except Exception as e:
            logger.error("Batch prediction failed for %s: %s", user.user_id, e)

    high_risk = sum(1 for p in predictions if p.risk_tier in ("high", "critical"))

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
    )
