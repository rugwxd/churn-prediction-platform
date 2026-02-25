"""Centralized configuration management."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class DataConfig(BaseModel):
    n_users: int = 10000
    n_months: int = 12
    churn_rate: float = 0.18
    seed: int = 42
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class FeaturesConfig(BaseModel):
    numerical: list[str] = Field(
        default_factory=lambda: [
            "login_frequency",
            "avg_session_duration_min",
            "feature_usage_score",
            "support_tickets_total",
            "days_since_last_login",
            "monthly_active_days",
            "pages_per_session",
        ]
    )
    categorical: list[str] = Field(
        default_factory=lambda: [
            "plan_tier",
            "billing_cycle",
            "signup_channel",
        ]
    )
    target: str = "churned"


class TrainingConfig(BaseModel):
    test_size: float = 0.2
    val_size: float = 0.15
    seed: int = 42
    cv_folds: int = 5


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: str = "auto"
    eval_metric: str = "aucpr"
    early_stopping_rounds: int = 30


class NeuralNetConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 50
    patience: int = 10
    weight_decay: float = 0.0001


class LogisticConfig(BaseModel):
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str = "balanced"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_name: str = "xgboost"
    model_version: str = "latest"
    risk_threshold_critical: float = 0.8
    risk_threshold_high: float = 0.6
    risk_threshold_medium: float = 0.3


class MonitoringConfig(BaseModel):
    drift_threshold: float = 0.05
    performance_window_days: int = 30
    alert_threshold_auc: float = 0.75


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file: str = "logs/churn.log"


class Settings(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    neural_net: NeuralNetConfig = Field(default_factory=NeuralNetConfig)
    logistic: LogisticConfig = Field(default_factory=LogisticConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str | None = None) -> Settings:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "default.yaml")

    data: dict[str, Any] = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}

    return Settings(**data)


def setup_logging(config: LoggingConfig) -> None:
    """Configure application-wide logging."""
    log_file = PROJECT_ROOT / config.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
