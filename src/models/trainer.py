"""Model training pipeline with XGBoost, PyTorch, and Logistic Regression.

Implements a unified training interface with cross-validation,
hyperparameter management, and comprehensive metric computation.
All models are evaluated on the same held-out test set for
fair comparison.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import (
    LogisticConfig,
    NeuralNetConfig,
    Settings,
    TrainingConfig,
    XGBoostConfig,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Comprehensive evaluation metrics for a trained model."""

    model_name: str
    auc_roc: float
    auc_pr: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    train_time_seconds: float = 0.0
    n_train: int = 0
    n_test: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class TrainedModel:
    """Container for a trained model with metadata."""

    name: str
    model: Any
    metrics: ModelMetrics
    feature_names: list[str]
    version: str
    created_at: datetime = field(default_factory=datetime.now)


class ChurnTabNet(nn.Module):
    """Tabular neural network for churn prediction.

    Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Output
    Uses batch normalization for training stability and dropout for regularization.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ModelTrainer:
    """Unified training pipeline for all churn prediction models."""

    def __init__(self, config: Settings) -> None:
        self.config = config
        self.training_config = config.training
        self.models: dict[str, TrainedModel] = {}

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> dict[str, TrainedModel]:
        """Train all three models and return comparison results.

        Args:
            X: Feature matrix (from feature store).
            y: Binary target array.
            feature_names: Ordered feature names.

        Returns:
            Dict mapping model name to TrainedModel.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.training_config.test_size,
            random_state=self.training_config.seed,
            stratify=y,
        )

        logger.info(
            "Training split: %d train, %d test (%.1f%% churn in train)",
            len(X_train), len(X_test), y_train.mean() * 100,
        )

        # Train each model
        self.models["xgboost"] = self._train_xgboost(
            X_train, y_train, X_test, y_test, feature_names
        )
        self.models["neural_net"] = self._train_neural_net(
            X_train, y_train, X_test, y_test, feature_names
        )
        self.models["logistic"] = self._train_logistic(
            X_train, y_train, X_test, y_test, feature_names
        )

        # Log comparison
        self._log_comparison()

        return self.models

    def _train_xgboost(
        self, X_train, y_train, X_test, y_test, feature_names
    ) -> TrainedModel:
        """Train XGBoost with early stopping and cross-validation."""
        cfg = self.config.xgboost
        logger.info("Training XGBoost (n_estimators=%d, max_depth=%d)",
                     cfg.n_estimators, cfg.max_depth)

        # Compute scale_pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Validation split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=self.training_config.val_size,
            random_state=self.training_config.seed,
            stratify=y_train,
        )

        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=cfg.early_stopping_rounds,
            random_state=self.training_config.seed,
            n_jobs=-1,
        )

        start = time.perf_counter()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        train_time = time.perf_counter() - start

        # Cross-validation AUC
        cv_auc = self._cross_validate_auc(model, X_train, y_train)

        metrics = self._compute_metrics(
            "xgboost", model, X_test, y_test, cv_auc, train_time
        )

        logger.info("XGBoost trained in %.1fs (AUC: %.4f, best_iter: %d)",
                     train_time, metrics.auc_roc, model.best_iteration)

        return TrainedModel(
            name="xgboost", model=model, metrics=metrics,
            feature_names=feature_names, version="1.0.0",
        )

    def _train_neural_net(
        self, X_train, y_train, X_test, y_test, feature_names
    ) -> TrainedModel:
        """Train PyTorch tabular neural network with early stopping."""
        cfg = self.config.neural_net
        logger.info("Training Neural Net (layers=%s, dropout=%.2f)",
                     cfg.hidden_dims, cfg.dropout)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=self.training_config.val_size,
            random_state=self.training_config.seed,
            stratify=y_train,
        )

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True
        )

        # Model
        model = ChurnTabNet(
            input_dim=X_train.shape[1],
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(device)

        # Class-weighted loss
        pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training loop with early stopping
        start = time.perf_counter()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg.epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.FloatTensor(X_val).to(device))
                val_loss = criterion(val_logits, torch.FloatTensor(y_val).to(device)).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        train_time = time.perf_counter() - start

        # Restore best weights
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # Wrap for sklearn-compatible predict_proba
        wrapper = NeuralNetWrapper(model, device)
        metrics = self._compute_metrics(
            "neural_net", wrapper, X_test, y_test, (0.0, 0.0), train_time
        )

        logger.info("Neural Net trained in %.1fs (AUC: %.4f)", train_time, metrics.auc_roc)

        return TrainedModel(
            name="neural_net", model=wrapper, metrics=metrics,
            feature_names=feature_names, version="1.0.0",
        )

    def _train_logistic(
        self, X_train, y_train, X_test, y_test, feature_names
    ) -> TrainedModel:
        """Train logistic regression baseline."""
        cfg = self.config.logistic
        logger.info("Training Logistic Regression (C=%.2f)", cfg.C)

        model = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            random_state=self.training_config.seed,
            solver="lbfgs",
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        cv_auc = self._cross_validate_auc(model, X_train, y_train)

        metrics = self._compute_metrics(
            "logistic", model, X_test, y_test, cv_auc, train_time
        )

        logger.info("Logistic Regression trained in %.1fs (AUC: %.4f)",
                     train_time, metrics.auc_roc)

        return TrainedModel(
            name="logistic", model=model, metrics=metrics,
            feature_names=feature_names, version="1.0.0",
        )

    def _compute_metrics(
        self,
        name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_auc: tuple[float, float],
        train_time: float,
    ) -> ModelMetrics:
        """Compute comprehensive metrics on the test set."""
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        return ModelMetrics(
            model_name=name,
            auc_roc=roc_auc_score(y_test, y_prob),
            auc_pr=average_precision_score(y_test, y_prob),
            f1=f1_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            accuracy=accuracy_score(y_test, y_pred),
            cv_auc_mean=cv_auc[0],
            cv_auc_std=cv_auc[1],
            train_time_seconds=train_time,
            n_train=len(y_test),  # approximate
            n_test=len(y_test),
        )

    def _cross_validate_auc(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float]:
        """Run stratified k-fold cross-validation for AUC."""
        try:
            skf = StratifiedKFold(
                n_splits=self.training_config.cv_folds,
                shuffle=True,
                random_state=self.training_config.seed,
            )
            aucs = []
            for train_idx, val_idx in skf.split(X, y):
                clone = model.__class__(**model.get_params())
                clone.fit(X[train_idx], y[train_idx])
                y_prob = clone.predict_proba(X[val_idx])[:, 1]
                aucs.append(roc_auc_score(y[val_idx], y_prob))
            return float(np.mean(aucs)), float(np.std(aucs))
        except Exception as e:
            logger.warning("Cross-validation failed: %s", e)
            return 0.0, 0.0

    def _log_comparison(self) -> None:
        """Log a comparison table of all trained models."""
        header = f"{'Model':<15} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time':>8}"
        logger.info("\n%s\n%s", header, "-" * len(header))
        for name, tm in self.models.items():
            m = tm.metrics
            logger.info(
                "%s %8.4f %8.4f %8.4f %10.4f %8.4f %6.1fs",
                f"{name:<15}", m.auc_roc, m.auc_pr, m.f1, m.precision, m.recall,
                m.train_time_seconds,
            )

    def save_models(self, output_dir: str | None = None) -> Path:
        """Save all trained models to the registry."""
        if output_dir is None:
            base = PROJECT_ROOT / "models" / "registry"
        else:
            base = Path(output_dir)

        base.mkdir(parents=True, exist_ok=True)

        for name, trained_model in self.models.items():
            model_dir = base / name / trained_model.version
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model artifact
            if name == "neural_net":
                torch.save(
                    trained_model.model.model.state_dict(),
                    model_dir / "model.pt",
                )
            else:
                joblib.dump(trained_model.model, model_dir / "model.joblib")

            # Save metrics
            with open(model_dir / "metrics.json", "w") as f:
                json.dump(trained_model.metrics.to_dict(), f, indent=2)

            # Save metadata
            metadata = {
                "name": name,
                "version": trained_model.version,
                "created_at": trained_model.created_at.isoformat(),
                "feature_names": trained_model.feature_names,
            }
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Saved model '%s' v%s to %s", name, trained_model.version, model_dir)

        return base


class NeuralNetWrapper:
    """Sklearn-compatible wrapper for the PyTorch neural network."""

    def __init__(self, model: ChurnTabNet, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(X).to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
