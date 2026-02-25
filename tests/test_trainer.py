"""Tests for the model trainer."""

import numpy as np
import pytest

from src.config import DataConfig, FeaturesConfig, load_config
from src.data.generator import ChurnDataGenerator
from src.features.store import FeatureStore
from src.models.trainer import ModelTrainer, ModelMetrics, ChurnTabNet


@pytest.fixture
def training_data():
    """Generate a small dataset for testing."""
    gen = ChurnDataGenerator(DataConfig(n_users=200, n_months=4, churn_rate=0.2, seed=42))
    df = gen.generate()
    store = FeatureStore(FeaturesConfig())
    X, y, names = store.fit_transform(df)
    return X, y, names


class TestModelTrainer:
    def test_train_all_returns_three_models(self, training_data):
        X, y, names = training_data
        config = load_config()
        # Reduce training time for tests
        config.xgboost.n_estimators = 10
        config.xgboost.early_stopping_rounds = 5
        config.neural_net.epochs = 3
        config.neural_net.patience = 2
        config.training.cv_folds = 2

        trainer = ModelTrainer(config)
        models = trainer.train_all(X, y, names)

        assert "xgboost" in models
        assert "neural_net" in models
        assert "logistic" in models

    def test_metrics_are_valid(self, training_data):
        X, y, names = training_data
        config = load_config()
        config.xgboost.n_estimators = 10
        config.xgboost.early_stopping_rounds = 5
        config.neural_net.epochs = 3
        config.neural_net.patience = 2
        config.training.cv_folds = 2

        trainer = ModelTrainer(config)
        models = trainer.train_all(X, y, names)

        for name, model in models.items():
            m = model.metrics
            assert 0 <= m.auc_roc <= 1, f"{name} AUC-ROC out of range"
            assert 0 <= m.f1 <= 1, f"{name} F1 out of range"
            assert 0 <= m.precision <= 1, f"{name} Precision out of range"
            assert 0 <= m.recall <= 1, f"{name} Recall out of range"
            assert m.train_time_seconds > 0, f"{name} missing train time"

    def test_xgboost_beats_logistic(self, training_data):
        X, y, names = training_data
        config = load_config()
        config.xgboost.n_estimators = 50
        config.xgboost.early_stopping_rounds = 10
        config.neural_net.epochs = 5
        config.neural_net.patience = 3
        config.training.cv_folds = 2

        trainer = ModelTrainer(config)
        models = trainer.train_all(X, y, names)

        xgb_auc = models["xgboost"].metrics.auc_roc
        log_auc = models["logistic"].metrics.auc_roc
        # XGBoost should generally outperform logistic on this data
        assert xgb_auc >= log_auc * 0.95  # Allow small margin

    def test_save_models(self, training_data, tmp_path):
        X, y, names = training_data
        config = load_config()
        config.xgboost.n_estimators = 5
        config.xgboost.early_stopping_rounds = 3
        config.neural_net.epochs = 2
        config.neural_net.patience = 1
        config.training.cv_folds = 2

        trainer = ModelTrainer(config)
        trainer.train_all(X, y, names)
        base = trainer.save_models(str(tmp_path))

        assert (tmp_path / "xgboost" / "1.0.0" / "model.joblib").exists()
        assert (tmp_path / "xgboost" / "1.0.0" / "metrics.json").exists()
        assert (tmp_path / "logistic" / "1.0.0" / "model.joblib").exists()
        assert (tmp_path / "neural_net" / "1.0.0" / "model.pt").exists()


class TestModelMetrics:
    def test_to_dict(self):
        m = ModelMetrics(
            model_name="test", auc_roc=0.85, auc_pr=0.75,
            f1=0.7, precision=0.8, recall=0.65, accuracy=0.82,
        )
        d = m.to_dict()
        assert d["model_name"] == "test"
        assert d["auc_roc"] == 0.85


class TestChurnTabNet:
    def test_forward_pass(self):
        import torch
        model = ChurnTabNet(input_dim=10, hidden_dims=[32, 16], dropout=0.1)
        x = torch.randn(5, 10)
        out = model(x)
        assert out.shape == (5,)

    def test_output_range_after_sigmoid(self):
        import torch
        model = ChurnTabNet(input_dim=10, hidden_dims=[32], dropout=0.0)
        model.eval()
        x = torch.randn(10, 10)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
        assert (probs >= 0).all() and (probs <= 1).all()
