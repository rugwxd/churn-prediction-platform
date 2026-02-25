"""Tests for configuration."""

from src.config import Settings, load_config, DataConfig, XGBoostConfig


class TestConfig:
    def test_load_config(self):
        config = load_config()
        assert isinstance(config, Settings)

    def test_default_data_config(self):
        config = DataConfig()
        assert config.n_users == 10000
        assert config.churn_rate == 0.18

    def test_default_xgboost_config(self):
        config = XGBoostConfig()
        assert config.n_estimators == 300
        assert config.learning_rate == 0.05

    def test_settings_nested(self):
        settings = Settings()
        assert settings.data.n_users == 10000
        assert settings.training.cv_folds == 5
        assert settings.xgboost.max_depth == 6
