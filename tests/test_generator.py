"""Tests for the data generator."""

import pytest
import pandas as pd

from src.config import DataConfig
from src.data.generator import ChurnDataGenerator


@pytest.fixture
def config():
    return DataConfig(n_users=200, n_months=6, churn_rate=0.2, seed=42)


@pytest.fixture
def generator(config):
    return ChurnDataGenerator(config)


class TestChurnDataGenerator:
    def test_generate_returns_dataframe(self, generator):
        df = generator.generate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_correct_number_of_users(self, generator):
        df = generator.generate()
        # May be slightly less than n_users due to filtering
        assert len(df) <= generator.config.n_users
        assert len(df) >= generator.config.n_users * 0.9

    def test_churn_rate_approximate(self, generator):
        df = generator.generate()
        actual_rate = df["churned"].mean()
        # Should be within 5% of target
        assert abs(actual_rate - generator.config.churn_rate) < 0.05

    def test_has_required_columns(self, generator):
        df = generator.generate()
        required = [
            "user_id", "login_frequency", "avg_session_duration_min",
            "feature_usage_score", "support_tickets_total",
            "days_since_last_login", "monthly_active_days",
            "pages_per_session", "plan_tier", "billing_cycle",
            "signup_channel", "churned",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_negative_values(self, generator):
        df = generator.generate()
        numerical = ["login_frequency", "avg_session_duration_min",
                      "feature_usage_score", "monthly_active_days"]
        for col in numerical:
            assert (df[col] >= 0).all(), f"Negative values in {col}"

    def test_target_is_binary(self, generator):
        df = generator.generate()
        assert set(df["churned"].unique()).issubset({0, 1})

    def test_deterministic_with_seed(self, config):
        gen1 = ChurnDataGenerator(config)
        gen2 = ChurnDataGenerator(config)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_plan_tiers_valid(self, generator):
        df = generator.generate()
        valid_tiers = {"free", "starter", "professional", "enterprise"}
        assert set(df["plan_tier"].unique()).issubset(valid_tiers)

    def test_save_and_load(self, generator, tmp_path):
        df = generator.generate()
        generator.config.raw_dir = str(tmp_path)

        from unittest.mock import patch
        with patch("src.data.generator.PROJECT_ROOT", tmp_path):
            path = generator.save(df)
            assert path.exists()
            loaded = pd.read_parquet(path)
            assert len(loaded) == len(df)
