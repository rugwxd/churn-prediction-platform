"""Tests for the feature store."""

import numpy as np
import pandas as pd
import pytest

from src.config import FeaturesConfig
from src.data.generator import ChurnDataGenerator
from src.config import DataConfig
from src.features.store import FeatureStore, FeatureView, FeatureDefinition


@pytest.fixture
def config():
    return FeaturesConfig()


@pytest.fixture
def store(config):
    return FeatureStore(config)


@pytest.fixture
def sample_df():
    gen = ChurnDataGenerator(DataConfig(n_users=100, n_months=6, churn_rate=0.2, seed=42))
    return gen.generate()


class TestFeatureStore:
    def test_fit_transform_returns_arrays(self, store, sample_df):
        X, y, names = store.fit_transform(sample_df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(names) > 0

    def test_feature_dimensions_match(self, store, sample_df):
        X, y, names = store.fit_transform(sample_df)
        assert X.shape[0] == len(sample_df)
        assert X.shape[1] == len(names)
        assert len(y) == len(sample_df)

    def test_target_is_binary(self, store, sample_df):
        _, y, _ = store.fit_transform(sample_df)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_transform_after_fit(self, store, sample_df):
        X_fit, _, names = store.fit_transform(sample_df)
        X_transform = store.transform(sample_df)
        assert X_transform.shape == X_fit.shape

    def test_transform_before_fit_raises(self, store, sample_df):
        with pytest.raises(RuntimeError, match="fitted"):
            store.transform(sample_df)

    def test_feature_names_populated(self, store, sample_df):
        store.fit_transform(sample_df)
        names = store.get_feature_names()
        assert len(names) > 0
        assert "login_frequency" in names

    def test_registry_populated(self, store, sample_df):
        store.fit_transform(sample_df)
        registry = store.get_registry()
        assert len(registry) == 1
        assert registry[0].n_rows == len(sample_df)

    def test_save_and_load(self, store, sample_df, tmp_path):
        X1, _, names1 = store.fit_transform(sample_df)
        store.save(str(tmp_path))

        store2 = FeatureStore(FeaturesConfig())
        store2.load(str(tmp_path / "feature_store.joblib"))
        X2 = store2.transform(sample_df)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_handles_unseen_categories(self, store, sample_df):
        store.fit_transform(sample_df)

        # Modify a category to something unseen
        modified = sample_df.copy()
        modified.loc[modified.index[0], "plan_tier"] = "unknown_tier"
        X = store.transform(modified)
        assert X.shape[0] == len(modified)


class TestFeatureView:
    def test_default_view_creation(self, store):
        view = store.get_default_feature_view()
        assert isinstance(view, FeatureView)
        assert len(view.features) > 0
        assert view.entity_column == "user_id"

    def test_feature_definition(self):
        feat = FeatureDefinition(name="test", dtype="numerical")
        assert feat.name == "test"
        assert feat.default_value is None
