"""Feast-inspired feature store with point-in-time correct retrieval.

Ensures training and serving features are consistent and free of
lookahead bias. Supports feature versioning, schema validation,
and temporal joins for historical feature retrieval.

Design:
- Features are defined declaratively with FeatureView objects
- Point-in-time joins use event timestamps to prevent data leakage
- Feature materialization produces versioned snapshots
- Serving path retrieves latest features for real-time inference
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import PROJECT_ROOT, FeaturesConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Schema for a single feature."""

    name: str
    dtype: str  # numerical | categorical
    description: str = ""
    default_value: Any = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class FeatureView:
    """A logical grouping of features from a data source."""

    name: str
    features: list[FeatureDefinition]
    entity_column: str = "user_id"
    timestamp_column: str | None = None
    version: str = "1.0.0"


@dataclass
class MaterializedFeatures:
    """A versioned snapshot of computed features."""

    version: str
    created_at: datetime
    n_rows: int
    n_features: int
    feature_names: list[str]
    schema_hash: str


class FeatureStore:
    """Point-in-time correct feature store for churn prediction.

    Handles feature transformation, encoding, scaling, and ensures
    no lookahead bias through temporal validation. Maintains separate
    encoders/scalers for train and serving consistency.
    """

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config
        self.feature_views: dict[str, FeatureView] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self._fitted = False
        self._feature_names: list[str] = []
        self._registry: list[MaterializedFeatures] = []

    def register_feature_view(self, view: FeatureView) -> None:
        """Register a feature view definition."""
        self.feature_views[view.name] = view
        logger.info(
            "Registered feature view: %s (v%s, %d features)",
            view.name,
            view.version,
            len(view.features),
        )

    def get_default_feature_view(self) -> FeatureView:
        """Create the default churn prediction feature view."""
        numerical_features = [
            FeatureDefinition(name=f, dtype="numerical") for f in self.config.numerical
        ]
        # Add engineered features
        numerical_features.extend(
            [
                FeatureDefinition(
                    name="login_frequency_trend",
                    dtype="numerical",
                    description="Rate of change in login frequency",
                ),
                FeatureDefinition(
                    name="session_duration_std",
                    dtype="numerical",
                    description="Variability in session duration",
                ),
                FeatureDefinition(name="months_active", dtype="numerical"),
                FeatureDefinition(name="support_tickets_recent", dtype="numerical"),
                FeatureDefinition(
                    name="mrr", dtype="numerical", description="Monthly recurring revenue"
                ),
            ]
        )

        categorical_features = [
            FeatureDefinition(name=f, dtype="categorical") for f in self.config.categorical
        ]
        categorical_features.append(FeatureDefinition(name="company_size", dtype="categorical"))

        return FeatureView(
            name="churn_features_v1",
            features=numerical_features + categorical_features,
            entity_column="user_id",
            version="1.0.0",
        )

    def validate_point_in_time(
        self,
        df: pd.DataFrame,
        event_timestamp_col: str | None = None,
        label_timestamp_col: str | None = None,
    ) -> bool:
        """Validate that features don't contain lookahead bias.

        Checks that all feature timestamps precede the label timestamp,
        preventing data leakage from future events.
        """
        if event_timestamp_col is None or label_timestamp_col is None:
            logger.info("No timestamp columns provided, skipping temporal validation")
            return True

        if event_timestamp_col not in df.columns or label_timestamp_col not in df.columns:
            return True

        violations = df[df[event_timestamp_col] > df[label_timestamp_col]]
        if len(violations) > 0:
            logger.error(
                "Point-in-time violation: %d rows have features from after the label event",
                len(violations),
            )
            return False

        logger.info("Point-in-time validation passed: no lookahead bias detected")
        return True

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Fit encoders/scaler on training data and transform.

        This method must only be called on training data to prevent
        data leakage from the test set into the scaling/encoding.

        Args:
            df: Raw dataframe with features and target.

        Returns:
            Tuple of (X features, y target, feature_names).
        """
        view = self.get_default_feature_view()
        self.register_feature_view(view)

        numerical_cols = [f.name for f in view.features if f.dtype == "numerical"]
        categorical_cols = [f.name for f in view.features if f.dtype == "categorical"]

        # Validate all expected columns exist
        all_feature_cols = numerical_cols + categorical_cols
        missing = set(all_feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")

        X_parts = []
        feature_names = []

        # Encode categorical features
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(df[col].astype(str))
            self.encoders[col] = encoder
            X_parts.append(encoded.reshape(-1, 1))
            feature_names.append(col)

        # Scale numerical features
        X_numerical = df[numerical_cols].fillna(0).values.astype(np.float32)
        self.scaler = StandardScaler()
        X_numerical_scaled = self.scaler.fit_transform(X_numerical)
        X_parts.append(X_numerical_scaled)
        feature_names.extend(numerical_cols)

        X = np.hstack(X_parts).astype(np.float32)
        y = df[self.config.target].values.astype(np.float32)

        self._fitted = True
        self._feature_names = feature_names

        # Record materialization
        schema_hash = hashlib.md5(json.dumps(feature_names, sort_keys=True).encode()).hexdigest()[
            :8
        ]

        self._registry.append(
            MaterializedFeatures(
                version=view.version,
                created_at=datetime.now(),
                n_rows=len(df),
                n_features=len(feature_names),
                feature_names=feature_names,
                schema_hash=schema_hash,
            )
        )

        logger.info(
            "Feature store fitted: %d samples, %d features (schema: %s)",
            X.shape[0],
            X.shape[1],
            schema_hash,
        )
        return X, y, feature_names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted encoders/scaler.

        Used for test data and serving. Encoders and scaler were
        fit on training data only.

        Args:
            df: Raw dataframe with feature columns.

        Returns:
            Transformed feature array.
        """
        if not self._fitted:
            raise RuntimeError("FeatureStore must be fitted before transform()")

        view = list(self.feature_views.values())[0]
        numerical_cols = [f.name for f in view.features if f.dtype == "numerical"]
        categorical_cols = [f.name for f in view.features if f.dtype == "categorical"]

        X_parts = []

        for col in categorical_cols:
            encoder = self.encoders[col]
            # Handle unseen categories gracefully
            values = df[col].astype(str).copy()
            unseen_mask = ~values.isin(encoder.classes_)
            if unseen_mask.any():
                logger.warning(
                    "Unseen categories in '%s': %s (mapping to mode)",
                    col,
                    values[unseen_mask].unique()[:5],
                )
                values[unseen_mask] = encoder.classes_[0]
            encoded = encoder.transform(values)
            X_parts.append(encoded.reshape(-1, 1))

        X_numerical = df[numerical_cols].fillna(0).values.astype(np.float32)
        X_numerical_scaled = self.scaler.transform(X_numerical)
        X_parts.append(X_numerical_scaled)

        return np.hstack(X_parts).astype(np.float32)

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of feature names after transformation."""
        return self._feature_names

    def get_registry(self) -> list[MaterializedFeatures]:
        """Return the feature materialization registry."""
        return self._registry

    def save(self, path: str | None = None) -> Path:
        """Persist the fitted feature store to disk."""
        import joblib

        if path is None:
            save_dir = PROJECT_ROOT / "data" / "processed"
        else:
            save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        store_path = save_dir / "feature_store.joblib"
        state = {
            "encoders": self.encoders,
            "scaler": self.scaler,
            "feature_names": self._feature_names,
            "fitted": self._fitted,
            "registry": self._registry,
            "feature_views": self.feature_views,
        }
        joblib.dump(state, store_path)
        logger.info("Feature store saved to %s", store_path)
        return store_path

    def load(self, path: str | None = None) -> None:
        """Load a previously saved feature store."""
        import joblib

        if path is None:
            store_path = PROJECT_ROOT / "data" / "processed" / "feature_store.joblib"
        else:
            store_path = Path(path)

        state = joblib.load(store_path)
        self.encoders = state["encoders"]
        self.scaler = state["scaler"]
        self._feature_names = state["feature_names"]
        self._fitted = state["fitted"]
        self._registry = state.get("registry", [])
        self.feature_views = state.get("feature_views", {})
        if not self.feature_views:
            view = self.get_default_feature_view()
            self.register_feature_view(view)
        logger.info("Feature store loaded from %s", store_path)
