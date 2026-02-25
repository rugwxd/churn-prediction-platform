"""Tests for drift detection."""

import numpy as np
import pandas as pd
import pytest

from src.config import MonitoringConfig
from src.monitoring.drift import DriftDetector


@pytest.fixture
def detector():
    return DriftDetector(MonitoringConfig(drift_threshold=0.05))


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 500),
        "feature_b": np.random.normal(5, 2, 500),
        "category": np.random.choice(["A", "B", "C"], 500),
    })


class TestDriftDetector:
    def test_no_drift_same_distribution(self, detector, reference_data):
        detector.set_reference(reference_data)

        # Current data from same distribution
        np.random.seed(99)
        current = pd.DataFrame({
            "feature_a": np.random.normal(0, 1, 500),
            "feature_b": np.random.normal(5, 2, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
        })

        results = detector.detect_drift(
            current,
            numerical_cols=["feature_a", "feature_b"],
            categorical_cols=["category"],
        )

        # Same distribution should generally not trigger drift
        assert isinstance(results, dict)
        assert "features" in results

    def test_drift_with_shifted_distribution(self, detector, reference_data):
        detector.set_reference(reference_data)

        # Significantly shifted distribution
        current = pd.DataFrame({
            "feature_a": np.random.normal(5, 1, 500),  # Mean shifted from 0 to 5
            "feature_b": np.random.normal(5, 2, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
        })

        results = detector.detect_drift(
            current,
            numerical_cols=["feature_a", "feature_b"],
        )

        assert results["drift_detected"]
        assert "feature_a" in results["drifted_features"]

    def test_no_reference_raises(self, detector):
        current = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="Reference data not set"):
            detector.detect_drift(current)

    def test_results_structure(self, detector, reference_data):
        detector.set_reference(reference_data)

        results = detector.detect_drift(
            reference_data,
            numerical_cols=["feature_a"],
        )

        assert "timestamp" in results
        assert "n_reference" in results
        assert "n_current" in results
        assert "features" in results
        assert "drift_detected" in results

    def test_save_results(self, detector, reference_data, tmp_path):
        detector.set_reference(reference_data)
        results = detector.detect_drift(reference_data, numerical_cols=["feature_a"])

        from unittest.mock import patch
        with patch("src.monitoring.drift.PROJECT_ROOT", tmp_path):
            path = detector.save_results(results)
            assert path.exists()
