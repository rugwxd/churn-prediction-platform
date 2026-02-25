"""Data and model drift detection using Evidently AI.

Monitors feature distributions and model performance over time,
generating alerts when statistical drift exceeds thresholds.
Critical for maintaining model reliability in production.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import MonitoringConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


def _try_import_evidently():
    """Import evidently with graceful fallback."""
    try:
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
        return Report, DataDriftPreset, TargetDriftPreset
    except ImportError:
        logger.warning("Evidently not installed. Install with: pip install evidently")
        return None, None, None


class DriftDetector:
    """Detects data and concept drift between reference and current data.

    Uses the Kolmogorov-Smirnov test for numerical features and
    chi-squared test for categorical features. Integrates with
    Evidently AI for comprehensive drift reports.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config
        self.reference_data: pd.DataFrame | None = None

    def set_reference(self, df: pd.DataFrame) -> None:
        """Set the reference (training) data distribution."""
        self.reference_data = df.copy()
        logger.info("Drift detector reference set: %d rows, %d columns",
                     len(df), len(df.columns))

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        numerical_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect feature drift between reference and current data.

        Uses KS-test for numerical and chi-squared for categorical features.

        Args:
            current_data: Current production data.
            numerical_cols: Numerical feature columns to check.
            categorical_cols: Categorical feature columns to check.

        Returns:
            Dict with drift detection results per feature.
        """
        if self.reference_data is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")

        from scipy import stats

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "n_reference": len(self.reference_data),
            "n_current": len(current_data),
            "features": {},
            "drift_detected": False,
            "drifted_features": [],
        }

        # Numerical drift (KS test)
        if numerical_cols:
            for col in numerical_cols:
                if col not in self.reference_data.columns or col not in current_data.columns:
                    continue

                ref_vals = self.reference_data[col].dropna().values
                cur_vals = current_data[col].dropna().values

                if len(ref_vals) == 0 or len(cur_vals) == 0:
                    continue

                stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
                is_drifted = p_value < self.config.drift_threshold

                results["features"][col] = {
                    "type": "numerical",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": is_drifted,
                    "ref_mean": float(np.mean(ref_vals)),
                    "cur_mean": float(np.mean(cur_vals)),
                    "ref_std": float(np.std(ref_vals)),
                    "cur_std": float(np.std(cur_vals)),
                }

                if is_drifted:
                    results["drifted_features"].append(col)

        # Categorical drift (Chi-squared)
        if categorical_cols:
            for col in categorical_cols:
                if col not in self.reference_data.columns or col not in current_data.columns:
                    continue

                ref_counts = self.reference_data[col].value_counts(normalize=True)
                cur_counts = current_data[col].value_counts(normalize=True)

                # Align categories
                all_cats = set(ref_counts.index) | set(cur_counts.index)
                ref_aligned = np.array([ref_counts.get(c, 0) for c in all_cats])
                cur_aligned = np.array([cur_counts.get(c, 0) for c in all_cats])

                # Avoid zero frequencies
                ref_aligned = ref_aligned + 1e-10
                cur_aligned = cur_aligned + 1e-10

                stat, p_value = stats.chisquare(cur_aligned, ref_aligned)
                is_drifted = p_value < self.config.drift_threshold

                results["features"][col] = {
                    "type": "categorical",
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": is_drifted,
                }

                if is_drifted:
                    results["drifted_features"].append(col)

        results["drift_detected"] = len(results["drifted_features"]) > 0

        if results["drift_detected"]:
            logger.warning(
                "Drift detected in %d features: %s",
                len(results["drifted_features"]), results["drifted_features"],
            )
        else:
            logger.info("No significant drift detected")

        return results

    def generate_evidently_report(
        self,
        current_data: pd.DataFrame,
        output_path: str | None = None,
    ) -> Path | None:
        """Generate a comprehensive Evidently drift report.

        Args:
            current_data: Current production data.
            output_path: Path to save HTML report.

        Returns:
            Path to the generated report, or None if Evidently unavailable.
        """
        Report, DataDriftPreset, TargetDriftPreset = _try_import_evidently()
        if Report is None:
            return None

        if self.reference_data is None:
            raise RuntimeError("Reference data not set")

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_data)

        if output_path is None:
            reports_dir = PROJECT_ROOT / "data" / "drift_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(reports_dir / f"drift_report_{timestamp}.html")

        report.save_html(output_path)
        logger.info("Evidently drift report saved to %s", output_path)
        return Path(output_path)

    def save_results(self, results: dict[str, Any], filename: str | None = None) -> Path:
        """Save drift detection results to JSON."""
        reports_dir = PROJECT_ROOT / "data" / "drift_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drift_results_{timestamp}.json"

        output_path = reports_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Drift results saved to %s", output_path)
        return output_path
