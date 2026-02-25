"""Performance tracking and business metrics monitoring.

Tracks model performance over time, prediction distributions,
and business KPIs (churn rate, revenue at risk) for the
monitoring dashboard.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks model performance and business metrics over time."""

    def __init__(self) -> None:
        self.prediction_log: list[dict[str, Any]] = []
        self.performance_snapshots: list[dict[str, Any]] = []

    def log_prediction(
        self,
        user_id: str,
        probability: float,
        prediction: bool,
        actual: bool | None = None,
        model_name: str = "",
        model_version: str = "",
    ) -> None:
        """Log a single prediction for tracking."""
        self.prediction_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "probability": probability,
            "prediction": prediction,
            "actual": actual,
            "model_name": model_name,
            "model_version": model_version,
        })

    def compute_business_metrics(
        self,
        predictions_df: pd.DataFrame,
        mrr_column: str = "mrr",
    ) -> dict[str, Any]:
        """Compute business-level metrics from predictions.

        Args:
            predictions_df: DataFrame with predictions and MRR data.
            mrr_column: Column name for monthly recurring revenue.

        Returns:
            Business metrics dictionary.
        """
        total_users = len(predictions_df)
        predicted_churners = predictions_df[predictions_df["churn_prediction"]].copy()

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_users": total_users,
            "predicted_churn_count": len(predicted_churners),
            "predicted_churn_rate": len(predicted_churners) / max(total_users, 1),
        }

        if mrr_column in predictions_df.columns:
            total_mrr = predictions_df[mrr_column].sum()
            at_risk_mrr = predicted_churners[mrr_column].sum() if len(predicted_churners) > 0 else 0
            metrics["total_mrr"] = float(total_mrr)
            metrics["revenue_at_risk"] = float(at_risk_mrr)
            metrics["revenue_at_risk_pct"] = float(at_risk_mrr / max(total_mrr, 1))

        # Risk tier distribution
        if "churn_probability" in predictions_df.columns:
            probs = predictions_df["churn_probability"]
            metrics["risk_distribution"] = {
                "low": int((probs < 0.3).sum()),
                "medium": int(((probs >= 0.3) & (probs < 0.6)).sum()),
                "high": int(((probs >= 0.6) & (probs < 0.8)).sum()),
                "critical": int((probs >= 0.8).sum()),
            }
            metrics["avg_churn_probability"] = float(probs.mean())
            metrics["median_churn_probability"] = float(probs.median())

        return metrics

    def get_prediction_distribution(self) -> dict[str, Any]:
        """Analyze the distribution of recent predictions."""
        if not self.prediction_log:
            return {"count": 0}

        probs = [p["probability"] for p in self.prediction_log]
        return {
            "count": len(probs),
            "mean": float(np.mean(probs)),
            "std": float(np.std(probs)),
            "median": float(np.median(probs)),
            "p10": float(np.percentile(probs, 10)),
            "p90": float(np.percentile(probs, 90)),
            "histogram": np.histogram(probs, bins=10, range=(0, 1))[0].tolist(),
        }

    def save_snapshot(self, metrics: dict[str, Any], path: str | None = None) -> Path:
        """Save a performance snapshot."""
        if path is None:
            output_dir = PROJECT_ROOT / "data" / "processed"
            output_dir.mkdir(parents=True, exist_ok=True)
            path = str(output_dir / "performance_snapshot.json")

        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info("Performance snapshot saved to %s", path)
        return Path(path)
