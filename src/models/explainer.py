"""Model explainability using SHAP values.

Provides global and local feature importance explanations to
understand what drives churn predictions. Essential for stakeholder
trust and model debugging.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import shap

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


class ChurnExplainer:
    """SHAP-based model explainability for churn predictions."""

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer: shap.Explainer | None = None

    def _build_explainer(self, X_background: np.ndarray) -> shap.Explainer:
        """Build the appropriate SHAP explainer for the model type."""
        model_type = type(self.model).__name__

        if model_type == "XGBClassifier":
            return shap.TreeExplainer(self.model)
        elif model_type == "LogisticRegression":
            return shap.LinearExplainer(self.model, X_background)
        else:
            # Use KernelExplainer for neural nets (sample background)
            background = shap.sample(X_background, min(100, len(X_background)))
            return shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1],
                background,
            )

    def compute_global_importance(
        self, X: np.ndarray, max_samples: int = 500
    ) -> dict[str, float]:
        """Compute global feature importance via mean absolute SHAP values.

        Args:
            X: Feature matrix for computing SHAP values.
            max_samples: Maximum samples to use (for efficiency).

        Returns:
            Dict mapping feature name to mean |SHAP value|, sorted descending.
        """
        X_sample = X[:max_samples]
        self._explainer = self._build_explainer(X_sample)

        shap_values = self._explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        mean_abs = np.abs(shap_values).mean(axis=0)

        importance = {}
        for i, name in enumerate(self.feature_names):
            if i < len(mean_abs):
                importance[name] = float(mean_abs[i])

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        logger.info("Top 5 features: %s",
                     {k: f"{v:.4f}" for k, v in list(importance.items())[:5]})
        return importance

    def explain_prediction(
        self, X_single: np.ndarray, X_background: np.ndarray
    ) -> dict[str, float]:
        """Explain a single prediction with per-feature SHAP contributions.

        Args:
            X_single: Single sample to explain (1, n_features).
            X_background: Background dataset for the explainer.

        Returns:
            Dict mapping feature name to SHAP contribution.
        """
        if self._explainer is None:
            self._explainer = self._build_explainer(X_background)

        shap_values = self._explainer.shap_values(X_single.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        contributions = {}
        for i, name in enumerate(self.feature_names):
            if i < shap_values.shape[1]:
                contributions[name] = float(shap_values[0, i])

        return contributions
