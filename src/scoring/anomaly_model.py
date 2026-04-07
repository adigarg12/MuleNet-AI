"""
Isolation Forest anomaly model.

Train once on a baseline feature matrix; score new accounts as additive boost.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.ensemble import IsolationForest
import yaml

_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "thresholds.yaml"
)

FEATURE_ORDER = [
    "in_degree",
    "out_degree",
    "betweenness",
    "pagerank",
    "clustering",
    "velocity",
    "retention_time",
    "burst_score",
    "fan_out_ratio",
    "fan_in_ratio",
    "flow_depth",
    "cross_channel_jumps",
    "amount_concentration",
]


def _load_config() -> Dict[str, Any]:
    with open(_THRESHOLDS_PATH, "r") as f:
        return yaml.safe_load(f)


def _features_to_vector(features: Dict[str, float]) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in FEATURE_ORDER], dtype=float)


class AnomalyModel:
    def __init__(self):
        cfg = _load_config()
        anomaly_cfg = cfg.get("anomaly", {})
        self._weight: float = anomaly_cfg.get("isolation_forest_weight", 0.15)
        self._contamination: float = anomaly_cfg.get("contamination", 0.05)
        self._model: Optional[IsolationForest] = None
        self._trained: bool = False

    def train(self, feature_dicts: List[Dict[str, float]]) -> None:
        """Fit Isolation Forest on a list of feature dicts."""
        if not feature_dicts:
            return
        X = np.array([_features_to_vector(f) for f in feature_dicts])
        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
        )
        self._model.fit(X)
        self._trained = True

    def anomaly_boost(self, features: Dict[str, float]) -> float:
        """
        Return an additive score boost in [0, weight].
        Returns 0.0 if the model has not been trained yet.
        """
        if not self._trained or self._model is None:
            return 0.0
        vec = _features_to_vector(features).reshape(1, -1)
        # score_samples returns negative values; more negative = more anomalous
        raw_score = self._model.score_samples(vec)[0]
        # Map to [0, 1]: typical range is roughly [-0.5, 0.5]
        # Anomalous (negative) → high boost; normal (positive) → low boost
        normalised = max(0.0, min(1.0, (0.5 - raw_score)))
        return round(normalised * self._weight, 4)

    def is_trained(self) -> bool:
        return self._trained


# Module-level singleton
anomaly_model = AnomalyModel()
