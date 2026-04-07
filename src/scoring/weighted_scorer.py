"""
Configurable weighted risk scorer.

Reads feature weights from config/scoring_weights.yaml.
Returns a weighted score in [0, ∞) before normalization.
"""

import os
import yaml
import math
from typing import Dict, Any

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "scoring_weights.yaml"
)


def _load_weights() -> Dict[str, float]:
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return {k: float(v) for k, v in raw.items()}


def _normalize_feature(value: float, scale: float = 1.0) -> float:
    """
    Sigmoid-clamp a raw feature value into (0, 1).
    scale controls the sensitivity.
    """
    return 1.0 / (1.0 + math.exp(-value / scale))


# Feature → (extraction key, normalisation scale)
FEATURE_MAP: Dict[str, tuple] = {
    "velocity":       ("velocity",         0.01),   # txns/sec; 0.01 → mid at 100 txns/s
    "fan_out":        ("fan_out_ratio",     2.0),    # ratio; mid at fan_out ~2
    "centrality":     ("betweenness",       0.1),    # betweenness [0,1]
    "retention_time": ("retention_time",    -600.0), # shorter hold → higher risk (inverted)
    "pagerank":       ("pagerank",          0.05),   # pagerank [0,1]
    "cross_channel":  ("cross_channel_jumps", 3.0),  # count; mid at 3 jumps
    "burst_score":    ("burst_score",       1.0),    # CoV; mid at 1.0
}


class WeightedScorer:
    def __init__(self, config_path: str = _CONFIG_PATH):
        self._config_path = config_path
        self._weights = _load_weights()

    def reload_weights(self) -> None:
        self._weights = _load_weights()

    def score(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute the weighted risk score.

        Returns:
            {
                "weighted_score": float,   # raw weighted sum in [0,1]
                "contributions":  dict,    # per-feature contribution
                "missing":        list,    # features not found in input
            }
        """
        total = 0.0
        contributions: Dict[str, float] = {}
        missing = []

        for weight_key, (feat_key, scale) in FEATURE_MAP.items():
            weight = self._weights.get(weight_key, 0.0)
            raw_value = features.get(feat_key)

            if raw_value is None:
                missing.append(feat_key)
                continue

            # Invert retention_time: shorter hold = higher risk
            if weight_key == "retention_time":
                # retention_time = seconds. Invert so 0s → 1.0, 600s → ~0.5
                norm = 1.0 - _normalize_feature(raw_value, 300.0)
            else:
                norm = _normalize_feature(raw_value, abs(scale))

            contribution = weight * norm
            contributions[weight_key] = round(norm, 4)
            total += contribution

        return {
            "weighted_score": round(min(total, 1.0), 4),
            "contributions":  contributions,
            "missing":        missing,
        }
