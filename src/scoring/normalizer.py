"""
Score normalizer + risk tier assignment.
Maps combined raw score → [0, 1] and assigns a risk tier label.
"""

import os
import yaml
from typing import Dict, Any, Tuple

_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "thresholds.yaml"
)


def _load_thresholds() -> Dict[str, list]:
    with open(_THRESHOLDS_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["risk_tiers"]


def normalize_score(raw_score: float) -> float:
    """Clamp score to [0, 1]."""
    return max(0.0, min(1.0, raw_score))


def assign_tier(score: float) -> str:
    """
    Map normalised score to a risk tier string.
    Tiers are ordered from critical → low to ensure boundaries work correctly.
    """
    tiers = _load_thresholds()
    for tier in ["critical", "high", "medium", "low"]:
        lo, hi = tiers[tier]
        if lo <= score <= hi:
            return tier.upper()
    return "LOW"


def build_risk_result(
    account_id: str,
    weighted_score: float,
    anomaly_boost: float,
    contributions: Dict[str, float],
    missing: list,
) -> Dict[str, Any]:
    """
    Combine weighted score + anomaly boost, normalise, tier, and package result.
    """
    combined = normalize_score(weighted_score + anomaly_boost)
    tier = assign_tier(combined)

    # Sort contributions by value descending for report readability
    top_drivers = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    return {
        "account_id":     account_id,
        "risk_score":     combined,
        "tier":           tier,
        "weighted_score": round(weighted_score, 4),
        "anomaly_boost":  round(anomaly_boost, 4),
        "top_drivers":    [{"feature": k, "score": v} for k, v in top_drivers[:5]],
        "missing_features": missing,
    }
