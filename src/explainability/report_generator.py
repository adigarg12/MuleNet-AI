"""
Human-readable risk report generator.

Produces both a structured dict and a formatted text report per account.
"""

from typing import Dict, Any, List, Optional


TIER_COLORS = {
    "LOW":      "  [LOW]     ",
    "MEDIUM":   "  [MEDIUM]  ",
    "HIGH":     "  [HIGH]    ",
    "CRITICAL": "  [CRITICAL]",
}

FEATURE_LABELS = {
    # GNN input feature importances (gradient × input attribution)
    "in_degree":         "Inbound connection count",
    "out_degree":        "Outbound connection count",
    "txn_count":         "Total transaction volume",
    "total_sent":        "Total amount sent",
    "total_received":    "Total amount received",
    "channel_diversity": "Payment channel diversity",
    "time_span":         "Activity time span",
    "avg_amount":        "Average transaction amount",
    # Legacy hand-crafted labels kept for backward compatibility
    "velocity":          "Transaction velocity",
    "fan_out":           "Fan-out ratio",
    "centrality":        "Betweenness centrality",
    "retention_time":    "Money retention time",
    "pagerank":          "PageRank influence",
    "cross_channel":     "Cross-channel jumps",
    "burst_score":       "Burst / timing anomaly",
}


def generate_report(
    account_id: str,
    risk_result: Dict[str, Any],
    cluster_info: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a human-readable text report for a single account.

    Args:
        account_id:  Account identifier
        risk_result: Output from normalizer.build_risk_result()
        cluster_info: Optional cluster metadata (from cluster_scorer)
        features:    Raw feature values for additional context

    Returns:
        Formatted multi-line string
    """
    score = risk_result["risk_score"]
    tier  = risk_result["tier"]
    tier_label = TIER_COLORS.get(tier, f"  [{tier}]     ")

    lines = [
        "─" * 60,
        f"  Account: {account_id:<20} Risk Score: {score:.2f}{tier_label}",
        "─" * 60,
        "  Top Risk Drivers:",
    ]

    for driver in risk_result.get("top_drivers", [])[:5]:
        feat  = driver["feature"]
        fscore = driver["score"]
        label  = FEATURE_LABELS.get(feat, feat)

        # Add contextual hint from raw features if available
        hint = ""
        if features:
            if feat == "velocity" and "velocity" in features:
                hint = f"  ({features['velocity']:.3f} txns/s)"
            elif feat == "fan_out" and "fan_out_ratio" in features:
                hint = f"  (ratio: {features['fan_out_ratio']:.1f})"
            elif feat == "retention_time" and "retention_time" in features:
                secs = features["retention_time"]
                hint = f"  (avg hold: {secs:.0f}s)"
            elif feat == "cross_channel" and "cross_channel_jumps" in features:
                hint = f"  ({int(features['cross_channel_jumps'])} jumps)"

        bar = _score_bar(fscore)
        lines.append(f"    • {label:<28} {fscore:.2f}  {bar}{hint}")

    if cluster_info:
        cid   = cluster_info.get("community_id", "?")
        csize = cluster_info.get("size", "?")
        crisk = cluster_info.get("cluster_risk_score", 0.0)
        tag   = "  *** MULE RING ***" if cluster_info.get("is_mule_ring") else ""
        lines.append("")
        lines.append(
            f"  Cluster #{cid}  ({csize} members, cluster risk: {crisk:.2f}){tag}"
        )

    missing = risk_result.get("missing_features", [])
    if missing:
        lines.append(f"  [!] Missing features: {', '.join(missing)}")

    lines.append("─" * 60)
    return "\n".join(lines)


def _score_bar(score: float, width: int = 10) -> str:
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def generate_report_dict(
    account_id: str,
    risk_result: Dict[str, Any],
    cluster_info: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Return the report as a structured dict (for API responses / JSON).
    """
    return {
        "account_id":    account_id,
        "risk_score":    risk_result["risk_score"],
        "tier":          risk_result["tier"],
        "top_drivers":   risk_result.get("top_drivers", []),
        "cluster":       cluster_info,
        "raw_features":  features,
        "text_report":   generate_report(account_id, risk_result, cluster_info, features),
    }
