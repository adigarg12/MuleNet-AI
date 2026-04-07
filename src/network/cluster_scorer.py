"""
Aggregate cluster-level risk scores and tag mule rings.
"""

import networkx as nx
from typing import Dict, Any, List, Set

from src.network.subgraph_extractor import has_suspicious_cycle, extract_subgraph
import yaml
import os

_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "thresholds.yaml"
)


def _load_cluster_cfg() -> Dict[str, Any]:
    with open(_THRESHOLDS_PATH, "r") as f:
        return yaml.safe_load(f).get("cluster", {})


def score_cluster(
    G: nx.DiGraph,
    members: Set[str],
    risk_scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute cluster-level risk metrics.

    Returns:
        {
            size, avg_risk, max_risk, density_weighted_risk,
            has_cycle, is_mule_ring, cluster_risk_score
        }
    """
    cfg = _load_cluster_cfg()
    scores = [risk_scores.get(m, 0.0) for m in members]
    avg_risk = sum(scores) / len(scores) if scores else 0.0
    max_risk = max(scores) if scores else 0.0

    sub = extract_subgraph(G, members)
    density = nx.density(sub)
    density_weighted = avg_risk * (1 + density)  # denser clusters are riskier

    has_cycle = has_suspicious_cycle(sub)

    # Fan-out check: count nodes with high fan-out
    high_fanout_nodes = [
        n for n in members
        if sub.out_degree(n) > 3 and sub.in_degree(n) <= 2
    ]
    fanout_ratio = len(high_fanout_nodes) / len(members) if members else 0.0

    mule_fanout_thresh = cfg.get("mule_ring_fan_out", 0.70)
    require_cycle = cfg.get("mule_ring_cycle", True)

    is_mule_ring = (
        fanout_ratio >= mule_fanout_thresh
        and (has_cycle if require_cycle else True)
    )

    cluster_risk = min(1.0, density_weighted)

    return {
        "size":                  len(members),
        "avg_risk":              round(avg_risk, 4),
        "max_risk":              round(max_risk, 4),
        "density":               round(density, 4),
        "density_weighted_risk": round(density_weighted, 4),
        "has_cycle":             has_cycle,
        "fanout_ratio":          round(fanout_ratio, 4),
        "is_mule_ring":          is_mule_ring,
        "cluster_risk_score":    round(cluster_risk, 4),
    }


def score_all_clusters(
    G: nx.DiGraph,
    communities: List[Set[str]],
    risk_scores: Dict[str, float],
    min_members: int = 3,
) -> List[Dict[str, Any]]:
    """Score every community and return sorted list."""
    results = []
    for idx, members in enumerate(communities):
        if len(members) < min_members:
            continue
        info = score_cluster(G, members, risk_scores)
        info["community_id"] = idx
        info["members"] = sorted(members)
        results.append(info)
    return sorted(results, key=lambda x: x["cluster_risk_score"], reverse=True)
