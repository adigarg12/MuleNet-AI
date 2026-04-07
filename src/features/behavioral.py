"""
Behavioural features that capture structural patterns specific to fraud.
"""

import networkx as nx
from typing import Dict, Any, List, Optional


def compute_fan_out_ratio(G: nx.DiGraph, node: str) -> float:
    """
    out_degree / (in_degree + 1).
    High ratio → disbursement node (smurfing source or mule).
    """
    out_d = G.out_degree(node)
    in_d  = G.in_degree(node)
    return out_d / (in_d + 1)


def compute_fan_in_ratio(G: nx.DiGraph, node: str) -> float:
    """
    in_degree / (out_degree + 1).
    High ratio → aggregation node (collection account).
    """
    in_d  = G.in_degree(node)
    out_d = G.out_degree(node)
    return in_d / (out_d + 1)


def compute_flow_depth(G: nx.DiGraph, node: str) -> int:
    """
    Maximum shortest-path length from any source node to this node.
    Deep nodes are buried in layering chains.
    """
    try:
        # Returns dict {source: length} in NetworkX 3.x
        lengths = dict(nx.single_target_shortest_path_length(G, node))
        depths = list(lengths.values())
        return max(depths) if depths else 0
    except (nx.NetworkXError, Exception):
        return 0


def compute_cross_channel_jumps(
    G: nx.DiGraph,
    node: str,
) -> int:
    """
    Count distinct channel transitions in transactions involving this node.
    E.g. ACH→P2P→WIRE counts as 2 jumps.
    A high cross-channel count suggests deliberate obfuscation.
    """
    txns: List[Dict[str, Any]] = []
    for _, _, data in G.out_edges(node, data=True):
        txns.extend(data.get("transactions", []))
    for _, _, data in G.in_edges(node, data=True):
        txns.extend(data.get("transactions", []))

    if not txns:
        return 0

    sorted_txns = sorted(txns, key=lambda t: t["timestamp"])
    channels = [t["channel"] for t in sorted_txns]
    jumps = sum(1 for i in range(1, len(channels)) if channels[i] != channels[i - 1])
    return jumps


def compute_amount_concentration(G: nx.DiGraph, node: str) -> float:
    """
    Gini-like measure of how concentrated outbound amounts are.
    0 = uniform dispersion, 1 = single large payment.
    """
    out_amounts = [
        data["total_amount"]
        for _, _, data in G.out_edges(node, data=True)
    ]
    if not out_amounts or len(out_amounts) == 1:
        return 0.0

    total = sum(out_amounts)
    if total == 0:
        return 0.0

    n = len(out_amounts)
    sorted_a = sorted(out_amounts)
    cumsum = 0.0
    gini_sum = 0.0
    for i, a in enumerate(sorted_a, 1):
        cumsum += a
        gini_sum += (2 * i - n - 1) * a
    return gini_sum / (n * total)


def get_behavioral_features(G: nx.DiGraph, node: str) -> Dict[str, float]:
    """Return all behavioural features for a single node."""
    return {
        "fan_out_ratio":        compute_fan_out_ratio(G, node),
        "fan_in_ratio":         compute_fan_in_ratio(G, node),
        "flow_depth":           float(compute_flow_depth(G, node)),
        "cross_channel_jumps":  float(compute_cross_channel_jumps(G, node)),
        "amount_concentration": compute_amount_concentration(G, node),
    }
