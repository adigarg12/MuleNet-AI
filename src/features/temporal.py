"""
Temporal features derived from transaction timestamps.

Inputs are either:
  - A list of transaction dicts (for node-level queries)
  - A NetworkX DiGraph (for graph-traversal queries)
"""

import statistics
from typing import List, Dict, Any, Optional
import networkx as nx


def _node_txns(G: nx.DiGraph, node: str) -> List[Dict[str, Any]]:
    """Collect all transactions (in + out) touching a node."""
    txns: List[Dict[str, Any]] = []
    for _, _, data in G.out_edges(node, data=True):
        txns.extend(data.get("transactions", []))
    for _, _, data in G.in_edges(node, data=True):
        txns.extend(data.get("transactions", []))
    return txns


def compute_velocity(
    txns: List[Dict[str, Any]],
    window_seconds: float = 600.0,
    node: Optional[str] = None,
) -> float:
    """
    Transactions per second within the most recent rolling window.
    If node is given, filter to only transactions involving that node.
    """
    if not txns:
        return 0.0

    filtered = txns
    if node:
        filtered = [
            t for t in txns
            if t.get("from_account") == node or t.get("to_account") == node
        ]
    if not filtered:
        return 0.0

    max_ts = max(t["timestamp"] for t in filtered)
    window_txns = [t for t in filtered if t["timestamp"] >= max_ts - window_seconds]
    if window_seconds <= 0:
        return float(len(window_txns))
    return len(window_txns) / window_seconds


def compute_avg_retention_time(G: nx.DiGraph, node: str) -> float:
    """
    Average time (seconds) between receiving money and sending it on.
    Short retention (< 60s) is a strong mule indicator.
    Returns 0.0 if the node has no in+out edges.
    """
    in_times = [
        t["timestamp"]
        for _, _, data in G.in_edges(node, data=True)
        for t in data.get("transactions", [])
    ]
    out_times = [
        t["timestamp"]
        for _, _, data in G.out_edges(node, data=True)
        for t in data.get("transactions", [])
    ]
    if not in_times or not out_times:
        return 0.0

    avg_in  = sum(in_times)  / len(in_times)
    avg_out = sum(out_times) / len(out_times)
    return max(0.0, avg_out - avg_in)


def compute_burst_score(txns: List[Dict[str, Any]]) -> float:
    """
    Coefficient of variation of inter-arrival times.
    High burst score → irregular timing → bot-like or adversarial behaviour.
    Returns 0.0 when there are fewer than 2 transactions.
    """
    if len(txns) < 2:
        return 0.0

    sorted_ts = sorted(t["timestamp"] for t in txns)
    inter_arrivals = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]

    if len(inter_arrivals) < 2:
        return 0.0

    mean = statistics.mean(inter_arrivals)
    if mean == 0:
        return 1.0  # all simultaneous — maximum burst
    stdev = statistics.stdev(inter_arrivals)
    return stdev / mean  # coefficient of variation


def get_temporal_features(
    G: nx.DiGraph,
    node: str,
    window_seconds: float = 600.0,
) -> Dict[str, float]:
    """Return all temporal features for a single node."""
    txns = _node_txns(G, node)
    return {
        "velocity":         compute_velocity(txns, window_seconds, node),
        "retention_time":   compute_avg_retention_time(G, node),
        "burst_score":      compute_burst_score(txns),
        "txn_count":        float(len(txns)),
    }
