"""
Raw node feature extractor — O(1) per node, no expensive graph algorithms.

These 8 features are the GNN's input. The GNN learns structural patterns
(what hand-crafted centrality/PageRank/fan-out captured) via message passing
over the neighbourhood, so no manual computation is needed.
"""

import networkx as nx
import numpy as np
from typing import Dict, List


FEATURE_NAMES: List[str] = [
    "in_degree",        # how many unique accounts send to this node
    "out_degree",       # how many unique accounts this node sends to
    "txn_count",        # total transactions in + out
    "total_sent",       # sum of outbound amounts
    "total_received",   # sum of inbound amounts
    "channel_diversity",# number of distinct payment channels used
    "time_span",        # seconds between first and last transaction
    "avg_amount",       # average transaction amount
]


def extract_raw_features(G: nx.DiGraph, node: str) -> Dict[str, float]:
    """Extract raw node features from graph edges."""
    in_edges  = list(G.in_edges(node,  data=True))
    out_edges = list(G.out_edges(node, data=True))

    in_degree  = float(len(in_edges))
    out_degree = float(len(out_edges))

    all_txns = []
    for _, _, d in out_edges:
        all_txns.extend(d.get("transactions", []))
    for _, _, d in in_edges:
        all_txns.extend(d.get("transactions", []))

    txn_count = float(len(all_txns))

    total_sent     = sum(d.get("total_amount", 0.0) for _, _, d in out_edges)
    total_received = sum(d.get("total_amount", 0.0) for _, _, d in in_edges)

    channels   = {t["channel"] for t in all_txns if "channel" in t}
    timestamps = [t["timestamp"] for t in all_txns if "timestamp" in t]

    channel_diversity = float(len(channels))
    time_span = float(max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
    avg_amount = (total_sent + total_received) / txn_count if txn_count > 0 else 0.0

    return {
        "in_degree":         in_degree,
        "out_degree":        out_degree,
        "txn_count":         txn_count,
        "total_sent":        total_sent,
        "total_received":    total_received,
        "channel_diversity": channel_diversity,
        "time_span":         time_span,
        "avg_amount":        avg_amount,
    }


def extract_feature_matrix(G: nx.DiGraph, node_order: List[str]) -> np.ndarray:
    """
    Returns (N, 8) float32 array of raw features for nodes in node_order.
    Applies log1p normalisation to suppress large-magnitude skew.
    """
    rows = []
    for node in node_order:
        f = extract_raw_features(G, node)
        rows.append([f[k] for k in FEATURE_NAMES])

    x = np.array(rows, dtype=np.float32)

    # log1p: degree, counts, amounts, time
    for col in range(len(FEATURE_NAMES)):
        if FEATURE_NAMES[col] == "channel_diversity":
            x[:, col] = x[:, col] / 5.0   # max 5 channels → [0, 1]
        else:
            x[:, col] = np.log1p(x[:, col])

    return x
