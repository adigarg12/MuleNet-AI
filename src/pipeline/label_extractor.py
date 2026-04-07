"""
Extract per-account fraud-ratio labels directly from a live graph.

Replaces the duplicated _extract_account_labels() in run.py and api/main.py.
Works from the graph's edge data (each edge stores a 'transactions' list)
so no re-reading of files is needed during incremental retraining.
"""

import networkx as nx
from typing import Dict


def extract_labels_from_graph(G: nx.DiGraph) -> Dict[str, float]:
    """
    Build {account_id: fraud_ratio} labels in [0, 1] from live graph edges.

    Each edge stores a list of raw transaction dicts under G[u][v]['transactions'].
    An account's label = (# fraud transactions it touched) / (# total transactions).
    """
    fraud_count: Dict[str, int] = {}
    total_count: Dict[str, int] = {}

    for u, v, data in G.edges(data=True):
        for txn in data.get("transactions", []):
            for acc in (txn.get("from_account"), txn.get("to_account")):
                if acc is None:
                    continue
                total_count[acc] = total_count.get(acc, 0) + 1
                if txn.get("label") in ("fraud", 1, "1"):
                    fraud_count[acc] = fraud_count.get(acc, 0) + 1

    if not total_count:
        return {}

    return {a: fraud_count.get(a, 0) / total_count[a] for a in total_count}
