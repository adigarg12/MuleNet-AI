"""
Graph builder — converts transaction dicts into a NetworkX DiGraph.

Nodes  : account IDs (strings)
Edges  : transactions (directed from_account → to_account)
         edge attributes: txn_id, amount, channel, timestamp, label
"""

import networkx as nx
from typing import Dict, Any, List


def build_graph(transactions: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Build a fresh DiGraph from a list of transaction dicts.
    Multiple transactions between the same pair create parallel edges
    (stored as a MultiDiGraph would), but here we use a DiGraph and
    accumulate edge attribute lists so all data is preserved.
    """
    G = nx.DiGraph()
    for txn in transactions:
        _add_transaction(G, txn)
    return G


def _add_transaction(G: nx.DiGraph, txn: Dict[str, Any]) -> None:
    src = txn["from_account"]
    dst = txn["to_account"]

    # Ensure nodes exist with metadata dict
    if src not in G:
        G.add_node(src, transactions=[])
    if dst not in G:
        G.add_node(dst, transactions=[])

    if G.has_edge(src, dst):
        # Accumulate multiple transactions on the same edge
        G[src][dst]["transactions"].append(txn)
        G[src][dst]["total_amount"] += txn["amount"]
        G[src][dst]["count"] += 1
        G[src][dst]["last_timestamp"] = max(
            G[src][dst]["last_timestamp"], txn["timestamp"]
        )
    else:
        G.add_edge(
            src,
            dst,
            transactions=[txn],
            total_amount=txn["amount"],
            count=1,
            first_timestamp=txn["timestamp"],
            last_timestamp=txn["timestamp"],
        )


def add_transaction(G: nx.DiGraph, txn: Dict[str, Any]) -> nx.DiGraph:
    """Incrementally add a single transaction to an existing graph."""
    _add_transaction(G, txn)
    return G


def graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Return basic stats about the current graph state."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "total_transactions": sum(
            d.get("count", 1) for _, _, d in G.edges(data=True)
        ),
        "is_directed": G.is_directed(),
    }
