"""
Structural graph features per node.

All functions accept a NetworkX DiGraph.
Expensive computations (betweenness, PageRank) are cached on the graph object
and invalidated when the graph changes.
"""

import functools
import networkx as nx
from typing import Dict


# ------------------------------------------------------------------
# Simple per-node metrics
# ------------------------------------------------------------------

def compute_in_degree(G: nx.DiGraph, node: str) -> int:
    return G.in_degree(node)


def compute_out_degree(G: nx.DiGraph, node: str) -> int:
    return G.out_degree(node)


def compute_total_degree(G: nx.DiGraph, node: str) -> int:
    return G.in_degree(node) + G.out_degree(node)


# ------------------------------------------------------------------
# Graph-wide metrics (cached as graph attributes)
# ------------------------------------------------------------------

def _cache_key(G: nx.DiGraph, metric: str) -> str:
    return f"_cache_{metric}_{G.number_of_edges()}"


def compute_betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """
    Betweenness centrality for all nodes.
    Cached on the graph object; recomputed when edge count changes.
    """
    key = _cache_key(G, "betweenness")
    if not hasattr(G, "_feature_cache"):
        G._feature_cache = {}
    if key not in G._feature_cache:
        G._feature_cache = {key: nx.betweenness_centrality(G, normalized=True)}
    return G._feature_cache[key]


def compute_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> Dict[str, float]:
    """PageRank for all nodes. Cached similarly."""
    key = _cache_key(G, f"pagerank_{alpha}")
    if not hasattr(G, "_feature_cache"):
        G._feature_cache = {}
    if key not in G._feature_cache:
        G._feature_cache[key] = nx.pagerank(G, alpha=alpha)
    return G._feature_cache[key]


def compute_clustering_coefficient(G: nx.DiGraph, node: str) -> float:
    """
    Local clustering coefficient.
    Uses undirected projection because nx.clustering needs undirected graph.
    """
    UG = G.to_undirected()
    return nx.clustering(UG, node)


def get_structural_features(G: nx.DiGraph, node: str) -> Dict[str, float]:
    """
    Return all structural features for a single node as a flat dict.
    """
    bc = compute_betweenness_centrality(G)
    pr = compute_pagerank(G)
    return {
        "in_degree":    float(compute_in_degree(G, node)),
        "out_degree":   float(compute_out_degree(G, node)),
        "total_degree": float(compute_total_degree(G, node)),
        "betweenness":  bc.get(node, 0.0),
        "pagerank":     pr.get(node, 0.0),
        "clustering":   compute_clustering_coefficient(G, node),
    }
