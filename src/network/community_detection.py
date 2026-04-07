"""
Community detection using NetworkX greedy modularity communities.

NetworkX's greedy_modularity_communities works on undirected graphs,
so we project the DiGraph to undirected before running detection.
"""

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from typing import List, Set, Dict, Any


def detect_communities(G: nx.DiGraph) -> List[Set[str]]:
    """
    Detect communities in the transaction graph.
    Returns a list of sets, each set containing account IDs.
    """
    if G.number_of_nodes() == 0:
        return []

    # Undirected projection (collapses edge direction)
    UG = G.to_undirected()

    # Only works on connected components; run on each component
    communities: List[Set[str]] = []
    for component in nx.connected_components(UG):
        sub = UG.subgraph(component).copy()
        if sub.number_of_nodes() < 2:
            communities.append(set(sub.nodes()))
            continue
        try:
            comms = list(greedy_modularity_communities(sub))
            communities.extend([set(c) for c in comms])
        except Exception:
            communities.append(set(sub.nodes()))

    return communities


def community_summary(communities: List[Set[str]]) -> List[Dict[str, Any]]:
    """Return a list of dicts summarising each community."""
    return [
        {"community_id": i, "members": sorted(c), "size": len(c)}
        for i, c in enumerate(communities)
    ]
