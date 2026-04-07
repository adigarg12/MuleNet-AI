"""
Extract suspicious subgraphs from the main transaction graph.
"""

import networkx as nx
from typing import List, Set, Dict, Any


def extract_subgraph(G: nx.DiGraph, members: Set[str]) -> nx.DiGraph:
    """Return the induced subgraph for a set of account IDs."""
    return G.subgraph(members).copy()


def detect_cycles(G: nx.DiGraph) -> List[List[str]]:
    """Return up to one cycle (fast O(V+E) check — avoids exponential enumeration)."""
    try:
        cycle_edges = nx.find_cycle(G)
        return [list({u for u, v in cycle_edges} | {v for u, v in cycle_edges})]
    except nx.NetworkXNoCycle:
        return []
    except Exception:
        return []


def has_suspicious_cycle(G: nx.DiGraph, min_length: int = 2) -> bool:
    """True if the graph contains any cycle of at least min_length nodes."""
    try:
        nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False
    except Exception:
        return False


def flag_suspicious_subgraphs(
    G: nx.DiGraph,
    communities: List[Set[str]],
    risk_scores: Dict[str, float],
    min_avg_risk: float = 0.60,
    min_members: int = 3,
) -> List[Dict[str, Any]]:
    """
    For each community, extract its subgraph and flag it if:
      - It has >= min_members accounts, AND
      - Average member risk score >= min_avg_risk

    Returns a list of flagged community descriptors.
    """
    flagged = []
    for idx, members in enumerate(communities):
        if len(members) < min_members:
            continue

        scores = [risk_scores.get(m, 0.0) for m in members]
        avg_risk = sum(scores) / len(scores)

        if avg_risk < min_avg_risk:
            continue

        sub = extract_subgraph(G, members)
        has_cycle = has_suspicious_cycle(sub)

        flagged.append({
            "community_id":  idx,
            "members":       sorted(members),
            "size":          len(members),
            "avg_risk":      round(avg_risk, 4),
            "max_risk":      round(max(scores), 4),
            "has_cycle":     has_cycle,
        })

    return sorted(flagged, key=lambda x: x["avg_risk"], reverse=True)
