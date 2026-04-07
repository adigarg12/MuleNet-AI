"""
GET /clusters/suspicious — list flagged suspicious clusters.
GET /graph/stats          — graph-level summary statistics.
"""

from fastapi import APIRouter, Query
from typing import List

from api.schemas import ClusterResponse, GraphStatsResponse
from src.graph.graph_store import graph_store
from src.graph.graph_builder import graph_summary
from src.network.community_detection import detect_communities
from src.network.cluster_scorer import score_all_clusters
from src.pipeline.state import pipeline_state
from src.gnn.embedder import gnn_embedder

router = APIRouter(tags=["clusters"])


@router.get("/clusters/suspicious", response_model=List[ClusterResponse])
def get_suspicious_clusters(
    min_risk: float = Query(0.60, ge=0.0, le=1.0, description="Minimum cluster risk score"),
    min_members: int = Query(3, ge=2, description="Minimum cluster size"),
):
    """Return suspicious clusters above the given risk threshold."""
    # Serve from pipeline state cache (populated after each retrain cycle)
    cached = pipeline_state.get_clusters()
    if cached:
        flagged = [
            c for c in cached
            if c["cluster_risk_score"] >= min_risk and len(c.get("members", [])) >= min_members
        ]
        return [_to_response(c) for c in flagged]

    # Fallback: live computation if pipeline hasn't run yet
    G = graph_store.get_graph()
    if G.number_of_nodes() == 0:
        return []

    all_scores  = gnn_embedder.get_all_scores()
    communities = detect_communities(G)
    all_clusters = score_all_clusters(G, communities, all_scores, min_members=min_members)
    flagged = [c for c in all_clusters if c["cluster_risk_score"] >= min_risk]
    return [_to_response(c) for c in flagged]


def _to_response(c: dict) -> ClusterResponse:
    return ClusterResponse(
        community_id      = c["community_id"],
        members           = c["members"],
        size              = c["size"],
        avg_risk          = c["avg_risk"],
        max_risk          = c["max_risk"],
        density           = c["density"],
        cluster_risk_score= c["cluster_risk_score"],
        is_mule_ring      = c["is_mule_ring"],
        has_cycle         = c["has_cycle"],
    )


@router.get("/graph/stats", response_model=GraphStatsResponse)
def get_graph_stats():
    """Return summary statistics about the current transaction graph."""
    G = graph_store.get_graph()
    stats = graph_summary(G)
    return GraphStatsResponse(**stats)
