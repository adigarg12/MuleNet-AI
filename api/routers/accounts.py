"""
GET /accounts/{account_id}/risk — score and explain a single account via GNN.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import RiskResponse, FeatureScoreItem
from src.graph.graph_store import graph_store
from src.gnn.embedder import gnn_embedder
from src.scoring.normalizer import build_risk_result
from src.network.community_detection import detect_communities
from src.network.cluster_scorer import score_cluster
from src.explainability.report_generator import generate_report

router = APIRouter(prefix="/accounts", tags=["accounts"])


def _score_account(account_id: str) -> dict:
    G = graph_store.get_graph()

    if account_id not in G:
        raise HTTPException(
            status_code=404,
            detail=f"Account {account_id!r} not found in graph.",
        )

    # GNN risk score + feature importances (replaces WeightedScorer + AnomalyModel)
    gnn_score     = gnn_embedder.get_risk_score(account_id)
    contributions = gnn_embedder.explain_node(account_id)

    risk_result = build_risk_result(
        account_id    = account_id,
        weighted_score= gnn_score,
        anomaly_boost = 0.0,
        contributions = contributions,
        missing       = [] if gnn_embedder.is_trained() else ["gnn_not_trained"],
    )

    # Cluster lookup (unchanged — still graph-topology-based)
    cluster_info = None
    try:
        communities = detect_communities(G)
        for idx, members in enumerate(communities):
            if account_id in members:
                member_scores = {m: gnn_embedder.get_risk_score(m) for m in members}
                cluster_info  = score_cluster(G, members, member_scores)
                cluster_info["community_id"] = idx
                cluster_info["members"]      = sorted(members)
                break
    except Exception:
        pass

    text_report = generate_report(
        account_id, risk_result, cluster_info,
        features=contributions,
    )
    risk_result["text_report"] = text_report
    risk_result["cluster"]     = cluster_info
    return risk_result


@router.get("/{account_id}/risk", response_model=RiskResponse)
def get_account_risk(account_id: str):
    """Return GNN risk score and feature attribution for a single account."""
    result = _score_account(account_id)
    return RiskResponse(
        account_id      = result["account_id"],
        risk_score      = result["risk_score"],
        tier            = result["tier"],
        weighted_score  = result["weighted_score"],
        anomaly_boost   = result["anomaly_boost"],
        top_drivers     = [
            FeatureScoreItem(feature=d["feature"], score=d["score"])
            for d in result.get("top_drivers", [])
        ],
        cluster         = result.get("cluster"),
        text_report     = result.get("text_report"),
        missing_features= result.get("missing_features", []),
    )
