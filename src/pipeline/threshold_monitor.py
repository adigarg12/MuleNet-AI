"""
ThresholdMonitor — checks scored clusters and fires webhook alerts for new crossings.

A cluster triggers an alert when:
  - Its cluster_risk_score >= ALERT_THRESHOLD
  - It has not already been alerted within the TTL window (24h by default)
"""

import datetime
import logging
import os
from typing import Dict, List, Any

import networkx as nx

from src.pipeline.state import pipeline_state
from src.pipeline.webhook_alerter import webhook_alerter
from src.sar.sar_store import sar_store
from src.sar.sar_generator import maybe_generate_sar

logger = logging.getLogger(__name__)

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))


def check_thresholds(scored_clusters: List[Dict], G: nx.DiGraph, all_scores: Dict[str, float]) -> List[Dict]:
    """
    Check each cluster against the alert threshold.
    Fires webhook for new crossings. Tracks flag counts. Generates SAR after N flags.
    Returns list of newly alerted clusters.
    """
    new_alerts = []
    for cluster in scored_clusters:
        if cluster["cluster_risk_score"] >= ALERT_THRESHOLD:
            fingerprint = pipeline_state.cluster_fingerprint(cluster)

            # Always increment flag count on each detection cycle
            flag_count = sar_store.increment_flag(fingerprint)

            # Fire webhook only on new crossings (deduped by TTL)
            if not pipeline_state.is_alerted(cluster):
                pipeline_state.mark_alerted(cluster)
                payload = build_alert_payload(cluster, G, all_scores)
                webhook_alerter.send_alert(payload)
                new_alerts.append(cluster)
                logger.info(
                    "ALERT fired — cluster %s | %d members | risk %.3f | pattern: %s | flagged %d times",
                    cluster.get("community_id"),
                    cluster.get("size", 0),
                    cluster["cluster_risk_score"],
                    _infer_pattern_type(cluster),
                    flag_count,
                )

            # Generate SAR draft after threshold flags
            maybe_generate_sar(cluster, G, all_scores, fingerprint, flag_count)

    return new_alerts


def build_alert_payload(cluster: Dict, G: nx.DiGraph, all_scores: Dict[str, float]) -> Dict[str, Any]:
    members = cluster.get("members", [])
    total_amount, earliest_ts, latest_ts = _compute_cluster_financials(G, members)
    duration = (latest_ts - earliest_ts) if latest_ts and earliest_ts else 0

    top_accounts = _top_accounts_by_centrality(G, members, all_scores)

    return {
        "alert_type":             "fraud_cluster_detected",
        "timestamp":              datetime.datetime.utcnow().isoformat() + "Z",
        "cluster_id":             cluster.get("community_id"),
        "members":                members,
        "member_count":           len(members),
        "pattern_type":           _infer_pattern_type(cluster),
        "cluster_risk_score":     round(cluster["cluster_risk_score"], 4),
        "is_mule_ring":           cluster.get("is_mule_ring", False),
        "has_cycle":              cluster.get("has_cycle", False),
        "total_amount_moved":     round(total_amount, 2),
        "timeframe": {
            "earliest_unix":      earliest_ts,
            "latest_unix":        latest_ts,
            "duration_seconds":   round(duration, 1),
        },
        "top_accounts_by_centrality": top_accounts,
        "threshold_used":         ALERT_THRESHOLD,
    }


def _infer_pattern_type(cluster: Dict) -> str:
    if cluster.get("is_mule_ring"):
        return "mule_ring"
    if cluster.get("has_cycle") and cluster.get("size", 0) > 4:
        return "layering"
    if cluster.get("fanout_ratio", 0) > 0.5:
        return "fan_out"
    return "general"


def _compute_cluster_financials(G: nx.DiGraph, members: List[str]):
    member_set   = set(members)
    total_amount = 0.0
    earliest_ts  = None
    latest_ts    = None

    for u, v, data in G.edges(data=True):
        if u in member_set and v in member_set:
            total_amount += data.get("total_amount", 0.0)
            first = data.get("first_timestamp")
            last  = data.get("last_timestamp")
            if first is not None:
                earliest_ts = first if earliest_ts is None else min(earliest_ts, first)
            if last is not None:
                latest_ts = last if latest_ts is None else max(latest_ts, last)

    return total_amount, earliest_ts, latest_ts


def _top_accounts_by_centrality(G: nx.DiGraph, members: List[str], all_scores: Dict[str, float], top_n: int = 5) -> List[Dict]:
    if not members:
        return []

    subgraph    = G.subgraph(members)
    try:
        centrality = nx.betweenness_centrality(subgraph, normalized=True)
    except Exception:
        centrality = {m: 0.0 for m in members}

    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [
        {
            "account_id":  acc,
            "betweenness": round(score, 4),
            "risk_score":  round(all_scores.get(acc, 0.0), 4),
        }
        for acc, score in ranked
    ]
