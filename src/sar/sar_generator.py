"""
SAR Generator — builds a Suspicious Activity Report draft from cluster data.

Triggered by threshold_monitor when a cluster has been flagged SAR_FLAG_THRESHOLD
times (default 3). Produces a structured draft stored in sar_store.
"""

import datetime
import logging
import os
import uuid
from typing import Dict, List, Any

import networkx as nx

from src.sar.sar_store import sar_store
from src.pipeline.state import pipeline_state

logger = logging.getLogger(__name__)

SAR_FLAG_THRESHOLD = int(os.getenv("SAR_FLAG_THRESHOLD", "3"))


def maybe_generate_sar(
    cluster:     Dict,
    G:           nx.DiGraph,
    all_scores:  Dict[str, float],
    fingerprint: str,
    flag_count:  int,
) -> bool:
    """
    Generate a SAR draft if the cluster has been flagged enough times
    and no SAR exists yet. Returns True if a new SAR was created.
    """
    if flag_count < SAR_FLAG_THRESHOLD:
        return False
    if sar_store.has_sar(fingerprint):
        return False

    sar = _build_sar(cluster, G, all_scores, fingerprint, flag_count)
    sar_store.save(sar)
    logger.info(
        "SAR DRAFT GENERATED — %s | cluster %s | %d members | $%.2f | flagged %d times",
        sar["sar_id"],
        cluster.get("community_id"),
        cluster.get("size", 0),
        sar["total_amount_moved"],
        flag_count,
    )
    return True


def _build_sar(
    cluster:     Dict,
    G:           nx.DiGraph,
    all_scores:  Dict[str, float],
    fingerprint: str,
    flag_count:  int,
) -> Dict[str, Any]:
    members  = cluster.get("members", [])
    pattern  = _infer_pattern_type(cluster)
    now      = datetime.datetime.utcnow()
    sar_id   = f"SAR-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    total_amount, earliest_ts, latest_ts = _cluster_financials(G, members)
    duration_days = ((latest_ts - earliest_ts) / 86400) if latest_ts and earliest_ts else 0

    subject_accounts = _top_accounts_by_centrality(G, members, all_scores, top_n=5)
    top_transactions = _top_transactions(G, members, top_n=10)

    narrative = _generate_narrative(
        sar_id, pattern, members, subject_accounts,
        total_amount, earliest_ts, latest_ts, duration_days,
        cluster, flag_count,
    )

    return {
        "sar_id":              sar_id,
        "status":              "pending",
        "created_at":          now.isoformat() + "Z",
        "reviewed_at":         None,
        "reviewed_by":         None,

        "cluster_fingerprint": fingerprint,
        "cluster_id":          cluster.get("community_id"),
        "times_flagged":       flag_count,

        "pattern_type":        pattern,
        "cluster_risk_score":  round(cluster["cluster_risk_score"], 4),
        "is_mule_ring":        cluster.get("is_mule_ring", False),
        "has_cycle":           cluster.get("has_cycle", False),
        "member_count":        len(members),
        "all_members":         members,

        "subject_accounts":    subject_accounts,

        "total_amount_moved":  round(total_amount, 2),
        "timeframe": {
            "earliest_unix":   earliest_ts,
            "latest_unix":     latest_ts,
            "duration_days":   round(duration_days, 1),
            "earliest_str":    _ts_to_str(earliest_ts),
            "latest_str":      _ts_to_str(latest_ts),
        },

        "evidence": {
            "top_transactions": top_transactions,
        },

        "narrative":           narrative,
    }


def _generate_narrative(
    sar_id, pattern, members, subject_accounts,
    total_amount, earliest_ts, latest_ts, duration_days,
    cluster, flag_count,
) -> str:
    pattern_desc = {
        "mule_ring":  "circular fund transfers consistent with a mule ring operation",
        "layering":   "multi-hop layering designed to obscure the origin of funds",
        "fan_out":    "fan-out smurfing — splitting funds across many accounts to avoid reporting thresholds",
        "general":    "coordinated suspicious transaction activity",
    }.get(pattern, "suspicious coordinated activity")

    subjects_str = ", ".join(
        f"{a['account_id']} (risk: {a['risk_score']:.2f})"
        for a in subject_accounts[:3]
    )

    earliest_str = _ts_to_str(earliest_ts) or "unknown"
    latest_str   = _ts_to_str(latest_ts)   or "unknown"

    mule_line = (
        "\nThis network has been confirmed as a mule ring with circular transaction cycles."
        if cluster.get("is_mule_ring") else ""
    )

    return (
        f"SUSPICIOUS ACTIVITY REPORT DRAFT — {sar_id}\n\n"
        f"Between {earliest_str} and {latest_str} (approximately {duration_days:.1f} days), "
        f"a coordinated network of {len(members)} accounts was identified engaging in "
        f"{pattern_desc}.\n\n"
        f"The network moved a total of ${total_amount:,.2f} during this period. "
        f"The system flagged this network {flag_count} times across independent detection cycles, "
        f"confirming persistent and ongoing activity.\n\n"
        f"Key operator accounts identified by network centrality analysis: {subjects_str}.{mule_line}\n\n"
        f"The cluster risk score at time of SAR generation was "
        f"{cluster['cluster_risk_score']:.3f} (threshold: 0.70). "
        f"All {len(members)} member accounts are listed in the attached evidence.\n\n"
        f"This draft requires review and approval by a compliance officer before filing."
    )


def _infer_pattern_type(cluster: Dict) -> str:
    if cluster.get("is_mule_ring"):
        return "mule_ring"
    if cluster.get("has_cycle") and cluster.get("size", 0) > 4:
        return "layering"
    if cluster.get("fanout_ratio", 0) > 0.5:
        return "fan_out"
    return "general"


def _cluster_financials(G: nx.DiGraph, members: List[str]):
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


def _top_accounts_by_centrality(G, members, all_scores, top_n=5):
    subgraph = G.subgraph(members)
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


def _top_transactions(G, members, top_n=10):
    member_set = set(members)
    txns = []
    for u, v, data in G.edges(data=True):
        if u in member_set and v in member_set:
            for t in data.get("transactions", []):
                txns.append({
                    "from":      u,
                    "to":        v,
                    "amount":    t.get("amount", 0),
                    "channel":   t.get("channel", ""),
                    "timestamp": _ts_to_str(t.get("timestamp")),
                })
    return sorted(txns, key=lambda x: x["amount"], reverse=True)[:top_n]


def _ts_to_str(ts) -> str:
    if ts is None:
        return ""
    try:
        return datetime.datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)
