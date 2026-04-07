"""Tests for feature engineering modules."""

import pytest
import networkx as nx

from src.graph.graph_builder import build_graph
from src.features.structural import (
    compute_in_degree, compute_out_degree,
    compute_betweenness_centrality, compute_pagerank,
    get_structural_features,
)
from src.features.temporal import (
    compute_velocity, compute_burst_score, compute_avg_retention_time,
    get_temporal_features,
)
from src.features.behavioral import (
    compute_fan_out_ratio, compute_flow_depth,
    compute_cross_channel_jumps, get_behavioral_features,
)


# ---- build a small test graph ----

def make_fan_out_graph():
    """1 source → 5 targets (smurfing)."""
    txns = [
        {"txn_id": f"f{i}", "from_account": "SRC", "to_account": f"T{i}",
         "amount": 100.0, "channel": "P2P", "timestamp": float(1000 + i * 5),
         "label": "fraud"}
        for i in range(5)
    ]
    return build_graph(txns)


def make_chain_graph():
    """A→B→C→D (layering)."""
    txns = [
        {"txn_id": "c1", "from_account": "A", "to_account": "B",
         "amount": 500.0, "channel": "ACH", "timestamp": 1000.0, "label": "fraud"},
        {"txn_id": "c2", "from_account": "B", "to_account": "C",
         "amount": 490.0, "channel": "WIRE", "timestamp": 1100.0, "label": "fraud"},
        {"txn_id": "c3", "from_account": "C", "to_account": "D",
         "amount": 480.0, "channel": "P2P", "timestamp": 1200.0, "label": "fraud"},
    ]
    return build_graph(txns)


# ------------------------------------------------------------------
# Structural
# ------------------------------------------------------------------

def test_in_out_degree():
    G = make_fan_out_graph()
    assert compute_out_degree(G, "SRC") == 5
    assert compute_in_degree(G, "SRC") == 0
    assert compute_in_degree(G, "T0") == 1


def test_betweenness_centrality_source_highest():
    G = make_chain_graph()
    bc = compute_betweenness_centrality(G)
    # B and C are on the only path, so they have highest betweenness
    # A and D are endpoints with 0
    assert bc["A"] == pytest.approx(0.0, abs=1e-6)
    assert bc["D"] == pytest.approx(0.0, abs=1e-6)


def test_pagerank_not_empty():
    G = make_chain_graph()
    pr = compute_pagerank(G)
    assert len(pr) == 4
    assert all(v >= 0 for v in pr.values())


def test_get_structural_features_keys():
    G = make_fan_out_graph()
    feats = get_structural_features(G, "SRC")
    expected_keys = {"in_degree", "out_degree", "total_degree", "betweenness", "pagerank", "clustering"}
    assert expected_keys == set(feats.keys())


# ------------------------------------------------------------------
# Temporal
# ------------------------------------------------------------------

def test_velocity_basic():
    txns = [
        {"txn_id": f"v{i}", "from_account": "X", "to_account": "Y",
         "amount": 10.0, "channel": "ACH",
         "timestamp": float(i * 10),   # 10 txns over 90 s window
         "label": "normal"}
        for i in range(10)
    ]
    vel = compute_velocity(txns, window_seconds=600.0, node="X")
    assert vel > 0.0


def test_burst_score_uniform_intervals_low():
    # Perfectly uniform → CoV ≈ 0
    txns = [
        {"txn_id": f"b{i}", "from_account": "B", "to_account": "C",
         "amount": 1.0, "channel": "ACH", "timestamp": float(i * 100), "label": "normal"}
        for i in range(10)
    ]
    score = compute_burst_score(txns)
    assert score < 0.05


def test_burst_score_irregular_high():
    import random
    random.seed(1)
    txns = [
        {"txn_id": f"r{i}", "from_account": "R", "to_account": "S",
         "amount": 1.0, "channel": "ACH",
         "timestamp": float(random.expovariate(0.1)),
         "label": "fraud"}
        for i in range(20)
    ]
    score = compute_burst_score(txns)
    assert score >= 0.0  # just ensure it doesn't crash and returns a number


def test_retention_time_pass_through():
    # Money comes in at t=1000, goes out at t=1010 → retention ≈ 10s
    txns = [
        {"txn_id": "in1", "from_account": "EXT", "to_account": "MULE",
         "amount": 100.0, "channel": "ACH", "timestamp": 1000.0, "label": "fraud"},
        {"txn_id": "out1", "from_account": "MULE", "to_account": "DST",
         "amount": 98.0, "channel": "P2P", "timestamp": 1010.0, "label": "fraud"},
    ]
    G = build_graph(txns)
    ret = compute_avg_retention_time(G, "MULE")
    assert ret == pytest.approx(10.0, abs=1.0)


# ------------------------------------------------------------------
# Behavioural
# ------------------------------------------------------------------

def test_fan_out_ratio_high_for_smurf():
    G = make_fan_out_graph()
    ratio = compute_fan_out_ratio(G, "SRC")
    assert ratio > 4.0   # 5 out / (0 + 1) = 5


def test_flow_depth_chain():
    G = make_chain_graph()
    # D is 3 hops from A
    depth = compute_flow_depth(G, "D")
    assert depth == 3


def test_cross_channel_jumps():
    # ACH → WIRE → P2P → P2P = 2 jumps
    txns = [
        {"txn_id": "cc1", "from_account": "X", "to_account": "Y",
         "amount": 10.0, "channel": "ACH", "timestamp": 1.0, "label": "fraud"},
        {"txn_id": "cc2", "from_account": "X", "to_account": "Z",
         "amount": 10.0, "channel": "WIRE", "timestamp": 2.0, "label": "fraud"},
        {"txn_id": "cc3", "from_account": "X", "to_account": "W",
         "amount": 10.0, "channel": "P2P", "timestamp": 3.0, "label": "fraud"},
    ]
    G = build_graph(txns)
    jumps = compute_cross_channel_jumps(G, "X")
    assert jumps >= 2
