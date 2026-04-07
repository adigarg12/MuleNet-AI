"""Tests for graph builder."""

import pytest
import networkx as nx

from src.graph.graph_builder import build_graph, add_transaction, graph_summary


SAMPLE_TXNS = [
    {"txn_id": "t1", "from_account": "A", "to_account": "B",
     "amount": 100.0, "channel": "ACH", "timestamp": 1000.0, "label": "normal"},
    {"txn_id": "t2", "from_account": "B", "to_account": "C",
     "amount": 200.0, "channel": "WIRE", "timestamp": 1010.0, "label": "normal"},
    {"txn_id": "t3", "from_account": "A", "to_account": "C",
     "amount": 50.0,  "channel": "CARD", "timestamp": 1020.0, "label": "normal"},
    {"txn_id": "t4", "from_account": "C", "to_account": "A",  # creates cycle
     "amount": 75.0,  "channel": "P2P",  "timestamp": 1030.0, "label": "fraud"},
]


def test_build_graph_nodes():
    G = build_graph(SAMPLE_TXNS)
    assert set(G.nodes()) == {"A", "B", "C"}


def test_build_graph_edges():
    G = build_graph(SAMPLE_TXNS)
    # A→B, B→C, A→C, C→A = 4 unique directed pairs
    assert G.number_of_edges() == 4


def test_parallel_transactions_accumulated():
    txns = [
        {"txn_id": "x1", "from_account": "X", "to_account": "Y",
         "amount": 10.0, "channel": "ACH", "timestamp": 1.0, "label": "normal"},
        {"txn_id": "x2", "from_account": "X", "to_account": "Y",
         "amount": 20.0, "channel": "ACH", "timestamp": 2.0, "label": "normal"},
    ]
    G = build_graph(txns)
    edge_data = G["X"]["Y"]
    assert edge_data["count"] == 2
    assert abs(edge_data["total_amount"] - 30.0) < 1e-6


def test_add_transaction_incremental():
    G = build_graph(SAMPLE_TXNS[:2])
    assert G.number_of_nodes() == 3
    add_transaction(G, SAMPLE_TXNS[2])
    # A→C added; still 3 nodes
    assert G.number_of_nodes() == 3
    assert G.has_edge("A", "C")


def test_graph_summary():
    G = build_graph(SAMPLE_TXNS)
    summary = graph_summary(G)
    assert summary["nodes"] == 3
    assert summary["edges"] == 4
    assert summary["is_directed"] is True
    assert summary["total_transactions"] == 4


def test_cycle_exists():
    G = build_graph(SAMPLE_TXNS)
    cycles = list(nx.simple_cycles(G))
    # A→B→C→A and A→C→A should exist
    assert any(len(c) >= 2 for c in cycles)
