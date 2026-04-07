"""Tests for simulation data generators."""

import pytest
from src.simulation.normal_patterns import (
    generate_p2p_payments, generate_business_payroll, generate_mixed_normal,
)
from src.simulation.fraud_patterns import (
    generate_mule_ring, generate_fan_out, generate_layering,
    generate_rapid_burst, generate_mixed_fraud,
)


REQUIRED_FIELDS = {"txn_id", "from_account", "to_account", "amount", "channel", "timestamp"}


def _check_txns(txns, min_count=1, label=None):
    assert len(txns) >= min_count
    for t in txns:
        assert REQUIRED_FIELDS.issubset(set(t.keys())), f"Missing fields in {t}"
        assert t["amount"] > 0
        assert t["timestamp"] > 0
        if label:
            assert t["label"] == label


# ------------------------------------------------------------------
# Normal patterns
# ------------------------------------------------------------------

def test_p2p_payments_schema():
    txns = generate_p2p_payments(n=50)
    _check_txns(txns, min_count=50, label="normal")


def test_p2p_unique_txn_ids():
    txns = generate_p2p_payments(n=100)
    ids = [t["txn_id"] for t in txns]
    assert len(set(ids)) == len(ids), "Duplicate txn_ids found"


def test_payroll_all_same_employer():
    employer = "EMP99999"
    txns = generate_business_payroll(n_employees=10, employer_id=employer)
    assert all(t["from_account"] == employer for t in txns)
    assert all(t["channel"] == "ACH" for t in txns)


def test_mixed_normal_count():
    txns = generate_mixed_normal(total=200, seed=1)
    assert len(txns) >= 100  # roughly half the total requested


# ------------------------------------------------------------------
# Fraud patterns
# ------------------------------------------------------------------

def test_mule_ring_schema():
    txns = generate_mule_ring(n_accounts=5, n_hops=2)
    _check_txns(txns, label="fraud")


def test_mule_ring_circular():
    """Verify that mule ring transactions form a cycle."""
    import networkx as nx
    from src.graph.graph_builder import build_graph

    txns = generate_mule_ring(n_accounts=4, n_hops=1)
    G = build_graph(txns)
    cycles = list(nx.simple_cycles(G))
    assert len(cycles) > 0, "Mule ring should contain at least one cycle"


def test_fan_out_single_source():
    src = "SMURF01"
    txns = generate_fan_out(n_targets=10, source_id=src)
    assert all(t["from_account"] == src for t in txns)
    assert len(set(t["to_account"] for t in txns)) == 10


def test_fan_out_amounts_below_threshold():
    """All fan-out amounts should be roughly below $500 reporting threshold."""
    txns = generate_fan_out(n_targets=20, amount_per_target=490.0)
    assert all(t["amount"] < 600.0 for t in txns)


def test_layering_decreasing_amounts():
    txns = generate_layering(depth=5, entry_amount=10000.0)
    amounts = [t["amount"] for t in txns]
    # Each hop should reduce amount (not guaranteed strictly, but on average)
    assert amounts[-1] < amounts[0]


def test_rapid_burst_time_window():
    window = 60.0
    txns = generate_rapid_burst(n_txns=20, window_seconds=window, start_ts=0.0)
    assert all(t["timestamp"] <= window for t in txns)


def test_mixed_fraud_all_labeled():
    txns = generate_mixed_fraud(seed=42)
    _check_txns(txns, label="fraud")
