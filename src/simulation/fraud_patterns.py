"""
Generate synthetic fraud transaction patterns.

Patterns implemented:
  - Mule ring    : circular transfers through a chain of accounts
  - Fan-out      : one source disbursing to many accounts (smurfing)
  - Layering     : deep hop chain (structuring / placement)
  - Rapid burst  : account firing many txns in a very short window
"""

import uuid
import random
import time
from typing import List, Dict, Any


CHANNELS = ["ACH", "WIRE", "CARD", "P2P", "ATM"]


def _txn(from_acc: str, to_acc: str, amount: float, channel: str, ts: float) -> Dict[str, Any]:
    return {
        "txn_id": str(uuid.uuid4()),
        "from_account": from_acc,
        "to_account": to_acc,
        "amount": round(amount, 2),
        "channel": channel,
        "timestamp": ts,
        "label": "fraud",
    }


def _account_id(prefix: str = "MUL") -> str:
    return f"{prefix}{random.randint(10000, 99999)}"


def generate_mule_ring(
    n_accounts: int = 6,
    n_hops: int = 3,
    base_amount: float = 5000.0,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Circular money flow: A→B→C→…→A.
    Each hop strips a small percentage (laundering fee).
    """
    if start_ts is None:
        start_ts = time.time() - 3600
    if n_accounts < 2:
        raise ValueError("Need at least 2 accounts for a mule ring")

    accounts = [_account_id() for _ in range(n_accounts)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts
    amount = base_amount

    for hop in range(n_hops):
        for i in range(len(accounts)):
            src = accounts[i]
            dst = accounts[(i + 1) % len(accounts)]
            channel = random.choice(CHANNELS)
            ts += random.uniform(5, 60)   # quick hops
            txns.append(_txn(src, dst, amount, channel, ts))
        amount *= random.uniform(0.90, 0.98)  # laundering fee each round

    return txns


def generate_fan_out(
    n_targets: int = 20,
    source_id: str | None = None,
    amount_per_target: float = 490.0,   # just below $500 reporting threshold
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    One source → many targets in rapid succession (smurfing / structuring).
    """
    if start_ts is None:
        start_ts = time.time() - 1800
    if source_id is None:
        source_id = _account_id("SRC")

    targets = [_account_id("TGT") for _ in range(n_targets)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts

    for tgt in targets:
        ts += random.uniform(1, 15)   # rapid fire
        amount = amount_per_target + random.uniform(-20, 20)  # slight variation
        channel = random.choice(["P2P", "ACH", "CARD"])
        txns.append(_txn(source_id, tgt, amount, channel, ts))

    return txns


def generate_layering(
    depth: int = 8,
    entry_amount: float = 50000.0,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Deep hop chain: A₀→A₁→A₂→…→Aₙ (placement through layers).
    Amount decreases slightly at each hop to obscure the trail.
    """
    if start_ts is None:
        start_ts = time.time() - 7200

    chain = [_account_id("LAY") for _ in range(depth + 1)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts
    amount = entry_amount

    for i in range(depth):
        channel = random.choice(CHANNELS)
        ts += random.uniform(30, 300)   # longer gaps to evade velocity checks
        txns.append(_txn(chain[i], chain[i + 1], amount, channel, ts))
        amount *= random.uniform(0.85, 0.97)

    return txns


def generate_rapid_burst(
    n_txns: int = 30,
    source_id: str | None = None,
    window_seconds: float = 120.0,
    avg_amount: float = 200.0,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """
    Single account sends many transactions in a very short time window.
    Simulates account takeover / automated fraud tooling.
    """
    if start_ts is None:
        start_ts = time.time() - 600
    if source_id is None:
        source_id = _account_id("BOT")

    txns: List[Dict[str, Any]] = []
    targets = [_account_id("EXT") for _ in range(n_txns)]

    for tgt in targets:
        ts = start_ts + random.uniform(0, window_seconds)
        amount = random.gauss(avg_amount, avg_amount * 0.1)
        channel = random.choice(CHANNELS)
        txns.append(_txn(source_id, tgt, abs(amount), channel, ts))

    txns.sort(key=lambda x: x["timestamp"])
    return txns


def generate_mixed_fraud(
    seed: int = 99,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """Combine all fraud patterns into one dataset."""
    random.seed(seed)
    if start_ts is None:
        start_ts = time.time() - 3600 * 6

    txns = (
        generate_mule_ring(n_accounts=6, n_hops=3, start_ts=start_ts)
        + generate_mule_ring(n_accounts=4, n_hops=2, start_ts=start_ts + 500)
        + generate_fan_out(n_targets=15, start_ts=start_ts + 1000)
        + generate_fan_out(n_targets=10, start_ts=start_ts + 1500)
        + generate_layering(depth=6, start_ts=start_ts + 2000)
        + generate_rapid_burst(n_txns=25, start_ts=start_ts + 3000)
    )

    random.shuffle(txns)
    return txns
