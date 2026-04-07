"""
Generate realistic normal transaction patterns for training / benchmarking.
All functions return lists of transaction dicts compatible with the ingestion layer.
"""

import uuid
import random
import time
from datetime import datetime, timezone, timedelta
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
        "label": "normal",
    }


def _account_id(prefix: str = "ACC") -> str:
    return f"{prefix}{random.randint(10000, 99999)}"


def generate_p2p_payments(
    n: int = 200,
    n_accounts: int = 50,
    start_ts: float | None = None,
    avg_amount: float = 150.0,
) -> List[Dict[str, Any]]:
    """Random peer-to-peer payments among a pool of accounts."""
    if start_ts is None:
        start_ts = time.time() - 86400  # yesterday

    accounts = [_account_id() for _ in range(n_accounts)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts

    for _ in range(n):
        src, dst = random.sample(accounts, 2)
        amount = random.gauss(avg_amount, avg_amount * 0.3)
        amount = max(1.0, amount)
        channel = random.choice(CHANNELS)
        ts += random.expovariate(1 / 30)  # ~30 s between transactions on average
        txns.append(_txn(src, dst, amount, channel, ts))

    return txns


def generate_business_payroll(
    n_employees: int = 30,
    employer_id: str | None = None,
    base_salary: float = 3000.0,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """One employer paying many employees (legitimate fan-out)."""
    if start_ts is None:
        start_ts = time.time()
    if employer_id is None:
        employer_id = _account_id("EMP")

    employees = [_account_id("EMP") for _ in range(n_employees)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts

    for emp in employees:
        amount = random.gauss(base_salary, base_salary * 0.05)
        ts += random.uniform(1, 5)  # payments sent seconds apart — normal batch payroll
        txns.append(_txn(employer_id, emp, amount, "ACH", ts))

    return txns


def generate_recurring_subscriptions(
    n_subscribers: int = 40,
    merchant_id: str | None = None,
    subscription_fee: float = 12.99,
    start_ts: float | None = None,
) -> List[Dict[str, Any]]:
    """Many accounts paying a single merchant (fan-in)."""
    if start_ts is None:
        start_ts = time.time()
    if merchant_id is None:
        merchant_id = _account_id("MER")

    subscribers = [_account_id() for _ in range(n_subscribers)]
    txns: List[Dict[str, Any]] = []
    ts = start_ts

    for sub in subscribers:
        ts += random.uniform(0.5, 3)
        txns.append(_txn(sub, merchant_id, subscription_fee, "CARD", ts))

    return txns


def generate_mixed_normal(
    total: int = 500,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Combine all normal pattern generators into one shuffled dataset."""
    random.seed(seed)
    base_ts = time.time() - 3600 * 24

    txns = (
        generate_p2p_payments(n=total // 2, start_ts=base_ts)
        + generate_business_payroll(n_employees=total // 5, start_ts=base_ts + 1000)
        + generate_recurring_subscriptions(n_subscribers=total // 5, start_ts=base_ts + 2000)
    )

    random.shuffle(txns)
    return txns
