"""
Batch loader — reads JSON or CSV transaction files and yields normalised dicts.

Expected transaction schema:
  {txn_id, from_account, to_account, amount, channel, timestamp}
"""

import csv
import json
import os
from typing import Generator, Dict, Any, List


REQUIRED_FIELDS = {"txn_id", "from_account", "to_account", "amount", "channel", "timestamp"}


def _validate(txn: Dict[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_FIELDS - set(txn.keys())
    if missing:
        raise ValueError(f"Transaction missing fields: {missing}  txn={txn}")
    return {
        "txn_id":       str(txn["txn_id"]),
        "from_account": str(txn["from_account"]),
        "to_account":   str(txn["to_account"]),
        "amount":       float(txn["amount"]),
        "channel":      str(txn["channel"]),
        "timestamp":    float(txn["timestamp"]),
        "label":        txn.get("label", "unknown"),
    }


def load_json(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Yield validated transactions from a JSON file (list of objects)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {filepath}")
    for raw in data:
        yield _validate(raw)


def load_csv(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Yield validated transactions from a CSV file."""
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _validate(row)


def load_file(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """Auto-detect format by extension and yield transactions."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        yield from load_json(filepath)
    elif ext == ".csv":
        yield from load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}  (expected .json or .csv)")


def load_all(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON/CSV files in a directory into a flat list."""
    txns: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".json", ".csv")):
            txns.extend(load_file(os.path.join(directory, fname)))
    return txns
