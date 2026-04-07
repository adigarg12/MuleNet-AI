"""
In-memory graph state manager (singleton).

Provides:
  - A single shared DiGraph for the running process
  - Snapshot capability for batch comparison
  - Thread-safe-ish access (GIL protects dict/list ops in CPython)
"""

import copy
import threading
import networkx as nx
from typing import Dict, Any, Optional

from src.graph.graph_builder import build_graph, add_transaction


class GraphStore:
    """Singleton in-memory graph store."""

    _instance: Optional["GraphStore"] = None

    def __new__(cls) -> "GraphStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._graph = nx.DiGraph()
            cls._instance._snapshots: Dict[str, nx.DiGraph] = {}
            cls._instance.lock = threading.Lock()
        return cls._instance

    # ------------------------------------------------------------------
    # Graph access
    # ------------------------------------------------------------------

    def get_graph(self) -> nx.DiGraph:
        return self._graph

    def ingest_transaction(self, txn: Dict[str, Any]) -> None:
        with self.lock:
            add_transaction(self._graph, txn)

    def ingest_batch(self, transactions) -> int:
        """Ingest an iterable of transactions. Returns count ingested."""
        count = 0
        for txn in transactions:
            with self.lock:
                add_transaction(self._graph, txn)
            count += 1
        return count

    def load_fresh(self, transactions) -> None:
        """Replace current graph with a freshly built one."""
        new_graph = build_graph(list(transactions))
        with self.lock:
            self._graph = new_graph

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self, name: str) -> None:
        """Save a deep copy of the current graph under the given name."""
        with self.lock:
            self._snapshots[name] = copy.deepcopy(self._graph)

    def get_snapshot(self, name: str) -> Optional[nx.DiGraph]:
        return self._snapshots.get(name)

    def list_snapshots(self):
        return list(self._snapshots.keys())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the graph (useful between test runs)."""
        with self.lock:
            self._graph = nx.DiGraph()
            self._snapshots.clear()


# Module-level singleton
graph_store = GraphStore()
