"""
GNNEmbedder — module-level singleton that owns the trained FraudGNN.

Replaces:
  - WeightedScorer   (rule-based weighted sum)
  - AnomalyModel     (Isolation Forest)
  - get_structural_features / get_temporal_features / get_behavioral_features

Public API
----------
  gnn_embedder.train(G, labels, epochs)   → fit model, cache scores
  gnn_embedder.get_risk_score(node)       → float in [0, 1]
  gnn_embedder.get_all_scores()           → {node: float}
  gnn_embedder.explain_node(node)         → {feature: importance}
  gnn_embedder.get_embedding(node)        → np.ndarray (32,)
  gnn_embedder.is_trained()               → bool
"""

import logging
import threading
from typing import Dict, List, Optional

import numpy as np
import torch
import networkx as nx

from src.gnn.model import FraudGNN, EMBED_DIM
from src.gnn.graph_to_pyg import to_pyg_data
from src.gnn.trainer import train as _train
from src.gnn.node_features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class GNNEmbedder:

    def __init__(self) -> None:
        self._lock         = threading.RLock()
        self._model:       Optional[FraudGNN]      = None
        self._trained:     bool                    = False
        self._data                                 = None   # PyG Data
        self._node_order:  List[str]               = []
        self._node_index:  Dict[str, int]          = {}
        self._embeddings:  Optional[torch.Tensor]  = None  # (N, 32)
        self._scores:      Optional[torch.Tensor]  = None  # (N,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        G:      nx.DiGraph,
        labels: Optional[Dict[str, int]] = None,
        epochs: int = 200,
    ) -> None:
        """
        Convert graph to PyG data, train FraudGNN, cache embeddings+scores.

        Args:
            G:      The current transaction graph.
            labels: {account_id: 0|1}.  Accounts absent from dict are unlabeled.
            epochs: Training epochs.
        """
        data, node_order, node_index = to_pyg_data(G, labels)

        model = FraudGNN()
        _train(model, data, epochs=epochs)

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(data.x, data.edge_index)
            scores     = model.classify(embeddings)

        with self._lock:
            self._model      = model
            self._data       = data
            self._node_order = node_order
            self._node_index = node_index
            self._embeddings = embeddings
            self._scores     = scores
            self._trained    = True

        logger.info(
            "GNN ready — score range [%.3f, %.3f]",
            float(scores.min()), float(scores.max()),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._trained

    def get_risk_score(self, node: str) -> float:
        """Fraud probability in [0, 1].  Returns 0.0 if untrained or unknown node."""
        with self._lock:
            if not self._trained or node not in self._node_index:
                return 0.0
            return float(self._scores[self._node_index[node]].item())

    def get_all_scores(self) -> Dict[str, float]:
        """Returns {node_id: fraud_probability} for every node."""
        with self._lock:
            if not self._trained:
                return {}
            return {
                n: float(self._scores[i].item())
                for i, n in enumerate(self._node_order)
            }

    def get_embedding(self, node: str) -> np.ndarray:
        """32-dim embedding vector.  Returns zeros if untrained or unknown."""
        with self._lock:
            if not self._trained or node not in self._node_index:
                return np.zeros(EMBED_DIM, dtype=np.float32)
            return self._embeddings[self._node_index[node]].numpy()

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_node(self, node: str) -> Dict[str, float]:
        """
        Gradient × input attribution.

        Returns {feature_name: importance_score} in [0, 1] for the 8 raw
        input features.  These replace the old 'contributions' dict from
        WeightedScorer and are used by build_risk_result / report_generator.
        """
        with self._lock:
            if not self._trained or node not in self._node_index:
                return {k: 0.0 for k in FEATURE_NAMES}

            idx = self._node_index[node]
            self._model.eval()
            importances = self._model.feature_importance(
                self._data.x, self._data.edge_index, idx
            )
            return {
                k: round(float(importances[i].item()), 4)
                for i, k in enumerate(FEATURE_NAMES)
            }


# Module-level singleton — import this everywhere
gnn_embedder = GNNEmbedder()
