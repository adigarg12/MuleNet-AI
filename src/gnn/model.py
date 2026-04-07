"""
FraudGNN — GraphSAGE encoder + binary classification head.

Architecture
------------
  SAGEConv(8  → 64) + ReLU + Dropout
  SAGEConv(64 → 32) + ReLU
  ─────────────────────────────────── encode() stops here (32-dim embedding)
  Linear(32 → 16) + ReLU
  Linear(16 → 1)  + Sigmoid
  ─────────────────────────────────── classify() produces fraud probability

The encoder replaces all hand-crafted structural feature functions
(betweenness, PageRank, clustering, degree ratios) by learning to aggregate
2-hop neighbourhood information directly from the graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.gnn.node_features import FEATURE_NAMES

IN_CHANNELS     = len(FEATURE_NAMES)   # 8
HIDDEN_CHANNELS = 64
EMBED_DIM       = 32


class FraudGNN(nn.Module):

    def __init__(
        self,
        in_channels:     int   = IN_CHANNELS,
        hidden_channels: int   = HIDDEN_CHANNELS,
        embed_dim:       int   = EMBED_DIM,
        dropout:         float = 0.3,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden_channels)
        self.conv2   = SAGEConv(hidden_channels, embed_dim)
        self.dropout = dropout

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns (N, 32) node embedding matrix."""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Maps (N, 32) embeddings → (N,) fraud probabilities in [0, 1]."""
        return self.head(z).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x, edge_index))

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        node_idx:   int,
    ) -> torch.Tensor:
        """
        Gradient × input attribution for a single node.

        Computes  importance_i = |∂P(fraud)/∂x_i  ×  x_i|
        and normalises to [0, 1].  Returns shape (8,).
        """
        # Fresh tensor with gradient tracking
        x_in = x.detach().clone().requires_grad_(True)
        probs = self.forward(x_in, edge_index)
        probs[node_idx].backward()

        grad       = x_in.grad[node_idx].abs()   # (8,)
        feat_mag   = x[node_idx].abs()
        importance = grad * feat_mag

        max_val = importance.max()
        if max_val > 0:
            importance = importance / max_val
        return importance.detach()
