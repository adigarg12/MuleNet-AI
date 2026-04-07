"""
GNN training loop — soft label regression.

Labels are continuous fraud ratios in [0, 1] (fraction of an account's
transactions that are fraudulent).  This produces a natural score spectrum
instead of binary 0/1 outputs.

Nodes with y == -1.0 are unlabeled and excluded from loss.
"""

import logging
from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.gnn.model import FraudGNN

logger = logging.getLogger(__name__)


def train(
    model:  FraudGNN,
    data:   Data,
    epochs: int   = 200,
    lr:     float = 0.005,
) -> List[float]:
    """
    Train model in-place using BCELoss on soft fraud-ratio labels.
    Labels are floats in [0, 1]; y == -1.0 means unlabeled (skipped).
    """
    if not hasattr(data, "y") or data.y is None:
        logger.warning("No labels — skipping GNN training.")
        return []

    labeled_mask = data.y >= 0.0
    if labeled_mask.sum() == 0:
        logger.warning("All nodes unlabeled — skipping GNN training.")
        return []

    y_labeled = data.y[labeled_mask]          # float [0, 1]
    n_labeled = int(labeled_mask.sum().item())

    logger.info("Training on %d labeled nodes  (fraud ratio mean=%.3f)",
                n_labeled, float(y_labeled.mean().item()))

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    losses: List[float] = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = model(data.x, data.edge_index)   # (N,)
        loss  = criterion(probs[labeled_mask], y_labeled)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            logger.info("  epoch %d/%d  loss=%.4f", epoch + 1, epochs, loss.item())

    return losses
