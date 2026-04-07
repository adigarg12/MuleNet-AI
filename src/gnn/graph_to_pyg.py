"""
Convert a NetworkX DiGraph to a PyTorch Geometric Data object.

The node ordering is fixed at conversion time and returned alongside
the Data so callers can map back from integer indices to account IDs.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx

from src.gnn.node_features import extract_feature_matrix


def to_pyg_data(
    G: nx.DiGraph,
    labels: Optional[Dict[str, int]] = None,
) -> Tuple[Data, List[str], Dict[str, int]]:
    """
    Convert NetworkX DiGraph to a PyG Data object.

    Args:
        G:      Directed transaction graph.
        labels: Optional {account_id: 0|1} dict. Unlabeled nodes get y=-1.

    Returns:
        data:        PyG Data(x, edge_index, [y])
        node_order:  List of account IDs in index order.
        node_index:  {account_id: int_index} mapping.
    """
    node_order: List[str] = list(G.nodes())
    node_index: Dict[str, int] = {n: i for i, n in enumerate(node_order)}

    # Node feature matrix (N, 8)
    x = extract_feature_matrix(G, node_order)
    x_tensor = torch.tensor(x, dtype=torch.float)

    # Edge index (2, E) — directed edges
    edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Labels: float in [0, 1] (fraud ratio). -1.0 = unlabeled.
    if labels is not None:
        y_list = [float(labels.get(n, -1.0)) for n in node_order]
        y = torch.tensor(y_list, dtype=torch.float)
        data = Data(x=x_tensor, edge_index=edge_index, y=y)
    else:
        data = Data(x=x_tensor, edge_index=edge_index)

    return data, node_order, node_index
