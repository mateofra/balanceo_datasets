from __future__ import annotations

import numpy as np
import torch


# 21 landmarks MediaPipe, indexados 0-20.
HAND_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def build_adjacency_matrix(
    num_nodes: int = 21,
    edges: list[tuple[int, int]] = HAND_EDGES,
) -> torch.Tensor:
    """Build a symmetric, normalized adjacency matrix for the hand graph."""

    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for source, target in edges:
        adjacency[source, target] = 1.0
        adjacency[target, source] = 1.0

    np.fill_diagonal(adjacency, 1.0)

    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid_degree = degree > 0
    degree_inv_sqrt[valid_degree] = degree[valid_degree] ** -0.5
    degree_matrix = np.diag(degree_inv_sqrt)

    adjacency_normalized = degree_matrix @ adjacency @ degree_matrix
    return torch.tensor(adjacency_normalized, dtype=torch.float32)