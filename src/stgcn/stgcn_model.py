from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Learn a per-node importance score for auditability."""

    def __init__(self, in_channels: int, num_joints: int = 21):
        super().__init__()
        self.num_joints = num_joints
        self.W = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attended features and node-level attention weights.

        Args:
            x: Tensor shaped (B, T, J, C).

        Returns:
            out: Tensor shaped (B, T, J, C).
            attn: Tensor shaped (B, J).
        """

        scores = self.W(x)  # (B, T, J, 1)
        scores = scores.mean(dim=1)  # (B, J, 1)
        attn = torch.softmax(scores, dim=1)  # (B, J, 1)
        out = x * attn.unsqueeze(1)
        return out, attn.squeeze(-1)


class GraphConvolution(nn.Module):
    """Fixed-adjacency graph convolution over hand joints."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("A", adjacency.float())
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply graph aggregation followed by a channel projection.

        Args:
            x: Tensor shaped (B, T, J, C).
        """

        aggregated = torch.einsum("ij,btjc->btic", self.A, x)
        return torch.relu(self.W(aggregated))


class STGCN(nn.Module):
    """ST-GCN con cabeza intercambiable para reconstrucción o clasificación."""

    def __init__(
        self,
        adjacency: torch.Tensor,
        in_channels: int = 3,
        hidden: int = 64,
        num_joints: int = 21,
        num_classes: int | None = None,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gcn1 = GraphConvolution(in_channels, hidden, adjacency)
        self.gcn2 = GraphConvolution(hidden, hidden, adjacency)
        self.spatial_attn = SpatialAttention(hidden, num_joints)
        self.temporal = nn.GRU(hidden * num_joints, hidden, batch_first=True)
        if num_classes is not None:
            self.head = nn.Linear(hidden, num_classes)
        else:
            self.head = nn.Linear(hidden, num_joints * in_channels)

    def forward(
        self,
        x: torch.Tensor,
        masked_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the forward pass.

        Args:
            x: Tensor shaped (B, T, J, C).
            masked_nodes: Optional boolean mask shaped (B, J).

        Returns:
            recon: Tensor shaped (B, J * C).
            attn_weights: Tensor shaped (B, J).
        """

        del masked_nodes

        h = self.gcn1(x)
        h = self.gcn2(h)
        h, attn_weights = self.spatial_attn(h)

        batch_size, num_frames, num_joints, hidden_channels = h.shape
        h = h.reshape(batch_size, num_frames, num_joints * hidden_channels)
        h, _ = self.temporal(h)
        out = self.head(h[:, -1, :])
        return out, attn_weights