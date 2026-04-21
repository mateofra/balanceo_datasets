from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    """Single ST-GCN block: graph spatial conv + temporal conv + residual."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("A", adjacency.float())

        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(temporal_kernel // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        # Aggregate neighboring joints
        x = torch.einsum("nctv,vw->nctw", x, self.A)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = x + residual
        return self.activation(x)

class RealSTGCN(nn.Module):
    """Stacked ST-GCN model for gesture recognition."""
    def __init__(
        self,
        num_classes: int,
        adjacency: torch.Tensor,
        in_channels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.shape[0])

        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, 64, adjacency, temporal_kernel=9, dropout=dropout),
                STGCNBlock(64, 64, adjacency, temporal_kernel=9, dropout=dropout),
                STGCNBlock(64, 128, adjacency, temporal_kernel=9, dropout=dropout),
            ]
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x shape: (N, C, T, V)
        b, c, t, v = x.shape
        x = x.reshape(b, c * v, t)
        x = self.data_bn(x)
        x = x.reshape(b, c, t, v)

        for block in self.blocks:
            x = block(x)

        # Global Average Pooling over Temporal (T) and Joints (V)
        x = F.avg_pool2d(x, (x.size(2), x.size(3))) # (N, 128, 1, 1)
        x = x.reshape(b, -1) # (N, 128)
        
        logits = self.classifier(x)
        return logits, None