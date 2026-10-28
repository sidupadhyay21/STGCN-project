"""Spatial GCN baseline: graph convolutions over the 13 joints of each frame, with no
temporal modelling, then an average over joints and frames.

Block takes a temporal kernel size so ST-GCN can reuse it. The Spatial GCN uses
t_kernel=1, so every layer acts on each frame independently.
"""
import torch
from torch import nn

from .graph import NUM_JOINTS, adjacency


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, A):  # A: (K, V, V), A[k, src, dst]
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv2d(c_in, c_out * A.shape[0], kernel_size=1)
        self.edge_importance = nn.Parameter(torch.ones_like(A))  # learnable edge weights, as in ST-GCN

    def forward(self, x):  # x: (N, C, T, V)
        N, _, T, V = x.shape
        x = self.conv(x).view(N, self.A.shape[0], -1, T, V)
        return torch.einsum("nkctv,kvw->nctw", x, self.A * self.edge_importance)


class Block(nn.Module):
    """Graph conv, then a temporal conv over t_kernel frames, with a residual connection."""

    def __init__(self, c_in, c_out, A, t_kernel=1, dropout=0.0):
        super().__init__()
        self.gcn = GraphConv(c_in, c_out, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        if c_in == c_out:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class SpatialGCN(nn.Module):
    def __init__(self, num_classes=15, in_channels=2, channels=(64, 64, 128, 128, 256, 256),
                 strategy="spatial", t_kernel=1, dropout=0.25):
        super().__init__()
        A = torch.from_numpy(adjacency(strategy))
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        widths = [in_channels, *channels]
        self.blocks = nn.ModuleList(Block(c_in, c_out, A, t_kernel, dropout)
                                    for c_in, c_out in zip(widths, widths[1:]))
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x, vis=None):  # x: (N, T, V, C); vis is unused until the visibility-aware variants
        N, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(N, V * C, T))
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)  # (N, C, T, V)
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean(dim=(2, 3)))
