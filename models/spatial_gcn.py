"""Spatial GCN baseline: graph convolutions over the 13 joints of each frame, with no
temporal modelling, then an average over joints and frames.

Block takes a temporal kernel size and stride, and GraphConv an optional visibility mask,
so ST-GCN (st_gcn.py) can reuse them. The Spatial GCN uses t_kernel=1, so every layer acts
on each frame independently.
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

    def forward(self, x, mask=None):  # x: (N, C, T, V); mask: (N, T, V), 1 = visible
        N, _, T, V = x.shape
        x = self.conv(x).view(N, self.A.shape[0], -1, T, V)
        A = self.A * self.edge_importance
        if mask is None:
            return torch.einsum("nkctv,kvw->nctw", x, A)
        # Invisible joints send nothing, and each joint's sum is divided by the visible share
        # of its neighbors (self included), so it averages over the visible ones only.
        visible_share = torch.einsum("ntv,vw->ntw", mask, self.A.sum(0))
        x = torch.einsum("nkctv,kvw->nctw", x * mask[:, None, None], A)
        return x / visible_share.clamp(min=1e-6)[:, None]


class Block(nn.Module):
    """Graph conv, then a temporal conv over t_kernel frames (stride frames apart), with a
    residual connection."""

    def __init__(self, c_in, c_out, A, t_kernel=1, dropout=0.0, stride=1):
        super().__init__()
        self.stride = stride
        self.gcn = GraphConv(c_in, c_out, A)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), stride=(stride, 1), padding=(t_kernel // 2, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        if c_in == c_out and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        return self.relu(self.tcn(self.gcn(x, mask)) + self.residual(x))


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

    def forward(self, x, vis=None):  # x: (N, T, V, C); vis is unused (see st_gcn.py for the variants that use it)
        N, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(N, V * C, T))
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)  # (N, C, T, V)
        for block in self.blocks:
            x = block(x)
        return self.fc(x.mean(dim=(2, 3)))
