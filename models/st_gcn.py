"""
ST-GCN is the Spatial GCN with a temporal convolution over t_kernel frames in every block.
As in ST-GCN, the time axis is halved wherever the width grows, so the default channels
(64, 64, 128, 128, 256, 256) take 64 frames down to 16.

Each variant is off by default and switched on through --model-args:
  vis_input      visibility flag as a third input channel
  vis_mask       invisible joints send no messages; each joint averages its visible neighbors
  joint_dropout  during training only, hide each visible joint for the whole clip with this
                 probability (e.g. 0.1), so the model learns to cope with missing joints
"""
import torch
from torch import nn

from .graph import NUM_JOINTS, adjacency
from .spatial_gcn import Block


class STGCN(nn.Module):
    def __init__(self, num_classes=15, in_channels=2, channels=(64, 64, 128, 128, 256, 256),
                 strategy="spatial", t_kernel=9, dropout=0.25,
                 vis_input=False, vis_mask=False, joint_dropout=0.0):
        super().__init__()
        self.vis_input, self.vis_mask, self.joint_dropout = vis_input, vis_mask, joint_dropout
        in_channels += int(vis_input)
        A = torch.from_numpy(adjacency(strategy))
        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)
        blocks, c_in = [], in_channels
        for i, c_out in enumerate(channels):
            stride = 2 if i > 0 and c_out > channels[i - 1] else 1
            blocks.append(Block(c_in, c_out, A, t_kernel, dropout, stride))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x, vis=None):  # x: (N, T, V, C); vis: (N, T, V) bool
        if vis is None:
            vis = torch.ones(x.shape[:3], dtype=torch.bool, device=x.device)
        if self.training and self.joint_dropout > 0:
            keep = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) >= self.joint_dropout
            vis = vis & keep
            x = x * vis[..., None]
        if self.vis_input:
            x = torch.cat([x, vis[..., None].float()], dim=-1)

        N, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(N, V * C, T))
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)  # (N, C, T, V)
        mask = vis.float() if self.vis_mask else None
        for block in self.blocks:
            x = block(x, mask)
            if mask is not None:
                mask = mask[:, ::block.stride]  # follow the block's temporal stride
        return self.fc(x.mean(dim=(2, 3)))
