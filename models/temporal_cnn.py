"""Temporal CNN baseline: the 13 joints x (x, y) of each frame are flattened into 26
channels and 1D convolutions run along time. It models motion but not the skeleton graph.

The time axis is halved wherever the width grows, so the default channels
(64, 64, 128, 128, 256, 256) use stride 2 in the third and fifth layers.
"""
from torch import nn

from .graph import NUM_JOINTS


class TemporalCNN(nn.Module):
    def __init__(self, num_classes=15, in_channels=2, channels=(64, 64, 128, 128, 256, 256),
                 kernel_size=9, dropout=0.3):
        super().__init__()
        layers, c_in = [], in_channels * NUM_JOINTS
        for i, c_out in enumerate(channels):
            stride = 2 if i > 0 and c_out > channels[i - 1] else 1
            layers += [nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=kernel_size // 2),
                       nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(dropout)]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x, vis=None):  # x: (N, T, V, C); vis is unused
        N, T, V, C = x.shape
        x = x.reshape(N, T, V * C).transpose(1, 2)  # (N, V*C, T): one channel per joint coordinate
        return self.fc(self.net(x).mean(dim=2))      # average over time
