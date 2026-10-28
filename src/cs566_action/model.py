"""Neural-network baselines for skeleton action recognition."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Flatten a fixed-length skeleton sequence and classify it with an MLP."""

    def __init__(
        self,
        input_channels: int,
        num_frames: int,
        num_joints: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if min(input_channels, num_frames, num_joints, num_classes) < 1:
            raise ValueError("all input and output dimensions must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_shape = (input_channels, num_frames, num_joints)
        input_dim = input_channels * num_frames * num_joints
        layers: list[nn.Module] = [nn.Flatten()]
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "MLP input must have shape (batch, channels, frames, joints); "
                f"got {tuple(inputs.shape)}"
            )
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise ValueError(
                f"expected per-sample shape {self.input_shape}, got {tuple(inputs.shape[1:])}"
            )
        return self.network(inputs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

