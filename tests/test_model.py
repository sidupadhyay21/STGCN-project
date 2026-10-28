import torch

from cs566_action.model import MLPClassifier


def test_mlp_output_shape() -> None:
    model = MLPClassifier(
        input_channels=2,
        num_frames=64,
        num_joints=13,
        num_classes=15,
        hidden_dims=(32, 16),
        dropout=0.0,
    )
    output = model(torch.zeros(4, 2, 64, 13))
    assert output.shape == (4, 15)

