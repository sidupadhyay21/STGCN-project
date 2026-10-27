"""Sanity checks for the Temporal CNN.

Run from the repo root: python -m tests.test_temporal_cnn
"""
import torch
import torch.nn.functional as F

from models import build_model
from models.graph import NUM_JOINTS


def test_output_shape():
    model = build_model("temporal_cnn").eval()
    for T in (32, 64, 100):  # the average over time makes any clip length work
        x = torch.randn(4, T, NUM_JOINTS, 2)
        assert model(x).shape == (4, 15)
        assert model(x, torch.ones(4, T, NUM_JOINTS, dtype=torch.bool)).shape == (4, 15)


def test_frame_order_matters():
    """Unlike the Spatial GCN, the temporal convolutions see the order of the frames."""
    torch.manual_seed(0)
    model = build_model("temporal_cnn").eval()
    x = torch.randn(2, 64, NUM_JOINTS, 2)
    assert (model(x) - model(x[:, torch.randperm(64)])).abs().max() > 1e-3


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = build_model("temporal_cnn", dropout=0.0)
    x, y = torch.randn(15, 32, NUM_JOINTS, 2), torch.arange(15)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (model(x).argmax(1) == y).all(), loss.item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
