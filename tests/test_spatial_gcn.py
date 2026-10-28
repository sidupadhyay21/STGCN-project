"""Sanity checks for the skeleton graph and the Spatial GCN.

Run from the repo root: python -m tests.test_spatial_gcn
"""
import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from models.graph import EDGES, NUM_JOINTS, adjacency, hop_distance_to_center


def test_hips_are_the_center():
    d = hop_distance_to_center()
    assert d[7] == d[8] == 0
    assert d[5] == d[6] == 3  # wrists are farthest


def test_partitions_split_the_uniform_graph():
    uniform, spatial = adjacency("uniform"), adjacency("spatial")
    assert spatial.shape == (3, NUM_JOINTS, NUM_JOINTS)
    np.testing.assert_allclose(spatial.sum(axis=0), uniform[0], atol=1e-6)
    np.testing.assert_allclose(uniform[0].sum(axis=0), 1.0, atol=1e-6)  # each joint averages its neighbors
    links = [(i, i) for i in range(NUM_JOINTS)] + EDGES + [(j, i) for i, j in EDGES]
    for src, dst in links:
        assert (spatial[:, src, dst] > 0).sum() == 1, (src, dst)


def test_output_shape():
    out = build_model("spatial_gcn")(torch.randn(4, 64, NUM_JOINTS, 2))
    assert out.shape == (4, 15)


def test_frame_order_does_not_matter():
    """With t_kernel=1 there is no temporal modelling, so shuffling frames changes nothing."""
    model = build_model("spatial_gcn").eval()
    x = torch.randn(2, 64, NUM_JOINTS, 2)
    torch.testing.assert_close(model(x), model(x[:, torch.randperm(64)]), atol=1e-5, rtol=1e-4)


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = build_model("spatial_gcn", dropout=0.0)
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
