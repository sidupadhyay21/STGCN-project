"""Sanity checks for ST-GCN and its visibility-aware variants.

Run from the repo root: python3 -m tests.test_st_gcn
"""
import torch
import torch.nn.functional as F

from models import build_model
from models.graph import NUM_JOINTS, adjacency
from models.spatial_gcn import GraphConv

ALL_VARIANTS = {"vis_input": True, "vis_mask": True, "joint_dropout": 0.2}


def random_vis(N, T, p_visible=0.8):
    return torch.rand(N, T, NUM_JOINTS) < p_visible


def test_output_shape():
    for args in ({}, {"vis_input": True}, {"vis_mask": True}, ALL_VARIANTS):
        model = build_model("st_gcn", **args).eval()
        for T in (64, 33):  # odd lengths survive the temporal strides
            x = torch.randn(4, T, NUM_JOINTS, 2)
            assert model(x, random_vis(4, T)).shape == (4, 15), args
            assert model(x).shape == (4, 15), args  # vis is optional


def test_frame_order_matters():
    """Unlike the Spatial GCN, the temporal convolutions see the order of the frames."""
    torch.manual_seed(0)
    model = build_model("st_gcn").eval()
    x = torch.randn(2, 64, NUM_JOINTS, 2)
    assert (model(x) - model(x[:, torch.randperm(64)])).abs().max() > 1e-3


def test_invisible_joints_send_no_messages():
    torch.manual_seed(0)
    conv = GraphConv(4, 8, torch.from_numpy(adjacency()))
    x = torch.randn(2, 4, 5, NUM_JOINTS)
    mask = torch.ones(2, 5, NUM_JOINTS)
    mask[:, :, 5] = 0  # hide the left wrist
    moved = x.clone()
    moved[..., 5] = 100.0
    torch.testing.assert_close(conv(x, mask), conv(moved, mask))
    torch.testing.assert_close(conv(x, torch.ones_like(mask)), conv(x))  # all visible = no masking


def test_joint_dropout_only_while_training():
    torch.manual_seed(0)
    model = build_model("st_gcn", joint_dropout=1.0, dropout=0.0)  # hide every joint
    x1, x2 = torch.randn(2, 2, 64, NUM_JOINTS, 2)
    torch.testing.assert_close(model.train()(x1), model(x2))  # the input is fully hidden
    assert (model.eval()(x1) - model(x2)).abs().max() > 1e-3  # but not at test time


def test_overfits_a_tiny_batch():
    torch.manual_seed(0)
    model = build_model("st_gcn", dropout=0.0, vis_input=True, vis_mask=True)
    x, y, vis = torch.randn(15, 32, NUM_JOINTS, 2), torch.arange(15), random_vis(15, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = F.cross_entropy(model(x, vis), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (model.eval()(x, vis).argmax(1) == y).all(), loss.item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
