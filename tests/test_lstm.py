import unittest

import torch
import torch.nn.functional as F

from models import build_model
from models.lstm import NUM_JOINTS


class LSTMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shapes_and_config(self):
        for layers in (1, 2):
            model = build_model("lstm", hidden=16, num_layers=layers)
            for batch, frames in ((1, 1), (2, 64), (3, 9)):
                x = torch.randn(batch, frames, NUM_JOINTS, 2)
                out = model(x, torch.ones(batch, frames, NUM_JOINTS, dtype=torch.bool))
                self.assertEqual(out.shape, (batch, 15))
                self.assertTrue(torch.isfinite(out).all())
        custom = build_model("lstm", num_classes=4, in_channels=3)
        self.assertEqual(custom(torch.randn(2, 8, NUM_JOINTS, 3)).shape, (2, 4))

    def test_bad_inputs(self):
        model = build_model("lstm")
        for shape in ((2, 64, 12, 2), (2, 64, 13), (2, 0, 13, 2)):
            with self.assertRaises(ValueError):
                model(torch.randn(*shape))
        for args in ({"hidden": 0}, {"num_layers": 0}, {"dropout": 1}):
            with self.assertRaises(ValueError):
                build_model("lstm", **args)

    def test_finite_gradients_and_all_invisible(self):
        model = build_model("lstm", hidden=16)
        x = torch.randn(2, 8, NUM_JOINTS, 2, requires_grad=True)
        F.cross_entropy(model(x), torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[:, 0].abs().sum().item(), 0)
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        out = model(torch.zeros_like(x), torch.zeros(2, 8, NUM_JOINTS, dtype=torch.bool))
        self.assertTrue(torch.isfinite(out).all())

    def test_order_matters_and_state_resets(self):
        model = build_model("lstm", hidden=16).eval()
        x = torch.randn(2, 12, NUM_JOINTS, 2)
        first = model(x)
        self.assertFalse(torch.allclose(first, model(x.flip(1))))
        model(torch.randn_like(x))
        torch.testing.assert_close(first, model(x))
        torch.testing.assert_close(first, torch.cat([model(x[i:i+1]) for i in range(2)]))

    def test_overfits_a_tiny_batch(self):
        model = build_model("lstm", hidden=32, num_layers=1, dropout=0)
        x, y = torch.randn(15, 8, NUM_JOINTS, 2), torch.arange(15)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(120):
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertTrue((model(x).argmax(1) == y).all(), loss.item())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
