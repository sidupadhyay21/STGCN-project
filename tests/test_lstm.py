"""Synthetic fixtures validate the pipeline without modifying real data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.io as sio
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

    def test_shared_pipeline(self):
        import evaluate
        import penn_action
        import train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "Penn_Action" / "labels"
            labels.mkdir(parents=True)
            val_ids = []
            rng = np.random.default_rng(0)
            # Create separate train, validation and test fixtures, each with all classes.
            for split in range(3):
                for cls, action in enumerate(penn_action.CLASSES):
                    vid = f"{split * 15 + cls + 1:04d}"
                    xy = rng.normal(size=(8, NUM_JOINTS, 2)).astype(np.float32)
                    sio.savemat(labels / f"{vid}.mat", {
                        "x": xy[..., 0], "y": xy[..., 1],
                        "visibility": np.ones((8, NUM_JOINTS)),
                        "action": action, "train": 1 if split < 2 else -1,
                    })
                    if split == 1:
                        val_ids.append(vid)
            split_file = root / "val_ids.txt"
            split_file.write_text("\n".join(val_ids))
            argv = ["train.py", "--model", "lstm", "--epochs", "2", "--device", "cpu",
                    "--batch-size", "15", "--out", str(root / "runs"), "--test",
                    "--model-args", '{"hidden": 8, "num_layers": 1, "dropout": 0}']
            # Redirect every data and output path to avoid touching real experiments.
            with patch.object(penn_action, "DATA_DIR", root / "Penn_Action"), \
                 patch.object(penn_action, "CACHE_DIR", root / "cache"), \
                 patch.object(penn_action, "VAL_SPLIT_FILE", split_file), \
                 patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                train.main()
                run = root / "runs" / "lstm_seed0"
                for name in ("config.json", "log.csv", "best.pt", "preds_val.csv", "preds_test.csv"):
                    self.assertTrue((run / name).is_file(), name)
                config = json.loads((run / "config.json").read_text())
                model = build_model(config["model"], **config["model_args"])
                model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
                for split in ("val", "test"):
                    loader = train.DataLoader(penn_action.PennAction(split), batch_size=15)
                    ids, targets, probabilities = train.predict(model, loader, "cpu")
                    y, pred = evaluate.read_preds(run / f"preds_{split}.csv")
                    self.assertEqual(len(ids), 15)
                    np.testing.assert_array_equal(y, targets)
                    np.testing.assert_array_equal(pred, probabilities.argmax(1))
                with patch("sys.argv", ["evaluate.py", str(run / "preds_val.csv"),
                                        "--plot", str(root / "cm.png")]):
                    evaluate.main()
                self.assertTrue((root / "cm.png").is_file())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
