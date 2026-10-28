"""Train a model on Penn Action, keeping the checkpoint with the best validation macro-F1.

Writes runs/<model>_seed<seed>/: config.json, log.csv, best.pt and preds_val.csv
(plus preds_test.csv with --test, which should wait for the final runs).

Example: python train.py --model lstm --seed 0
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from models import MODELS, build_model
from penn_action import CLASSES, PennAction


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=next(iter(MODELS)), choices=MODELS)
    p.add_argument("--model-args", type=json.loads, default={},
                   help='extra model arguments as JSON, e.g. \'{"dropout": 0.25}\'')
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--invisible", choices=["zero", "keep"], default="zero")
    p.add_argument("--test", action="store_true", help="also write test-set predictions")
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    p.add_argument("--out", type=Path, default=Path("runs"))
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, labels, probs = [], [], []
    for b in loader:
        probs.append(model(b["x"].to(device), b["vis"].to(device)).softmax(-1).cpu())
        labels.append(b["label"])
        ids += b["video_id"]
    return ids, torch.cat(labels).numpy(), torch.cat(probs).numpy()


def write_preds(path, ids, labels, probs):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label", "pred"] + [f"p_{c}" for c in CLASSES])
        for vid, y, p in zip(ids, labels, probs):
            w.writerow([vid, CLASSES[y], CLASSES[p.argmax()]] + [f"{v:.5f}" for v in p])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or pick_device())
    run_dir = args.out / f"{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "device": str(device)}, indent=2, default=str))

    splits = ["train", "val"] + (["test"] if args.test else [])
    data = {s: PennAction(s, invisible=args.invisible) for s in splits}
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loaders = {s: DataLoader(data[s], batch_size=128) for s in splits[1:]}

    model = build_model(args.model, **args.model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters, {len(data['train'])} train / {len(data['val'])} val clips, {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_f1, best_epoch, log = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            loss = F.cross_entropy(model(b["x"].to(device), b["vis"].to(device)), b["label"].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b["label"])
        sched.step()

        _, y, p = predict(model, eval_loaders["val"], device)
        acc, f1 = (p.argmax(1) == y).mean(), f1_score(y, p.argmax(1), average="macro")
        log.append({"epoch": epoch, "train_loss": total_loss / len(data["train"]), "val_acc": acc, "val_f1": f1})
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(model.state_dict(), run_dir / "best.pt")
        print(f"epoch {epoch:3d}  loss {log[-1]['train_loss']:.3f}  val acc {acc:.3f}  val F1 {f1:.3f}")

    with open(run_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    for split, loader in eval_loaders.items():
        write_preds(run_dir / f"preds_{split}.csv", *predict(model, loader, device))
    print(f"best val F1 {best_f1:.3f} at epoch {best_epoch}; results in {run_dir}")


if __name__ == "__main__":
    main()
