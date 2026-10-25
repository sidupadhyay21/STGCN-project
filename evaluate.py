"""Summarize prediction files written by train.py: accuracy, macro-F1, per-class accuracy
and a confusion matrix summed over all files (e.g. one per seed).

Example: python evaluate.py runs/spatial_gcn_seed*/preds_val.csv --plot runs/spatial_gcn_val_cm.png
"""
import argparse
import csv

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from penn_action import CLASSES


def read_preds(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([CLASSES.index(r["label"]) for r in rows]),
            np.array([CLASSES.index(r["pred"]) for r in rows]))


def plot_confusion(cm, path):
    import matplotlib.pyplot as plt

    rates = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                color="white" if rates[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("preds", nargs="+", help="preds_*.csv files")
    p.add_argument("--plot", help="save the confusion matrix as an image")
    args = p.parse_args()

    accs, f1s = [], []
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for path in args.preds:
        y, pred = read_preds(path)
        accs.append((y == pred).mean())
        f1s.append(f1_score(y, pred, labels=range(len(CLASSES)), average="macro", zero_division=0))
        cm += confusion_matrix(y, pred, labels=range(len(CLASSES)))
        print(f"{path}: acc {accs[-1]:.3f}  macro-F1 {f1s[-1]:.3f}")
    if len(args.preds) > 1:
        print(f"mean over {len(args.preds)} files: acc {np.mean(accs):.3f} ± {np.std(accs):.3f}  "
              f"macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    print("\nper-class accuracy (worst first), most common mistake:")
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for i in np.argsort(per_class):
        off = cm[i].copy()
        off[i] = 0
        mistake = f"-> {CLASSES[off.argmax()]} ({off.max()})" if off.any() else ""
        print(f"  {CLASSES[i]:16s} {per_class[i]:.2f}  {mistake}")

    if args.plot:
        plot_confusion(cm, args.plot)
        print(f"\nconfusion matrix saved to {args.plot}")


if __name__ == "__main__":
    main()
