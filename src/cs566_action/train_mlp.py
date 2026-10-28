"""Train the fixed-length MLP baseline on the shared Penn Action cache."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .constants import TEST_FLAG, TRAIN_FLAG
from .metrics import classification_metrics
from .model import MLPClassifier, count_trainable_parameters


class SkeletonDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory view of selected videos and input channels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        use_visibility: bool,
    ) -> None:
        channels = 3 if use_visibility else 2
        selected = np.ascontiguousarray(features[indices, :channels], dtype=np.float32)
        selected_labels = np.ascontiguousarray(labels[indices], dtype=np.int64)
        self.features = torch.from_numpy(selected)
        self.labels = torch.from_numpy(selected_labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def stratified_train_val_split(
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    validation_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split official training videos while retaining every class in both sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    labels = np.asarray(labels, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []

    for class_index in sorted(np.unique(labels[candidate_indices]).tolist()):
        class_indices = candidate_indices[labels[candidate_indices] == class_index].copy()
        rng.shuffle(class_indices)
        validation_count = int(round(len(class_indices) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(class_indices) - 1)
        validation_parts.append(class_indices[:validation_count])
        train_parts.append(class_indices[validation_count:])

    train_indices = np.concatenate(train_parts)
    validation_indices = np.concatenate(validation_parts)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def load_processed_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"processed dataset not found: {path}. Run cs566_action.preprocess first."
        )
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "labels", "split_flags", "action_names", "sequence_ids"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"processed archive is missing fields: {sorted(missing)}")
        data = {name: archive[name].copy() for name in archive.files}

    features = data["features"]
    labels = data["labels"]
    split_flags = data["split_flags"]
    if features.ndim != 4 or features.shape[1] != 3 or features.shape[3] != 13:
        raise ValueError(f"expected features (N, 3, T, 13), got {features.shape}")
    if labels.shape != (features.shape[0],) or split_flags.shape != labels.shape:
        raise ValueError("labels and split_flags must have one entry per sequence")
    return data


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for inputs, batch_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, batch_targets)
        total_loss += float(loss.item()) * batch_targets.shape[0]
        total_examples += int(batch_targets.shape[0])
        targets.append(batch_targets.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())

    metrics = classification_metrics(
        np.concatenate(targets),
        np.concatenate(predictions),
        num_classes,
    )
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


@torch.inference_mode()
def benchmark_latency_ms(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    model.eval()
    sample = sample.unsqueeze(0).to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_one_seed(
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=args.deterministic)
    features = data["features"]
    labels = data["labels"].astype(np.int64)
    action_names = [str(name) for name in data["action_names"]]
    num_classes = len(action_names)

    train_dataset = SkeletonDataset(
        features, labels, train_indices, use_visibility=args.use_visibility
    )
    validation_dataset = SkeletonDataset(
        features, labels, validation_indices, use_visibility=args.use_visibility
    )
    test_dataset = SkeletonDataset(
        features, labels, test_indices, use_visibility=args.use_visibility
    )
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        validation_dataset,
        args.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    input_channels = 3 if args.use_visibility else 2
    model = MLPClassifier(
        input_channels=input_channels,
        num_frames=int(features.shape[2]),
        num_joints=int(features.shape[3]),
        num_classes=num_classes,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    class_weights = None
    if args.class_weighted:
        counts = np.bincount(labels[train_indices], minlength=num_classes).astype(np.float64)
        weights = counts.sum() / (num_classes * counts)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.scheduler_patience
    )

    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        training_loss = 0.0
        training_examples = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * targets.shape[0]
            training_examples += int(targets.shape[0])

        validation_metrics = evaluate_model(
            model, validation_loader, criterion, device, num_classes
        )
        scheduler.step(validation_metrics["macro_f1"])
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": training_loss / max(training_examples, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        print(
            f"seed={seed} epoch={epoch:03d} "
            f"train_loss={epoch_record['train_loss']:.4f} "
            f"val_acc={epoch_record['validation_accuracy']:.4f} "
            f"val_f1={epoch_record['validation_macro_f1']:.4f}"
        )

        score = float(validation_metrics["macro_f1"])
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "input_channels": input_channels,
                    "num_frames": int(features.shape[2]),
                    "num_joints": int(features.shape[3]),
                    "num_classes": num_classes,
                    "hidden_dims": list(args.hidden_dims),
                    "dropout": args.dropout,
                    "action_names": action_names,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"seed={seed} early stopping after epoch {epoch}")
                break

    _write_history(seed_dir / "history.csv", history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics = evaluate_model(
        model, validation_loader, criterion, device, num_classes
    )

    result: dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "parameter_count": count_trainable_parameters(model),
        "inference_latency_ms_single_sample": benchmark_latency_ms(
            model, validation_dataset[0][0], device
        ),
        "use_visibility": args.use_visibility,
        "class_weighted": args.class_weighted,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
        "validation": validation_metrics,
        "action_names": action_names,
    }
    if args.evaluate_test:
        result["test"] = evaluate_model(model, test_loader, criterion, device, num_classes)

    (seed_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _aggregate(results: list[dict[str, Any]], include_test: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": results}
    for split in ("validation", "test"):
        if split == "test" and not include_test:
            continue
        for metric in ("accuracy", "macro_f1"):
            values = np.asarray([run[split][metric] for run in results], dtype=np.float64)
            summary[f"{split}_{metric}_mean"] = float(values.mean())
            summary[f"{split}_{metric}_std"] = float(values.std(ddof=0))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("processed/penn_action_64.npz")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mlp"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=566)
    parser.add_argument("--seeds", nargs="+", type=int, default=[566, 567, 568])
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--use-visibility",
        action="store_true",
        help="include the visibility mask as a third MLP input channel",
    )
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the held-out official test set; use only for final runs",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.data = args.data.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    data = load_processed_dataset(args.data)
    official_train = np.flatnonzero(data["split_flags"] == TRAIN_FLAG)
    official_test = np.flatnonzero(data["split_flags"] == TEST_FLAG)
    train_indices, validation_indices = stratified_train_val_split(
        data["labels"],
        official_train,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    if set(train_indices) & set(validation_indices):
        raise RuntimeError("train/validation overlap detected")
    if set(official_train) & set(official_test):
        raise RuntimeError("official train/test overlap detected")

    device = resolve_device(args.device)
    print(
        f"device={device} train={len(train_indices)} "
        f"validation={len(validation_indices)} test={len(official_test)}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        train_one_seed(
            data,
            train_indices,
            validation_indices,
            official_test,
            args,
            seed,
            device,
        )
        for seed in args.seeds
    ]
    summary = _aggregate(results, include_test=args.evaluate_test)
    summary["configuration"] = {
        "data": str(args.data),
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "use_visibility": args.use_visibility,
        "class_weighted": args.class_weighted,
        "evaluate_test": args.evaluate_test,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()

