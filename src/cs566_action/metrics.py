"""Dependency-light classification metrics shared by all future baselines."""

from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.size and (
        targets.min() < 0
        or predictions.min() < 0
        or targets.max() >= num_classes
        or predictions.max() >= num_classes
    ):
        raise ValueError("class index outside [0, num_classes)")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return matrix


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = matrix.sum()
    accuracy = float(true_positive.sum() / total) if total else 0.0
    populated = support > 0
    macro_f1 = float(f1[populated].mean()) if populated.any() else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.astype(np.int64).tolist(),
        "confusion_matrix": matrix.tolist(),
    }

