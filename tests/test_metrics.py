import numpy as np

from cs566_action.metrics import classification_metrics


def test_classification_metrics() -> None:
    targets = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0, 1, 1, 1])
    metrics = classification_metrics(targets, predictions, num_classes=2)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0.0 < metrics["macro_f1"] < 1.0

