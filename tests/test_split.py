import numpy as np

from cs566_action.train_mlp import stratified_train_val_split


def test_stratified_split_is_reproducible_and_disjoint() -> None:
    labels = np.repeat(np.arange(3), 10)
    candidates = np.arange(labels.size)
    train_a, validation_a = stratified_train_val_split(labels, candidates, 0.2, 566)
    train_b, validation_b = stratified_train_val_split(labels, candidates, 0.2, 566)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert not set(train_a) & set(validation_a)
    assert set(train_a) | set(validation_a) == set(candidates)
    assert np.bincount(labels[validation_a], minlength=3).tolist() == [2, 2, 2]

