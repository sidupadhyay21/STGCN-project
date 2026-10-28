import numpy as np

from cs566_action.preprocess import normalize_with_bbox, uniform_frame_indices


def test_uniform_frame_indices_include_endpoints() -> None:
    indices = uniform_frame_indices(nframes=18, target_frames=64)
    assert indices.shape == (64,)
    assert indices[0] == 0
    assert indices[-1] == 17
    assert np.all(np.diff(indices) >= 0)


def test_bbox_normalization_and_visibility_mask() -> None:
    x = np.tile(np.arange(1, 14, dtype=np.float32), (2, 1))
    y = np.tile(np.arange(2, 15, dtype=np.float32), (2, 1))
    visibility = np.ones((2, 13), dtype=bool)
    visibility[0, 3] = False
    bbox = np.asarray([[1, 1, 21, 11], [1, 1, 21, 11]], dtype=np.float32)

    x_norm, y_norm, repaired = normalize_with_bbox(x, y, visibility, bbox)

    assert x_norm.shape == (2, 13)
    assert y_norm.shape == (2, 13)
    assert x_norm[0, 3] == 0.0
    assert y_norm[0, 3] == 0.0
    np.testing.assert_allclose(x_norm[1, 0], -0.5)
    np.testing.assert_allclose(y_norm[1, 0], -0.2)
    assert repaired == 0


def test_degenerate_bbox_is_interpolated() -> None:
    x = np.ones((3, 13), dtype=np.float32)
    y = np.ones((3, 13), dtype=np.float32)
    visibility = np.ones((3, 13), dtype=bool)
    bbox = np.asarray(
        [[1, 1, 11, 21], [5, 5, 5, 5], [3, 3, 13, 23]], dtype=np.float32
    )

    x_norm, y_norm, repaired = normalize_with_bbox(x, y, visibility, bbox)

    assert repaired == 1
    assert np.isfinite(x_norm).all()
    assert np.isfinite(y_norm).all()

