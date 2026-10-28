"""Convert raw Penn Action MATLAB annotations into one shared tensor cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .constants import ACTION_NAMES, JOINT_NAMES, TEST_FLAG, TRAIN_FLAG


def uniform_frame_indices(nframes: int, target_frames: int) -> np.ndarray:
    """Return deterministic, uniformly spaced frame indices including endpoints."""
    if nframes < 1:
        raise ValueError(f"nframes must be positive, got {nframes}")
    if target_frames < 1:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    return np.rint(np.linspace(0, nframes - 1, target_frames)).astype(np.int64)


def normalize_with_bbox(
    x: np.ndarray,
    y: np.ndarray,
    visibility: np.ndarray,
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Center and scale joints using the per-frame person bounding box.

    Penn Action coordinates use MATLAB's one-based pixel convention. Both the
    joints and box are converted to zero-based coordinates before normalization.
    Invisible coordinates are zeroed only after the geometric transform so they
    cannot affect centering or scaling.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=bool)
    bbox = np.asarray(bbox, dtype=np.float32)

    if x.shape != y.shape or x.shape != visibility.shape:
        raise ValueError(
            "x, y, and visibility must have identical (frames, joints) shapes; "
            f"got {x.shape}, {y.shape}, and {visibility.shape}"
        )
    if x.ndim != 2 or x.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"expected (frames, {len(JOINT_NAMES)}) joints, got {x.shape}")
    if bbox.shape != (x.shape[0], 4):
        raise ValueError(f"expected bbox shape {(x.shape[0], 4)}, got {bbox.shape}")

    x_zero = x - 1.0
    y_zero = y - 1.0
    bbox_zero = bbox - 1.0

    # A few Penn Action frames have only one annotated joint, producing a
    # zero-area box. Interpolate each box coordinate from valid neighboring
    # frames; np.interp uses the nearest valid endpoint for leading/trailing
    # runs. This keeps the video and avoids division by zero without deriving a
    # body scale from an isolated joint.
    raw_width = bbox_zero[:, 2] - bbox_zero[:, 0]
    raw_height = bbox_zero[:, 3] - bbox_zero[:, 1]
    valid_bbox = np.maximum(raw_width, raw_height) > 0
    repaired_bbox_frames = int((~valid_bbox).sum())
    if repaired_bbox_frames:
        if not valid_bbox.any():
            raise ValueError("sequence has no positive-area bounding box")
        frame_indices = np.arange(bbox_zero.shape[0])
        repaired = bbox_zero.copy()
        for coordinate in range(4):
            repaired[:, coordinate] = np.interp(
                frame_indices,
                frame_indices[valid_bbox],
                bbox_zero[valid_bbox, coordinate],
            )
        bbox_zero = repaired

    center_x = (bbox_zero[:, 0] + bbox_zero[:, 2]) * 0.5
    center_y = (bbox_zero[:, 1] + bbox_zero[:, 3]) * 0.5
    width = bbox_zero[:, 2] - bbox_zero[:, 0]
    height = bbox_zero[:, 3] - bbox_zero[:, 1]
    scale = np.maximum(width, height)
    if np.any(scale <= 0):
        bad = np.flatnonzero(scale <= 0)[:10].tolist()
        raise ValueError(f"non-positive bounding-box scale after repair in frames {bad}")

    x_norm = (x_zero - center_x[:, None]) / scale[:, None]
    y_norm = (y_zero - center_y[:, None]) / scale[:, None]
    x_norm[~visibility] = 0.0
    y_norm[~visibility] = 0.0
    return (
        x_norm.astype(np.float32),
        y_norm.astype(np.float32),
        repaired_bbox_frames,
    )


def _scalar(annotation: dict[str, Any], key: str) -> Any:
    value = np.asarray(annotation[key])
    if value.size != 1:
        raise ValueError(f"annotation field {key!r} is not scalar: shape={value.shape}")
    return value.item()


def process_annotation(
    annotation_path: Path,
    target_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and transform one .mat annotation file."""
    annotation = loadmat(
        annotation_path,
        variable_names=(
            "action",
            "pose",
            "x",
            "y",
            "visibility",
            "train",
            "bbox",
            "dimensions",
            "nframes",
        ),
        squeeze_me=True,
    )

    action = str(_scalar(annotation, "action"))
    pose = str(_scalar(annotation, "pose"))
    split_flag = int(_scalar(annotation, "train"))
    nframes = int(_scalar(annotation, "nframes"))
    dimensions = np.asarray(annotation["dimensions"], dtype=np.int64).reshape(-1)
    x = np.asarray(annotation["x"], dtype=np.float32)
    y = np.asarray(annotation["y"], dtype=np.float32)
    visibility = np.asarray(annotation["visibility"], dtype=bool)
    bbox = np.asarray(annotation["bbox"], dtype=np.float32)

    if action not in ACTION_NAMES:
        raise ValueError(f"unknown action {action!r} in {annotation_path}")
    if split_flag not in (TRAIN_FLAG, TEST_FLAG):
        raise ValueError(f"unexpected split flag {split_flag} in {annotation_path}")
    if dimensions.shape != (3,):
        raise ValueError(f"expected dimensions [height, width, frames], got {dimensions}")
    if (
        nframes != int(dimensions[2])
        or x.shape != (nframes, len(JOINT_NAMES))
        or y.shape != x.shape
        or visibility.shape != x.shape
    ):
        raise ValueError(
            f"frame-count mismatch in {annotation_path}: nframes={nframes}, "
            f"dimensions={dimensions.tolist()}, x={x.shape}, y={y.shape}, "
            f"visibility={visibility.shape}"
        )

    # Two released sequences (1154 and 1865 in this copy) contain one fewer
    # bbox row than skeleton frames. The missing row location is not encoded,
    # so deterministically extend the final valid box and record the repair.
    padded_bbox_frames = 0
    if bbox.shape == (nframes - 1, 4):
        bbox = np.vstack((bbox, bbox[-1]))
        padded_bbox_frames = 1
    elif bbox.shape != (nframes, 4):
        raise ValueError(
            f"bbox length mismatch in {annotation_path}: expected {(nframes, 4)}, "
            f"got {bbox.shape}"
        )

    x_norm, y_norm, repaired_bbox_frames = normalize_with_bbox(
        x, y, visibility, bbox
    )
    indices = uniform_frame_indices(nframes, target_frames)

    # Shared representation: channels x time x joints. Coordinate-only models
    # select channels 0:2; visibility-aware variants also consume channel 2.
    features = np.stack(
        (
            x_norm[indices],
            y_norm[indices],
            visibility[indices].astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32)

    metadata = {
        "sequence_id": annotation_path.stem,
        "action": action,
        "label": ACTION_NAMES.index(action),
        "pose": pose,
        "split_flag": split_flag,
        "split": "train" if split_flag == TRAIN_FLAG else "test",
        "nframes": nframes,
        "height": int(dimensions[0]),
        "width": int(dimensions[1]),
        "visibility_fraction": float(visibility.mean()),
        "repaired_bbox_frames": repaired_bbox_frames,
        "padded_bbox_frames": padded_bbox_frames,
    }
    return features, metadata


def build_dataset(
    dataset_root: Path,
    output_path: Path,
    target_frames: int = 64,
    verify_frames: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the processed cache and a human-readable sequence manifest."""
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()
    label_dir = dataset_root / "labels"
    frame_dir = dataset_root / "frames"

    if not label_dir.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {label_dir}")
    if verify_frames and not frame_dir.is_dir():
        raise FileNotFoundError(f"frame directory not found: {frame_dir}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite")

    annotation_paths = sorted(label_dir.glob("*.mat"))
    if not annotation_paths:
        raise FileNotFoundError(f"no .mat files found in {label_dir}")

    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for number, annotation_path in enumerate(annotation_paths, start=1):
        sequence_features, metadata = process_annotation(annotation_path, target_frames)
        if verify_frames:
            images = list((frame_dir / metadata["sequence_id"]).glob("*.jpg"))
            if len(images) != metadata["nframes"]:
                raise ValueError(
                    f"{metadata['sequence_id']} has {len(images)} images but "
                    f"the annotation declares {metadata['nframes']}"
                )
        features.append(sequence_features)
        rows.append(metadata)
        if number % 250 == 0 or number == len(annotation_paths):
            print(f"processed {number}/{len(annotation_paths)} sequences")

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    split_flags = np.asarray([row["split_flag"] for row in rows], dtype=np.int8)
    feature_array = np.stack(features, axis=0)
    train_count = int((split_flags == TRAIN_FLAG).sum())
    test_count = int((split_flags == TEST_FLAG).sum())

    present_actions = tuple(sorted({row["action"] for row in rows}))
    if present_actions != tuple(sorted(ACTION_NAMES)):
        raise ValueError(
            "dataset action set does not match the expected 15 classes: "
            f"found {present_actions}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        features=feature_array,
        labels=labels,
        split_flags=split_flags,
        sequence_ids=np.asarray([row["sequence_id"] for row in rows]),
        poses=np.asarray([row["pose"] for row in rows]),
        original_nframes=np.asarray([row["nframes"] for row in rows], dtype=np.int32),
        visibility_fraction=np.asarray(
            [row["visibility_fraction"] for row in rows], dtype=np.float32
        ),
        action_names=np.asarray(ACTION_NAMES),
        joint_names=np.asarray(JOINT_NAMES),
        target_frames=np.asarray(target_frames, dtype=np.int32),
        normalization=np.asarray("per-frame bbox center/max-side scale"),
    )
    temporary_path.replace(output_path)

    manifest_path = output_path.with_name("manifest.csv")
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": str(dataset_root),
        "output": str(output_path),
        "manifest": str(manifest_path),
        "sequences": len(rows),
        "train_sequences": train_count,
        "test_sequences": test_count,
        "classes": len(ACTION_NAMES),
        "target_frames": target_frames,
        "feature_shape": list(feature_array.shape),
        "verified_frame_files": verify_frames,
        "repaired_bbox_frames": int(
            sum(row["repaired_bbox_frames"] for row in rows)
        ),
        "padded_bbox_frames": int(sum(row["padded_bbox_frames"] for row in rows)),
    }
    summary_path = output_path.with_name("preprocessing_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Penn_Action"),
        help="folder containing labels/ and frames/ (default: Penn_Action)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/penn_action_64.npz"),
        help="processed .npz cache",
    )
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument(
        "--verify-frames",
        action="store_true",
        help="verify every sequence has the annotated number of JPEG frames",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_dataset(
        dataset_root=args.dataset_root,
        output_path=args.output,
        target_frames=args.target_frames,
        verify_frames=args.verify_frames,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

