# Penn Action data-processing plan

## Goal and shared protocol

The project classifies a complete Penn Action video using only its 13 annotated
2D joints. Every model (MLP, LSTM, temporal CNN, spatial GCN, ST-GCN, and the
visibility-aware variants) must consume the same processed sequences and use the
same train/validation/test membership. This makes architecture comparisons
meaningful.

The raw dataset stays unchanged under `Penn_Action/`. Generated tensors,
manifests, checkpoints, and metrics go under ignored `processed/` and `outputs/`
directories.

## Stage 1: audit the raw data

The local copy has already been checked:

- 2,326 annotation files and 2,326 frame directories
- 163,841 image files, equal to the sum of all annotated frame counts
- 15 action labels
- 1,258 official training videos (`train == 1`)
- 1,068 official test videos (`train == -1`)
- sequence lengths from 18 to 663 frames
- 185 zero-area bounding-box rows and two sequences with one missing box row;
  both cases are repaired deterministically and logged in the manifest

The preprocessing command can repeat the per-sequence JPEG-count check with
`--verify-frames`.

## Stage 2: build one canonical skeleton tensor

For every annotation file:

1. Load `x`, `y`, `visibility`, `bbox`, `action`, `pose`, `train`, and
   `nframes`.
2. Convert MATLAB one-based coordinates to zero-based coordinates.
3. For each frame, subtract the center of its person bounding box and divide by
   the larger box side. This removes camera translation and scale while keeping
   pose and motion.
4. If a frame has a zero-area box, linearly interpolate its box coordinates from
   valid neighboring frames (nearest valid box for a leading or trailing run)
   and record the repair count in the manifest. Two released sequences contain
   one fewer box than joint frames; extend their final valid box through the
   final frame and record that repair separately.
5. Zero the normalized coordinates of invisible joints and retain the
   visibility mask as a separate channel.
6. Uniformly resample the full video to 64 time steps, including its first and
   final frames. Short clips repeat some frames; long clips are subsampled.
7. Save a `(3, 64, 13)` float tensor with channels `(x, y, visibility)`.

The resulting archive has shape `(2326, 3, 64, 13)`. Coordinate-only baselines
select the first two channels. Visibility-aware variants consume all three. This
single cache avoids model-specific preprocessing differences.

## Stage 3: split without leakage

- Preserve Penn Action's official test set exactly.
- Create one stratified 15% validation subset from the official training videos
  using split seed 566.
- Keep that validation membership fixed across model types and experimental
  seeds.
- Never generate clips or frames that cross video-level split boundaries.
- Do not inspect test metrics during tuning. The training command evaluates the
  test set only when `--evaluate-test` is explicitly supplied.

With the current settings, there are 1,070 optimization videos, 188 validation
videos, and 1,068 held-out test videos.

## Stage 4: baseline and visibility ablation

The first baseline is a sequence-level MLP:

- flatten `(2, 64, 13)` into 1,664 inputs
- hidden widths 512 and 256
- LayerNorm, ReLU, and 0.30 dropout after each hidden layer
- 15 output logits and cross-entropy loss
- AdamW, validation macro-F1 model selection, learning-rate reduction, and
  early stopping

The default MLP is coordinate-only. Re-running with `--use-visibility` provides
the first controlled visibility ablation without changing any other data.

## Stage 5: evaluation and reproducibility

Run each final configuration with seeds 566, 567, and 568. Save and report:

- top-1 accuracy and macro-F1
- per-class precision, recall, F1, and support
- confusion matrix
- trainable parameter count
- single-sequence inference latency
- mean and standard deviation across the three seeds

Before final test evaluation, select hyperparameters using validation macro-F1.
After the protocol is frozen, add `--evaluate-test` and run each final model once
per seed.

## Stage 6: later models

All later models should read the same `(N, C, T, V)` archive:

- LSTM: reshape each time step to `C * V` features.
- Temporal CNN: use temporal convolutions over `T` with joint/channel features.
- Spatial GCN: apply the fixed 13-joint adjacency graph per frame.
- ST-GCN: use the tensor directly and add a singleton person dimension if the
  implementation expects `(N, C, T, V, M)`.
- Visibility-gated ST-GCN: use channel 3 to remove messages from hidden joints.

Training-only augmentations can later include horizontal reflection with
left/right joint swapping, small coordinate rotation/scale jitter, temporal
crop jitter, and random joint dropout. They must be identical across comparable
models except when an augmentation itself is the ablation under study.

