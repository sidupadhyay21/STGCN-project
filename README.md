# STGCN-project
CS566 Group Project
Website: https://cs566-penn-action.vercel.app/

## Data
The Penn Action dataset (3.2 GB) is not tracked in git. From the repo root, download it and extract it:

```
curl -L https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz | tar -xz
```

Expected layout:

```
Penn_Action/
  frames/0001/000001.jpg ...
  labels/0001.mat ... 2326.mat
  tools/
  README
```

Only `labels/` is needed to train the skeleton models; `frames/` is for visualizations.

## Running
```
pip install -r requirements.txt
python3 -m tests.test_spatial_gcn     # sanity checks for the Spatial GCN
python3 train.py --model spatial_gcn --seed 0
python3 evaluate.py runs/spatial_gcn_seed*/preds_val.csv --plot runs/spatial_gcn_val_cm.png
```

## Shared tools
Every model uses these, so results are comparable.

- **`penn_action.py`** loads the data. `PennAction("train" | "val" | "test")` returns clips as `{"x": [64, 13, 2], "vis": [64, 13], "label": int, "video_id": str}`. Each clip is centered on its visible joints, scaled by their extent and resampled to 64 frames, and invisible joints are set to (0, 0). The first use builds `cache/penn_action_T64.npz`. The data choices are constants at the top of the file.
- **`splits/val_ids.txt`** lists the 189 official training clips held out for validation. It is committed so everyone validates on the same clips; don't edit or delete it.
- **`models/__init__.py`** is the model registry. Every model takes `x` of shape `[N, T, 13, 2]` and an optional visibility mask `vis` of shape `[N, T, 13]`, and returns logits `[N, 15]` (no softmax).
- **`train.py`** trains any registered model the same way (AdamW with a cosine schedule) and keeps the epoch with the best validation macro-F1. It writes `runs/<model>_seed<seed>/` with `config.json`, `log.csv`, `best.pt` and `preds_val.csv`, plus `preds_test.csv` with `--test`. `python3 train.py --help` lists the flags; model-specific settings go in `--model-args` as JSON, e.g. `--model-args '{"hidden": 256}'`.
- **`evaluate.py`** reads one or more `preds_*.csv` files and prints accuracy and macro-F1 (mean ± std across files), per-class accuracy with each class's most common mistake, and optionally saves a confusion matrix with `--plot`.

## Adding a model
1. Create `models/<name>.py` with a `torch.nn.Module` that follows the interface above.
2. Register it in `models/__init__.py`: `MODELS = {..., "<name>": YourModel}`.
3. Train with seeds 0, 1 and 2, then evaluate:
   ```
   python3 train.py --model <name> --seed 0
   python3 evaluate.py runs/<name>_seed*/preds_val.csv --plot runs/<name>_val_cm.png
   ```

Tune through flags rather than changing `penn_action.py` or `train.py`, so every model sees the same data and training loop. Hold out the test set (`--test`) until the final runs.
