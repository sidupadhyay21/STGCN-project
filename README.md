# STGCN-project

CS566 Group Project — LSTM baseline for Penn Action classification.

## Data

Download [Penn Action](https://dreamdragon.github.io/PennAction/) and extract it in
the project root. Only `Penn_Action/labels/0001.mat` through `2326.mat` are required.
The dataset is not tracked in Git. Keep `splits/val_ids.txt` unchanged.

## Running

Use Python 3.10+ and run commands from the project root:

```bash
python -m pip install -r requirements.txt
python -m tests.test_lstm
python train.py --model lstm --seed 0
python evaluate.py runs/lstm_seed0/preds_val.csv --plot runs/lstm_val.png
```

Repeat training with seeds 1 and 2. Add `--test` only for final experiments:

```bash
python train.py --model lstm --seed 0 --test
python evaluate.py runs/lstm_seed0/preds_test.csv --plot runs/lstm_test.png
```

## Model and shared tools

`models/lstm.py` uses a two-layer LSTM with 128 hidden units and dropout 0.25.
Input: `[N, 64, 13, 2]`; output: logits `[N, 15]`.

The branch includes the shared `penn_action.py`, `train.py`, `evaluate.py`,
`models/graph.py`, and fixed validation split. No other branch is required.

## Outputs

`runs/lstm_seed<seed>/` contains `config.json`, `log.csv`, `best.pt`, and
`preds_val.csv`; `--test` also exports `preds_test.csv`.
The best checkpoint is selected by validation macro-F1. Evaluation prints
accuracy, macro-F1 and per-class accuracy; `--plot` saves a confusion matrix.
