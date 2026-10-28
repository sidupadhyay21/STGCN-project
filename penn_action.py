"""Penn Action skeletons: parse the label files, normalize, cache, and serve clips.

The data decisions live in the constants below and in PennAction's arguments, so they
can be changed in one place.
"""
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

JOINTS = ["head", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
# Names as stored in the .mat files (some differ from the dataset README), sorted alphabetically.
CLASSES = ["baseball_pitch", "baseball_swing", "bench_press", "bowl", "clean_and_jerk",
           "golf_swing", "jump_rope", "jumping_jacks", "pullup", "pushup", "situp", "squat",
           "strum_guitar", "tennis_forehand", "tennis_serve"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Penn_Action"
CACHE_DIR = ROOT / "cache"
VAL_SPLIT_FILE = ROOT / "splits" / "val_ids.txt"

NUM_FRAMES = 64      # every clip is resampled to this many evenly spaced frames
VAL_FRACTION = 0.15  # share of the official training clips held out for validation
SPLIT_SEED = 0


def load_clip(path):
    """Read one label file. Returns (xy [n, V, 2], vis [n, V], label, is_train)."""
    m = sio.loadmat(path, squeeze_me=True)
    xy = np.stack([m["x"], m["y"]], axis=-1).astype(np.float32)
    vis = m["visibility"].astype(bool)
    return xy, vis, CLASSES.index(str(m["action"])), int(m["train"]) == 1


def normalize(xy, vis):
    """Center on the mean visible joint and divide by the larger side of the visible joints'
    extent. Both are computed once per clip, so movement within the clip is preserved."""
    pts = xy[vis] if vis.any() else xy.reshape(-1, 2)
    center = pts.mean(axis=0)
    scale = max((pts.max(axis=0) - pts.min(axis=0)).max(), 1e-6)
    return (xy - center) / scale


def resample(arr, num_frames):
    """Pick num_frames evenly spaced frames (short clips repeat frames)."""
    idx = np.linspace(0, len(arr) - 1, num_frames).round().astype(int)
    return arr[idx]


def build_cache(path, num_frames):
    files = sorted((DATA_DIR / "labels").glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No label files in {DATA_DIR / 'labels'}; see the README for the download.")
    xs, viss, labels, is_train = [], [], [], []
    for f in files:
        xy, vis, label, train = load_clip(f)
        xs.append(resample(normalize(xy, vis), num_frames))
        viss.append(resample(vis, num_frames))
        labels.append(label)
        is_train.append(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.stack(xs), vis=np.stack(viss), label=np.array(labels),
                        is_train=np.array(is_train), video_id=np.array([f.stem for f in files]))


def make_val_split(video_ids, labels):
    """Hold out a class-stratified share of the training clips and save their ids."""
    _, val_ids = train_test_split(video_ids, test_size=VAL_FRACTION, stratify=labels,
                                  random_state=SPLIT_SEED)
    VAL_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    VAL_SPLIT_FILE.write_text("\n".join(sorted(val_ids)) + "\n")


class PennAction(Dataset):
    """Clips from one split: "train", "val" or "test".

    Each item is {"x": float [T, V, 2], "vis": bool [T, V], "label": int, "video_id": str}.
    invisible="zero" moves invisible joints to (0, 0) after normalization (the clip center);
    "keep" passes the annotation through, which is often a placeholder near the image corner.
    """

    def __init__(self, split, num_frames=NUM_FRAMES, invisible="zero"):
        cache = CACHE_DIR / f"penn_action_T{num_frames}.npz"
        if not cache.exists():
            build_cache(cache, num_frames)
        d = dict(np.load(cache))
        if not VAL_SPLIT_FILE.exists():
            make_val_split(d["video_id"][d["is_train"]], d["label"][d["is_train"]])
        in_val = np.isin(d["video_id"], VAL_SPLIT_FILE.read_text().split())
        mask = {"train": d["is_train"] & ~in_val, "val": in_val, "test": ~d["is_train"]}[split]

        x, vis = d["x"][mask], d["vis"][mask]
        if invisible == "zero":
            x = np.where(vis[..., None], x, 0.0)
        self.x = torch.from_numpy(x).float()
        self.vis = torch.from_numpy(vis)
        self.label = torch.from_numpy(d["label"][mask]).long()
        self.video_id = d["video_id"][mask].tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {"x": self.x[i], "vis": self.vis[i], "label": self.label[i], "video_id": self.video_id[i]}
