"""Penn Action skeleton graph and the adjacency partitions used by graph convolutions."""
from collections import deque

import numpy as np

NUM_JOINTS = 13
# Joint order follows the dataset: 0 head, 1 l_shoulder, 2 r_shoulder, 3 l_elbow, 4 r_elbow,
# 5 l_wrist, 6 r_wrist, 7 l_hip, 8 r_hip, 9 l_knee, 10 r_knee, 11 l_ankle, 12 r_ankle.
EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
         (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
# Penn Action has no pelvis joint, so the two hips together act as the body center.
CENTER = (7, 8)


def hop_distance_to_center():
    neighbors = [[] for _ in range(NUM_JOINTS)]
    for i, j in EDGES:
        neighbors[i].append(j)
        neighbors[j].append(i)
    dist = np.full(NUM_JOINTS, -1)
    dist[list(CENTER)] = 0
    queue = deque(CENTER)
    while queue:
        i = queue.popleft()
        for j in neighbors[i]:
            if dist[j] < 0:
                dist[j] = dist[i] + 1
                queue.append(j)
    return dist


def adjacency(strategy="spatial"):
    """Return A with shape (K, V, V), where A[k, src, dst] weights the message src -> dst.

    Each column is divided by the destination's degree (self-loop included), so every joint
    averages over itself and its neighbors. "uniform" gives K=1. "spatial" is ST-GCN's
    partitioning into K=3 groups: neighbors as far from the center as the joint (including
    itself), neighbors closer to the center, and neighbors farther away.
    """
    A = np.eye(NUM_JOINTS)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1
    A = A / A.sum(axis=0, keepdims=True)
    if strategy == "uniform":
        return A[None].astype(np.float32)
    if strategy != "spatial":
        raise ValueError(f"unknown strategy {strategy!r}")
    d = hop_distance_to_center()
    src, dst = d[:, None], d[None, :]
    return np.stack([A * (src == dst), A * (src < dst), A * (src > dst)]).astype(np.float32)
