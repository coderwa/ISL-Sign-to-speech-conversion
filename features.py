"""Shared landmark -> feature conversion and augmentation.

One implementation is used by prepare_dataset.py, train.py and main.py so the
training and live-inference feature spaces can never drift apart.

Raw landmarks are stored as (2, 21, 3) arrays in *pixel-like* units
(x*width, y*height, z*width) plus a (2,) boolean presence mask.
"""
import numpy as np
from config import INPUT_SIZE, HAND_FEATURES, MAX_HANDS


def _ordered_hands(hands, mask):
    """Present hands sorted left->right by wrist x (mirror-consistent)."""
    idx = [i for i in range(MAX_HANDS) if mask[i]]
    idx.sort(key=lambda i: float(hands[i, 0, 0]))
    return [hands[i] for i in idx]


def hands_to_features(hands, mask):
    out = np.zeros(INPUT_SIZE, dtype=np.float32)
    present = _ordered_hands(hands, mask)
    wrists, scales = [], []
    for slot, pts in enumerate(present[:MAX_HANDS]):
        rel = pts - pts[0:1]
        s = float(np.linalg.norm(rel[9]))                  # wrist -> middle MCP
        if s < 1e-6:
            s = float(np.max(np.linalg.norm(rel, axis=1)))
        s = max(s, 1e-6)
        out[slot * HAND_FEATURES:(slot + 1) * HAND_FEATURES] = (rel / s).reshape(-1)
        out[MAX_HANDS * HAND_FEATURES + 3 + slot] = 1.0     # presence flag
        wrists.append(pts[0]); scales.append(s)
    if len(present) >= 2:
        off = (wrists[1] - wrists[0]) / (0.5 * (scales[0] + scales[1]))
        out[MAX_HANDS * HAND_FEATURES:MAX_HANDS * HAND_FEATURES + 3] = np.clip(off, -10, 10)
    return out


def _rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def augment(hands, mask, rng):
    """Geometry-aware augmentation on raw landmarks (returns a new array)."""
    h = hands.copy()
    if rng.random() < 0.5:                                  # mirror (left/right-handed, selfie view)
        h[..., 0] *= -1.0
    present = [i for i in range(MAX_HANDS) if mask[i]]
    if not present:
        return h
    centre = h[present].reshape(-1, 3).mean(0)
    size = max(float(np.linalg.norm(h[present[0], 9] - h[present[0], 0])), 1e-6)
    d = np.deg2rad
    R = _rotation(d(rng.uniform(-8, 8)), d(rng.uniform(-8, 8)), d(rng.uniform(-15, 15)))
    stretch = np.array([rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), rng.uniform(0.7, 1.3)], np.float32)
    for i in present:
        pts = (h[i] - centre) @ R.T
        pts = pts * stretch + centre
        pts += rng.normal(0, 0.01 * size, pts.shape).astype(np.float32) * np.array([1, 1, 2], np.float32)
        h[i] = pts
    return h


def pixel_hands_from_mediapipe(hand_landmarks_list, width, height):
    """MediaPipe result.hand_landmarks -> ((2,21,3) array, (2,) mask)."""
    hands = np.zeros((MAX_HANDS, 21, 3), dtype=np.float32)
    mask = np.zeros(MAX_HANDS, dtype=bool)
    for i, lm in enumerate(hand_landmarks_list[:MAX_HANDS]):
        hands[i] = np.asarray([[p.x * width, p.y * height, p.z * width] for p in lm], np.float32)
        mask[i] = True
    return hands, mask
