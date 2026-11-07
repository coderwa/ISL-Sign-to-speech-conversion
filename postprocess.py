"""Temporal probability smoothing and stable character emission."""
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class StabilizerResult:
    emitted: Optional[str]
    dominant: Optional[str]
    confidence: float
    vote_ratio: float


class SignStabilizer:
    def __init__(self, class_names, window_size=9, min_votes=5, confidence_threshold=0.70,
                 cooldown_frames=8, release_frames=4):
        self.class_names = list(class_names); self.window = deque(maxlen=window_size)
        self.min_votes = min_votes; self.threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames; self.release_frames = release_frames
        self.reset()

    def reset(self):
        self.window.clear(); self.cooldown = 0; self.last_emitted = None; self.missing = 0

    def update(self, probabilities):
        if probabilities is None:
            self.window.clear(); self.cooldown = max(0, self.cooldown - 1); self.missing += 1
            if self.missing >= self.release_frames:
                self.last_emitted = None          # hand left the frame: the same char may be typed again
            return StabilizerResult(None, None, 0.0, 0.0)
        self.missing = 0
        p = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if len(p) != len(self.class_names) or not np.all(np.isfinite(p)) or p.sum() <= 0:
            return StabilizerResult(None, None, 0.0, 0.0)
        self.window.append(p / p.sum())
        avg = np.mean(np.stack(self.window), axis=0); idx = int(np.argmax(avg)); label = self.class_names[idx]
        ratio = Counter(int(np.argmax(v)) for v in self.window)[idx] / len(self.window)
        confidence = float(avg[idx]); emitted = None
        if self.cooldown > 0:
            self.cooldown -= 1
        elif (len(self.window) >= self.min_votes and confidence >= self.threshold
              and ratio >= 0.60 and label != self.last_emitted):
            emitted = label; self.last_emitted = label; self.cooldown = self.cooldown_frames
        return StabilizerResult(emitted, label, confidence, float(ratio))
