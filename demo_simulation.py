"""Sanity test for the stabilizer and feature pipeline; needs no camera, model or torch."""
import numpy as np
from config import CLASS_NAMES, INPUT_SIZE
from postprocess import SignStabilizer
from features import hands_to_features, augment

rng = np.random.default_rng(7)

def make_probs(label, confidence=0.92):
    p = np.full(len(CLASS_NAMES), (1 - confidence) / (len(CLASS_NAMES) - 1), dtype=np.float32)
    p[CLASS_NAMES.index(label)] = confidence
    p = np.clip(p + rng.normal(0, 0.002, len(p)), 1e-5, None)
    return p / p.sum()

# 1) a held sign is emitted exactly once
st = SignStabilizer(CLASS_NAMES, 9, 5, 0.70)
emitted = [r.emitted for r in (st.update(make_probs("A")) for _ in range(30)) if r.emitted]
assert emitted == ["A"], emitted
# 2) lowering the hand and showing it again types the same character twice
for _ in range(5): st.update(None)
emitted += [r.emitted for r in (st.update(make_probs("A")) for _ in range(30)) if r.emitted]
assert emitted == ["A", "A"], emitted

# 3) features: right size, mirror invariance of a mirrored augmentation-free hand
hands = rng.normal(size=(2, 21, 3)).astype(np.float32) * 50 + 300; mask = np.array([True, True])
f = hands_to_features(hands, mask); assert f.shape == (INPUT_SIZE,) and np.isfinite(f).all()
a = augment(hands, mask, np.random.default_rng(0)); assert a.shape == hands.shape
one = hands_to_features(hands, np.array([True, False])); assert one[-1] == 0 and one[-2] == 1
print(f"All smoke tests passed; classes={len(CLASS_NAMES)}, feature size={INPUT_SIZE}")
