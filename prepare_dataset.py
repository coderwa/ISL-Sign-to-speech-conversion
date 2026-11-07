"""Extract MediaPipe hand landmarks from an ISL image dataset -> data/processed/*.npz

Supported layouts (class folders must be named 1-9 / A-Z, case-insensitive):

  A) pre-split:   root/Training (A-Z)/A/*.jpg, root/Validation.../, root/Testing.../
                  (also train/val/test, Train/Valid/Test ...)
  B) flat:        root/A/*.jpg, root/1/*.jpg ...   -> random stratified 70/15/15 split

Usage:
  python download_hand_model.py
  python prepare_dataset.py "path/to/dataset" [--zero-as-o]
"""
import argparse, json, re
from collections import Counter
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import (CLASS_NAMES, HAND_LANDMARKER_PATH, PROCESSED_DIR, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
                    MAX_HANDS, STATIC_DETECTION_CONFIDENCE, SEED, ensure_dirs)
from features import pixel_hands_from_mediapipe

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}


def split_key(folder_name):
    token = re.sub(r"[^a-z]", " ", folder_name.lower()).split()
    return SPLIT_ALIASES.get(token[0]) if token else None


def class_label(folder_name, zero_as_o):
    name = folder_name.strip().upper()
    if name in CLASS_NAMES:
        return name
    if name == "0" and zero_as_o:
        return "O"          # ISL zero and letter O share one handshape
    return None


def discover(root, zero_as_o):
    """Return {split: [(path, label), ...]} (split may be 'all' for flat layouts)."""
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    splits = {split_key(d.name): d for d in subdirs if split_key(d.name)}
    skipped = Counter()
    result = {}

    def collect(base):
        items = []
        for cdir in sorted(p for p in base.iterdir() if p.is_dir()):
            label = class_label(cdir.name, zero_as_o)
            if label is None:
                skipped[cdir.name] += 1
                continue
            for f in sorted(cdir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                    items.append((f, label))
        return items

    if "train" in splits:
        for name, d in splits.items():
            result[name] = collect(d)
    else:
        result["all"] = collect(root)
    return result, skipped


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)           # unicode/Windows-path safe
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def video_frames(path, count):
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); frames = []
    for pos in (np.linspace(0, max(n - 1, 0), count).astype(int) if n > 0 else []):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos)); ok, bgr = cap.read()
        if ok: frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release(); return frames


def detect(landmarker, rgb):
    """Landmarks for one RGB image; retries on the mirrored image if nothing is found."""
    h, w = rgb.shape[:2]
    if max(h, w) > 1280:
        s = 1280 / max(h, w); rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA); h, w = rgb.shape[:2]
    for candidate in (rgb, rgb[:, ::-1]):
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(candidate)))
        if res.hand_landmarks:
            return pixel_hands_from_mediapipe(res.hand_landmarks, w, h)
    return None


def stratified_split(items, fractions=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(SEED); out = {"train": [], "val": [], "test": []}
    by_label = {}
    for it in items: by_label.setdefault(it[1], []).append(it)
    for label, lst in by_label.items():
        idx = rng.permutation(len(lst)); n = len(lst)
        n_val, n_test = int(round(n * fractions[1])), int(round(n * fractions[2]))
        for k, i in enumerate(idx):
            out["val" if k < n_val else "test" if k < n_val + n_test else "train"].append(lst[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_root"); ap.add_argument("--zero-as-o", action="store_true",
                    help="merge a '0' folder into class O (identical sign in ISL)")
    ap.add_argument("--video-frames", type=int, default=10, help="frames sampled per video file")
    args = ap.parse_args()
    ensure_dirs()
    root = Path(args.dataset_root).resolve()
    if not root.is_dir(): raise SystemExit(f"Dataset folder not found: {root}")
    if not HAND_LANDMARKER_PATH.exists(): raise SystemExit("Run `python download_hand_model.py` first.")

    splits, skipped = discover(root, args.zero_as_o)
    if skipped: print("Ignored folders (not 1-9/A-Z):", ", ".join(f"{k}" for k in skipped))
    if "all" in splits:
        splits = stratified_split(splits["all"]); print("No train/val/test folders found: created a 70/15/15 split.")
    if not splits.get("train"): raise SystemExit("No training images found. Check the folder layout in the README.")

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
        running_mode=vision.RunningMode.IMAGE, num_hands=MAX_HANDS,
        min_hand_detection_confidence=STATIC_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=STATIC_DETECTION_CONFIDENCE)
    summary = {}
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for split in ("train", "val", "test"):
            items = splits.get(split, [])
            if not items: continue
            H, M, Y, P, failed = [], [], [], [], Counter()
            for n, (path, label) in enumerate(items, 1):
                frames = ([read_image(path)] if path.suffix.lower() in IMAGE_EXTENSIONS
                          else video_frames(path, args.video_frames))
                for rgb in frames:
                    r = detect(landmarker, rgb) if rgb is not None else None
                    if r is None: failed[label] += 1; continue
                    H.append(r[0]); M.append(r[1]); Y.append(CLASS_NAMES.index(label)); P.append(str(path.name))
                if n % 200 == 0: print(f"  [{split}] {n}/{len(items)}")
            np.savez_compressed(PROCESSED_DIR / f"{split}.npz", hands=np.stack(H) if H else np.zeros((0, 2, 21, 3), np.float32),
                                mask=np.stack(M) if M else np.zeros((0, 2), bool), y=np.asarray(Y, np.int64), names=np.asarray(P))
            counts = Counter(CLASS_NAMES[i] for i in Y)
            summary[split] = {"kept": len(Y), "no_hand_detected": sum(failed.values()),
                              "per_class": dict(sorted(counts.items())), "failed_per_class": dict(failed),
                              "two_hand_samples": int(sum(m.sum() == 2 for m in M))}
            print(f"{split}: kept {len(Y)}, skipped (no hand) {sum(failed.values())}, two-hand samples {summary[split]['two_hand_samples']}")
            absent = [c for c in CLASS_NAMES if c not in counts]
            if absent: print(f"  WARNING classes with no samples in {split}: {', '.join(absent)}")
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved to {PROCESSED_DIR}. Next: python train.py")


if __name__ == "__main__":
    main()
