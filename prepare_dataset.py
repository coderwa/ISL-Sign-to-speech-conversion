"""Prepare a downloaded ISL alphabet dataset for training.

Supported layouts include the RealSign dataset:
  Dataset/Training (A-Z)/A/*.jpg
  Dataset/Testing (A-Z)/A/*.jpg
  Dataset/Validation (A-Z)/A/*.jpg

After extracting the dataset, pass its root directory to this script. It
creates data/train, data/val and data/test in ImageFolder-compatible form.
"""
from pathlib import Path
import argparse
import shutil

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split(root, candidates):
    for p in candidates:
        q = root / p
        if q.exists():
            return q
    return None


def copy_split(src, dst):
    if src is None:
        return 0
    count = 0
    dst.mkdir(parents=True, exist_ok=True)
    for letter in LETTERS:
        src_class = src / letter
        if not src_class.exists():
            continue
        out_class = dst / letter
        out_class.mkdir(parents=True, exist_ok=True)
        for f in src_class.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                target = out_class / f.name
                if target.exists():
                    target = out_class / f"{count}_{f.name}"
                shutil.copy2(f, target)
                count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="Extracted ISL dataset root")
    args = ap.parse_args()
    root = Path(args.dataset_root).resolve()
    project = Path(__file__).resolve().parent
    data = project / "data"

    train = find_split(root, ["Training (A-Z)", "Training", "train", "Train"])
    val = find_split(root, ["Validation (A-Z)", "Validation", "val", "Valid", "validation"])
    test = find_split(root, ["Testing (A-Z)", "Testing", "test", "Test"])

    if train is None:
        raise SystemExit("Could not find an ISL training folder containing A-Z class folders.")

    for name in ("train", "val", "test"):
        shutil.rmtree(data / name, ignore_errors=True)

    n_train = copy_split(train, data / "train")
    n_val = copy_split(val, data / "val")
    n_test = copy_split(test, data / "test")

    # If no validation split exists, train.py will create a validation split.
    print(f"Prepared ISL dataset: train={n_train}, val={n_val}, test={n_test}")
    print(f"Output: {data}")


if __name__ == "__main__":
    main()
