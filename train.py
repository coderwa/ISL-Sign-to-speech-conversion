"""Train/evaluate the 35-class ISL landmark classifier from data/processed/*.npz."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from config import (PROCESSED_DIR, MODEL_DIR, CHECKPOINT_PATH, LABELS_PATH, CLASS_NAMES, INPUT_SIZE,
                    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, NUM_WORKERS, PATIENCE, SEED,
                    LABEL_SMOOTHING, HIDDEN_SIZE, NUM_BLOCKS, DROPOUT, ensure_dirs)
from features import hands_to_features, augment
from model import build_model

NUM = len(CLASS_NAMES)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class LandmarkDataset(Dataset):
    def __init__(self, hands, mask, y, train=False, seed=SEED):
        self.hands, self.mask, self.y, self.train = hands, mask, y, train
        self.rng = np.random.default_rng(seed)
        # Evaluation features are fixed, so compute once. Training features are re-augmented each epoch.
        self.cache = None if train else np.stack([hands_to_features(h, m) for h, m in zip(hands, mask)]).astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        if self.train:
            f = hands_to_features(augment(self.hands[i], self.mask[i], self.rng), self.mask[i])
        else:
            f = self.cache[i]
        return torch.from_numpy(f), int(self.y[i])


def load_npz(name):
    path = PROCESSED_DIR / f"{name}.npz"
    if not path.exists(): return None
    with np.load(path, allow_pickle=False) as d:
        if len(d["y"]) == 0: return None
        return d["hands"].astype(np.float32), d["mask"].astype(bool), d["y"].astype(np.int64)


def stratified_holdout(hands, mask, y, fraction=0.15):
    rng = np.random.default_rng(SEED); tr, va = [], []
    for cls in np.unique(y):
        ids = np.where(y == cls)[0]; rng.shuffle(ids)
        n = int(round(len(ids) * fraction)) if len(ids) >= 4 else 0
        va.extend(ids[:n].tolist()); tr.extend(ids[n:].tolist())
    return (hands[tr], mask[tr], y[tr]), (hands[va], mask[va], y[va])


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval(); total_loss = 0.0; n = 0; truths, preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device); logits = model(x)
        total_loss += criterion(logits, y).item() * len(y); n += len(y)
        preds.extend(logits.argmax(1).cpu().tolist()); truths.extend(y.cpu().tolist())
    return (total_loss / max(n, 1), accuracy_score(truths, preds),
            f1_score(truths, preds, average="macro", zero_division=0), np.asarray(truths), np.asarray(preds))


def report(name, model, loader, criterion, device):
    loss, acc, f1, truth, pred = evaluate(model, loader, criterion, device)
    print(f"{name.upper()} loss={loss:.4f} accuracy={acc:.4f} macro_f1={f1:.4f}")
    labels = list(range(NUM))
    (MODEL_DIR / f"classification_report_{name}.json").write_text(json.dumps(
        classification_report(truth, pred, labels=labels, target_names=CLASS_NAMES, output_dict=True, zero_division=0), indent=2), encoding="utf-8")
    cm = confusion_matrix(truth, pred, labels=labels)
    np.savetxt(MODEL_DIR / f"confusion_matrix_{name}.csv", cm, fmt="%d", delimiter=",")
    off = [(cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(NUM) for j in range(NUM) if i != j and cm[i, j] > 0]
    if off:
        print("  Most confused (true -> predicted): " + ", ".join(f"{a}->{b} x{c}" for c, a, b in sorted(off, reverse=True)[:8]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS); ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE); ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--resume", action="store_true", help="start from the saved checkpoint weights")
    ap.add_argument("--allow-missing-classes", action="store_true")
    args = ap.parse_args()
    ensure_dirs(); seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; classes={NUM}; input={INPUT_SIZE}")

    train = load_npz("train")
    if train is None: raise SystemExit("Missing data/processed/train.npz. Run prepare_dataset.py first.")
    missing = [CLASS_NAMES[i] for i in range(NUM) if not np.any(train[2] == i)]
    if missing and not args.allow_missing_classes:
        raise SystemExit("Training split has no samples for: " + ", ".join(missing) +
                         ". Fix the dataset (or the hand detection failures in data/processed/summary.json), "
                         "or pass --allow-missing-classes.")
    if missing: print("WARNING: these classes cannot be recognised:", ", ".join(missing))
    val = load_npz("val")
    if val is None:
        train, val = stratified_holdout(*train); print("No val.npz: held out 15% of train for validation.")
    test = load_npz("test")
    print(f"samples: train={len(train[2])} val={len(val[2])} test={len(test[2]) if test else 0}")

    counts = np.bincount(train[2], minlength=NUM); weights = 1.0 / np.maximum(counts[train[2]], 1)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = DataLoader(LandmarkDataset(*train, train=True), batch_size=args.batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=device.type == "cuda", drop_last=len(train[2]) > args.batch_size)
    val_loader = DataLoader(LandmarkDataset(*val), batch_size=256, num_workers=NUM_WORKERS)
    test_loader = DataLoader(LandmarkDataset(*test), batch_size=256, num_workers=NUM_WORKERS) if test else None

    cfg = {"hidden_size": HIDDEN_SIZE, "num_blocks": NUM_BLOCKS, "dropout": DROPOUT}
    model = build_model(num_classes=NUM, **cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.resume and CHECKPOINT_PATH.exists():
        ck = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        if ck.get("class_names") != CLASS_NAMES: raise SystemExit("Checkpoint classes differ from config; train from scratch.")
        model.load_state_dict(ck["model_state_dict"]); print("Loaded weights from existing checkpoint.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    steps = max(len(train_loader), 1) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()

    best_f1, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); running = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); running.append(loss.item())
        vl, va, vf, *_ = evaluate(model, val_loader, eval_criterion, device)
        print(f"Epoch {epoch:03d}/{args.epochs} train_loss={np.mean(running):.4f} val_loss={vl:.4f} val_acc={va:.4f} val_macro_f1={vf:.4f}")
        if vf > best_f1:
            best_f1, stale = vf, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state_dict": best_state, "class_names": CLASS_NAMES, "num_classes": NUM,
                        "input_size": INPUT_SIZE, "model_config": cfg, "best_val_macro_f1": best_f1}, CHECKPOINT_PATH)
        else:
            stale += 1
            if stale >= args.patience: print("Early stopping."); break

    model.load_state_dict(best_state)
    report("val", model, val_loader, eval_criterion, device)
    if test_loader: report("test", model, test_loader, eval_criterion, device)
    else: print("No test split found: val metrics above were also used for model selection, so treat them as optimistic.")
    LABELS_PATH.write_text(json.dumps({"classes": CLASS_NAMES, "class_to_idx": {c: i for i, c in enumerate(CLASS_NAMES)}}, indent=2), encoding="utf-8")
    print(f"Best checkpoint (val macro-F1 {best_f1:.4f}): {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
