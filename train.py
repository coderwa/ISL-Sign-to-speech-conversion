"""Train a ResNet18 ISL A-Z classifier.

Expected data layout:
  data/train/A ... Z
  data/val/A ... Z   (optional)
  data/test/A ... Z  (optional)

If data/val is missing, 15% of data/train is used for validation.
"""
from pathlib import Path
import argparse, random, json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import build_model
from config import *


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_datasets():
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0)),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
        transforms.RandomHorizontalFlip(p=0.20),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"
    if not train_dir.exists():
        raise SystemExit("Missing data/train. Prepare the ISL dataset first.")

    full = datasets.ImageFolder(train_dir, transform=train_tf)
    if val_dir.exists() and any(val_dir.iterdir()):
        train_ds = full
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    else:
        n_val = max(1, int(0.15 * len(full)))
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(SEED)
        train_ds, val_raw = random_split(full, [n_train, n_val], generator=gen)
        # random_split keeps train transforms for val; replace via wrapper.
        val_base = datasets.ImageFolder(train_dir, transform=eval_tf)
        val_ds = torch.utils.data.Subset(val_base, val_raw.indices)

    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf) if test_dir.exists() and any(test_dir.iterdir()) else None
    return train_ds, val_ds, test_ds, full.classes


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss = total = correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward(); optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item(); total += x.size(0)
    return total_loss / max(1,total), correct / max(1,total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    seed_everything(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds, test_ds, classes = make_datasets()
    if classes != CLASS_NAMES:
        raise SystemExit(f"Expected A-Z classes, found: {classes}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS) if test_ds else None

    model = build_model(NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    best_acc = -1
    start_epoch = 0
    checkpoint = MODEL_DIR / "checkpoint.pth"
    if checkpoint.exists() and not args.fresh:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1; best_acc = ck.get("best_acc", -1)
        print(f"Resuming from epoch {start_epoch}; best val accuracy={best_acc:.2%}")

    print(f"Device: {device}; train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds) if test_ds else 0}")
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        print(f"Epoch {epoch+1}/{args.epochs} | train {tr_acc:.2%} | val {va_acc:.2%} | val loss {va_loss:.4f}")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_acc": best_acc}, checkpoint)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": best_acc * 100, "classes": classes}, MODEL_PATH)
            print(f"  Saved best model: {MODEL_PATH} ({best_acc:.2%})")

    if test_loader:
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device)
        print(f"Test accuracy: {te_acc:.2%}")
    LABELS_PATH.write_text("\n".join(classes), encoding="utf-8")
    print("Training complete.")


if __name__ == "__main__": main()
