"""Train VisionGuard ResNet-18 with early stopping and TensorBoard."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import make_datasets
from src.models import build_model
from src.utils import get_device, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 for easiest Windows compatibility")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    train_ds, val_ds, _ = make_datasets(args.data_dir, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(num_classes=2, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = checkpoint_dir / "best.pth", checkpoint_dir / "last.pth"
    start_epoch, best_val_loss, stale = 0, float("inf"), 0

    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch + 1}")

    writer = SummaryWriter("runs/visionguard")
    history = []
    try:
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("Loss/validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
            writer.add_scalar("Accuracy/validation", val_acc, epoch + 1)
            row = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc,
                   "val_loss": val_loss, "val_accuracy": val_acc}
            history.append(row)
            print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f}, acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f}, acc {val_acc:.4f}")

            save_checkpoint(last_path, model, optimizer, epoch, best_val_loss, train_ds.classes)
            if val_loss < best_val_loss:
                best_val_loss, stale = val_loss, 0
                save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, train_ds.classes)
            else:
                stale += 1
            if stale >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break
    finally:
        writer.close()

    if history:
        history_path = Path("outputs/training_history.csv")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Best checkpoint: {best_path.resolve()}")


if __name__ == "__main__":
    main()
