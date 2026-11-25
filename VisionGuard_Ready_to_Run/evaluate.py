"""Evaluate the best checkpoint on the held-out test set."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import make_datasets
from src.models import build_model
from src.utils import get_device, load_checkpoint, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = get_device()
    _, _, test_ds = make_datasets(args.data_dir, args.image_size)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(pretrained=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    model.eval()

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())

    labels = [0, 1]
    class_names = checkpoint.get("class_names", test_ds.classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else None
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0)
    metrics = {
        "accuracy": float(accuracy),
        "normal_precision": float(precision[0]), "normal_recall": float(recall[0]), "normal_f1": float(f1[0]),
        "defective_precision": float(precision[1]), "defective_recall": float(recall[1]),
        "defective_f1": float(f1[1]), "roc_auc": float(auc) if auc is not None else None,
        "test_images": len(y_true),
    }

    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set(title="VisionGuard — Confusion Matrix", xlabel="Predicted label", ylabel="True label",
           xticks=[0, 1], yticks=[0, 1], xticklabels=class_names, yticklabels=class_names)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(title="VisionGuard — ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "roc_curve.png", dpi=160)
        plt.close(fig)

    print("\nTEST METRICS")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\n{report}")
    print(f"Reports saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
