"""Create reproducible stratified train/validation/test image folders."""
import argparse
import random
import shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--output", default="data/split")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-8:
        parser.error("--train + --val + --test must equal 1.0")

    source, output = Path(args.source), Path(args.output)
    classes = ["normal", "defective"]
    rng = random.Random(args.seed)

    if output.exists():
        shutil.rmtree(output)
    for split in ["train", "val", "test"]:
        for cls in classes:
            (output / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        class_dir = source / cls
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        images = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
        if len(images) < 7:
            raise ValueError(f"Class '{cls}' has {len(images)} images. Add at least 7; more is strongly recommended.")
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # Keep at least one sample for each split.
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train -= 1
        if n_train < 1:
            raise ValueError(f"Not enough images in {cls} for all three splits.")
        groups = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:],
        }
        for split, paths in groups.items():
            for index, src in enumerate(paths):
                # Prefix index to avoid same-name collisions from nested folders.
                dst = output / split / cls / f"{index:06d}_{src.name}"
                shutil.copy2(src, dst)
        print(f"{cls}: total={n}, train={len(groups['train'])}, "
              f"val={len(groups['val'])}, test={len(groups['test'])}")

    print(f"\nDataset split created at: {output.resolve()}")


if __name__ == "__main__":
    main()
