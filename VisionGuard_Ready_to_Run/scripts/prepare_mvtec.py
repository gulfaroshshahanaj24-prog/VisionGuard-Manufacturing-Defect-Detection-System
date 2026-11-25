"""Convert one extracted MVTec AD category to binary normal/defective folders."""
import argparse
import shutil
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def copy_images(source, destination, prefix):
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTS:
            shutil.copy2(path, destination / f"{prefix}_{count:06d}_{path.name}")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Folder containing MVTec category folders")
    parser.add_argument("--category", required=True, help="For example: bottle, capsule, carpet, grid")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    category = Path(args.dataset_root) / args.category
    if not category.is_dir():
        raise FileNotFoundError(f"Category folder not found: {category}")

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)
    normal_count = 0
    defective_count = 0

    train_good = category / "train" / "good"
    test_root = category / "test"
    if train_good.exists():
        normal_count += copy_images(train_good, output / "normal", "train_good")
    if test_root.exists():
        good_test = test_root / "good"
        if good_test.exists():
            normal_count += copy_images(good_test, output / "normal", "test_good")
        for defect_dir in test_root.iterdir():
            if defect_dir.is_dir() and defect_dir.name.lower() != "good":
                defective_count += copy_images(defect_dir, output / "defective", defect_dir.name)
    if normal_count == 0 or defective_count == 0:
        raise ValueError("Could not find both normal and defective images. Check the extracted MVTec category structure.")
    print(f"Prepared category '{args.category}': normal={normal_count}, defective={defective_count}")
    print(f"Output: {output.resolve()}")
    print("Note: MVTec ground-truth masks are not used by this image-level classifier.")


if __name__ == "__main__":
    main()
