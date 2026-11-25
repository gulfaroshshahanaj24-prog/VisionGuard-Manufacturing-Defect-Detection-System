"""Dataset and transform helpers."""
from pathlib import Path
from torchvision import datasets, transforms

CLASS_NAMES = ["normal", "defective"]


def build_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def make_datasets(data_dir, image_size=224):
    data_dir = Path(data_dir)
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    expected = CLASS_NAMES
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds.classes != expected:
            raise ValueError(
                f"{name} class folders must be exactly {expected}; found {ds.classes}. "
                "Check folder names and spelling."
            )
    return train_ds, val_ds, test_ds
