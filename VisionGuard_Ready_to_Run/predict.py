"""Classify one product image using a trained VisionGuard checkpoint."""
import argparse
import torch
from PIL import Image

from src.data import build_transforms
from src.models import build_model
from src.utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to one product image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pth")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = get_device()
    _, transform = build_transforms(args.image_size)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model = build_model(pretrained=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    class_names = checkpoint.get("class_names", ["normal", "defective"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()

    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    print(f"Prediction: {class_names[predicted_index]}")
    print("Class probabilities:")
    for name, probability in zip(class_names, probabilities):
        print(f"  {name}: {probability:.4f} ({probability*100:.2f}%)")
    print("\nEducational output only — not a certified quality-control decision.")


if __name__ == "__main__":
    main()
