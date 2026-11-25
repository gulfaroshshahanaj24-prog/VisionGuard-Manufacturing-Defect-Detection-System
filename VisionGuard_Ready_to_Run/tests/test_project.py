from pathlib import Path
from PIL import Image
import tempfile

from src.data import build_transforms
from src.models import build_model


def test_transform_output_shape():
    _, transform = build_transforms(224)
    image = Image.new("RGB", (300, 200), color="white")
    tensor = transform(image)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_model_output_shape():
    import torch
    model = build_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert tuple(output.shape) == (1, 2)
