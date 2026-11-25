"""Model definitions for VisionGuard."""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True):
    """Create a ResNet-18 transfer-learning classifier."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze the pretrained feature extractor first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final residual block and the new classification head.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(in_features, num_classes),
    )
    return model
