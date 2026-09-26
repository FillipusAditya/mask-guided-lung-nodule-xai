"""ResNet-50 construction for pseudo-RGB adjacent CT slices."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def resolve_weights(name: str):
    name = name.upper()
    if name == "DEFAULT":
        return ResNet50_Weights.DEFAULT
    if name == "IMAGENET1K_V2":
        return ResNet50_Weights.IMAGENET1K_V2
    if name == "NONE":
        return None
    raise ValueError(f"Unsupported pretrained weights: {name}")


def build_model(config: dict, load_pretrained: bool = True) -> nn.Module:
    model_config = config["model"]
    weights = resolve_weights(model_config["pretrained_weights"]) if load_pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(float(model_config["classifier_dropout"])),
        nn.Linear(model.fc.in_features, int(model_config["num_classes"])),
    )
    return model

