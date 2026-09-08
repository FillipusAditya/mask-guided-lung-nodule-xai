"""Residual probability-attention ResNet-50 architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class ProbabilityAttention(nn.Module):
    """Project a one-channel probability map to a layer-3 attention tensor."""

    def __init__(
        self,
        channels: int = 1024,
        hidden_channels: int = 64,
        alpha_initial_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        # Alpha=0 makes the initial network exactly equivalent to its ResNet
        # backbone and lets training introduce mask guidance gradually.
        self.alpha = nn.Parameter(torch.tensor(float(alpha_initial_value)))

    def forward(
        self,
        probability_map: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        probability_map = F.interpolate(
            probability_map,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.encoder(probability_map)


class SegmentationGuidedResNet50(nn.Module):
    """ResNet-50 with residual multiplicative attention after layer3.

    The input is a four-channel tensor: normalized CT RGB channels followed by
    the unnormalized U-Net foreground probability map in [0, 1].
    """

    architecture_name = "SegmentationGuidedResNet50"

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT,
        attention_hidden_channels: int = 64,
        attention_alpha_initial_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = models.resnet50(weights=weights)
        self.attention = ProbabilityAttention(
            channels=1024,
            hidden_channels=attention_hidden_channels,
            alpha_initial_value=attention_alpha_initial_value,
        )
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.fc.in_features, num_classes),
        )

    def forward_features(
        self,
        ct: torch.Tensor,
        probability_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return guided layer-3 features and their learned attention map."""

        backbone = self.backbone
        features = backbone.conv1(ct)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)
        features = backbone.layer1(features)
        features = backbone.layer2(features)
        features = backbone.layer3(features)
        attention = self.attention(probability_map, features.shape[-2:])
        guided_features = features * (1.0 + self.attention.alpha * attention)
        return guided_features, attention

    def forward_from_inputs(
        self,
        ct: torch.Tensor,
        probability_map: torch.Tensor,
    ) -> torch.Tensor:
        """Classify separate CT and probability-map tensors."""

        features, _ = self.forward_features(ct, probability_map)
        features = self.backbone.layer4(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return self.backbone.fc(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 4:
            raise ValueError(
                "Expected [B, 4, H, W]: three CT channels plus one "
                f"probability channel, received {tuple(inputs.shape)}."
            )
        return self.forward_from_inputs(inputs[:, :3], inputs[:, 3:4])


class FixedAttentionInputModel(nn.Module):
    """Expose the CT branch as a three-channel model for LRP attribution."""

    def __init__(
        self,
        model: SegmentationGuidedResNet50,
        probability_map: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("probability_map", probability_map.detach().clone())

    def forward(self, ct: torch.Tensor) -> torch.Tensor:
        probability_map = self.probability_map.expand(ct.shape[0], -1, -1, -1)
        return self.model.forward_from_inputs(ct, probability_map)
