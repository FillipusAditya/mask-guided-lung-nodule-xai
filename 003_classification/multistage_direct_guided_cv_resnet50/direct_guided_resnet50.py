"""ResNet-50 with direct probability-map guidance at three stages."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


def resize_probability_map(
    probability_map: Tensor,
    target_size: tuple[int, int],
    mode: str = "max",
) -> Tensor:
    """Resize a probability map to a feature map's spatial size."""

    if mode not in {"max", "bilinear"}:
        raise ValueError(
            "mode must be 'max' or 'bilinear', "
            f"received {mode!r}."
        )

    if probability_map.shape[-2:] == target_size:
        return probability_map

    input_height, input_width = probability_map.shape[-2:]
    target_height, target_width = target_size

    if (
        mode == "max"
        and target_height <= input_height
        and target_width <= input_width
    ):
        return F.adaptive_max_pool2d(probability_map, target_size)

    # Max pooling is only meaningful for reduction. Bilinear interpolation is
    # used when a probability map must be enlarged.
    return F.interpolate(
        probability_map,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )


def apply_direct_guidance(
    feature_map: Tensor,
    probability_map: Tensor,
    alpha: float | Tensor,
) -> Tensor:
    """Amplify guided locations while preserving features where P is zero."""

    return feature_map * (1.0 + alpha * probability_map)


class MultiStageDirectGuidedResNet50(nn.Module):
    """ResNet-50 with direct guidance after layers 1, 2, and 3."""

    architecture_name = "MultiStageDirectGuidedResNet50"

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        weights: ResNet50_Weights | None = ResNet50_Weights.DEFAULT,
        alpha_layer1: float = 1.0,
        alpha_layer2: float = 1.0,
        alpha_layer3: float = 1.0,
        learnable_alpha: bool = False,
        guidance_resize_mode: str = "max",
    ) -> None:
        super().__init__()

        if guidance_resize_mode not in {"max", "bilinear"}:
            raise ValueError(
                "guidance_resize_mode must be 'max' or 'bilinear', "
                f"received {guidance_resize_mode!r}."
            )

        self.guidance_resize_mode = guidance_resize_mode
        self.learnable_alpha = learnable_alpha

        backbone = models.resnet50(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

        self._store_alpha("alpha_layer1", alpha_layer1)
        self._store_alpha("alpha_layer2", alpha_layer2)
        self._store_alpha("alpha_layer3", alpha_layer3)

    def _store_alpha(self, name: str, value: float) -> None:
        """Store an alpha value as a fixed buffer or trainable parameter."""

        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

        alpha = torch.tensor(value, dtype=torch.float32)
        if self.learnable_alpha:
            self.register_parameter(name, nn.Parameter(alpha))
        else:
            self.register_buffer(name, alpha)

    @staticmethod
    def _validate_inputs(ct_image: Tensor, probability_map: Tensor) -> None:
        """Check the required CT and probability-map dimensions."""

        if ct_image.ndim != 4:
            raise ValueError(
                "CT input must have shape [batch_size, channels, height, width]."
            )
        if ct_image.shape[1] not in {1, 3}:
            raise ValueError(
                "CT input must contain either one or three channels, "
                f"received {ct_image.shape[1]}."
            )
        if probability_map.ndim != 4:
            raise ValueError(
                "Probability map must have shape "
                "[batch_size, 1, height, width]."
            )
        if probability_map.shape[1] != 1:
            raise ValueError(
                "Probability map must contain one channel, "
                f"received {probability_map.shape[1]}."
            )
        if probability_map.shape[0] != ct_image.shape[0]:
            raise ValueError(
                "Probability-map batch size must match CT batch size, "
                f"received {probability_map.shape[0]} and {ct_image.shape[0]}."
            )

    def _resize_guidance(
        self,
        probability_map: Tensor,
        feature_map: Tensor,
    ) -> Tensor:
        """Resize guidance to one ResNet stage and match its data type."""

        resized_map = resize_probability_map(
            probability_map=probability_map,
            target_size=feature_map.shape[-2:],
            mode=self.guidance_resize_mode,
        )
        return resized_map.to(dtype=feature_map.dtype)

    def forward(
        self,
        ct_image: Tensor,
        probability_map: Tensor,
        return_guidance_maps: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return raw logits and optionally the three resized guidance maps."""

        self._validate_inputs(ct_image, probability_map)

        if ct_image.shape[1] == 1:
            ct_image = ct_image.repeat(1, 3, 1, 1)

        features = self.conv1(ct_image)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)

        features = self.layer1(features)
        guidance_layer1 = self._resize_guidance(probability_map, features)
        features = apply_direct_guidance(
            features,
            guidance_layer1,
            self.alpha_layer1,
        )

        features = self.layer2(features)
        guidance_layer2 = self._resize_guidance(probability_map, features)
        features = apply_direct_guidance(
            features,
            guidance_layer2,
            self.alpha_layer2,
        )

        features = self.layer3(features)
        guidance_layer3 = self._resize_guidance(probability_map, features)
        features = apply_direct_guidance(
            features,
            guidance_layer3,
            self.alpha_layer3,
        )

        features = self.layer4(features)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        logits = self.classifier(features)

        if not return_guidance_maps:
            return logits

        guidance_maps = {
            "layer1": guidance_layer1,
            "layer2": guidance_layer2,
            "layer3": guidance_layer3,
        }
        return logits, guidance_maps


if __name__ == "__main__":
    # No weights are needed for this offline shape-only smoke test.
    model = MultiStageDirectGuidedResNet50(weights=None)
    model.eval()

    dummy_ct_batch = torch.randn(2, 1, 224, 224)
    dummy_probability_batch = torch.rand(2, 1, 512, 512)

    with torch.no_grad():
        output_logits, output_guidance_maps = model(
            dummy_ct_batch,
            dummy_probability_batch,
            return_guidance_maps=True,
        )

    print("CT input shape:", tuple(dummy_ct_batch.shape))
    print("Probability-map input shape:", tuple(dummy_probability_batch.shape))
    print("Output logits shape:", tuple(output_logits.shape))
    for stage_name, guidance_map in output_guidance_maps.items():
        print(f"{stage_name} guidance shape:", tuple(guidance_map.shape))
