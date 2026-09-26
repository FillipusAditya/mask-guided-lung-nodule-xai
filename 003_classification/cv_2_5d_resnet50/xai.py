"""Grad-CAM and channel-preserving LRP for 2.5D ResNet-50."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_unsigned(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1)
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def normalize_signed_channels(values: torch.Tensor) -> torch.Tensor:
    """Normalize [B,3,H,W] once per window, preserving channel magnitudes."""

    scale = values.abs().flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    return values / scale.clamp_min(1e-12)


class GradCAM:
    """Standard Grad-CAM: one spatial map for the complete 2.5D window."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = model.layer4[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def generate(self, inputs: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        scores = self.model(inputs)
        scores.gather(1, classes[:, None]).sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        maps = F.interpolate(maps, inputs.shape[-2:], mode="bilinear", align_corners=False)
        return maps[:, 0].detach()

    def close(self) -> None:
        self.handle.remove()


def generate_lrp(
    model: nn.Module, inputs: torch.Tensor, classes: torch.Tensor
) -> torch.Tensor:
    """Return signed input relevance without collapsing the three slices."""

    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonPlusFlat
        from zennit.torchvision import ResNetCanonizer
    except ImportError as error:
        raise RuntimeError("LRP requires zennit==0.5.1.") from error
    relevance_input = inputs.detach().requires_grad_(True)
    target = torch.zeros((len(inputs), 2), device=inputs.device)
    target.scatter_(1, classes[:, None], 1.0)
    model.zero_grad(set_to_none=True)
    with Gradient(
        model=model,
        composite=EpsilonPlusFlat(canonizers=[ResNetCanonizer()]),
    ) as attributor:
        _, relevance = attributor(relevance_input, target)
    return relevance.detach()

