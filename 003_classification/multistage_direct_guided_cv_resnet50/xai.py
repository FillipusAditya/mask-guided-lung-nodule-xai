"""Multi-stage Grad-CAM, LRP, and study-level visualization helpers."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from tqdm.auto import tqdm

if TYPE_CHECKING:
    try:
        from .dataset import DirectGuidedClassificationDataset
    except ImportError:
        from dataset import DirectGuidedClassificationDataset

try:
    from .direct_guided_resnet50 import (
        MultiStageDirectGuidedResNet50,
        apply_direct_guidance,
        resize_probability_map,
    )
except ImportError:
    from direct_guided_resnet50 import (
        MultiStageDirectGuidedResNet50,
        apply_direct_guidance,
        resize_probability_map,
    )


STAGE_NAMES = ("layer1", "layer2", "layer3", "layer4")
SAMPLE_FILENAME_PATTERN = re.compile(
    r"^(?P<study>.+)_(?P<nodule_kind>finding|cluster)_"
    r"(?P<nodule_number>\d+)_slice_(?P<slice_index>\d+)\.npy$"
)
GROUND_TRUTH_FILL_COLOR = "#00FFFF"
GROUND_TRUTH_INNER_OUTLINE_COLOR = "#FFFF00"
GROUND_TRUTH_OUTER_OUTLINE_COLOR = "#000000"
GROUND_TRUTH_FILL_ALPHA = 0.25


def normalize_unsigned(values: Tensor) -> Tensor:
    """Normalize each non-negative map independently to [0, 1]."""

    flat = values.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def normalize_signed(values: Tensor) -> Tensor:
    """Normalize each signed map independently to [-1, 1]."""

    scale = values.abs().flatten(1).max(dim=1).values[:, None, None]
    return values / scale.clamp_min(1e-12)


class MultiStageGradCAM:
    """Capture Grad-CAM maps from ResNet layers 1 through 4."""

    def __init__(self, model: MultiStageDirectGuidedResNet50) -> None:
        self.model = model
        self.activations: dict[str, Tensor] = {}
        self.gradients: dict[str, Tensor] = {}
        self.capture_enabled = False
        self.handles = []

        for stage_name in STAGE_NAMES:
            stage = getattr(model, stage_name)
            self.handles.append(
                stage[-1].register_forward_hook(
                    self._make_forward_hook(stage_name)
                )
            )

    def _make_forward_hook(self, stage_name: str):
        def forward_hook(module, inputs, output) -> None:
            del module, inputs
            if not self.capture_enabled:
                return
            self.activations[stage_name] = output
            output.register_hook(
                lambda gradient, name=stage_name: self.gradients.__setitem__(
                    name,
                    gradient,
                )
            )

        return forward_hook

    def generate(
        self,
        ct_images: Tensor,
        probability_maps: Tensor,
        class_indices: Tensor,
    ) -> dict[str, Tensor]:
        """Generate four input-resolution Grad-CAM maps."""

        self.activations.clear()
        self.gradients.clear()
        self.capture_enabled = True
        self.model.zero_grad(set_to_none=True)

        try:
            logits = self.model(ct_images, probability_maps)
            logits.gather(1, class_indices[:, None]).sum().backward()

            maps = {}
            for stage_name in STAGE_NAMES:
                if (
                    stage_name not in self.activations
                    or stage_name not in self.gradients
                ):
                    raise RuntimeError(
                        f"Grad-CAM did not capture {stage_name} tensors."
                    )
                activation = self.activations[stage_name]
                gradient = self.gradients[stage_name]
                channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
                stage_map = torch.relu(
                    (channel_weights * activation).sum(dim=1, keepdim=True)
                )
                stage_map = F.interpolate(
                    stage_map,
                    size=ct_images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                maps[stage_name] = normalize_unsigned(stage_map[:, 0]).detach()
            return maps
        finally:
            self.capture_enabled = False
            self.model.zero_grad(set_to_none=True)

    def close(self) -> None:
        """Remove all stage hooks."""

        for handle in self.handles:
            handle.remove()


def extract_guided_stage_features(
    model: MultiStageDirectGuidedResNet50,
    ct_images: Tensor,
    probability_maps: Tensor,
) -> dict[str, Tensor]:
    """Return the representation used after each ResNet stage."""

    if ct_images.shape[1] == 1:
        ct_images = ct_images.repeat(1, 3, 1, 1)

    features = model.conv1(ct_images)
    features = model.bn1(features)
    features = model.relu(features)
    features = model.maxpool(features)

    stage_features = {}
    features = model.layer1(features)
    probability_layer1 = resize_probability_map(
        probability_maps,
        features.shape[-2:],
        model.guidance_resize_mode,
    ).to(dtype=features.dtype)
    features = apply_direct_guidance(
        features,
        probability_layer1,
        model.alpha_layer1,
    )
    stage_features["layer1"] = features

    features = model.layer2(features)
    probability_layer2 = resize_probability_map(
        probability_maps,
        features.shape[-2:],
        model.guidance_resize_mode,
    ).to(dtype=features.dtype)
    features = apply_direct_guidance(
        features,
        probability_layer2,
        model.alpha_layer2,
    )
    stage_features["layer2"] = features

    features = model.layer3(features)
    probability_layer3 = resize_probability_map(
        probability_maps,
        features.shape[-2:],
        model.guidance_resize_mode,
    ).to(dtype=features.dtype)
    features = apply_direct_guidance(
        features,
        probability_layer3,
        model.alpha_layer3,
    )
    stage_features["layer3"] = features

    features = model.layer4(features)
    stage_features["layer4"] = features
    return stage_features


class StageTailModel(nn.Module):
    """Classify from one guided intermediate ResNet representation."""

    def __init__(
        self,
        model: MultiStageDirectGuidedResNet50,
        start_stage: str,
        probability_maps: Tensor,
    ) -> None:
        super().__init__()
        if start_stage not in STAGE_NAMES:
            raise ValueError(f"Unsupported stage: {start_stage}")

        self.start_stage = start_stage
        self.guidance_resize_mode = model.guidance_resize_mode
        self.register_buffer(
            "probability_maps",
            probability_maps.detach().clone(),
        )

        if start_stage == "layer1":
            self.layer2 = model.layer2
            self.register_buffer(
                "alpha_layer2",
                model.alpha_layer2.detach().clone(),
            )
        if start_stage in {"layer1", "layer2"}:
            self.layer3 = model.layer3
            self.register_buffer(
                "alpha_layer3",
                model.alpha_layer3.detach().clone(),
            )
        if start_stage != "layer4":
            self.layer4 = model.layer4

        self.avgpool = model.avgpool
        self.dropout = model.dropout
        self.classifier = model.classifier

    def _apply_guidance(self, features: Tensor, alpha: Tensor) -> Tensor:
        probability_map = resize_probability_map(
            self.probability_maps,
            features.shape[-2:],
            self.guidance_resize_mode,
        ).to(dtype=features.dtype)
        return apply_direct_guidance(features, probability_map, alpha)

    def forward(self, features: Tensor) -> Tensor:
        if self.start_stage == "layer1":
            features = self.layer2(features)
            features = self._apply_guidance(features, self.alpha_layer2)
        if self.start_stage in {"layer1", "layer2"}:
            features = self.layer3(features)
            features = self._apply_guidance(features, self.alpha_layer3)
        if self.start_stage != "layer4":
            features = self.layer4(features)

        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        return self.classifier(features)


def generate_multistage_lrp(
    model: MultiStageDirectGuidedResNet50,
    ct_images: Tensor,
    probability_maps: Tensor,
    class_indices: Tensor,
) -> dict[str, Tensor]:
    """Generate Zennit LRP maps for layers 1 through 4."""

    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonPlusFlat
        from zennit.torchvision import ResNetCanonizer
    except ImportError as error:
        raise RuntimeError(
            "LRP requires zennit==0.5.1. Install notebook dependencies."
        ) from error

    with torch.no_grad():
        stage_features = extract_guided_stage_features(
            model,
            ct_images,
            probability_maps,
        )

    target = torch.zeros(
        (ct_images.shape[0], model.classifier.out_features),
        device=ct_images.device,
    )
    target.scatter_(1, class_indices[:, None], 1.0)
    maps = {}

    for stage_name in STAGE_NAMES:
        stage_input = stage_features[stage_name].detach().requires_grad_(True)
        tail_model = StageTailModel(
            model,
            start_stage=stage_name,
            probability_maps=probability_maps,
        ).to(ct_images.device)
        tail_model.eval()
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
        with Gradient(model=tail_model, composite=composite) as attributor:
            _, relevance = attributor(stage_input, target)

        stage_map = relevance.sum(dim=1, keepdim=True)
        stage_map = F.interpolate(
            stage_map,
            size=ct_images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        maps[stage_name] = normalize_signed(stage_map[:, 0]).detach()
        model.zero_grad(set_to_none=True)

    return maps


def normalize_ct(image: np.ndarray) -> np.ndarray:
    """Normalize one CT image for display."""

    image = np.asarray(image, dtype=np.float32)
    finite_values = image[np.isfinite(image)]
    lower, upper = np.percentile(finite_values, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(image)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def resize_map_for_display(
    values: np.ndarray,
    target_shape: tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """Resize a 2D XAI or mask array to the original CT grid."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D map, received {values.shape}.")
    if values.shape == target_shape:
        return values

    tensor = torch.from_numpy(values)[None, None]
    if mode == "nearest":
        resized = F.interpolate(tensor, size=target_shape, mode=mode)
    else:
        resized = F.interpolate(
            tensor,
            size=target_shape,
            mode=mode,
            align_corners=False,
        )
    return resized[0, 0].numpy()


def load_display_array(path: Path, description: str) -> np.ndarray:
    """Load and validate one 2D visualization array."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {description}: {path}")
    return values


def add_ground_truth_overlay(axis, mask: np.ndarray) -> None:
    """Draw a translucent mask with a double outline."""

    binary_mask = np.asarray(mask, dtype=bool)
    if not binary_mask.any():
        return

    colored_mask = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
    colored_mask[binary_mask] = to_rgba(
        GROUND_TRUTH_FILL_COLOR,
        alpha=GROUND_TRUTH_FILL_ALPHA,
    )
    axis.imshow(colored_mask, interpolation="nearest")
    mask_values = binary_mask.astype(np.float32)
    axis.contour(
        mask_values,
        levels=[0.5],
        colors=[GROUND_TRUTH_OUTER_OUTLINE_COLOR],
        linewidths=3.0,
    )
    axis.contour(
        mask_values,
        levels=[0.5],
        colors=[GROUND_TRUTH_INNER_OUTLINE_COLOR],
        linewidths=1.5,
    )


def parse_sample_identifiers(filename: str) -> tuple[str, str, int]:
    """Return study, nodule, and slice identifiers."""

    match = SAMPLE_FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported sample filename: {filename}")
    nodule = f"{match.group('nodule_kind')}_{match.group('nodule_number')}"
    return match.group("study"), nodule, int(match.group("slice_index"))


def save_study_figure(
    study_id: str,
    study_frame: pd.DataFrame,
    dataset: DirectGuidedClassificationDataset,
    gradcam_directories: dict[str, Path],
    lrp_directories: dict[str, Path],
    save_path: Path,
    dpi: int,
) -> None:
    """Save CT, mask, probability, and eight XAI panels per slice."""

    nodule_groups = list(study_frame.groupby("xai_nodule", sort=False))
    total_slices = len(study_frame)
    section_heights = [max(1, len(frame)) for _, frame in nodule_groups]
    figure = plt.figure(
        figsize=(36, max(5.0, 1.2 + total_slices * 2.2)),
        facecolor="#f6f7fb",
        layout="constrained",
    )
    sections = figure.subfigures(
        len(nodule_groups),
        1,
        squeeze=False,
        height_ratios=section_heights,
    )
    column_titles = (
        "Parenchyma CT",
        "Ground-truth mask",
        "U-Net probability",
        "Grad-CAM L1",
        "Grad-CAM L2",
        "Grad-CAM L3",
        "Grad-CAM L4",
        "LRP L1",
        "LRP L2",
        "LRP L3",
        "LRP L4",
    )

    for section, (nodule_id, nodule_frame) in zip(
        sections.flat,
        nodule_groups,
        strict=True,
    ):
        nodule_frame = nodule_frame.sort_values("xai_slice_index")
        axes = section.subplots(
            len(nodule_frame),
            len(column_titles),
            squeeze=False,
        )
        section.suptitle(
            f"Nodule: {nodule_id} | slices: {len(nodule_frame)}",
            fontsize=14,
            weight="bold",
        )

        for row_index, (_, row) in enumerate(nodule_frame.iterrows()):
            filename = Path(str(row["filename"])).name
            ct = load_display_array(
                resolve_metadata_path(
                    dataset.root_dir,
                    row[dataset.ct_path_column],
                ),
                "CT",
            )
            mask = load_display_array(
                resolve_metadata_path(dataset.root_dir, row["mask_path"]),
                "ground-truth mask",
            )
            probability = load_display_array(
                dataset.probability_root / filename,
                "probability map",
            )
            display_ct = normalize_ct(ct)
            target_shape = tuple(int(value) for value in display_ct.shape)
            display_mask = (
                resize_map_for_display(mask, target_shape, mode="nearest") >= 0.5
            )
            display_probability = resize_map_for_display(
                probability,
                target_shape,
            )

            row_axes = axes[row_index]
            row_axes[0].imshow(display_ct, cmap="gray", vmin=0.0, vmax=1.0)
            row_axes[1].imshow(display_mask, cmap="gray", vmin=0.0, vmax=1.0)
            row_axes[2].imshow(
                display_probability,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
            )

            for offset, stage_name in enumerate(STAGE_NAMES, start=3):
                gradcam = load_display_array(
                    gradcam_directories[stage_name] / filename,
                    f"Grad-CAM {stage_name}",
                )
                row_axes[offset].imshow(
                    display_ct,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                row_axes[offset].imshow(
                    resize_map_for_display(gradcam, target_shape),
                    cmap="jet",
                    alpha=0.45,
                    vmin=0.0,
                    vmax=1.0,
                )
                add_ground_truth_overlay(row_axes[offset], display_mask)

            for offset, stage_name in enumerate(STAGE_NAMES, start=7):
                relevance = load_display_array(
                    lrp_directories[stage_name] / filename,
                    f"LRP {stage_name}",
                )
                row_axes[offset].imshow(
                    display_ct,
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                row_axes[offset].imshow(
                    resize_map_for_display(relevance, target_shape),
                    cmap="seismic",
                    alpha=0.50,
                    vmin=-1.0,
                    vmax=1.0,
                )
                add_ground_truth_overlay(row_axes[offset], display_mask)

            if row_index == 0:
                for axis, title in zip(row_axes, column_titles, strict=True):
                    axis.set_title(title, fontsize=10, weight="bold")

            predicted_class = str(row["predicted_class"])
            predicted_probability = float(
                row[f"probability_{predicted_class.lower()}"]
            )
            row_axes[0].text(
                -0.04,
                0.5,
                f"Slice {int(row['xai_slice_index'])}\n"
                f"Pred: {predicted_class}\n"
                f"p={predicted_probability:.3f}",
                transform=row_axes[0].transAxes,
                ha="right",
                va="center",
                fontsize=8,
                weight="bold",
            )
            for axis in row_axes:
                axis.axis("off")

    figure.suptitle(
        f"Study: {study_id} | nodules: {len(nodule_groups)} | "
        f"slices: {total_slices}",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def resolve_metadata_path(root_dir: Path, value: object) -> Path:
    """Resolve one metadata path against its dataset root."""

    path = Path(str(value))
    return path if path.is_absolute() else root_dir / path


def save_study_visualizations(
    predictions: pd.DataFrame,
    dataset: DirectGuidedClassificationDataset,
    gradcam_directories: dict[str, Path],
    lrp_directories: dict[str, Path],
    visualization_dir: Path,
    dpi: int,
) -> None:
    """Render one multi-panel XAI figure per study."""

    parsed = predictions["filename"].map(parse_sample_identifiers)
    predictions = predictions.copy()
    predictions["xai_study"] = parsed.map(lambda values: values[0])
    predictions["xai_nodule"] = parsed.map(lambda values: values[1])
    predictions["xai_slice_index"] = parsed.map(lambda values: values[2])
    predictions["xai_nodule_order"] = predictions["xai_nodule"].map(
        lambda value: int(str(value).rsplit("_", 1)[1])
    )
    predictions = predictions.sort_values(
        ["xai_study", "xai_nodule_order", "xai_slice_index"]
    )

    groups = list(predictions.groupby("xai_study", sort=False))
    for study_id, study_frame in tqdm(
        groups,
        desc="Rendering study visualizations",
        unit="study",
    ):
        save_study_figure(
            study_id=str(study_id),
            study_frame=study_frame,
            dataset=dataset,
            gradcam_directories=gradcam_directories,
            lrp_directories=lrp_directories,
            save_path=visualization_dir / f"{study_id}.png",
            dpi=dpi,
        )
