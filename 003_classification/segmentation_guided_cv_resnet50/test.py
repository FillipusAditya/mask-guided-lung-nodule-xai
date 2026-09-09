"""Evaluate a five-fold segmentation-guided ensemble with Grad-CAM and LRP."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import (
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    update_confusion_matrix,
)
from .dataset import ProbabilityGuidedClassificationDataset
from .model import FixedAttentionInputModel, SegmentationGuidedResNet50
from .transforms import build_val_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_THRESHOLD = 0.5
CONFIG_PATH = (
    PROJECT_ROOT
    / "003_classification"
    / "configs"
    / "segmentation_guided_cv_resnet50.json"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "000_dataset/_segmentation_dataset_v2"
CT_INPUT_COLUMNS = {
    "windowed": "ct_windowed_path",
    "parenchyma": "ct_parenchyma_path",
}
SAMPLE_FILENAME_PATTERN = re.compile(
    r"^(?P<study>.+)_(?P<nodule_kind>finding|cluster)_"
    r"(?P<nodule_number>\d+)_slice_(?P<slice_index>\d+)\.npy$"
)


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


with CONFIG_PATH.open("r", encoding="utf-8") as file:
    DEFAULT_CONFIG = json.load(file)

DEFAULT_RESULT_DIR = (
    resolve_path(DEFAULT_CONFIG["output"]["root_directory"])
    / str(DEFAULT_CONFIG["experiment"]["id"])
    / str(DEFAULT_CONFIG["experiment"]["component"])
)
DEFAULT_SEGMENTATION_RUN_DIR = resolve_path(
    DEFAULT_CONFIG["data"]["probability_root"]
).parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation-guided five-fold ResNet-50 ensemble."
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULT_DIR,
        help=(
            "Completed CV result directory. Defaults to the experiment and "
            "component selected in the JSON configuration."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. The default is conservative for local GPUs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker count. Zero is the safest default for local and "
            "notebook execution."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()

def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a clear error for invalid top-level data."""

    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return values


def validate_run_config(
    fold_config: dict[str, Any],
    run_config: dict[str, Any],
    path: Path,
) -> None:
    """Reject a run-level config that does not belong to these checkpoints."""

    architecture = run_config.get("model", {}).get("architecture")
    if architecture != SegmentationGuidedResNet50.architecture_name:
        raise ValueError(
            f"Unexpected model architecture in {path}: {architecture!r}."
        )

    fold_experiment = fold_config.get("experiment", {})
    run_experiment = run_config.get("experiment", {})
    fold_experiment_id = fold_experiment.get("experiment_id")
    run_experiment_id = run_experiment.get(
        "experiment_id", run_experiment.get("id")
    )
    if (
        fold_experiment_id is not None
        and run_experiment_id is not None
        and str(fold_experiment_id) != str(run_experiment_id)
    ):
        raise ValueError(
            "Experiment ID mismatch between fold configuration and "
            f"{path}: {fold_experiment_id!r} != {run_experiment_id!r}."
        )


def normalize_ct_input_config(data: dict[str, Any]) -> dict[str, Any]:
    """Validate new profiles and infer input types for historical runs."""

    data = dict(data)
    ct_path_column = str(data["ct_path_column"])
    ct_input_type = data.get("ct_input_type")
    if ct_input_type is None:
        matching_input_types = [
            input_type
            for input_type, column in CT_INPUT_COLUMNS.items()
            if column == ct_path_column
        ]
        if not matching_input_types:
            raise ValueError(
                "Cannot infer ct_input_type from ct_path_column="
                f"{ct_path_column!r}."
            )
        ct_input_type = matching_input_types[0]
    else:
        ct_input_type = str(ct_input_type).strip().lower()
    if ct_input_type not in CT_INPUT_COLUMNS:
        raise ValueError(
            f"ct_input_type must be one of {sorted(CT_INPUT_COLUMNS)}, "
            f"received {ct_input_type!r}."
        )
    expected_column = CT_INPUT_COLUMNS[ct_input_type]
    if ct_path_column != expected_column:
        raise ValueError(
            f"ct_input_type={ct_input_type!r} requires "
            f"ct_path_column={expected_column!r}, received "
            f"{ct_path_column!r}."
        )
    data["ct_input_type"] = ct_input_type
    data["ct_path_column"] = ct_path_column
    return data


def load_run_config(result_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    """Combine fold details with authoritative run-level data settings.

    Historical Colab runs can contain stale path fields in each fold's copied
    training configuration. The root snapshot and ``cv_config.json`` describe
    the data that was actually selected for the complete run, while the fold
    configuration remains authoritative for model and transform details.
    """

    path = result_dir / "fold_0" / "training_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fold training configuration not found: {path}")
    config = load_json(path)
    if (
        config.get("model", {}).get("architecture")
        != SegmentationGuidedResNet50.architecture_name
    ):
        raise ValueError("The result directory is not a segmentation-guided ResNet-50 run.")

    data = dict(config["data"])
    sources = [path]

    snapshot_path = result_dir / "segmentation_guided_cv_resnet50.json"
    if snapshot_path.is_file():
        snapshot = load_json(snapshot_path)
        validate_run_config(config, snapshot, snapshot_path)
        snapshot_data = snapshot.get("data", {})
        for key in (
            "dataset_root",
            "metadata_path",
            "probability_root",
            "ct_input_type",
            "ct_path_column",
        ):
            if key in snapshot_data:
                data[key] = snapshot_data[key]
        sources.append(snapshot_path)

    cv_config_path = result_dir / "cv_config.json"
    if cv_config_path.is_file():
        cv_config = load_json(cv_config_path)
        validate_run_config(config, cv_config, cv_config_path)
        cv_data = cv_config.get("data", {})
        for key in (
            "dataset_root",
            "probability_root",
            "ct_input_type",
            "ct_path_column",
        ):
            if key in cv_data:
                data[key] = cv_data[key]
        cv_metadata_path = cv_config.get("cross_validation", {}).get(
            "metadata_path"
        )
        if cv_metadata_path:
            data["metadata_path"] = cv_metadata_path
        sources.append(cv_config_path)

    config["data"] = normalize_ct_input_config(data)
    return config, sources


def relocate_colab_data_paths(
    config: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """Replace unavailable absolute Colab paths with their local equivalents."""

    config = dict(config)
    data = dict(config["data"])

    configured_dataset_root = resolve_path(data["dataset_root"])
    dataset_candidates = (configured_dataset_root, DEFAULT_DATASET_ROOT)
    dataset_root = next(
        (path.resolve() for path in dataset_candidates if path.is_dir()),
        None,
    )
    if dataset_root is None:
        raise FileNotFoundError(
            "Dataset root was not found. Checked: "
            + ", ".join(str(path) for path in dataset_candidates)
        )

    configured_metadata_path = resolve_path(data["metadata_path"])
    metadata_candidates = (
        configured_metadata_path,
        dataset_root / configured_metadata_path.name,
    )
    metadata_path = next(
        (path.resolve() for path in metadata_candidates if path.is_file()),
        None,
    )
    if metadata_path is None:
        raise FileNotFoundError(
            "CV metadata was not found. Checked: "
            + ", ".join(str(path) for path in metadata_candidates)
        )

    configured_probability_root = resolve_path(data["probability_root"])
    probability_candidates = (
        configured_probability_root,
        result_dir.parent / "inference/probability_npy",
        result_dir.parents[1] / "segmentation/unet/inference/probability_npy",
        DEFAULT_SEGMENTATION_RUN_DIR / "inference/probability_npy",
    )
    probability_root = next(
        (path.resolve() for path in probability_candidates if path.is_dir()),
        None,
    )
    if probability_root is None:
        raise FileNotFoundError(
            "U-Net probability-map directory was not found. Checked: "
            + ", ".join(str(path) for path in probability_candidates)
        )

    data["dataset_root"] = str(dataset_root)
    data["metadata_path"] = str(metadata_path)
    data["probability_root"] = str(probability_root)
    config["data"] = data
    return config


def build_test_loader(
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> DataLoader:
    data = config["data"]
    dataset = ProbabilityGuidedClassificationDataset(
        root_dir=Path(data["dataset_root"]),
        metadata_path=Path(data["metadata_path"]),
        split="test",
        cv_fold=int(config["cross_validation"]["fold"]),
        probability_root=Path(data["probability_root"]),
        transform=build_val_transform(
            int(data["image_height"]),
            int(data["image_width"]),
            tuple(float(value) for value in data["normalization_mean"]),
            tuple(float(value) for value in data["normalization_std"]),
            int(config["training"]["seed"]),
        ),
        class_to_idx={
            key: int(value) for key, value in data["class_to_idx"].items()
        },
        ct_path_column=str(data["ct_path_column"]),
    )
    if "mask_path" not in dataset.metadata.columns:
        raise ValueError("Test metadata must contain the mask_path column.")
    missing_masks = [
        resolve_metadata_path(dataset.root_dir, value)
        for value in dataset.metadata["mask_path"]
        if not resolve_metadata_path(dataset.root_dir, value).is_file()
    ]
    if missing_masks:
        preview = ", ".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} ground-truth masks; examples: {preview}"
        )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset.metadata = dataset.metadata.iloc[:max_samples].reset_index(drop=True)
        dataset.targets = dataset.targets[:max_samples]

    missing_ct = [
        dataset.get_ct_path(index)
        for index in range(len(dataset))
        if not dataset.get_ct_path(index).is_file()
    ]
    if missing_ct:
        preview = ", ".join(str(path) for path in missing_ct[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_ct)} CT files selected through "
            f"{data['ct_path_column']!r}; examples: {preview}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def load_models(
    result_dir: Path,
    config: dict[str, Any],
    device: torch.device,
) -> list[SegmentationGuidedResNet50]:
    num_classes = int(config["model"]["num_classes"])
    dropout = float(config["model"]["classifier"]["dropout_probability"])
    attention_config = config["model"]["attention"]
    fold_ids = tuple(int(fold) for fold in config["cross_validation"]["all_folds"])
    models = []
    for fold in tqdm(fold_ids, desc="Loading fold models", unit="fold"):
        weights_path = result_dir / f"fold_{fold}" / "best_model.pth"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Best model not found: {weights_path}")
        model = SegmentationGuidedResNet50(
            num_classes=num_classes,
            dropout=dropout,
            weights=None,
            attention_hidden_channels=int(
                attention_config["attention_hidden_channels"]
            ),
            attention_alpha_initial_value=float(
                attention_config["alpha_initial_value"]
            ),
        )
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        model.to(device).eval()
        models.append(model)
    return models


class GradCAM:
    """Grad-CAM for the final convolutional stage of the guided classifier."""

    def __init__(self, model: SegmentationGuidedResNet50) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = model.backbone.layer4[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output) -> None:
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def generate(
        self,
        inputs: torch.Tensor,
        class_indices: torch.Tensor,
    ) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        logits.gather(1, class_indices[:, None]).sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        maps = F.interpolate(maps, inputs.shape[-2:], mode="bilinear", align_corners=False)
        result = normalize_unsigned(maps[:, 0]).detach()
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        return result

    def close(self) -> None:
        self.handle.remove()


def normalize_unsigned(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def normalize_signed(values: torch.Tensor) -> torch.Tensor:
    scale = values.abs().flatten(1).max(dim=1).values[:, None, None]
    return values / scale.clamp_min(1e-12)


def generate_lrp(
    model: SegmentationGuidedResNet50,
    inputs: torch.Tensor,
    class_indices: torch.Tensor,
) -> torch.Tensor:
    """Generate signed EpsilonPlusFlat LRP relevance for the CT channels."""

    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonPlusFlat
        from zennit.torchvision import ResNetCanonizer
    except ImportError as error:
        raise RuntimeError(
            "LRP requires Zennit. Install requirements.txt from this directory."
        ) from error

    ct = inputs[:, :3].detach().requires_grad_(True)
    fixed_model = FixedAttentionInputModel(model, inputs[:, 3:4]).eval()
    target = torch.zeros(
        (inputs.shape[0], model.backbone.fc[-1].out_features),
        device=inputs.device,
    )
    target.scatter_(1, class_indices[:, None], 1.0)
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    with Gradient(model=fixed_model, composite=composite) as attributor:
        _, relevance = attributor(ct, target)
    return normalize_signed(relevance.sum(dim=1)).detach()


def normalize_ct(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lower, upper = np.percentile(image[np.isfinite(image)], (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(image)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def resize_map_for_display(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a model-space map to the original CT grid for aligned plotting."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {values.shape}.")
    if values.shape == target_shape:
        return values
    resized = F.interpolate(
        torch.from_numpy(values)[None, None],
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def resize_mask_for_display(
    values: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a binary mask without introducing interpolated class values."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {values.shape}.")
    if values.shape != target_shape:
        values = F.interpolate(
            torch.from_numpy(values)[None, None],
            size=target_shape,
            mode="nearest",
        )[0, 0].numpy()
    return values >= 0.5


def parse_sample_identifiers(filename: str) -> tuple[str, str, int]:
    """Return study, nodule, and slice identifiers from a dataset filename."""

    match = SAMPLE_FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported segmentation sample filename: {filename}")
    nodule = f"{match.group('nodule_kind')}_{match.group('nodule_number')}"
    return match.group("study"), nodule, int(match.group("slice_index"))


def resolve_metadata_path(root_dir: Path, value: object) -> Path:
    """Resolve an absolute or dataset-relative path stored in metadata."""

    path = Path(str(value))
    return path if path.is_absolute() else root_dir / path


def load_display_array(path: Path, description: str) -> np.ndarray:
    """Load and validate one two-dimensional visualization source."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2D {description}, got shape {values.shape}: {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {description}: {path}")
    return values


def save_study_figure(
    study_id: str,
    study_frame: pd.DataFrame,
    dataset_root: Path,
    ct_path_column: str,
    ct_panel_title: str,
    probability_root: Path,
    gradcam_npy_dir: Path,
    lrp_npy_dir: Path,
    save_path: Path,
    dpi: int,
) -> None:
    """Save every nodule and slice from one study in a single PNG."""

    nodule_groups = list(study_frame.groupby("xai_nodule", sort=False))
    total_slices = len(study_frame)
    section_heights = [max(1, len(frame)) for _, frame in nodule_groups]
    figure_height = max(5.0, 1.2 + total_slices * 2.35 + len(nodule_groups) * 0.5)
    figure = plt.figure(
        figsize=(18, figure_height),
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
        ct_panel_title,
        "Ground-truth nodule mask",
        "U-Net probability heatmap",
        "Grad-CAM overlay",
        "LRP overlay",
    )

    for section, (nodule_id, nodule_frame) in zip(
        sections.flat, nodule_groups, strict=True
    ):
        nodule_frame = nodule_frame.sort_values("xai_slice_index")
        labels = ", ".join(sorted(nodule_frame["label"].astype(str).unique()))
        section.suptitle(
            f"Nodule section: {nodule_id}  |  Label: {labels}  |  "
            f"Slices: {len(nodule_frame)}",
            fontsize=14,
            weight="bold",
        )
        axes = section.subplots(len(nodule_frame), 5, squeeze=False)
        probability_artists = []

        for row_index, (_, row) in enumerate(nodule_frame.iterrows()):
            filename = Path(str(row["filename"])).name
            ct = load_display_array(
                resolve_metadata_path(dataset_root, row[ct_path_column]),
                "CT scan",
            )
            mask = load_display_array(
                resolve_metadata_path(dataset_root, row["mask_path"]),
                "ground-truth mask",
            )
            probability = load_display_array(
                probability_root / filename,
                "U-Net probability map",
            )
            gradcam = load_display_array(
                gradcam_npy_dir / filename,
                "Grad-CAM map",
            )
            lrp = load_display_array(lrp_npy_dir / filename, "LRP map")

            display = normalize_ct(ct)
            display_shape = tuple(int(value) for value in display.shape)
            mask_display = resize_mask_for_display(mask, display_shape)
            probability_display = resize_map_for_display(
                probability, display_shape
            )
            gradcam_display = resize_map_for_display(gradcam, display_shape)
            lrp_display = resize_map_for_display(lrp, display_shape)

            row_axes = axes[row_index]
            row_axes[0].imshow(
                display,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            row_axes[1].imshow(
                mask_display,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            probability_artists.append(
                row_axes[2].imshow(
                    probability_display,
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="bilinear",
                )
            )
            row_axes[3].imshow(
                display,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            row_axes[3].imshow(
                gradcam_display,
                cmap="jet",
                alpha=0.45,
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            row_axes[4].imshow(
                display,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            row_axes[4].imshow(
                lrp_display,
                cmap="seismic",
                alpha=0.50,
                vmin=-1.0,
                vmax=1.0,
                interpolation="bilinear",
            )

            if row_index == 0:
                for axis, column_title in zip(
                    row_axes, column_titles, strict=True
                ):
                    axis.set_title(
                        column_title, fontsize=11, weight="bold", pad=8
                    )

            prediction_probability = float(
                row[f"probability_{str(row['predicted_class']).lower()}"]
            )
            row_axes[0].text(
                -0.04,
                0.5,
                f"Slice {int(row['xai_slice_index'])}\n"
                f"Pred: {row['predicted_class']}\n"
                f"p={prediction_probability:.3f}",
                transform=row_axes[0].transAxes,
                ha="right",
                va="center",
                fontsize=9,
                weight="bold",
            )
            for axis in row_axes:
                axis.axis("off")

        section.colorbar(
            probability_artists[0],
            ax=axes[:, 2].tolist(),
            fraction=0.018,
            pad=0.012,
            shrink=0.88,
            label="Nodule probability",
        )

    figure.suptitle(
        f"Study: {study_id}  |  Nodules: {len(nodule_groups)}  |  "
        f"Slices: {total_slices}",
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


def save_study_visualizations(
    predictions: pd.DataFrame,
    dataset: ProbabilityGuidedClassificationDataset,
    gradcam_npy_dir: Path,
    lrp_npy_dir: Path,
    visualization_dir: Path,
    dpi: int,
) -> None:
    """Aggregate all evaluated slices into one visualization per study."""

    if "mask_path" not in predictions.columns:
        raise ValueError("Metadata must contain mask_path for XAI visualization.")

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

    study_groups = list(predictions.groupby("xai_study", sort=False))
    ct_panel_title = {
        "ct_windowed_path": "Full-area windowed CT",
        "ct_parenchyma_path": "Lung-parenchyma CT",
    }.get(dataset.ct_path_column, "CT scan")
    for study_id, study_frame in tqdm(
        study_groups,
        desc="Rendering study visualizations",
        unit="study",
    ):
        save_study_figure(
            study_id=str(study_id),
            study_frame=study_frame,
            dataset_root=dataset.root_dir,
            ct_path_column=dataset.ct_path_column,
            ct_panel_title=ct_panel_title,
            probability_root=dataset.probability_root,
            gradcam_npy_dir=gradcam_npy_dir,
            lrp_npy_dir=lrp_npy_dir,
            save_path=visualization_dir / f"{study_id}.png",
            dpi=dpi,
        )


def evaluate_and_explain(
    models: list[SegmentationGuidedResNet50],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_npy_dir = output_dir / "gradcam_npy"
    lrp_npy_dir = output_dir / "lrp_npy"
    visualization_dir = output_dir / "visualization"
    artifact_directories = (
        gradcam_npy_dir,
        lrp_npy_dir,
        visualization_dir,
    )
    for directory in artifact_directories:
        directory.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros(
        (len(dataset.classes), len(dataset.classes)), dtype=torch.int64
    )
    records: list[dict[str, Any]] = []
    targets_all, probabilities_all = [], []
    total_loss = 0.0
    sample_index = 0
    gradcams = [GradCAM(model) for model in models]

    try:
        for inputs, labels in tqdm(loader, desc="Test inference + XAI", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                fold_probabilities = [
                    torch.softmax(model(inputs), dim=1) for model in models
                ]
                probabilities = torch.stack(fold_probabilities).mean(dim=0)
            predictions = binary_probabilities_to_predictions(
                probabilities,
                CLASSIFICATION_THRESHOLD,
                dataset.class_to_idx["malignant"],
            )
            total_loss += float(
                criterion(probabilities.clamp_min(1e-12).log(), labels)
            )
            confusion = update_confusion_matrix(
                confusion, predictions, labels, len(dataset.classes)
            )
            targets_all.append(labels.cpu())
            probabilities_all.append(probabilities.cpu())

            gradcam_maps = torch.stack(
                [cam.generate(inputs.detach().clone(), predictions) for cam in gradcams]
            ).mean(dim=0)
            gradcam_maps = normalize_unsigned(gradcam_maps).cpu()
            lrp_maps = torch.stack(
                [generate_lrp(model, inputs, predictions) for model in models]
            ).mean(dim=0)
            lrp_maps = normalize_signed(lrp_maps).cpu()

            for batch_index in range(inputs.shape[0]):
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                prediction = int(predictions[batch_index])
                probability_values = probabilities[batch_index].cpu()
                gradcam = gradcam_maps[batch_index].numpy().astype(np.float32)
                lrp = lrp_maps[batch_index].numpy().astype(np.float32)
                np.save(gradcam_npy_dir / filename, gradcam, allow_pickle=False)
                np.save(lrp_npy_dir / filename, lrp, allow_pickle=False)

                record = dict(row)
                record.update(
                    {
                        "true_index": int(labels[batch_index]),
                        "predicted_index": prediction,
                        "predicted_class": dataset.classes[prediction],
                        "probability_benign": float(
                            probability_values[dataset.class_to_idx["benign"]]
                        ),
                        "probability_malignant": float(
                            probability_values[dataset.class_to_idx["malignant"]]
                        ),
                    }
                )
                records.append(record)
                sample_index += 1
    finally:
        for gradcam in gradcams:
            gradcam.close()

    targets = torch.cat(targets_all)
    probabilities = torch.cat(probabilities_all)
    metrics = compute_classification_metrics(confusion)
    metrics["loss"] = total_loss / len(dataset)
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    predictions = pd.DataFrame(records)
    save_study_visualizations(
        predictions,
        dataset,
        gradcam_npy_dir,
        lrp_npy_dir,
        visualization_dir,
        dpi,
    )
    return predictions, metrics, confusion


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError(
            "Batch size and DPI must be positive; workers cannot be negative."
        )
    result_dir = resolve_path(args.result_dir)
    output_dir = result_dir / "test"
    config, config_sources = load_run_config(result_dir)
    config = relocate_colab_data_paths(config, result_dir)
    device = select_device(args.device)
    loader = build_test_loader(
        config, args.batch_size, args.num_workers, args.max_samples
    )
    models = load_models(result_dir, config, device)

    print(f"Classification run : {result_dir}")
    print(
        "Configuration      : "
        + ", ".join(str(path.relative_to(result_dir)) for path in config_sources)
    )
    print(f"Dataset root       : {config['data']['dataset_root']}")
    print(f"CT input type      : {config['data']['ct_input_type']}")
    print(f"CT path column     : {config['data']['ct_path_column']}")
    print(f"Probability maps   : {config['data']['probability_root']}")

    started = time.perf_counter()
    predictions, metrics, confusion = evaluate_and_explain(
        models, loader, device, output_dir, args.dpi
    )
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    plot_confusion_matrix(confusion, loader.dataset.classes, output_dir)
    targets = torch.tensor(predictions["true_index"].to_numpy())
    probabilities = torch.tensor(
        predictions[["probability_benign", "probability_malignant"]].to_numpy()
    )
    plot_roc_curve(
        targets.numpy(),
        probabilities.numpy(),
        loader.dataset.classes,
        output_dir,
    )
    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": SegmentationGuidedResNet50.architecture_name,
        "ct_input_type": config["data"]["ct_input_type"],
        "ct_path_column": config["data"]["ct_path_column"],
        "ensemble_folds": len(models),
        "samples": len(loader.dataset),
        "metrics": {
            name: float(value) if np.isfinite(value) else None
            for name, value in metrics.items()
        },
        "runtime_seconds": elapsed,
        "attention_alpha_per_fold": [
            float(model.attention.alpha.detach().cpu()) for model in models
        ],
        "xai": {
            "gradcam_target": "backbone.layer4[-1]",
            "lrp_rule": "EpsilonPlusFlat with ResNetCanonizer",
            "lrp_target": "CT channels conditioned on the U-Net probability map",
            "visualization_grouping": "one PNG per study with one section per nodule",
            "visualization_directory": str(output_dir / "visualization"),
            "panels": [
                (
                    "full-area windowed CT"
                    if config["data"]["ct_input_type"] == "windowed"
                    else "lung-parenchyma CT"
                ),
                "ground-truth nodule mask",
                "U-Net probability heatmap",
                "Grad-CAM overlay",
                "LRP overlay",
            ],
        },
    }
    with (output_dir / "test_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Test samples : {len(loader.dataset)}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"AUC          : {metrics['auc']:.4f}")
    print(f"Outputs      : {output_dir}")


if __name__ == "__main__":
    main()
