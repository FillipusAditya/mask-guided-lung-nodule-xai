"""Run U-Net inference and export train/validation/test probability maps.

Example
-------
python 002_segmentation/unet_holdout/test.py \
    experiment_results/dc730a13-5813-4d87-b15c-3b630deb32b5/segmentation/unet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unet_matplotlib"))
warnings.filterwarnings("ignore")

import albumentations as A
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_arch import UNET
from unet_utils import (  # noqa: E402
    IoULoss,
    compute_segmentation_metrics,
    merge_tiles,
    update_confusion_matrix,
)
from unet_utils.dataset import LungDataset  # noqa: E402


METRIC_NAMES = ("dice", "iou", "precision", "sensitivity", "specificity")
METRIC_LABELS = {
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for inference artifact export."""

    parser = argparse.ArgumentParser(
        description=(
            "Export train/validation/test U-Net inference artifacts and evaluate "
            "the held-out test split."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Training result directory containing unet_holdout.json and model weights.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: <result_dir>/best_model.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Inference output directory (default: <result_dir>/inference).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of full scans per batch (default: 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: value from the training config).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Probability threshold (default: value from the training config).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally process only the first N scans of each split.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Visualization resolution (default: 180).",
    )
    return parser.parse_args()


def resolve_from_project(path: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""

    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_run_config(result_dir: Path) -> tuple[dict[str, Any], Path]:
    """Load the exact configuration copied into a training run."""

    candidates = (result_dir / "unet_holdout.json", result_dir / "config.json")
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(
            f"Training config not found in {result_dir}. Expected unet_holdout.json."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    for section in ("data", "training", "dataloader"):
        if section not in config:
            raise ValueError(f"Training config is missing the '{section}' section.")

    return config, config_path


def select_device(requested: str) -> torch.device:
    """Resolve and validate the requested inference device."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device) -> UNET:
    """Load either best-model weights or a complete training checkpoint."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Also accept checkpoints saved from torch.nn.DataParallel.
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

    model = UNET(features=[16, 32, 64, 128])
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def create_split_data(
    config: dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int | None,
    max_samples: int | None,
) -> tuple[DataLoader, pd.DataFrame]:
    """Build a deterministic loader and aligned metadata for one dataset split."""

    data_config = config["data"]
    loader_config = config["dataloader"]
    dataset_root = resolve_from_project(data_config["dataset_root"])
    transform = A.Compose(
        [
            A.Resize(
                height=int(data_config["input_height"]),
                width=int(data_config["input_width"]),
            )
        ]
    )
    dataset = LungDataset(
        root_dir=dataset_root,
        split=split,
        split_method=str(data_config["split_method"]),
        metadata_filename=data_config["metadata_filename"],
        fold=data_config.get("fold"),
        image_path_column=str(data_config["image_path_column"]),
        tile_grid_size=int(data_config["tile_grid_size"]),
        transform=transform,
    )

    # Keep all rows from the same study contiguous. This allows visualization
    # canvases to be written one study at a time without retaining a full split
    # of 512x512 arrays in memory.
    ordered_metadata = dataset.metadata.copy()
    ordered_metadata["_study_sort"] = [
        study_id_from_filename(str(row["filename"]), str(row["dataset"]))
        for _, row in ordered_metadata.iterrows()
    ]
    ordered_metadata["_nodule_sort"] = [
        nodule_id_from_filename(str(filename))
        for filename in ordered_metadata["filename"]
    ]
    ordered_metadata["_slice_sort"] = [
        int(match.group(1)) if (match := re.search(r"_slice_(\d+)", str(filename))) else -1
        for filename in ordered_metadata["filename"]
    ]
    dataset.metadata = (
        ordered_metadata.sort_values(
            ["_study_sort", "_nodule_sort", "_slice_sort"],
            kind="stable",
        )
        .drop(columns=["_study_sort", "_nodule_sort", "_slice_sort"])
        .reset_index(drop=True)
    )

    if len(dataset) == 0:
        raise ValueError(f"The configured {split} split contains no samples.")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be greater than zero.")
        sample_count = min(max_samples, len(dataset))
        loader_dataset = Subset(dataset, range(sample_count))
        metadata = dataset.metadata.iloc[:sample_count].reset_index(drop=True)
    else:
        loader_dataset = dataset
        metadata = dataset.metadata.reset_index(drop=True)

    workers = int(loader_config["num_workers"] if num_workers is None else num_workers)
    if workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    loader = DataLoader(
        loader_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            bool(loader_config.get("persistent_workers", True)) if workers > 0 else False
        ),
        prefetch_factor=(
            int(loader_config.get("prefetch_factor", 2)) if workers > 0 else None
        ),
    )
    return loader, metadata


def nodule_id_from_filename(filename: str) -> str:
    """Return the stable nodule/finding ID shared by all of its slices."""

    return re.sub(r"_slice_\d+$", "", Path(filename).stem)


def study_id_from_filename(filename: str, dataset_name: str) -> str:
    """Extract a study ID from the normalized LIDC-IDRI or LNDb filename."""

    stem = Path(filename).stem
    if dataset_name == "LIDC-IDRI":
        return re.sub(r"_cluster_\d+_slice_\d+$", "", stem)
    if dataset_name == "LNDb":
        return re.sub(r"_finding_\d+_slice_\d+$", "", stem)
    return re.sub(r"_(?:cluster|finding)_\d+_slice_\d+$", "", stem)


def normalize_ct(image: np.ndarray) -> np.ndarray:
    """Robustly normalize one CT image to [0, 1] for display only."""

    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lower, upper = np.percentile(finite, (1.0, 99.0))
    if upper <= lower:
        lower, upper = float(finite.min()), float(finite.max())
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def overlay_binary_mask(
    axis: plt.Axes,
    mask: np.ndarray,
    color: str,
    alpha: float = 0.38,
) -> None:
    """Draw a translucent binary mask and a crisp contour when non-empty."""

    mask = mask.astype(bool)
    if not mask.any():
        return
    axis.imshow(
        np.ma.masked_where(~mask, mask),
        cmap=ListedColormap([color]),
        alpha=alpha,
        interpolation="none",
    )
    axis.contour(mask.astype(np.uint8), levels=[0.5], colors=[color], linewidths=1.2)


def save_study_canvas(
    study_id: str,
    representatives: list[dict[str, Any]],
    study_metrics: dict[str, float],
    threshold: float,
    save_path: Path,
    dpi: int,
) -> None:
    """Save one study canvas using the largest-mask slice of every nodule."""

    representatives = sorted(representatives, key=lambda item: item["nodule_id"])
    row_count = len(representatives)
    figure_height = 4.25 * row_count + 1.6
    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(18, figure_height),
        squeeze=False,
        facecolor="#f6f7fb",
    )

    column_titles = (
        "Representative CT Slice",
        "Ground Truth",
        f"Model Prediction  (p > {threshold:.2f})",
        "Agreement Map",
    )
    column_colors = ("#202431", "#18864b", "#087f8c", "#202431")
    for row_index, item in enumerate(representatives):
        display_image = normalize_ct(item["image"])
        target = item["target"].astype(bool)
        prediction = item["prediction"].astype(bool)
        true_positive = target & prediction
        false_positive = ~target & prediction
        false_negative = target & ~prediction

        for column_index, axis in enumerate(axes[row_index]):
            axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(
                    column_titles[column_index],
                    fontsize=12,
                    weight="bold",
                    color=column_colors[column_index],
                    pad=8,
                )

        overlay_binary_mask(axes[row_index, 1], target, "#2ecc71")
        overlay_binary_mask(axes[row_index, 2], prediction, "#00bcd4")
        overlay_binary_mask(axes[row_index, 3], true_positive, "#2ecc71", alpha=0.48)
        overlay_binary_mask(axes[row_index, 3], false_positive, "#ff9f43", alpha=0.58)
        overlay_binary_mask(axes[row_index, 3], false_negative, "#e84393", alpha=0.58)

        axes[row_index, 0].text(
            0.02,
            0.03,
            (
                f"Nodule: {item['nodule_id']}\n"
                f"Class: {item['class']}  |  "
                f"Representative slice: {item['slice_index']}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#202431",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d7dae3",
                "alpha": 0.90,
            },
        )
        metrics = item["metrics"]
        axes[row_index, 3].text(
            0.5,
            -0.035,
            (
                f"Dice {metrics['dice']:.3f}  |  IoU {metrics['iou']:.3f}  |  "
                f"Precision {metrics['precision']:.3f}  |  "
                f"Sensitivity {metrics['sensitivity']:.3f}"
            ),
            transform=axes[row_index, 3].transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color="#414552",
        )

    first = representatives[0]
    figure.suptitle(
        f"Study: {study_id}   |   Patient: {first['patient_id']}   |   "
        f"Dataset: {first['dataset']}   |   Nodules: {row_count}",
        fontsize=14,
        weight="bold",
        color="#202431",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Study metrics (all slices):  Dice {study_metrics['dice']:.4f}   |   "
            f"IoU {study_metrics['iou']:.4f}   |   "
            f"Precision {study_metrics['precision']:.4f}   |   "
            f"Sensitivity {study_metrics['sensitivity']:.4f}   |   "
            f"Specificity {study_metrics['specificity']:.4f}     •     "
            "Agreement: TP green, FP orange, FN magenta"
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#303440",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d7dae3"},
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        top=1.0 - 1.05 / figure_height,
        bottom=0.62 / figure_height,
        hspace=0.10,
        wspace=0.035,
    )
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_probability_study_canvas(
    study_id: str,
    slices: list[dict[str, Any]],
    study_metrics: dict[str, float],
    threshold: float,
    save_path: Path,
    dpi: int,
) -> None:
    """Save every inference slice from one study in a five-panel canvas."""

    slices = sorted(
        slices,
        key=lambda item: (
            item["nodule_id"],
            int(item["slice_index"]) if item["slice_index"] != "N/A" else -1,
        ),
    )
    row_count = len(slices)
    figure_height = 4.25 * row_count + 1.6
    figure, axes = plt.subplots(
        row_count,
        5,
        figsize=(22.5, figure_height),
        squeeze=False,
        facecolor="#f6f7fb",
    )
    column_titles = (
        "CT Slice",
        "Ground Truth",
        "Probability Map",
        f"Model Prediction  (p > {threshold:.2f})",
        "Agreement Map",
    )
    column_colors = ("#202431", "#18864b", "#6a3d9a", "#087f8c", "#202431")
    for row_index, item in enumerate(slices):
        display_image = normalize_ct(item["image"])
        probability = item["probability"]
        target = item["target"].astype(bool)
        prediction = item["prediction"].astype(bool)
        true_positive = target & prediction
        false_positive = ~target & prediction
        false_negative = target & ~prediction

        for column_index, axis in enumerate(axes[row_index]):
            if column_index == 2:
                axis.imshow(
                    probability,
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
            else:
                axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(
                    column_titles[column_index],
                    fontsize=12,
                    weight="bold",
                    color=column_colors[column_index],
                    pad=8,
                )

        overlay_binary_mask(axes[row_index, 1], target, "#2ecc71")
        overlay_binary_mask(axes[row_index, 3], prediction, "#00bcd4")
        overlay_binary_mask(axes[row_index, 4], true_positive, "#2ecc71", alpha=0.48)
        overlay_binary_mask(axes[row_index, 4], false_positive, "#ff9f43", alpha=0.58)
        overlay_binary_mask(axes[row_index, 4], false_negative, "#e84393", alpha=0.58)
        axes[row_index, 2].text(
            0.97,
            0.03,
            "p: 0 → 1",
            transform=axes[row_index, 2].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.65},
        )

        axes[row_index, 0].text(
            0.02,
            0.03,
            (
                f"Nodule: {item['nodule_id']}\n"
                f"Class: {item['class']}  |  "
                f"Slice: {item['slice_index']}"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="#202431",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#d7dae3",
                "alpha": 0.90,
            },
        )
        metrics = item["metrics"]
        axes[row_index, 4].text(
            0.5,
            -0.035,
            (
                f"Dice {metrics['dice']:.3f}  |  IoU {metrics['iou']:.3f}  |  "
                f"Precision {metrics['precision']:.3f}  |  "
                f"Sensitivity {metrics['sensitivity']:.3f}"
            ),
            transform=axes[row_index, 4].transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color="#414552",
        )

    first = slices[0]
    nodule_count = len({item["nodule_id"] for item in slices})
    figure.suptitle(
        f"Study: {study_id}   |   Patient: {first['patient_id']}   |   "
        f"Dataset: {first['dataset']}   |   Nodules: {nodule_count}   |   "
        f"Slices: {row_count}",
        fontsize=14,
        weight="bold",
        color="#202431",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Study metrics (all slices):  Dice {study_metrics['dice']:.4f}   |   "
            f"IoU {study_metrics['iou']:.4f}   |   "
            f"Precision {study_metrics['precision']:.4f}   |   "
            f"Sensitivity {study_metrics['sensitivity']:.4f}   |   "
            f"Specificity {study_metrics['specificity']:.4f}     •     "
            "Probability: 0–1   •   Agreement: TP green, FP orange, FN magenta"
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#303440",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d7dae3"},
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.965,
        top=1.0 - 1.05 / figure_height,
        bottom=0.62 / figure_height,
        hspace=0.10,
        wspace=0.035,
    )
    # PNG dimensions are limited by the renderer. Preserve one canvas per study
    # while reducing DPI only when an unusually large number of slices requires it.
    max_dimension_dpi = int(65_000 / figure_height)
    max_pixel_dpi = int((40_000_000 / (22.5 * figure_height)) ** 0.5)
    render_dpi = max(1, min(dpi, max_dimension_dpi, max_pixel_dpi))
    figure.savefig(
        save_path,
        dpi=render_dpi,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def save_sectioned_study_canvas(
    study_id: str,
    slices: list[dict[str, Any]],
    study_metrics: dict[str, float],
    threshold: float,
    save_path: Path,
    dpi: int,
) -> None:
    """Save all slices grouped into clearly separated nodule sections."""

    slices = sorted(
        slices,
        key=lambda item: (
            item["nodule_id"],
            int(item["slice_index"]) if item["slice_index"] != "N/A" else -1,
        ),
    )
    grouped_slices: list[tuple[str, list[dict[str, Any]]]] = []
    for item in slices:
        if not grouped_slices or grouped_slices[-1][0] != item["nodule_id"]:
            grouped_slices.append((item["nodule_id"], []))
        grouped_slices[-1][1].append(item)

    row_count = len(slices)
    nodule_count = len(grouped_slices)
    figure_height = 1.8 + 0.62 * nodule_count + 4.25 * row_count
    figure = plt.figure(figsize=(22.5, figure_height), facecolor="#f6f7fb")

    height_ratios: list[float] = [0.58]
    for _, nodule_slices in grouped_slices:
        height_ratios.append(0.48)
        height_ratios.extend([4.0] * len(nodule_slices))

    grid = figure.add_gridspec(
        len(height_ratios),
        5,
        height_ratios=height_ratios,
        left=0.015,
        right=0.985,
        top=1.0 - 0.90 / figure_height,
        bottom=0.62 / figure_height,
        hspace=0.11,
        wspace=0.045,
    )

    column_titles = (
        "CT Slice",
        "Ground Truth",
        "Probability Map",
        f"Model Prediction  (p > {threshold:.2f})",
        "Agreement Map",
    )
    column_colors = ("#202431", "#18864b", "#6a3d9a", "#087f8c", "#202431")
    header_axes: list[plt.Axes] = []
    for column_index, title in enumerate(column_titles):
        header_axis = figure.add_subplot(grid[0, column_index])
        header_axis.axis("off")
        header_axis.text(
            0.5,
            0.70,
            title,
            transform=header_axis.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color=column_colors[column_index],
        )
        header_axes.append(header_axis)

    probability_scale_axis = header_axes[2].inset_axes([0.16, 0.02, 0.68, 0.22])
    probability_gradient = np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, :]
    probability_scale_axis.imshow(
        probability_gradient,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    probability_scale_axis.set_yticks([])
    probability_scale_axis.set_xticks([0, 128, 255], labels=["0", "0.5", "1"])
    probability_scale_axis.tick_params(axis="x", labelsize=7, length=2, pad=1)
    for spine in probability_scale_axis.spines.values():
        spine.set_color("#6a3d9a")
        spine.set_linewidth(0.7)

    grid_row = 1
    for section_index, (nodule_id, nodule_slices) in enumerate(grouped_slices, start=1):
        classes = ", ".join(sorted({str(item["class"]) for item in nodule_slices}))
        slice_numbers = [
            int(item["slice_index"])
            for item in nodule_slices
            if item["slice_index"] != "N/A"
        ]
        slice_range = (
            f"{min(slice_numbers)}–{max(slice_numbers)}" if slice_numbers else "N/A"
        )
        section_axis = figure.add_subplot(grid[grid_row, :])
        section_axis.set_facecolor("#e9e3f3" if section_index % 2 else "#e4eef2")
        section_axis.set_xticks([])
        section_axis.set_yticks([])
        for spine in section_axis.spines.values():
            spine.set_visible(False)
        section_axis.text(
            0.012,
            0.5,
            (
                f"Nodule {section_index}: {nodule_id}   •   Class: {classes}   •   "
                f"Slices: {slice_range} ({len(nodule_slices)} images)"
            ),
            transform=section_axis.transAxes,
            ha="left",
            va="center",
            fontsize=10.5,
            weight="bold",
            color="#303440",
        )
        grid_row += 1

        for item in nodule_slices:
            axes = [figure.add_subplot(grid[grid_row, index]) for index in range(5)]
            display_image = normalize_ct(item["image"])
            probability = item["probability"]
            target = item["target"].astype(bool)
            prediction = item["prediction"].astype(bool)
            true_positive = target & prediction
            false_positive = ~target & prediction
            false_negative = target & ~prediction

            for column_index, axis in enumerate(axes):
                if column_index == 2:
                    axis.imshow(
                        probability,
                        cmap="magma",
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                else:
                    axis.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)
                axis.axis("off")

            overlay_binary_mask(axes[1], target, "#2ecc71")
            overlay_binary_mask(axes[3], prediction, "#00bcd4")
            overlay_binary_mask(axes[4], true_positive, "#2ecc71", alpha=0.48)
            overlay_binary_mask(axes[4], false_positive, "#ff9f43", alpha=0.58)
            overlay_binary_mask(axes[4], false_negative, "#e84393", alpha=0.58)

            axes[0].text(
                0.02,
                0.03,
                f"Slice: {item['slice_index']}   •   {item['scan_id']}",
                transform=axes[0].transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#202431",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#d7dae3",
                    "alpha": 0.90,
                },
            )
            metrics = item["metrics"]
            axes[4].text(
                0.98,
                0.03,
                (
                    f"Dice {metrics['dice']:.3f} | IoU {metrics['iou']:.3f}\n"
                    f"Precision {metrics['precision']:.3f} | "
                    f"Sensitivity {metrics['sensitivity']:.3f}"
                ),
                transform=axes[4].transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#303440",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "#d7dae3",
                    "alpha": 0.88,
                },
            )
            grid_row += 1

    first = slices[0]
    figure.suptitle(
        f"Study: {study_id}   |   Patient: {first['patient_id']}   |   "
        f"Dataset: {first['dataset']}   |   Nodules: {nodule_count}   |   "
        f"Slices: {row_count}",
        fontsize=14,
        weight="bold",
        color="#202431",
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Study metrics: Dice {study_metrics['dice']:.4f}   |   "
            f"IoU {study_metrics['iou']:.4f}   |   "
            f"Precision {study_metrics['precision']:.4f}   |   "
            f"Sensitivity {study_metrics['sensitivity']:.4f}   |   "
            f"Specificity {study_metrics['specificity']:.4f}     •     "
            "Agreement: TP green, FP orange, FN magenta"
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#303440",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d7dae3"},
    )

    max_dimension_dpi = int(65_000 / figure_height)
    max_pixel_dpi = int((40_000_000 / (22.5 * figure_height)) ** 0.5)
    render_dpi = max(1, min(dpi, max_dimension_dpi, max_pixel_dpi))
    figure.savefig(save_path, dpi=render_dpi, facecolor=figure.get_facecolor())
    plt.close(figure)


def aggregate_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Calculate micro-averaged overall and per-class segmentation metrics."""

    totals = {name: sum(int(record[name]) for record in records) for name in ("tp", "fp", "tn", "fn")}
    overall = compute_segmentation_metrics(
        totals["tp"], totals["fp"], totals["tn"], totals["fn"]
    )
    overall.update(totals)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["class"])].append(record)

    per_class = []
    for class_name in sorted(grouped):
        class_records = grouped[class_name]
        class_totals = {
            name: sum(int(record[name]) for record in class_records)
            for name in ("tp", "fp", "tn", "fn")
        }
        values = compute_segmentation_metrics(
            class_totals["tp"],
            class_totals["fp"],
            class_totals["tn"],
            class_totals["fn"],
        )
        per_class.append(
            {"class": class_name, "samples": len(class_records), **values, **class_totals}
        )
    return overall, per_class


def add_bar_labels(axis: plt.Axes, bars: Any) -> None:
    """Annotate metric bars with compact values."""

    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.035),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            weight="bold",
        )


def save_metrics_plot(
    overall: dict[str, float],
    per_class: list[dict[str, Any]],
    per_scan: pd.DataFrame,
    save_path: Path,
    dpi: int,
) -> None:
    """Save a clean summary plot of global, class, and scan-level metrics."""

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.8), facecolor="#f6f7fb")
    colors = ["#3657a7", "#4979c5", "#2aa198", "#ef8354", "#6c5ce7"]
    x = np.arange(len(METRIC_NAMES))
    labels = [METRIC_LABELS[name] for name in METRIC_NAMES]

    bars = axes[0].bar(x, [overall[name] for name in METRIC_NAMES], color=colors, width=0.7)
    add_bar_labels(axes[0], bars)
    axes[0].set_title("Overall Test Metrics (Micro)", weight="bold")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")

    class_count = max(len(per_class), 1)
    width = min(0.8 / class_count, 0.36)
    for index, class_values in enumerate(per_class):
        offset = (index - (len(per_class) - 1) / 2) * width
        class_bars = axes[1].bar(
            x + offset,
            [class_values[name] for name in METRIC_NAMES],
            width=width,
            label=f"{class_values['class']} (n={class_values['samples']})",
        )
        add_bar_labels(axes[1], class_bars)
    axes[1].set_title("Metrics by Nodule Class (Micro)", weight="bold")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].legend(frameon=False, fontsize=9)

    box_data = [per_scan[name].to_numpy(dtype=float) for name in METRIC_NAMES]
    boxplot = axes[2].boxplot(box_data, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in boxplot["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    axes[2].set_title("Per-Scan Metric Distribution", weight="bold")
    axes[2].tick_params(axis="x", rotation=20)

    for axis in axes:
        axis.set_ylim(0.0, 1.08)
        axis.set_ylabel("Score")
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        f"U-Net Segmentation — Held-Out Test Set ({len(per_scan):,} scans)",
        fontsize=16,
        weight="bold",
        color="#202431",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.0)
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def run_inference(
    model: UNET,
    loader: DataLoader,
    metadata: pd.DataFrame,
    device: torch.device,
    threshold: float,
    tile_grid_size: int,
    probability_npy_dir: Path,
    visualization_dir: Path,
    mask_dir: Path,
    dpi: int,
    split: str,
) -> tuple[pd.DataFrame, dict[str, float], list[dict[str, Any]], float]:
    """Run inference and write probability maps, masks, and five-panel canvases."""

    probability_npy_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    criterion = IoULoss()
    records: list[dict[str, Any]] = []
    current_study_id: str | None = None
    current_study_records: list[dict[str, Any]] = []
    current_study_slices: list[dict[str, Any]] = []
    running_loss = 0.0
    sample_index = 0
    amp_enabled = device.type == "cuda"
    study_count = len(
        {
            study_id_from_filename(str(row["filename"]), str(row["dataset"]))
            for _, row in metadata.iterrows()
        }
    )
    study_progress = tqdm(
        total=study_count,
        desc=f"{split.capitalize()} study canvases",
        unit="study",
        file=sys.stdout,
        disable=False,
        dynamic_ncols=True,
    )

    def flush_study_canvas() -> None:
        """Write the completed current study and release its image arrays."""

        if current_study_id is None or not current_study_slices:
            return
        study_metrics, _ = aggregate_metrics(current_study_records)
        save_sectioned_study_canvas(
            study_id=current_study_id,
            slices=current_study_slices,
            study_metrics=study_metrics,
            threshold=threshold,
            save_path=visualization_dir / f"{current_study_id}.png",
            dpi=dpi,
        )
        study_progress.update(1)

    inference_progress = tqdm(
        loader,
        desc=f"{split.capitalize()} inference",
        unit="batch",
        file=sys.stdout,
        disable=False,
        dynamic_ncols=True,
    )
    with torch.inference_mode():
        for images, masks in inference_progress:
            batch_size, num_tiles = images.shape[:2]
            logit_tiles = []
            for tile_index in range(num_tiles):
                image_tiles = images[:, tile_index].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    tile_logits = model(image_tiles)
                logit_tiles.append(tile_logits.float().cpu())

            logits = merge_tiles(torch.stack(logit_tiles, dim=1), grid_size=tile_grid_size)
            full_images = merge_tiles(images, grid_size=tile_grid_size)
            targets = (merge_tiles(masks, grid_size=tile_grid_size) > 0.5).float()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            batch_loss = criterion(logits, targets)
            running_loss += float(batch_loss.item()) * batch_size

            for local_index in range(batch_size):
                row = metadata.iloc[sample_index]
                target = targets[local_index : local_index + 1]
                prediction = predictions[local_index : local_index + 1]
                tp, fp, tn, fn = update_confusion_matrix(prediction, target)
                metrics = compute_segmentation_metrics(tp, fp, tn, fn)
                filename = Path(
                    str(row.get("filename", f"scan_{sample_index:05d}.npy"))
                ).name
                scan_id = Path(filename).stem
                nodule_id = nodule_id_from_filename(filename)
                dataset_name = str(row.get("dataset", "N/A"))
                study_id = study_id_from_filename(filename, dataset_name)

                if current_study_id is not None and study_id != current_study_id:
                    flush_study_canvas()
                    current_study_records.clear()
                    current_study_slices.clear()
                current_study_id = study_id

                record: dict[str, Any] = {
                    "scan_id": scan_id,
                    "study_id": study_id,
                    "nodule_id": nodule_id,
                    "patient_id": str(row.get("patient_id", "N/A")),
                    "dataset": dataset_name,
                    "class": str(row.get("label", "N/A")),
                    **metrics,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "ground_truth_pixels": int(target.sum().item()),
                    "predicted_pixels": int(prediction.sum().item()),
                }
                records.append(record)
                current_study_records.append(record)

                probability_array = (
                    probabilities[local_index]
                    .squeeze()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                np.save(
                    probability_npy_dir / filename,
                    probability_array,
                    allow_pickle=False,
                )
                prediction_array = prediction.squeeze().numpy().astype(np.uint8, copy=False)
                np.save(mask_dir / filename, prediction_array, allow_pickle=False)

                # Retain every slice so each study canvas shows the complete set,
                # rather than only the largest-mask representative slice.
                slice_match = re.search(r"_slice_(\d+)$", scan_id)
                current_study_slices.append(
                    {
                        "study_id": study_id,
                        "nodule_id": nodule_id,
                        "scan_id": scan_id,
                        "slice_index": slice_match.group(1) if slice_match else "N/A",
                        "patient_id": record["patient_id"],
                        "dataset": dataset_name,
                        "class": record["class"],
                        "ground_truth_pixels": record["ground_truth_pixels"],
                        "metrics": metrics,
                        "image": (
                            full_images[local_index]
                            .squeeze()
                            .numpy()
                            .astype(np.float16, copy=True)
                        ),
                        "target": target.squeeze().numpy().astype(bool, copy=True),
                        "prediction": prediction_array.astype(bool, copy=True),
                        "probability": probability_array.astype(np.float16, copy=True),
                    }
                )
                sample_index += 1

    if sample_index != len(metadata):
        raise RuntimeError(
            f"Inference output count mismatch: generated {sample_index}, expected {len(metadata)}."
        )

    flush_study_canvas()
    inference_progress.close()
    study_progress.close()

    overall, per_class = aggregate_metrics(records)
    return pd.DataFrame.from_records(records), overall, per_class, running_loss / sample_index


def main() -> None:
    """Export all inference artifacts and evaluate the held-out test split."""

    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero.")

    result_dir = resolve_from_project(args.result_dir).resolve()
    checkpoint_path = (
        resolve_from_project(args.checkpoint).resolve()
        if args.checkpoint is not None
        else result_dir / "best_model.pth"
    )
    output_dir = (
        resolve_from_project(args.output_dir).resolve()
        if args.output_dir is not None
        else result_dir / "inference"
    )
    config, _ = load_run_config(result_dir)
    threshold = float(
        config["training"]["prediction_threshold"]
        if args.threshold is None
        else args.threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    device = select_device(args.device)
    model = load_model(checkpoint_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_npy_dir = output_dir / "probability_npy"
    visualization_root = output_dir / "visualization"
    mask_root = output_dir / "mask"

    test_results = None
    seen_probability_filenames: set[str] = set()
    for split in ("train", "val", "test"):
        loader, metadata = create_split_data(
            config=config,
            split=split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        split_filenames = {Path(str(filename)).name for filename in metadata["filename"]}
        duplicates = seen_probability_filenames & split_filenames
        if duplicates:
            duplicate = sorted(duplicates)[0]
            raise ValueError(
                f"Duplicate filename across dataset splits: {duplicate}. "
                "A flat probability_npy directory would overwrite this file."
            )
        seen_probability_filenames.update(split_filenames)
        output_group = "test" if split == "test" else "train"
        split_results = run_inference(
            model=model,
            loader=loader,
            metadata=metadata,
            device=device,
            threshold=threshold,
            tile_grid_size=int(config["data"]["tile_grid_size"]),
            probability_npy_dir=probability_npy_dir,
            visualization_dir=visualization_root / output_group,
            mask_dir=mask_root / output_group,
            dpi=args.dpi,
            split=split,
        )
        if split == "test":
            test_results = split_results

    if test_results is None:
        raise RuntimeError("Test inference did not produce results.")
    per_scan, overall, per_class, _ = test_results

    plot_path = output_dir / "metrics_summary.png"
    with tqdm(
        total=1,
        desc="Test metrics summary",
        unit="plot",
        file=sys.stdout,
        disable=False,
        dynamic_ncols=True,
    ) as progress:
        save_metrics_plot(overall, per_class, per_scan, plot_path, args.dpi)
        progress.update(1)


if __name__ == "__main__":
    main()
