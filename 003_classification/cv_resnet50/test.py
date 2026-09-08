"""Evaluate the five-fold ResNet-50 baseline with Grad-CAM and LRP."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
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
from torchvision import models
from tqdm import tqdm

from ..utils import (
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    create_dataloader,
    plot_confusion_matrix,
    plot_roc_curve,
    update_confusion_matrix,
)
from ..fulltuning_resnet50 import train as transform_utils


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "003_classification/configs/cv_resnet50.json"
with CONFIG_PATH.open("r", encoding="utf-8") as file:
    DEFAULT_CONFIG = json.load(file)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DEFAULT_RESULT_DIR = (
    resolve_path(DEFAULT_CONFIG["output"]["root_directory"])
    / str(DEFAULT_CONFIG["experiment"]["id"])
    / str(DEFAULT_CONFIG["experiment"]["component"])
)
DEFAULT_DATASET_ROOT = resolve_path(DEFAULT_CONFIG["data"]["dataset_root"])
SAMPLE_FILENAME_PATTERN = re.compile(
    r"^(?P<study>.+)_(?P<nodule_kind>finding|cluster)_"
    r"(?P<nodule_number>\d+)_slice_(?P<slice_index>\d+)\.npy$"
)


def normalize_unsigned(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    return (values - minimum) / (maximum - minimum).clamp_min(1e-12)


def normalize_signed(values: torch.Tensor) -> torch.Tensor:
    scale = values.abs().flatten(1).max(dim=1).values[:, None, None]
    return values / scale.clamp_min(1e-12)


def normalize_ct(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    lower, upper = np.percentile(finite, (1.0, 99.0))
    if upper <= lower:
        return np.zeros_like(values)
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def resize_map(values: np.ndarray, shape: tuple[int, int], mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D map, received {values.shape}.")
    if values.shape == shape:
        return values
    options = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(
        torch.from_numpy(values)[None, None], size=shape, mode=mode, **options
    )[0, 0].numpy()


def parse_sample_identifiers(filename: str) -> tuple[str, str, int]:
    match = SAMPLE_FILENAME_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"Unsupported sample filename: {filename}")
    nodule = f"{match.group('nodule_kind')}_{match.group('nodule_number')}"
    return match.group("study"), nodule, int(match.group("slice_index"))


def resolve_metadata_path(root_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root_dir / path


def load_display_array(path: Path, description: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid {description}: {path} ({values.shape})")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
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


def load_fold_config(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "fold_0/training_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fold training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("model", {}).get("architecture") != "ResNet50":
        raise ValueError("The result directory is not a ResNet-50 baseline run.")
    return config


def relocate_dataset_paths(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    data = dict(config["data"])
    configured_root = resolve_path(data["dataset_root"])
    dataset_root = next(
        (path for path in (configured_root, DEFAULT_DATASET_ROOT) if path.is_dir()),
        None,
    )
    if dataset_root is None:
        raise FileNotFoundError("Dataset root was not found.")
    configured_metadata = resolve_path(data["metadata_path"])
    metadata_path = next(
        (
            path
            for path in (configured_metadata, dataset_root / configured_metadata.name)
            if path.is_file()
        ),
        None,
    )
    if metadata_path is None:
        raise FileNotFoundError("CV metadata was not found.")
    data["dataset_root"] = str(dataset_root.resolve())
    data["metadata_path"] = str(metadata_path.resolve())
    config["data"] = data
    return config


def build_test_loader(
    config: dict[str, Any], batch_size: int, num_workers: int, max_samples: int | None
) -> DataLoader:
    data = config["data"]
    transform_utils.INPUT_HEIGHT = int(data["image_height"])
    transform_utils.INPUT_WIDTH = int(data["image_width"])
    transform_utils.IMAGENET_MEAN = tuple(data["normalization_mean"])
    transform_utils.IMAGENET_STD = tuple(data["normalization_std"])
    transform_utils.TRANSFORM_SEED = int(config["training"]["seed"])
    loader = create_dataloader(
        root_dir=Path(data["dataset_root"]),
        metadata_path=Path(data["metadata_path"]),
        split="test",
        cv_fold=int(config["cross_validation"]["fold"]),
        transform=transform_utils.build_val_transform(),
        class_to_idx={key: int(value) for key, value in data["class_to_idx"].items()},
        ct_path_column=str(data["ct_path_column"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2,
        drop_last=False,
    )
    dataset = loader.dataset
    if "mask_path" not in dataset.metadata.columns:
        raise ValueError("Test metadata must contain mask_path for visualization.")
    missing_masks = [
        resolve_metadata_path(dataset.root_dir, value)
        for value in dataset.metadata["mask_path"]
        if not resolve_metadata_path(dataset.root_dir, value).is_file()
    ]
    if missing_masks:
        raise FileNotFoundError(f"Missing {len(missing_masks)} ground-truth masks.")
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset.metadata = dataset.metadata.iloc[:max_samples].reset_index(drop=True)
        dataset.targets = dataset.targets[:max_samples]
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    return loader


def build_model(config: dict[str, Any]) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(float(config["model"]["classifier"]["dropout_probability"])),
        nn.Linear(model.fc.in_features, int(config["model"]["num_classes"])),
    )
    return model


def load_models(
    result_dir: Path, config: dict[str, Any], device: torch.device
) -> list[nn.Module]:
    fold_ids = tuple(int(value) for value in config["cross_validation"]["all_folds"])
    loaded = []
    for fold in tqdm(fold_ids, desc="Loading fold models", unit="fold"):
        path = result_dir / f"fold_{fold}/best_model.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Best model not found: {path}")
        model = build_model(config)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        loaded.append(model.to(device).eval())
    return loaded


class GradCAM:
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
        self.model(inputs).gather(1, classes[:, None]).sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        maps = F.interpolate(maps, inputs.shape[-2:], mode="bilinear", align_corners=False)
        return normalize_unsigned(maps[:, 0]).detach()

    def close(self) -> None:
        self.handle.remove()


def generate_lrp(
    model: nn.Module, inputs: torch.Tensor, classes: torch.Tensor
) -> torch.Tensor:
    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonPlusFlat
        from zennit.torchvision import ResNetCanonizer
    except ImportError as error:
        raise RuntimeError("LRP requires zennit==0.5.1.") from error
    relevance_input = inputs.detach().requires_grad_(True)
    target = torch.zeros((len(inputs), model.fc[-1].out_features), device=inputs.device)
    target.scatter_(1, classes[:, None], 1.0)
    with Gradient(
        model=model,
        composite=EpsilonPlusFlat(canonizers=[ResNetCanonizer()]),
    ) as attributor:
        _, relevance = attributor(relevance_input, target)
    return normalize_signed(relevance.sum(dim=1)).detach()


def save_study_figure(
    study_id: str,
    frame: pd.DataFrame,
    dataset,
    gradcam_dir: Path,
    lrp_dir: Path,
    output_path: Path,
    dpi: int,
) -> None:
    groups = list(frame.groupby("xai_nodule", sort=False))
    figure = plt.figure(
        figsize=(15, max(5.0, 1.2 + len(frame) * 2.35 + len(groups) * 0.5)),
        facecolor="#f6f7fb",
        layout="constrained",
    )
    sections = figure.subfigures(
        len(groups), 1, squeeze=False, height_ratios=[max(1, len(g)) for _, g in groups]
    )
    titles = ("Full CT scan", "Ground-truth nodule mask", "Grad-CAM overlay", "LRP overlay")
    for section, (nodule_id, nodule_frame) in zip(sections.flat, groups, strict=True):
        nodule_frame = nodule_frame.sort_values("xai_slice_index")
        labels = ", ".join(sorted(nodule_frame["label"].astype(str).unique()))
        section.suptitle(
            f"Nodule section: {nodule_id}  |  Label: {labels}  |  Slices: {len(nodule_frame)}",
            fontsize=14,
            weight="bold",
        )
        axes = section.subplots(len(nodule_frame), 4, squeeze=False)
        for row_index, (_, row) in enumerate(nodule_frame.iterrows()):
            filename = Path(str(row["filename"])).name
            ct = load_display_array(
                resolve_metadata_path(dataset.root_dir, row[dataset.ct_path_column]),
                "CT scan",
            )
            mask = load_display_array(
                resolve_metadata_path(dataset.root_dir, row["mask_path"]),
                "ground-truth mask",
            )
            gradcam = load_display_array(gradcam_dir / filename, "Grad-CAM map")
            lrp = load_display_array(lrp_dir / filename, "LRP map")
            display = normalize_ct(ct)
            shape = tuple(display.shape)
            mask = resize_map(mask, shape, "nearest") >= 0.5
            gradcam = resize_map(gradcam, shape, "bilinear")
            lrp = resize_map(lrp, shape, "bilinear")
            row_axes = axes[row_index]
            row_axes[0].imshow(display, cmap="gray", vmin=0, vmax=1)
            row_axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            row_axes[2].imshow(display, cmap="gray", vmin=0, vmax=1)
            row_axes[2].imshow(gradcam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            row_axes[3].imshow(display, cmap="gray", vmin=0, vmax=1)
            row_axes[3].imshow(lrp, cmap="seismic", alpha=0.50, vmin=-1, vmax=1)
            if row_index == 0:
                for axis, title in zip(row_axes, titles, strict=True):
                    axis.set_title(title, fontsize=11, weight="bold", pad=8)
            probability = float(row[f"probability_{str(row['predicted_class']).lower()}"])
            row_axes[0].text(
                -0.04, 0.5,
                f"Slice {int(row['xai_slice_index'])}\nPred: {row['predicted_class']}\np={probability:.3f}",
                transform=row_axes[0].transAxes,
                ha="right", va="center", fontsize=9, weight="bold",
            )
            for axis in row_axes:
                axis.axis("off")
    figure.suptitle(
        f"Study: {study_id}  |  Nodules: {len(groups)}  |  Slices: {len(frame)}",
        fontsize=16,
        weight="bold",
    )
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def save_visualizations(
    predictions: pd.DataFrame, dataset, gradcam_dir: Path, lrp_dir: Path,
    visualization_dir: Path, dpi: int
) -> None:
    parsed = predictions["filename"].map(parse_sample_identifiers)
    frame = predictions.copy()
    frame["xai_study"] = parsed.map(lambda value: value[0])
    frame["xai_nodule"] = parsed.map(lambda value: value[1])
    frame["xai_slice_index"] = parsed.map(lambda value: value[2])
    frame["xai_nodule_order"] = frame["xai_nodule"].map(
        lambda value: int(str(value).rsplit("_", 1)[1])
    )
    frame = frame.sort_values(["xai_study", "xai_nodule_order", "xai_slice_index"])
    groups = list(frame.groupby("xai_study", sort=False))
    for study_id, study_frame in tqdm(groups, desc="Rendering study visualizations", unit="study"):
        save_study_figure(
            str(study_id), study_frame, dataset, gradcam_dir, lrp_dir,
            visualization_dir / f"{study_id}.png", dpi,
        )


def evaluate_and_explain(
    model_list: list[nn.Module], loader: DataLoader, device: torch.device,
    output_dir: Path, dpi: int, classification_threshold: float
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_dir = output_dir / "gradcam_npy"
    lrp_dir = output_dir / "lrp_npy"
    visualization_dir = output_dir / "visualization"
    for directory in (gradcam_dir, lrp_dir, visualization_dir):
        directory.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros((len(dataset.classes), len(dataset.classes)), dtype=torch.int64)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    records, targets_all, probabilities_all = [], [], []
    total_loss, sample_index = 0.0, 0
    cameras = [GradCAM(model) for model in model_list]
    try:
        for inputs, labels in tqdm(loader, desc="Test inference + XAI", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                fold_probabilities = [torch.softmax(model(inputs), dim=1) for model in model_list]
                probabilities = torch.stack(fold_probabilities).mean(dim=0)
            predictions = binary_probabilities_to_predictions(
                probabilities,
                classification_threshold,
                dataset.class_to_idx["malignant"],
            )
            total_loss += float(criterion(probabilities.clamp_min(1e-12).log(), labels))
            confusion = update_confusion_matrix(confusion, predictions, labels, len(dataset.classes))
            targets_all.append(labels.cpu())
            probabilities_all.append(probabilities.cpu())
            gradcam_maps = normalize_unsigned(
                torch.stack([cam.generate(inputs.detach().clone(), predictions) for cam in cameras]).mean(0)
            ).cpu()
            lrp_maps = normalize_signed(
                torch.stack([generate_lrp(model, inputs, predictions) for model in model_list]).mean(0)
            ).cpu()
            for batch_index in range(len(inputs)):
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                prediction = int(predictions[batch_index])
                np.save(gradcam_dir / filename, gradcam_maps[batch_index].numpy().astype(np.float32))
                np.save(lrp_dir / filename, lrp_maps[batch_index].numpy().astype(np.float32))
                record = dict(row)
                record.update({
                    "true_index": int(labels[batch_index]),
                    "predicted_index": prediction,
                    "predicted_class": dataset.classes[prediction],
                    "probability_benign": float(probabilities[batch_index, dataset.class_to_idx["benign"]]),
                    "probability_malignant": float(probabilities[batch_index, dataset.class_to_idx["malignant"]]),
                })
                records.append(record)
                sample_index += 1
    finally:
        for camera in cameras:
            camera.close()
    targets = torch.cat(targets_all)
    probabilities = torch.cat(probabilities_all)
    metrics = compute_classification_metrics(confusion)
    metrics["loss"] = total_loss / len(dataset)
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    predictions_frame = pd.DataFrame(records)
    save_visualizations(predictions_frame, dataset, gradcam_dir, lrp_dir, visualization_dir, dpi)
    return predictions_frame, metrics, confusion


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError("Batch size and DPI must be positive; workers cannot be negative.")
    result_dir = resolve_path(args.result_dir)
    output_dir = result_dir / "test"
    config = relocate_dataset_paths(load_fold_config(result_dir))
    device = select_device(args.device)
    loader = build_test_loader(config, args.batch_size, args.num_workers, args.max_samples)
    model_list = load_models(result_dir, config, device)
    started = time.perf_counter()
    predictions, metrics, confusion = evaluate_and_explain(
        model_list,
        loader,
        device,
        output_dir,
        args.dpi,
        float(config.get("metrics", {}).get("classification_threshold", 0.5)),
    )
    elapsed = time.perf_counter() - started
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    plot_confusion_matrix(confusion, loader.dataset.classes, output_dir)
    targets = predictions["true_index"].to_numpy()
    probabilities = predictions[["probability_benign", "probability_malignant"]].to_numpy()
    plot_roc_curve(targets, probabilities, loader.dataset.classes, output_dir)
    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": "ResNet50",
        "segmentation_guided": False,
        "ensemble_folds": len(model_list),
        "samples": len(loader.dataset),
        "metrics": {
            name: float(value) if np.isfinite(value) else None
            for name, value in metrics.items()
        },
        "runtime_seconds": elapsed,
        "xai": {
            "gradcam_target": "layer4[-1]",
            "lrp_rule": "EpsilonPlusFlat with ResNetCanonizer",
            "visualization_grouping": "one PNG per study with one section per nodule",
            "panels": ["full CT scan", "ground-truth nodule mask", "Grad-CAM overlay", "LRP overlay"],
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
