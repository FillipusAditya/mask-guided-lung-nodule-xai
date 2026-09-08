"""Evaluate a five-fold segmentation-guided ensemble with Grad-CAM and LRP."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation-guided five-fold ResNet-50 ensemble."
    )
    parser.add_argument("result_dir", type=Path, help="Completed CV result directory.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def load_fold_config(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "fold_0" / "training_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fold training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if (
        config.get("model", {}).get("architecture")
        != SegmentationGuidedResNet50.architecture_name
    ):
        raise ValueError("The result directory is not a segmentation-guided ResNet-50 run.")
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
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset.metadata = dataset.metadata.iloc[:max_samples].reset_index(drop=True)
        dataset.targets = dataset.targets[:max_samples]
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
        return normalize_unsigned(maps[:, 0]).detach()

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


def save_xai_figure(
    ct: np.ndarray,
    probability_map: np.ndarray,
    gradcam: np.ndarray,
    lrp: np.ndarray,
    title: str,
    save_path: Path,
    dpi: int,
) -> None:
    display = normalize_ct(ct)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#f6f7fb")
    panels = (
        (display, "gray", "Lung-parenchyma CT", 0.0, 1.0),
        (probability_map, "magma", "U-Net probability map", 0.0, 1.0),
        (gradcam, "jet", "Grad-CAM", 0.0, 1.0),
        (display, "gray", "Grad-CAM overlay", 0.0, 1.0),
        (lrp, "seismic", "LRP relevance", -1.0, 1.0),
        (display, "gray", "LRP overlay", 0.0, 1.0),
    )
    panel_images = []
    for axis, (array, cmap, panel_title, vmin, vmax) in zip(axes.flat, panels):
        panel_images.append(axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax))
        axis.set_title(panel_title, weight="bold")
        axis.axis("off")
    axes[1, 0].imshow(gradcam, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[1, 2].imshow(lrp, cmap="seismic", alpha=0.50, vmin=-1.0, vmax=1.0)
    for index in (1, 2, 4):
        figure.colorbar(
            panel_images[index],
            ax=axes.flat[index],
            fraction=0.046,
            pad=0.025,
        )
    figure.suptitle(title, fontsize=12, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def evaluate_and_explain(
    models: list[SegmentationGuidedResNet50],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_dir = output_dir / "gradcam"
    gradcam_npy_dir = output_dir / "gradcam_npy"
    lrp_dir = output_dir / "lrp"
    lrp_npy_dir = output_dir / "lrp_npy"
    combined_dir = output_dir / "xai"
    artifact_directories = (
        gradcam_dir,
        gradcam_npy_dir,
        lrp_dir,
        lrp_npy_dir,
        combined_dir,
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
                stem = Path(filename).stem
                prediction = int(predictions[batch_index])
                probability_values = probabilities[batch_index].cpu()
                gradcam = gradcam_maps[batch_index].numpy().astype(np.float32)
                lrp = lrp_maps[batch_index].numpy().astype(np.float32)
                np.save(gradcam_npy_dir / filename, gradcam, allow_pickle=False)
                np.save(lrp_npy_dir / filename, lrp, allow_pickle=False)

                ct = np.load(dataset.get_ct_path(sample_index), allow_pickle=False)
                segmentation_probability = np.load(
                    dataset.get_probability_path(sample_index), allow_pickle=False
                )
                title = (
                    f"{stem} | True: {row['label']} | "
                    f"Pred: {dataset.classes[prediction]} "
                    f"({float(probability_values[prediction]):.3f})"
                )
                save_xai_figure(
                    ct, segmentation_probability, gradcam, lrp, title,
                    combined_dir / f"{stem}.png", dpi,
                )
                save_single_heatmap(
                    ct,
                    gradcam,
                    "Grad-CAM",
                    gradcam_dir / f"{stem}.png",
                    dpi,
                )
                save_single_heatmap(
                    ct,
                    lrp,
                    "LRP",
                    lrp_dir / f"{stem}.png",
                    dpi,
                    signed=True,
                )
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
    return pd.DataFrame(records), metrics, confusion


def save_single_heatmap(
    ct: np.ndarray,
    heatmap: np.ndarray,
    method: str,
    save_path: Path,
    dpi: int,
    signed: bool = False,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    display = normalize_ct(ct)
    cmap, vmin = ("seismic", -1.0) if signed else ("jet", 0.0)
    axes[0].imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CT")
    heatmap_artist = axes[1].imshow(
        heatmap, cmap=cmap, vmin=vmin, vmax=1.0
    )
    axes[1].set_title(f"{method} heatmap")
    axes[2].imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(heatmap, cmap=cmap, alpha=0.48, vmin=vmin, vmax=1.0)
    axes[2].set_title(f"{method} overlay")
    figure.colorbar(
        heatmap_artist, ax=axes[1], fraction=0.046, pad=0.025
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError(
            "Batch size and DPI must be positive; workers cannot be negative."
        )
    result_dir = resolve_path(args.result_dir)
    output_dir = result_dir / "test"
    config = load_fold_config(result_dir)
    device = select_device(args.device)
    loader = build_test_loader(
        config, args.batch_size, args.num_workers, args.max_samples
    )
    models = load_models(result_dir, config, device)

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
