"""Evaluate the five-fold 2.5D ResNet-50 ensemble on holdout data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
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
from .aggregation import aggregate_nodule_predictions
from .dataset import TwoPointFiveDLungDataset
from .model import build_model
from .transforms import build_eval_transform
from .visualization import save_visualizations
from .xai import GradCAM, generate_lrp, normalize_signed_channels, normalize_unsigned


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "experiment_results/a0d90f9e-3dd4-4de0-98af-12858696f613"
    / "classification/cv_2_5d_resnet50"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--rows-per-figure", type=int, default=12)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def load_config(
    result_dir: Path,
    dataset_root: Path | None,
    metadata_path: Path | None,
) -> dict[str, Any]:
    path = result_dir / "fold_0/training_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config["model"]["architecture"] != "2.5D ResNet50":
        raise ValueError("The selected result is not a 2.5D ResNet-50 run.")
    if dataset_root is not None:
        config["data"]["dataset_root"] = str(resolve_path(dataset_root))
    if metadata_path is not None:
        config["data"]["metadata_path"] = str(resolve_path(metadata_path))
    for name in ("dataset_root", "metadata_path"):
        candidate = Path(config["data"][name])
        if not candidate.exists():
            fallback = resolve_path(candidate.name if name == "metadata_path" else candidate)
            if fallback.exists():
                config["data"][name] = str(fallback)
    return config


def make_test_dataset(config: dict, max_samples: int | None) -> TwoPointFiveDLungDataset:
    data, cv = config["data"], config["cross_validation"]
    dataset = TwoPointFiveDLungDataset(
        root_dir=data["dataset_root"], metadata_path=data["metadata_path"], split="test",
        cv_fold=0, transform=build_eval_transform(config),
        class_to_idx={name: int(index) for name, index in data["class_to_idx"].items()},
        ct_path_column=data["ct_path_column"], nodule_column=cv["nodule_column"],
        group_column=cv["group_column"], role_column=cv["role_column"],
        fold_column=cv["fold_column"], development_role=cv["development_role"],
        holdout_role=cv["holdout_role"], slice_offsets=tuple(data["slice_offsets"]),
        boundary_mode=data["boundary_mode"],
    )
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        dataset.metadata = dataset.metadata.iloc[:max_samples].reset_index(drop=True)
        dataset.windows = dataset.windows[:max_samples]
        dataset.targets = dataset.targets[:max_samples]
    return dataset


def make_loader(dataset: TwoPointFiveDLungDataset, batch_size: int, workers: int) -> DataLoader:
    if batch_size <= 0 or workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative.")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def load_models(result_dir: Path, config: dict, device: torch.device) -> list[nn.Module]:
    models = []
    for fold in range(int(config["cross_validation"]["num_folds"])):
        checkpoint = result_dir / f"fold_{fold}/best_model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Best model not found: {checkpoint}")
        model = build_model(config, load_pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        models.append(model.to(device).eval())
    return models


def metric_bundle(
    targets: torch.Tensor, probabilities: torch.Tensor, threshold: float
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    predictions = binary_probabilities_to_predictions(probabilities, threshold, 1)
    confusion = update_confusion_matrix(
        torch.zeros((2, 2), dtype=torch.int64), predictions, targets, 2
    )
    metrics = compute_classification_metrics(confusion)
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    metrics["loss"] = float(
        F.nll_loss(probabilities.clamp_min(1e-12).log(), targets)
    )
    return metrics, confusion, predictions


def evaluate(
    model_list: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_dir: Path,
    generate_xai_maps: bool,
    xai_target: str,
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_dir, lrp_dir = output_dir / "gradcam_npy", output_dir / "lrp_window_npy"
    if generate_xai_maps:
        gradcam_dir.mkdir(parents=True, exist_ok=True)
        lrp_dir.mkdir(parents=True, exist_ok=True)
    cameras = [GradCAM(model) for model in model_list] if generate_xai_maps else []
    records, all_targets, all_probabilities = [], [], []
    try:
        for inputs, targets, indices in tqdm(loader, desc="Holdout inference", unit="batch"):
            inputs = inputs.to(device)
            targets_device = targets.to(device)
            with torch.no_grad():
                fold_probabilities = [torch.softmax(model(inputs), dim=1) for model in model_list]
                probabilities = torch.stack(fold_probabilities).mean(dim=0)
            predictions = binary_probabilities_to_predictions(probabilities, threshold, 1)
            all_targets.append(targets)
            all_probabilities.append(probabilities.cpu())

            if generate_xai_maps:
                attribution_classes = (
                    predictions if xai_target == "predicted_class" else targets_device
                )
                gradcam = normalize_unsigned(
                    torch.stack(
                        [
                            camera.generate(inputs.detach().clone(), attribution_classes)
                            for camera in cameras
                        ]
                    ).mean(dim=0)
                ).cpu()
                raw_lrp = torch.stack(
                    [generate_lrp(model, inputs, attribution_classes) for model in model_list]
                ).mean(dim=0)
                lrp = normalize_signed_channels(raw_lrp).cpu()

            for batch_index, sample_index_tensor in enumerate(indices):
                sample_index = int(sample_index_tensor)
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                prediction = int(predictions[batch_index])
                if generate_xai_maps:
                    np.save(gradcam_dir / filename, gradcam[batch_index].numpy().astype(np.float32))
                    np.save(lrp_dir / filename, lrp[batch_index].numpy().astype(np.float32))
                record = dict(row)
                record.update(
                    {
                        "sample_index": sample_index,
                        "true_index": int(targets_device[batch_index]),
                        "predicted_index": prediction,
                        "predicted_class": dataset.classes[prediction],
                        "probability_benign": float(probabilities[batch_index, 0]),
                        "probability_malignant": float(probabilities[batch_index, 1]),
                        "input_slice_indices": "|".join(
                            map(str, dataset.windows[sample_index]["input_slice_indices"])
                        ),
                        "input_filenames": "|".join(
                            dataset.windows[sample_index]["input_filenames"]
                        ),
                        "boundary_replicated": dataset.windows[sample_index]["boundary_replicated"],
                    }
                )
                records.append(record)
    finally:
        for camera in cameras:
            camera.close()
    targets_tensor = torch.cat(all_targets)
    probabilities_tensor = torch.cat(all_probabilities)
    metrics, confusion, _ = metric_bundle(targets_tensor, probabilities_tensor, threshold)
    return pd.DataFrame(records), metrics, confusion


def aggregate_lrp_by_physical_slice(
    predictions: pd.DataFrame,
    dataset: TwoPointFiveDLungDataset,
    window_dir: Path,
    output_dir: Path,
) -> None:
    """Average repeated LRP contributions for each physical slice."""

    references: dict[str, list[tuple[str, int]]] = {}
    for _, row in predictions.iterrows():
        center_filename = Path(str(row["filename"])).name
        sample_index = int(row["sample_index"])
        for channel, physical_filename in enumerate(dataset.windows[sample_index]["input_filenames"]):
            references.setdefault(Path(physical_filename).name, []).append((center_filename, channel))
    output_dir.mkdir(parents=True, exist_ok=True)
    for physical_filename, locations in tqdm(references.items(), desc="Aggregate LRP", unit="slice"):
        maps = [
            np.load(window_dir / center, allow_pickle=False)[channel]
            for center, channel in locations
        ]
        mean_map = np.mean(maps, axis=0).astype(np.float32)
        scale = float(np.max(np.abs(mean_map)))
        if scale > 0:
            mean_map /= scale
        np.save(output_dir / physical_filename, mean_map)


def save_level_outputs(
    frame: pd.DataFrame, metrics: dict, confusion: torch.Tensor,
    output_dir: Path, prefix: str,
) -> None:
    figure_dir = output_dir / f"{prefix}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(confusion, ["benign", "malignant"], figure_dir)
    plot_roc_curve(
        frame["true_index"].to_numpy(),
        frame[["probability_benign", "probability_malignant"]].to_numpy(),
        ["benign", "malignant"], figure_dir,
    )
    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {name: float(value) if np.isfinite(value) else None for name, value in metrics.items()},
            file, indent=2,
        )
        file.write("\n")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0 or args.rows_per_figure <= 0:
        raise ValueError("DPI and rows per figure must be positive.")
    result_dir = resolve_path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    output_dir = result_dir / "test"
    cached_path = output_dir / "test_predictions.csv"
    completed_results = output_dir / "test_results.json"
    if cached_path.is_file() and completed_results.is_file() and not args.force_inference:
        print(f"Cached test predictions already exist: {cached_path}")
        print("Use --force-inference to run the ensemble again.")
        return
    config = load_config(result_dir, args.dataset_root, args.metadata_path)
    dataset = make_test_dataset(config, args.max_samples)
    loader = make_loader(dataset, args.batch_size, args.num_workers)
    device = select_device(args.device)
    model_list = load_models(result_dir, config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["training"]["classification_threshold"])
    print("2.5D ResNet-50 ensemble holdout evaluation")
    print(f"Windows: {len(dataset):,}; device: {device}; XAI: {not args.skip_xai}")
    started = time.perf_counter()
    predictions, window_metrics, window_confusion = evaluate(
        model_list, loader, device, threshold, output_dir, not args.skip_xai,
        config["xai"]["target"],
    )
    predictions.to_csv(cached_path, index=False)
    nodules = aggregate_nodule_predictions(
        predictions, config["cross_validation"]["nodule_column"], threshold
    )
    nodules.to_csv(output_dir / "test_nodule_predictions.csv", index=False)
    nodule_targets = torch.tensor(nodules["true_index"].to_numpy(), dtype=torch.long)
    nodule_probabilities = torch.tensor(
        nodules[["probability_benign", "probability_malignant"]].to_numpy(), dtype=torch.float32
    )
    nodule_metrics, nodule_confusion, _ = metric_bundle(
        nodule_targets, nodule_probabilities, threshold
    )
    save_level_outputs(predictions, window_metrics, window_confusion, output_dir, "window")
    save_level_outputs(nodules, nodule_metrics, nodule_confusion, output_dir, "nodule")

    if not args.skip_xai:
        aggregate_lrp_by_physical_slice(
            predictions, dataset, output_dir / "lrp_window_npy",
            output_dir / "lrp_slice_aggregated_npy",
        )
        save_visualizations(
            predictions, dataset, output_dir / "gradcam_npy", output_dir / "lrp_window_npy",
            output_dir / "visualization", args.dpi, args.rows_per_figure,
        )

    results = {
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "architecture": "2.5D ResNet50", "ensemble_folds": len(model_list),
        "slice_offsets": config["data"]["slice_offsets"],
        "boundary_mode": config["data"]["boundary_mode"],
        "window_level": {
            "samples": len(predictions),
            "metrics": {name: float(value) if np.isfinite(value) else None for name, value in window_metrics.items()},
        },
        "nodule_level": {
            "samples": len(nodules),
            "aggregation": config["aggregation"]["nodule_method"],
            "metrics": {name: float(value) if np.isfinite(value) else None for name, value in nodule_metrics.items()},
        },
        "xai": {
            "enabled": not args.skip_xai,
            "target": config["xai"]["target"],
            "gradcam": "one map per 2.5D window, displayed on the central slice",
            "lrp": "three channel-preserving maps per window plus per-physical-slice averages",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    with (output_dir / "test_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Window accuracy: {window_metrics['accuracy']:.4f}; AUC: {window_metrics['auc']:.4f}")
    print(f"Nodule accuracy: {nodule_metrics['accuracy']:.4f}; AUC: {nodule_metrics['auc']:.4f}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
