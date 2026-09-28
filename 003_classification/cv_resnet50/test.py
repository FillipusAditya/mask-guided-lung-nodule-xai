from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from ..utils import (
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    create_dataloader,
    plot_confusion_matrix,
    plot_roc_curve,
    update_confusion_matrix,
)
from .xai import (
    STAGE_NAMES,
    LayerwiseGradCAM,
    generate_layerwise_lrp,
    normalize_signed,
    normalize_unsigned,
    save_study_visualizations,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "003_classification/configs/cv_resnet50.json"
with CONFIG_PATH.open("r", encoding="utf-8") as file:
    DEFAULT_CONFIG = json.load(file)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DEFAULT_EXPERIMENT_ID = str(DEFAULT_CONFIG["experiment"]["id"])
DEFAULT_RESULT_COMPONENT = str(DEFAULT_CONFIG["experiment"]["component"])
DEFAULT_RESULT_DIR = (
    resolve_path(DEFAULT_CONFIG["output"]["root_directory"])
    / DEFAULT_EXPERIMENT_ID
    / DEFAULT_RESULT_COMPONENT
)
DEFAULT_DATASET_ROOT = resolve_path(DEFAULT_CONFIG["data"]["dataset_root"])
DEFAULT_METADATA_PATH = resolve_path(DEFAULT_CONFIG["data"]["metadata_path"])
def resolve_metadata_path(root_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root_dir / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_RESULT_DIR,
        help=(
            "Completed baseline result directory. The default is "
            f"{DEFAULT_RESULT_DIR}."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Local dataset root override. By default the script uses "
            f"{DEFAULT_DATASET_ROOT}."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help=(
            "Local CV metadata override. By default the script uses "
            f"{DEFAULT_METADATA_PATH}."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size. One is safest for five-model XAI.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Zero is the safest local default.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available, otherwise use the CPU.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally limit holdout samples, for example 8 for a smoke test.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI used for Grad-CAM and LRP study figures.",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help=(
            "Run model inference again even when cached predictions, "
            "Grad-CAM maps, and LRP maps are available."
        ),
    )
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


def first_existing_path(
    candidates: tuple[Path, ...],
    path_type: str,
) -> Path:
    """Return the first existing file or directory from the candidates."""

    unique_candidates = tuple(dict.fromkeys(path.resolve() for path in candidates))
    if path_type == "directory":
        existing_path = next(
            (path for path in unique_candidates if path.is_dir()),
            None,
        )
    elif path_type == "file":
        existing_path = next(
            (path for path in unique_candidates if path.is_file()),
            None,
        )
    else:
        raise ValueError("path_type must be 'directory' or 'file'.")

    if existing_path is None:
        checked_paths = "\n".join(f"  - {path}" for path in unique_candidates)
        raise FileNotFoundError(
            f"No existing {path_type} was found. Checked:\n{checked_paths}"
        )
    return existing_path


def relocate_dataset_paths(
    config: dict[str, Any],
    dataset_root_override: Path | None = None,
    metadata_path_override: Path | None = None,
) -> dict[str, Any]:
    """Replace unavailable Colab paths with local dataset paths."""

    config = dict(config)
    data = dict(config["data"])
    configured_root = resolve_path(data["dataset_root"])
    root_candidates = tuple(
        path
        for path in (
            (
                resolve_path(dataset_root_override)
                if dataset_root_override is not None
                else None
            ),
            configured_root,
            DEFAULT_DATASET_ROOT,
        )
        if path is not None
    )
    dataset_root = first_existing_path(root_candidates, "directory")

    configured_metadata = resolve_path(data["metadata_path"])
    metadata_candidates = tuple(
        path
        for path in (
            (
                resolve_path(metadata_path_override)
                if metadata_path_override is not None
                else None
            ),
            configured_metadata,
            dataset_root / configured_metadata.name,
            DEFAULT_METADATA_PATH,
        )
        if path is not None
    )
    metadata_path = first_existing_path(metadata_candidates, "file")

    data["dataset_root"] = str(dataset_root)
    data["metadata_path"] = str(metadata_path)
    config["data"] = data
    return config


def build_evaluation_transform(data: dict[str, Any], seed: int) -> A.Compose:
    """Create the same deterministic preprocessing used for validation."""

    return A.Compose(
        [
            A.Resize(
                height=int(data["image_height"]),
                width=int(data["image_width"]),
            ),
            A.Normalize(
                mean=tuple(data["normalization_mean"]),
                std=tuple(data["normalization_std"]),
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=seed,
    )


def build_test_loader(
    config: dict[str, Any], batch_size: int, num_workers: int, max_samples: int | None
) -> DataLoader:
    data = config["data"]
    loader = create_dataloader(
        root_dir=Path(data["dataset_root"]),
        metadata_path=Path(data["metadata_path"]),
        split="test",
        cv_fold=int(config["cross_validation"]["fold"]),
        transform=build_evaluation_transform(
            data,
            seed=int(config["training"]["seed"]),
        ),
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


def load_cached_predictions(
    output_dir: Path,
    max_samples: int | None,
    ct_path_column: str,
) -> pd.DataFrame | None:
    """Load completed inference artifacts for visualization-only execution."""

    predictions_path = output_dir / "test_predictions.csv"
    if not predictions_path.is_file():
        return None

    predictions = pd.read_csv(predictions_path)
    required_columns = {
        "filename",
        ct_path_column,
        "mask_path",
        "label",
        "predicted_class",
        "probability_benign",
        "probability_malignant",
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Cached test predictions are missing columns: " f"{sorted(missing_columns)}"
        )
    if predictions.empty:
        raise ValueError(f"Cached predictions are empty: {predictions_path}")
    if max_samples is not None:
        predictions = predictions.iloc[:max_samples].copy()

    gradcam_directories = {
        stage: output_dir / "gradcam_npy" / stage for stage in STAGE_NAMES
    }
    lrp_directories = {
        stage: output_dir / "lrp_npy" / stage for stage in STAGE_NAMES
    }
    missing_gradcam: list[str] = []
    missing_lrp: list[str] = []
    for filename in predictions["filename"]:
        name = Path(str(filename)).name
        for stage in STAGE_NAMES:
            if not (gradcam_directories[stage] / name).is_file():
                missing_gradcam.append(f"{stage}/{name}")
            if not (lrp_directories[stage] / name).is_file():
                missing_lrp.append(f"{stage}/{name}")

    if missing_gradcam or missing_lrp:
        raise FileNotFoundError(
            "Cached inference is incomplete: "
            f"missing Grad-CAM={len(missing_gradcam)}, "
            f"missing LRP={len(missing_lrp)}. "
            "Use --force-inference to regenerate all inference artifacts."
        )
    return predictions


def evaluate_and_explain(
    model_list: list[nn.Module],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dpi: int,
    classification_threshold: float,
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_directories = {
        stage: output_dir / "gradcam_npy" / stage for stage in STAGE_NAMES
    }
    lrp_directories = {
        stage: output_dir / "lrp_npy" / stage for stage in STAGE_NAMES
    }
    visualization_dir = output_dir / "visualization"
    for directory in (
        *gradcam_directories.values(),
        *lrp_directories.values(),
        visualization_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros(
        (len(dataset.classes), len(dataset.classes)), dtype=torch.int64
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    records, targets_all, probabilities_all = [], [], []
    total_loss, sample_index = 0.0, 0
    cameras = [LayerwiseGradCAM(model) for model in model_list]
    try:
        for inputs, labels in tqdm(loader, desc="Test inference + XAI", unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                fold_probabilities = [
                    torch.softmax(model(inputs), dim=1) for model in model_list
                ]
                probabilities = torch.stack(fold_probabilities).mean(dim=0)
            predictions = binary_probabilities_to_predictions(
                probabilities,
                classification_threshold,
                dataset.class_to_idx["malignant"],
            )
            total_loss += float(criterion(probabilities.clamp_min(1e-12).log(), labels))
            confusion = update_confusion_matrix(
                confusion, predictions, labels, len(dataset.classes)
            )
            targets_all.append(labels.cpu())
            probabilities_all.append(probabilities.cpu())
            fold_gradcam_maps = [
                camera.generate(inputs.detach().clone(), predictions)
                for camera in cameras
            ]
            gradcam_maps = {
                stage: normalize_unsigned(
                    torch.stack(
                        [fold_maps[stage] for fold_maps in fold_gradcam_maps]
                    ).mean(dim=0)
                ).cpu()
                for stage in STAGE_NAMES
            }
            fold_lrp_maps = [
                generate_layerwise_lrp(model, inputs, predictions)
                for model in model_list
            ]
            lrp_maps = {
                stage: normalize_signed(
                    torch.stack(
                        [fold_maps[stage] for fold_maps in fold_lrp_maps]
                    ).mean(dim=0)
                ).cpu()
                for stage in STAGE_NAMES
            }
            for batch_index in range(len(inputs)):
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                prediction = int(predictions[batch_index])
                for stage in STAGE_NAMES:
                    np.save(
                        gradcam_directories[stage] / filename,
                        gradcam_maps[stage][batch_index]
                        .numpy()
                        .astype(np.float32),
                    )
                    np.save(
                        lrp_directories[stage] / filename,
                        lrp_maps[stage][batch_index]
                        .numpy()
                        .astype(np.float32),
                    )
                record = dict(row)
                record.update(
                    {
                        "true_index": int(labels[batch_index]),
                        "predicted_index": prediction,
                        "predicted_class": dataset.classes[prediction],
                        "probability_benign": float(
                            probabilities[batch_index, dataset.class_to_idx["benign"]]
                        ),
                        "probability_malignant": float(
                            probabilities[
                                batch_index, dataset.class_to_idx["malignant"]
                            ]
                        ),
                    }
                )
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
    save_study_visualizations(
        predictions=predictions_frame,
        dataset=dataset,
        gradcam_directories=gradcam_directories,
        lrp_directories=lrp_directories,
        visualization_dir=visualization_dir,
        dpi=dpi,
    )
    return predictions_frame, metrics, confusion


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError(
            "Batch size and DPI must be positive; workers cannot be negative."
        )

    result_dir = resolve_path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_dir = result_dir / "test"
    config = relocate_dataset_paths(
        config=load_fold_config(result_dir),
        dataset_root_override=args.dataset_root,
        metadata_path_override=args.metadata_path,
    )
    loader = build_test_loader(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    cached_predictions = None
    if not args.force_inference:
        cached_predictions = load_cached_predictions(
            output_dir=output_dir,
            max_samples=args.max_samples,
            ct_path_column=str(config["data"]["ct_path_column"]),
        )

    print("Local ResNet-50 baseline test")
    print(f"Result directory : {result_dir}")
    print(f"Dataset root     : {config['data']['dataset_root']}")
    print(f"Metadata         : {config['data']['metadata_path']}")
    print(f"CT path column   : {config['data']['ct_path_column']}")
    print(f"Test samples     : {len(loader.dataset):,}")
    print(f"Output directory : {output_dir}")

    if cached_predictions is not None:
        gradcam_directories = {
            stage: output_dir / "gradcam_npy" / stage for stage in STAGE_NAMES
        }
        lrp_directories = {
            stage: output_dir / "lrp_npy" / stage for stage in STAGE_NAMES
        }
        visualization_dir = output_dir / "visualization"
        print("Mode             : visualization only (cached inference)")
        print("Model inference  : skipped")
        print()
        save_study_visualizations(
            predictions=cached_predictions,
            dataset=loader.dataset,
            gradcam_directories=gradcam_directories,
            lrp_directories=lrp_directories,
            visualization_dir=visualization_dir,
            dpi=args.dpi,
        )
        print(f"Visualized samples: {len(cached_predictions):,} from cached maps")
        print(f"Visualizations   : {visualization_dir}")
        return

    device = select_device(args.device)
    print("Mode             : inference and visualization")
    print(f"Device           : {device}")
    print()

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
    probabilities = predictions[
        ["probability_benign", "probability_malignant"]
    ].to_numpy()
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
            "gradcam_targets": [f"{stage}[-1]" for stage in STAGE_NAMES],
            "lrp_stages": list(STAGE_NAMES),
            "lrp_rule": "EpsilonPlusFlat with ResNetCanonizer",
            "visualization_grouping": ("one PNG per study with one section per nodule"),
            "panels": [
                "full CT scan",
                "ground-truth nodule mask",
                *[f"Grad-CAM {stage}" for stage in STAGE_NAMES],
                *[f"LRP {stage}" for stage in STAGE_NAMES],
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
