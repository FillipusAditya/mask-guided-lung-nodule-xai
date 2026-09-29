"""Evaluate a five-fold segmentation-guided ensemble with Grad-CAM and LRP."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from .model import SegmentationGuidedResNet50
from .transforms import build_val_transform
from .xai import (
    GROUND_TRUTH_FILL_ALPHA,
    GROUND_TRUTH_FILL_COLOR,
    GROUND_TRUTH_INNER_OUTLINE_COLOR,
    GROUND_TRUTH_OUTER_OUTLINE_COLOR,
    STAGE_NAMES,
    LayerwiseGradCAM,
    generate_layerwise_lrp,
    normalize_signed,
    normalize_unsigned,
    save_study_visualizations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_EXPERIMENT_ID = "fa028196-fd4c-441f-a5d3-3efb9707e7f9"
DEFAULT_RESULT_COMPONENT = "classification/guided_resnet50"
CONFIG_PATH = (
    PROJECT_ROOT
    / "003_classification"
    / "configs"
    / "segmentation_guided_cv_resnet50.json"
)
CT_INPUT_COLUMNS = {
    "windowed": "ct_windowed_path",
    "parenchyma": "ct_parenchyma_path",
}


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


with CONFIG_PATH.open("r", encoding="utf-8") as file:
    DEFAULT_CONFIG = json.load(file)

DEFAULT_RESULT_DIR = (
    resolve_path(DEFAULT_CONFIG["output"]["root_directory"])
    / DEFAULT_EXPERIMENT_ID
    / DEFAULT_RESULT_COMPONENT
)
DEFAULT_DATASET_ROOT = resolve_path(DEFAULT_CONFIG["data"]["dataset_root"])
DEFAULT_METADATA_PATH = resolve_path(DEFAULT_CONFIG["data"]["metadata_path"])
DEFAULT_PROBABILITY_ROOT = resolve_path(
    DEFAULT_CONFIG["data"]["probability_root"]
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
            "Completed guided result directory. The default is "
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
        "--probability-root",
        type=Path,
        default=None,
        help=(
            "Local U-Net probability-map directory override. By default the "
            f"script uses {DEFAULT_PROBABILITY_ROOT}."
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
    dataset_root_override: Path | None = None,
    metadata_path_override: Path | None = None,
    probability_root_override: Path | None = None,
) -> dict[str, Any]:
    """Replace unavailable absolute Colab paths with their local equivalents."""

    config = dict(config)
    data = dict(config["data"])

    configured_dataset_root = resolve_path(data["dataset_root"])
    dataset_candidates = tuple(
        path
        for path in (
            resolve_path(dataset_root_override)
            if dataset_root_override is not None
            else None,
            configured_dataset_root,
            DEFAULT_DATASET_ROOT,
        )
        if path is not None
    )
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
    metadata_candidates = tuple(
        path
        for path in (
            resolve_path(metadata_path_override)
            if metadata_path_override is not None
            else None,
            configured_metadata_path,
            dataset_root / configured_metadata_path.name,
            DEFAULT_METADATA_PATH,
        )
        if path is not None
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
    probability_candidates = tuple(
        path
        for path in (
            resolve_path(probability_root_override)
            if probability_root_override is not None
            else None,
            configured_probability_root,
            result_dir.parents[1]
            / "segmentation"
            / "unet"
            / "inference"
            / "probability_npy",
            DEFAULT_PROBABILITY_ROOT,
            DEFAULT_SEGMENTATION_RUN_DIR / "inference/probability_npy",
        )
        if path is not None
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


def evaluate_and_explain(
    models: list[SegmentationGuidedResNet50],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, dict[str, float], torch.Tensor]:
    dataset = loader.dataset
    gradcam_directories = {
        stage_name: output_dir / "gradcam_npy" / stage_name
        for stage_name in STAGE_NAMES
    }
    lrp_directories = {
        stage_name: output_dir / "lrp_npy" / stage_name
        for stage_name in STAGE_NAMES
    }
    visualization_dir = output_dir / "visualization"
    artifact_directories = (
        list(gradcam_directories.values())
        + list(lrp_directories.values())
        + [visualization_dir]
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
    gradcams = [LayerwiseGradCAM(model) for model in models]

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

            fold_gradcam_maps = [
                camera.generate(inputs.detach().clone(), predictions)
                for camera in gradcams
            ]
            gradcam_maps = {
                stage_name: normalize_unsigned(
                    torch.stack(
                        [
                            fold_maps[stage_name]
                            for fold_maps in fold_gradcam_maps
                        ]
                    ).mean(dim=0)
                ).cpu()
                for stage_name in STAGE_NAMES
            }
            fold_lrp_maps = [
                generate_layerwise_lrp(model, inputs, predictions)
                for model in models
            ]
            lrp_maps = {
                stage_name: normalize_signed(
                    torch.stack(
                        [fold_maps[stage_name] for fold_maps in fold_lrp_maps]
                    ).mean(dim=0)
                ).cpu()
                for stage_name in STAGE_NAMES
            }

            for batch_index in range(inputs.shape[0]):
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                prediction = int(predictions[batch_index])
                probability_values = probabilities[batch_index].cpu()
                for stage_name in STAGE_NAMES:
                    gradcam = (
                        gradcam_maps[stage_name][batch_index]
                        .numpy()
                        .astype(np.float32)
                    )
                    relevance = (
                        lrp_maps[stage_name][batch_index]
                        .numpy()
                        .astype(np.float32)
                    )
                    np.save(
                        gradcam_directories[stage_name] / filename,
                        gradcam,
                        allow_pickle=False,
                    )
                    np.save(
                        lrp_directories[stage_name] / filename,
                        relevance,
                        allow_pickle=False,
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
    predictions = pd.DataFrame(records)
    save_study_visualizations(
        predictions=predictions,
        dataset=dataset,
        gradcam_directories=gradcam_directories,
        lrp_directories=lrp_directories,
        visualization_dir=visualization_dir,
        dpi=dpi,
    )
    return predictions, metrics, confusion


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError(
            "Batch size and DPI must be positive; workers cannot be negative."
        )
    result_dir = resolve_path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Classification result not found: {result_dir}")
    output_dir = result_dir / "test"
    config, config_sources = load_run_config(result_dir)
    config = relocate_colab_data_paths(
        config,
        result_dir,
        dataset_root_override=args.dataset_root,
        metadata_path_override=args.metadata_path,
        probability_root_override=args.probability_root,
    )
    device = select_device(args.device)
    loader = build_test_loader(
        config, args.batch_size, args.num_workers, args.max_samples
    )

    print(f"Classification run : {result_dir}")
    print(
        "Configuration      : "
        + ", ".join(str(path.relative_to(result_dir)) for path in config_sources)
    )
    print(f"Dataset root       : {config['data']['dataset_root']}")
    print(f"CT input type      : {config['data']['ct_input_type']}")
    print(f"CT path column     : {config['data']['ct_path_column']}")
    print(f"Probability maps   : {config['data']['probability_root']}")
    print(f"Test samples       : {len(loader.dataset)}")
    print(f"Device             : {device}")
    print(f"Output directory   : {output_dir}")

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
            "gradcam_targets": [
                f"backbone.{stage_name}[-1]"
                for stage_name in STAGE_NAMES
            ],
            "lrp_rule": "EpsilonPlusFlat with ResNetCanonizer",
            "lrp_stages": list(STAGE_NAMES),
            "lrp_target": (
                "backbone stage representations conditioned on the "
                "U-Net probability map"
            ),
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
                *[
                    f"Grad-CAM {stage_name} + ground-truth mask"
                    for stage_name in STAGE_NAMES
                ],
                *[
                    f"LRP {stage_name} + ground-truth mask"
                    for stage_name in STAGE_NAMES
                ],
            ],
            "gradcam_directories": {
                stage_name: str(
                    output_dir / "gradcam_npy" / stage_name
                )
                for stage_name in STAGE_NAMES
            },
            "lrp_directories": {
                stage_name: str(output_dir / "lrp_npy" / stage_name)
                for stage_name in STAGE_NAMES
            },
            "ground_truth_overlay": {
                "fill_color": GROUND_TRUTH_FILL_COLOR,
                "fill_alpha": GROUND_TRUTH_FILL_ALPHA,
                "outer_outline_color": GROUND_TRUTH_OUTER_OUTLINE_COLOR,
                "inner_outline_color": GROUND_TRUTH_INNER_OUTLINE_COLOR,
            },
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
