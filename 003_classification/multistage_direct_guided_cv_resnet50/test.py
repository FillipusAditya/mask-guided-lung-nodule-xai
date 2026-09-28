"""Smoke-test or evaluate the direct-guided ensemble with multi-stage XAI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_ROOT))

from utils import (  # noqa: E402
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    update_confusion_matrix,
)

try:
    from .dataset import DirectGuidedClassificationDataset
    from .direct_guided_resnet50 import (
        MultiStageDirectGuidedResNet50,
        apply_direct_guidance,
    )
    from .transforms import build_evaluation_transform
    from .xai import (
        MultiStageGradCAM,
        STAGE_NAMES,
        generate_multistage_lrp,
        normalize_signed,
        normalize_unsigned,
        save_study_visualizations,
    )
except ImportError:
    from dataset import DirectGuidedClassificationDataset
    from direct_guided_resnet50 import (
        MultiStageDirectGuidedResNet50,
        apply_direct_guidance,
    )
    from transforms import build_evaluation_transform
    from xai import (
        MultiStageGradCAM,
        STAGE_NAMES,
        generate_multistage_lrp,
        normalize_signed,
        normalize_unsigned,
        save_study_visualizations,
    )


CLASSIFICATION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse smoke-test or completed-run evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the model, or evaluate a trained five-fold ensemble "
            "with layer1-layer4 Grad-CAM and LRP."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Completed direct-guided training result directory.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--probability-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    """Resolve a repository-relative path."""

    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def select_device(requested: str) -> torch.device:
    """Select the requested inference device."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return values


def load_run_config(
    result_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Load training settings and direct-guidance model options."""

    run_config_path = result_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Run configuration not found: {run_config_path}")

    run_config = load_json(run_config_path)
    if run_config.get("architecture") != (
        MultiStageDirectGuidedResNet50.architecture_name
    ):
        raise ValueError("Result directory uses a different model architecture.")

    config = run_config.get("comparison_settings")
    model_options = run_config.get("model_options")
    if not isinstance(config, dict) or not isinstance(model_options, dict):
        raise ValueError("run_config.json is missing training or model settings.")

    sources = [run_config_path]
    snapshot_name = str(
        config.get("output", {}).get(
            "config_snapshot_filename",
            "multistage_direct_guided_cv_resnet50.json",
        )
    )
    snapshot_path = result_dir / snapshot_name
    if snapshot_path.is_file():
        snapshot = load_json(snapshot_path)
        if snapshot.get("model", {}).get("architecture") != (
            MultiStageDirectGuidedResNet50.architecture_name
        ):
            raise ValueError(f"Invalid architecture in {snapshot_path}.")
        config = snapshot
        sources.append(snapshot_path)

    return config, model_options, sources


def apply_data_overrides(
    config: dict[str, Any],
    dataset_root: Path | None,
    metadata_path: Path | None,
    probability_root: Path | None,
) -> dict[str, Any]:
    """Apply explicit Colab or local data paths."""

    config = dict(config)
    data = dict(config["data"])
    if dataset_root is not None:
        data["dataset_root"] = str(resolve_path(dataset_root))
    if metadata_path is not None:
        data["metadata_path"] = str(resolve_path(metadata_path))
    if probability_root is not None:
        data["probability_root"] = str(resolve_path(probability_root))

    for key in ("dataset_root", "metadata_path", "probability_root"):
        data[key] = str(resolve_path(data[key]))
    config["data"] = data
    return config


def resolve_metadata_path(root_dir: Path, value: object) -> Path:
    """Resolve one metadata path against the dataset root."""

    path = Path(str(value))
    return path if path.is_absolute() else root_dir / path


def build_test_loader(
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> DataLoader:
    """Build the unchanged independent holdout DataLoader."""

    data = config["data"]
    dataset = DirectGuidedClassificationDataset(
        root_dir=Path(data["dataset_root"]),
        metadata_path=Path(data["metadata_path"]),
        split="test",
        cv_fold=0,
        probability_root=Path(data["probability_root"]),
        transform=build_evaluation_transform(
            height=int(data["input_height"]),
            width=int(data["input_width"]),
            mean=tuple(float(value) for value in data["normalization_mean"]),
            std=tuple(float(value) for value in data["normalization_std"]),
            seed=int(config["training"]["transform_seed"]),
        ),
        class_to_idx={
            str(name): int(index)
            for name, index in data["class_to_idx"].items()
        },
        ct_path_column=str(data["ct_path_column"]),
    )
    if "mask_path" not in dataset.metadata.columns:
        raise ValueError("Test metadata must contain mask_path.")

    missing_masks = [
        resolve_metadata_path(dataset.root_dir, value)
        for value in dataset.metadata["mask_path"]
        if not resolve_metadata_path(dataset.root_dir, value).is_file()
    ]
    if missing_masks:
        examples = ", ".join(str(path) for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks; examples: {examples}"
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


def create_model(
    config: dict[str, Any],
    model_options: dict[str, Any],
) -> MultiStageDirectGuidedResNet50:
    """Create the architecture without downloading pretrained weights."""

    return MultiStageDirectGuidedResNet50(
        num_classes=len(config["data"]["class_to_idx"]),
        dropout=float(config["model"]["classifier_dropout"]),
        weights=None,
        alpha_layer1=float(model_options["alpha_layer1"]),
        alpha_layer2=float(model_options["alpha_layer2"]),
        alpha_layer3=float(model_options["alpha_layer3"]),
        learnable_alpha=bool(model_options["learnable_alpha"]),
        guidance_resize_mode=str(model_options["guidance_resize_mode"]),
    )


def load_models(
    result_dir: Path,
    config: dict[str, Any],
    model_options: dict[str, Any],
    device: torch.device,
) -> list[MultiStageDirectGuidedResNet50]:
    """Load the best checkpoint from every fold."""

    models = []
    number_of_folds = int(config["cross_validation"]["num_folds"])
    for fold in tqdm(
        range(number_of_folds),
        desc="Loading fold models",
        unit="fold",
    ):
        weights_path = result_dir / f"fold_{fold}" / "best_model.pt"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Best model not found: {weights_path}")
        model = create_model(config, model_options)
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        model.to(device).eval()
        models.append(model)
    return models


def evaluate_and_explain(
    models: list[MultiStageDirectGuidedResNet50],
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, dict[str, float], Tensor]:
    """Evaluate and create layer1-layer4 Grad-CAM and LRP maps."""

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

    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    records: list[dict[str, Any]] = []
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    total_loss = 0.0
    sample_index = 0
    gradcam_generators = [MultiStageGradCAM(model) for model in models]

    try:
        for ct_images, probability_maps, labels in tqdm(
            loader,
            desc="Holdout inference + multi-stage XAI",
            unit="batch",
        ):
            ct_images = ct_images.to(device, non_blocking=True)
            probability_maps = probability_maps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                fold_probabilities = [
                    torch.softmax(model(ct_images, probability_maps), dim=1)
                    for model in models
                ]
                probabilities = torch.stack(fold_probabilities).mean(dim=0)

            predictions = binary_probabilities_to_predictions(
                probabilities,
                threshold=CLASSIFICATION_THRESHOLD,
                positive_class_index=dataset.class_to_idx["malignant"],
            )
            total_loss += float(
                criterion(probabilities.clamp_min(1e-12).log(), labels)
            )
            confusion = update_confusion_matrix(
                confusion,
                predictions,
                labels,
                num_classes=2,
            )
            all_targets.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

            gradcam_per_fold = []
            lrp_per_fold = []
            for model, gradcam_generator in tqdm(
                zip(models, gradcam_generators, strict=True),
                total=len(models),
                desc="Fold XAI",
                unit="fold",
                leave=False,
            ):
                gradcam_per_fold.append(
                    gradcam_generator.generate(
                        ct_images.detach().clone(),
                        probability_maps.detach().clone(),
                        predictions,
                    )
                )
                lrp_per_fold.append(
                    generate_multistage_lrp(
                        model,
                        ct_images,
                        probability_maps,
                        predictions,
                    )
                )

            gradcam_maps = {
                stage: normalize_unsigned(
                    torch.stack(
                        [fold_maps[stage] for fold_maps in gradcam_per_fold]
                    ).mean(dim=0)
                ).cpu()
                for stage in STAGE_NAMES
            }
            lrp_maps = {
                stage: normalize_signed(
                    torch.stack(
                        [fold_maps[stage] for fold_maps in lrp_per_fold]
                    ).mean(dim=0)
                ).cpu()
                for stage in STAGE_NAMES
            }

            for batch_index in range(ct_images.shape[0]):
                row = dataset.metadata.iloc[sample_index]
                filename = Path(str(row["filename"])).name
                predicted_index = int(predictions[batch_index].item())
                probability_values = probabilities[batch_index].cpu()

                for stage_name in STAGE_NAMES:
                    np.save(
                        gradcam_directories[stage_name] / filename,
                        gradcam_maps[stage_name][batch_index]
                        .numpy()
                        .astype(np.float32),
                        allow_pickle=False,
                    )
                    np.save(
                        lrp_directories[stage_name] / filename,
                        lrp_maps[stage_name][batch_index]
                        .numpy()
                        .astype(np.float32),
                        allow_pickle=False,
                    )

                record = dict(row)
                record.update(
                    {
                        "true_index": int(labels[batch_index].item()),
                        "predicted_index": predicted_index,
                        "predicted_class": dataset.classes[predicted_index],
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
        for generator in gradcam_generators:
            generator.close()

    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    metrics = compute_classification_metrics(confusion)
    metrics["loss"] = total_loss / len(dataset)
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    prediction_frame = pd.DataFrame(records)
    save_study_visualizations(
        prediction_frame,
        dataset,
        gradcam_directories,
        lrp_directories,
        visualization_dir,
        dpi,
    )
    return prediction_frame, metrics, confusion


def run_smoke_test() -> None:
    """Run shape and direct-guidance behavior checks."""

    feature_map = torch.ones(1, 2, 4, 4)
    zero_map = torch.zeros(1, 1, 4, 4)
    assert torch.equal(
        apply_direct_guidance(feature_map, zero_map, alpha=1.0),
        feature_map,
    )

    regional_map = torch.zeros(1, 1, 4, 4)
    regional_map[:, :, 1:3, 1:3] = 1.0
    regional_features = apply_direct_guidance(
        feature_map,
        regional_map,
        alpha=1.0,
    )
    assert torch.equal(
        regional_features[:, :, 1:3, 1:3],
        feature_map[:, :, 1:3, 1:3] * 2.0,
    )

    batch_size = 2
    one_channel_ct = torch.randn(batch_size, 1, 224, 224)
    three_channel_ct = torch.randn(batch_size, 3, 224, 224)
    probability_map = torch.rand(batch_size, 1, 512, 512)
    model = MultiStageDirectGuidedResNet50()
    model.eval()
    with torch.no_grad():
        logits, guidance_maps = model(
            one_channel_ct,
            probability_map,
            return_guidance_maps=True,
        )
        three_channel_logits = model(three_channel_ct, probability_map)

    assert logits.shape == (batch_size, 2)
    assert three_channel_logits.shape == (batch_size, 2)
    assert guidance_maps["layer1"].shape == (batch_size, 1, 56, 56)
    assert guidance_maps["layer2"].shape == (batch_size, 1, 28, 28)
    assert guidance_maps["layer3"].shape == (batch_size, 1, 14, 14)
    print("CT input shape:", tuple(one_channel_ct.shape))
    print("Probability-map input shape:", tuple(probability_map.shape))
    for stage_name, guidance_map in guidance_maps.items():
        print(f"{stage_name} guidance shape:", tuple(guidance_map.shape))
    print("Output-logit shape:", tuple(logits.shape))
    print("All smoke tests passed.")


def main() -> None:
    """Run smoke checks or full holdout ensemble evaluation."""

    args = parse_args()
    if args.result_dir is None:
        run_smoke_test()
        return
    if args.batch_size <= 0 or args.num_workers < 0 or args.dpi <= 0:
        raise ValueError(
            "Batch size and DPI must be positive; workers cannot be negative."
        )

    result_dir = resolve_path(args.result_dir)
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Classification result not found: {result_dir}")
    config, model_options, config_sources = load_run_config(result_dir)
    config = apply_data_overrides(
        config,
        dataset_root=args.dataset_root,
        metadata_path=args.metadata_path,
        probability_root=args.probability_root,
    )
    device = select_device(args.device)
    loader = build_test_loader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    models = load_models(result_dir, config, model_options, device)
    output_dir = result_dir / "test"

    print(f"Classification run : {result_dir}")
    print("Configuration      : " + ", ".join(str(path) for path in config_sources))
    print(f"Dataset root       : {config['data']['dataset_root']}")
    print(f"Probability maps   : {config['data']['probability_root']}")
    print(f"Test samples       : {len(loader.dataset)}")
    print(f"Device             : {device}")
    print(f"Output directory   : {output_dir}")

    started = time.perf_counter()
    predictions, metrics, confusion = evaluate_and_explain(
        models,
        loader,
        device,
        output_dir,
        args.dpi,
    )
    elapsed_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    plot_confusion_matrix(confusion, loader.dataset.classes, output_dir)
    plot_roc_curve(
        predictions["true_index"].to_numpy(),
        predictions[
            ["probability_benign", "probability_malignant"]
        ].to_numpy(),
        loader.dataset.classes,
        output_dir,
    )

    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "architecture": MultiStageDirectGuidedResNet50.architecture_name,
        "ensemble_folds": len(models),
        "samples": len(loader.dataset),
        "metrics": {
            name: float(value) if np.isfinite(value) else None
            for name, value in metrics.items()
        },
        "runtime_seconds": elapsed_seconds,
        "xai": {
            "stages": list(STAGE_NAMES),
            "gradcam": "Grad-CAM at the final block of each ResNet stage",
            "lrp": (
                "Zennit EpsilonPlusFlat relevance with ResNetCanonizer, "
                "attributed from each guided stage representation"
            ),
            "visualization_directory": str(output_dir / "visualization"),
            "gradcam_directory": str(output_dir / "gradcam_npy"),
            "lrp_directory": str(output_dir / "lrp_npy"),
        },
    }
    with (output_dir / "test_results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Test samples : {len(loader.dataset)}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"AUC          : {metrics['auc']:.4f}")
    print(f"Outputs      : {output_dir}")


if __name__ == "__main__":
    main()
