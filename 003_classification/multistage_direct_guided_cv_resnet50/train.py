"""Train the multi-stage direct-guided ResNet-50 with five-fold CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from tqdm.auto import tqdm


CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_ROOT))

from utils import (  # noqa: E402
    EarlyStopping,
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    save_best_model,
    save_training_config,
    set_seed,
    update_confusion_matrix,
)

try:  # Support both ``python train.py`` and package execution.
    from .dataset import create_direct_guided_dataloader
    from .direct_guided_resnet50 import MultiStageDirectGuidedResNet50
    from .transforms import build_evaluation_transform, build_train_transform
except ImportError:
    from dataset import create_direct_guided_dataloader
    from direct_guided_resnet50 import MultiStageDirectGuidedResNet50
    from transforms import build_evaluation_transform, build_train_transform


DEFAULT_CONFIG_PATH = (
    CLASSIFICATION_ROOT
    / "configs"
    / "multistage_direct_guided_cv_resnet50.json"
)
OUTPUT_COMPONENT = "classification/multistage_direct_guided_cv_resnet50"


def parse_args() -> argparse.Namespace:
    """Read configuration and direct-guidance options from the command line."""

    parser = argparse.ArgumentParser(
        description="Train multi-stage direct-guided ResNet-50 with five-fold CV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Fair-comparison data and training configuration JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument("--alpha-layer1", type=float, default=None)
    parser.add_argument("--alpha-layer2", type=float, default=None)
    parser.add_argument("--alpha-layer3", type=float, default=None)
    parser.add_argument("--learnable-alpha", action="store_true", default=None)
    parser.add_argument(
        "--guidance-resize-mode",
        choices=("max", "bilinear"),
        default=None,
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configuration path relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    """Load one JSON experiment configuration."""

    path = resolve_project_path(path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return config


def select_device(requested_device: str) -> torch.device:
    """Select CPU or CUDA from the configured device name."""

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but is not available.")
    return torch.device(requested_device)


def select_weights(name: str) -> ResNet50_Weights | None:
    """Convert a configuration name to torchvision ResNet-50 weights."""

    if name == "DEFAULT":
        return ResNet50_Weights.DEFAULT
    if name == "IMAGENET1K_V2":
        return ResNet50_Weights.IMAGENET1K_V2
    if name == "NONE":
        return None
    raise ValueError(f"Unsupported pretrained_weights value: {name!r}")


def create_fold_dataloaders(
    config: dict[str, Any],
    fold: int,
) -> tuple[DataLoader, DataLoader]:
    """Create patient-isolated training and validation DataLoaders."""

    data_config = config["data"]
    training_config = config["training"]
    loader_config = config["dataloader"]

    mean = tuple(float(value) for value in data_config["normalization_mean"])
    std = tuple(float(value) for value in data_config["normalization_std"])
    height = int(data_config["input_height"])
    width = int(data_config["input_width"])
    transform_seed = int(training_config["transform_seed"])

    train_transform = build_train_transform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        seed=transform_seed,
    )
    evaluation_transform = build_evaluation_transform(
        height=height,
        width=width,
        mean=mean,
        std=std,
        seed=transform_seed,
    )

    common_arguments = {
        "root_dir": resolve_project_path(data_config["dataset_root"]),
        "batch_size": int(training_config["batch_size"]),
        "probability_root": resolve_project_path(data_config["probability_root"]),
        "class_to_idx": {
            str(name): int(index)
            for name, index in data_config["class_to_idx"].items()
        },
        "ct_path_column": str(data_config["ct_path_column"]),
        "num_workers": int(loader_config["num_workers"]),
        "pin_memory": bool(loader_config["pin_memory"])
        and torch.cuda.is_available(),
        "persistent_workers": bool(loader_config["persistent_workers"]),
        "prefetch_factor": int(loader_config["prefetch_factor"]),
        "metadata_path": resolve_project_path(data_config["metadata_path"]),
        "cv_fold": fold,
    }

    train_loader = create_direct_guided_dataloader(
        split="train",
        transform=train_transform,
        shuffle=bool(loader_config["train_shuffle"]),
        drop_last=bool(loader_config["train_drop_last"]),
        **common_arguments,
    )
    validation_loader = create_direct_guided_dataloader(
        split="val",
        transform=evaluation_transform,
        shuffle=bool(loader_config["val_shuffle"]),
        drop_last=bool(loader_config["val_drop_last"]),
        **common_arguments,
    )
    return train_loader, validation_loader


def create_holdout_dataloader(config: dict[str, Any]) -> DataLoader:
    """Create the unchanged independent holdout-test DataLoader."""

    data_config = config["data"]
    training_config = config["training"]
    loader_config = config["dataloader"]
    transform = build_evaluation_transform(
        height=int(data_config["input_height"]),
        width=int(data_config["input_width"]),
        mean=tuple(float(value) for value in data_config["normalization_mean"]),
        std=tuple(float(value) for value in data_config["normalization_std"]),
        seed=int(training_config["transform_seed"]),
    )
    return create_direct_guided_dataloader(
        root_dir=resolve_project_path(data_config["dataset_root"]),
        split="test",
        batch_size=int(training_config["batch_size"]),
        probability_root=resolve_project_path(data_config["probability_root"]),
        transform=transform,
        class_to_idx={
            str(name): int(index)
            for name, index in data_config["class_to_idx"].items()
        },
        ct_path_column=str(data_config["ct_path_column"]),
        shuffle=False,
        num_workers=int(loader_config["num_workers"]),
        pin_memory=bool(loader_config["pin_memory"])
        and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=bool(loader_config["persistent_workers"]),
        prefetch_factor=int(loader_config["prefetch_factor"]),
        metadata_path=resolve_project_path(data_config["metadata_path"]),
        cv_fold=0,
    )


def create_model(
    config: dict[str, Any],
    model_options: dict[str, Any],
    use_pretrained_weights: bool = True,
) -> MultiStageDirectGuidedResNet50:
    """Create one independent model for a fold."""

    data_config = config["data"]
    model_config = config["model"]
    weights = None
    if use_pretrained_weights:
        weights = select_weights(str(model_config["pretrained_weights"]))

    return MultiStageDirectGuidedResNet50(
        num_classes=len(data_config["class_to_idx"]),
        dropout=float(model_config["classifier_dropout"]),
        weights=weights,
        alpha_layer1=float(model_options["alpha_layer1"]),
        alpha_layer2=float(model_options["alpha_layer2"]),
        alpha_layer3=float(model_options["alpha_layer3"]),
        learnable_alpha=bool(model_options["learnable_alpha"]),
        guidance_resize_mode=str(model_options["guidance_resize_mode"]),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    description: str = "Training",
) -> dict[str, float]:
    """Train the full model for one epoch."""

    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=description, unit="batch", leave=False)
    for ct_images, probability_maps, labels in progress_bar:
        ct_images = ct_images.to(device, non_blocking=True)
        probability_maps = probability_maps.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(ct_images, probability_maps)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        predictions = binary_probabilities_to_predictions(
            probabilities=probabilities,
            threshold=threshold,
            positive_class_index=1,
        )
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size
        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            accuracy=f"{correct_predictions / total_samples:.4f}",
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct_predictions / total_samples,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    description: str = "Evaluation",
) -> tuple[dict[str, float], Tensor, Tensor]:
    """Evaluate one model and return metrics, targets, and probabilities."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros((2, 2), dtype=torch.int64)
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=description,
            unit="batch",
            leave=False,
        )
        for ct_images, probability_maps, labels in progress_bar:
            ct_images = ct_images.to(device, non_blocking=True)
            probability_maps = probability_maps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(ct_images, probability_maps)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = binary_probabilities_to_predictions(
                probabilities=probabilities,
                threshold=threshold,
                positive_class_index=1,
            )

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            confusion_matrix = update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=labels,
                num_classes=2,
            )
            all_targets.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
            progress_bar.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = total_loss / total_samples
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    return metrics, targets, probabilities


def train_fold(
    fold: int,
    config: dict[str, Any],
    model_options: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    """Train one fold and save its best validation-loss checkpoint."""

    training_config = config["training"]
    optimizer_config = config["optimizer"]
    early_config = config["early_stopping"]
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    train_loader, validation_loader = create_fold_dataloaders(config, fold)
    model = create_model(config, model_options).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        nesterov=bool(optimizer_config["nesterov"]),
    )
    early_stopping = EarlyStopping(
        patience=int(early_config["patience"]),
        mode=str(early_config["mode"]),
        min_delta=float(early_config["min_delta"]),
        verbose=bool(early_config["verbose"]),
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    total_epochs = int(training_config["num_epochs"])
    threshold = float(training_config["classification_threshold"])

    for epoch in range(1, total_epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            threshold,
            description=f"Fold {fold} epoch {epoch}/{total_epochs} [train]",
        )
        validation_metrics, _, _ = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            threshold,
            description=f"Fold {fold} epoch {epoch}/{total_epochs} [validation]",
        )

        is_best = validation_metrics["loss"] < best_validation_loss
        if is_best:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            save_best_model(model, fold_dir / "best_model.pt")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["accuracy"],
                "val_sensitivity": validation_metrics["sensitivity"],
                "val_specificity": validation_metrics["specificity"],
                "val_precision": validation_metrics["precision"],
                "val_f1_score": validation_metrics["f1_score"],
                "val_auc": validation_metrics["auc"],
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"Fold {fold} | epoch {epoch:03d}/{total_epochs} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} | "
            f"val accuracy {validation_metrics['accuracy']:.4f} | "
            f"val AUC {validation_metrics['auc']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.3e}"
        )

        latest_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "model_options": model_options,
        }
        torch.save(latest_checkpoint, fold_dir / "latest_checkpoint.pt")

        if early_stopping(validation_metrics["loss"], epoch):
            print(f"Early stopping fold {fold} after epoch {epoch}.")
            break

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    best_state = torch.load(
        fold_dir / "best_model.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    best_metrics, _, _ = evaluate(
        model,
        validation_loader,
        criterion,
        device,
        threshold,
        description=f"Fold {fold} [best validation]",
    )
    best_metrics["fold"] = float(fold)
    best_metrics["best_epoch"] = float(best_epoch)
    return best_metrics


def evaluate_holdout_ensemble(
    config: dict[str, Any],
    model_options: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    """Average the five best-fold probabilities on the holdout set."""

    holdout_loader = create_holdout_dataloader(config)
    number_of_folds = int(config["cross_validation"]["num_folds"])
    fold_probabilities: list[Tensor] = []
    targets: Tensor | None = None

    for fold in tqdm(
        range(number_of_folds),
        desc="Holdout ensemble",
        unit="fold",
    ):
        model = create_model(
            config,
            model_options,
            use_pretrained_weights=False,
        ).to(device)
        state = torch.load(
            output_dir / f"fold_{fold}" / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        _, fold_targets, probabilities = evaluate(
            model,
            holdout_loader,
            nn.CrossEntropyLoss(),
            device,
            float(config["training"]["classification_threshold"]),
            description=f"Holdout fold {fold}",
        )
        if targets is None:
            targets = fold_targets
        elif not torch.equal(targets, fold_targets):
            raise RuntimeError("Holdout sample order changed between folds.")
        fold_probabilities.append(probabilities)

    if targets is None:
        raise RuntimeError("No holdout predictions were produced.")

    ensemble_probabilities = torch.stack(fold_probabilities).mean(dim=0)
    threshold = float(config["training"]["classification_threshold"])
    predictions = binary_probabilities_to_predictions(
        ensemble_probabilities,
        threshold=threshold,
        positive_class_index=1,
    )
    confusion_matrix = update_confusion_matrix(
        torch.zeros((2, 2), dtype=torch.int64),
        predictions,
        targets,
        num_classes=2,
    )
    metrics = compute_classification_metrics(confusion_matrix)
    metrics["auc"] = compute_auc(
        targets.numpy(),
        ensemble_probabilities.numpy(),
    )

    prediction_frame = holdout_loader.dataset.metadata.copy()
    prediction_frame["target"] = targets.numpy()
    prediction_frame["prediction"] = predictions.numpy()
    prediction_frame["probability_benign"] = ensemble_probabilities[:, 0].numpy()
    prediction_frame["probability_malignant"] = ensemble_probabilities[:, 1].numpy()
    prediction_frame.to_csv(output_dir / "holdout_predictions.csv", index=False)
    save_training_config(metrics, output_dir / "holdout_metrics.json")
    return metrics


def main() -> None:
    """Run five-fold training followed by holdout ensemble evaluation."""

    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    if model_config["architecture"] != MultiStageDirectGuidedResNet50.architecture_name:
        raise ValueError(
            "Configuration architecture must be "
            f"{MultiStageDirectGuidedResNet50.architecture_name!r}."
        )
    set_seed(int(config["training"]["seed"]), deterministic=True)
    device = select_device(str(config["training"]["device"]))

    model_options = {
        "alpha_layer1": (
            model_config["alpha_layer1"]
            if args.alpha_layer1 is None
            else args.alpha_layer1
        ),
        "alpha_layer2": (
            model_config["alpha_layer2"]
            if args.alpha_layer2 is None
            else args.alpha_layer2
        ),
        "alpha_layer3": (
            model_config["alpha_layer3"]
            if args.alpha_layer3 is None
            else args.alpha_layer3
        ),
        "learnable_alpha": (
            model_config["learnable_alpha"]
            if args.learnable_alpha is None
            else args.learnable_alpha
        ),
        "guidance_resize_mode": (
            model_config["guidance_resize_mode"]
            if args.guidance_resize_mode is None
            else args.guidance_resize_mode
        ),
    }
    if args.output_dir is None:
        output_dir = (
            resolve_project_path(config["output"]["root_directory"])
            / str(config["experiment"]["id"])
            / OUTPUT_COMPONENT
        )
    else:
        output_dir = resolve_project_path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "source_config": str(resolve_project_path(args.config)),
        "architecture": MultiStageDirectGuidedResNet50.architecture_name,
        "output_component": OUTPUT_COMPONENT,
        "model_options": model_options,
        "comparison_settings": config,
    }
    save_training_config(run_config, output_dir / "run_config.json")
    snapshot_name = str(config["output"]["config_snapshot_filename"])
    save_training_config(config, output_dir / snapshot_name)

    fold_summaries = []
    for fold in range(int(config["cross_validation"]["num_folds"])):
        set_seed(int(config["training"]["seed"]), deterministic=True)
        summary = train_fold(
            fold,
            config,
            model_options,
            output_dir,
            device,
        )
        fold_summaries.append(summary)

    pd.DataFrame(fold_summaries).to_csv(
        output_dir / "cross_validation_summary.csv",
        index=False,
    )
    holdout_metrics = evaluate_holdout_ensemble(
        config,
        model_options,
        output_dir,
        device,
    )
    print("Holdout ensemble metrics:")
    print(json.dumps(holdout_metrics, indent=2))


if __name__ == "__main__":
    main()
