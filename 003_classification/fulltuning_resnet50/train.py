"""Fully fine-tune a pretrained ResNet-50 for CT classification."""

from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..utils import (
    append_training_log,
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    create_dataloader,
    create_training_log,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_roc_curve,
    plot_validation_metrics_curve,
    save_best_model,
    save_training_config,
    set_seed,
    update_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Project and output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = f"result_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_result"
    / "fulltuning_resnet50"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
TRAINING_LOG_PATH = OUTPUT_DIR / "training_log.csv"
TRAINING_CONFIG_PATH = OUTPUT_DIR / "training_config.json"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_latest.pth"


# ---------------------------------------------------------------------------
# Dataset and preprocessing
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
METADATA_PATH = DATASET_ROOT / "001_holdout_split_lidc_lndb.csv"
CT_PATH_COLUMN = "ct_windowed_path"
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TRAIN_DROP_LAST = False
VAL_DROP_LAST = False


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL_ARCHITECTURE = "ResNet50"
TRAINING_STRATEGY = "full_fine_tuning"
TRAINABLE_COMPONENT = "entire_model"
CLASSIFIER_DROPOUT = 0.3
CLASSIFICATION_THRESHOLD = 0.5

NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()

SEED = 42
TRANSFORM_SEED = SEED
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100
WEIGHT_DECAY_OPTM = 1e-4
MOMENTUM_OPTM = 0.9
NESTEROV_OPTM = False

BEST_MODEL_MONITOR = "val_loss"
BEST_MODEL_MODE = "min"
SAVE_LATEST_CHECKPOINT = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_train_transform() -> A.Compose:
    """Build the stochastic preprocessing used for training."""

    return A.Compose(
        [
            A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=TRANSFORM_SEED,
    )


def build_val_transform() -> A.Compose:
    """Build deterministic validation preprocessing."""

    return A.Compose(
        [
            A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=TRANSFORM_SEED,
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader, A.Compose, A.Compose]:
    """Create train and validation DataLoaders from the configured metadata."""

    train_transform = build_train_transform()
    val_transform = build_val_transform()
    train_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        metadata_path=METADATA_PATH,
        split=TRAIN_SPLIT,
        transform=train_transform,
        class_to_idx=CLASS_TO_IDX,
        ct_path_column=CT_PATH_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=TRAIN_DROP_LAST,
    )
    val_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        metadata_path=METADATA_PATH,
        split=VAL_SPLIT,
        transform=val_transform,
        class_to_idx=CLASS_TO_IDX,
        ct_path_column=CT_PATH_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=VAL_DROP_LAST,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train and validation DataLoaders must not be empty.")
    if train_loader.dataset.class_to_idx != val_loader.dataset.class_to_idx:
        raise ValueError("Train and validation class mappings do not match.")
    return train_loader, val_loader, train_transform, val_transform


def build_model(num_classes: int) -> nn.Module:
    """Build pretrained ResNet-50 with every parameter trainable."""

    model = models.resnet50(weights=WEIGHTS)
    classifier_input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT),
        nn.Linear(
            in_features=classifier_input_features,
            out_features=num_classes,
        ),
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    assert_full_model_trainable(model)
    return model.to(DEVICE)


def assert_full_model_trainable(model: nn.Module) -> None:
    """Verify that no ResNet-50 parameter is frozen."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("The model does not contain parameters.")
    frozen_names = [
        name
        for name, parameter in named_parameters
        if not parameter.requires_grad
    ]
    if frozen_names:
        raise RuntimeError(
            "Full fine-tuning requires every parameter to be trainable; "
            f"found frozen parameters: {frozen_names}"
        )


def set_full_fine_tuning_mode(model: nn.Module) -> None:
    """Enable training behavior for backbone, BatchNorm, and classifier."""

    model.train()
    assert_full_model_trainable(model)
    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_layers or any(
        not module.training for module in batch_norm_layers
    ):
        raise RuntimeError(
            "All ResNet-50 BatchNorm layers must be in training mode."
        )


def train_one_epoch(
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    classification_threshold: float,
    positive_class_index: int,
) -> tuple[float, float, int]:
    """Train the complete model for one epoch."""

    set_full_fine_tuning_mode(model)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        unit="batch",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite training loss at epoch {epoch + 1}."
            )
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)
        predictions = binary_probabilities_to_predictions(
            probabilities=probabilities,
            threshold=classification_threshold,
            positive_class_index=positive_class_index,
        )
        current_batch_size = labels.size(0)
        running_loss += loss.item() * current_batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += current_batch_size
        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{100 * correct_predictions / total_samples:.2f}%",
        )

    if total_samples == 0:
        raise RuntimeError("Training epoch processed no samples.")
    return (
        running_loss / total_samples,
        correct_predictions / total_samples,
        total_samples,
    )


def validate_one_epoch(
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    classification_threshold: float,
    positive_class_index: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one validation epoch without updating model parameters."""

    model.eval()
    running_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]",
        unit="batch",
        leave=False,
    )

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite validation loss at epoch {epoch + 1}."
                )
            probabilities = torch.softmax(outputs, dim=1)
            predictions = binary_probabilities_to_predictions(
                probabilities=probabilities,
                threshold=classification_threshold,
                positive_class_index=positive_class_index,
            )
            current_batch_size = labels.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size
            confusion_matrix = update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=labels,
                num_classes=num_classes,
            )
            all_targets.append(labels.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
            running_accuracy = (
                confusion_matrix.diag().sum().item() / total_samples
            )
            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{100 * running_accuracy:.2f}%",
            )

    if total_samples == 0:
        raise RuntimeError("Validation epoch processed no samples.")
    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = running_loss / total_samples
    metrics["auc"] = compute_auc(
        targets=targets.numpy(),
        probabilities=probabilities.numpy(),
    )
    return metrics, confusion_matrix, targets, probabilities


def save_latest_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_metrics: dict[str, float],
    best_metric: float,
    num_classes: int,
) -> None:
    """Save resumable state without scheduler or early-stopping state."""

    torch.save(
        {
            "epoch": epoch,
            "architecture": MODEL_ARCHITECTURE,
            "training_strategy": TRAINING_STRATEGY,
            "num_classes": num_classes,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": BATCH_SIZE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "early_stopping_state_dict": None,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "sensitivity": val_metrics["sensitivity"],
            "specificity": val_metrics["specificity"],
            "precision": val_metrics["precision"],
            "f1_score": val_metrics["f1_score"],
            "auc": val_metrics["auc"],
            "best_metric": best_metric,
            "best_metric_name": BEST_MODEL_MONITOR,
            "best_metric_mode": BEST_MODEL_MODE,
        },
        CHECKPOINT_PATH,
    )


def build_training_config(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_transform: A.Compose,
    val_transform: A.Compose,
    model: nn.Module,
    optimizer: torch.optim.SGD,
    criterion: nn.Module,
    positive_class_index: int,
) -> dict[str, object]:
    """Build the configuration consumed by the compatible test pipeline."""

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)
    optimizer_group = optimizer.param_groups[0]
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "experiment": {
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "training_strategy": TRAINING_STRATEGY,
            "pretrained_weights": str(WEIGHTS),
            "num_classes": len(train_dataset.classes),
            "backbone_frozen": False,
            "batch_norm_frozen": False,
            "batch_norm_mode_during_training": "train",
            "trainable_component": TRAINABLE_COMPONENT,
            "classifier": {
                "architecture": "dropout_linear",
                "dropout_probability": CLASSIFIER_DROPOUT,
            },
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": total_parameters - trainable_parameters,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "metadata_path": str(METADATA_PATH),
            "ct_path_column": CT_PATH_COLUMN,
            "image_height": INPUT_HEIGHT,
            "image_width": INPUT_WIDTH,
            "input_channels": 3,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "val_class_to_idx": val_dataset.class_to_idx,
            "train_class_distribution": {
                class_name: train_counts[class_index]
                for class_name, class_index in train_dataset.class_to_idx.items()
            },
            "val_class_distribution": {
                class_name: val_counts[class_index]
                for class_name, class_index in val_dataset.class_to_idx.items()
            },
            "train_transforms": str(train_transform),
            "val_transforms": str(val_transform),
            "transform_seed": TRANSFORM_SEED,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "seed": SEED,
            "learning_rate_behavior": "constant",
        },
        "loss": {"name": criterion.__class__.__name__},
        "metrics": {
            "names": [
                "confusion_matrix",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1_score",
                "roc_curve",
                "auc",
            ],
            "binary_positive_class_index": positive_class_index,
            "binary_positive_class_name": (
                train_dataset.classes[positive_class_index]
            ),
            "classification_threshold": CLASSIFICATION_THRESHOLD,
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "optimized_parameter_scope": "entire_model",
            "initial_learning_rate": optimizer_group["lr"],
            "weight_decay": optimizer_group["weight_decay"],
            "momentum": optimizer_group["momentum"],
            "dampening": optimizer_group["dampening"],
            "nesterov": optimizer_group["nesterov"],
        },
        "scheduler": None,
        "early_stopping": None,
        "checkpoint": {
            "best_model_monitor": BEST_MODEL_MONITOR,
            "best_model_mode": BEST_MODEL_MODE,
            "save_latest_checkpoint": SAVE_LATEST_CHECKPOINT,
            "best_model_path": str(BEST_MODEL_PATH),
            "checkpoint_path": str(CHECKPOINT_PATH),
        },
        "dataloader": {
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "train_shuffle": TRAIN_SHUFFLE,
            "val_shuffle": VAL_SHUFFLE,
            "train_drop_last": TRAIN_DROP_LAST,
            "val_drop_last": VAL_DROP_LAST,
        },
        "device": {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def _synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    """Run fixed-LR full fine-tuning and save best-model artifacts."""

    set_seed(seed=SEED, deterministic=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)
    create_training_log(TRAINING_LOG_PATH)

    train_loader, val_loader, train_transform, val_transform = (
        build_dataloaders()
    )
    train_dataset = train_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]
    model = build_model(num_classes)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM_OPTM,
        weight_decay=WEIGHT_DECAY_OPTM,
        nesterov=NESTEROV_OPTM,
    )
    criterion = nn.CrossEntropyLoss()
    assert_full_model_trainable(model)

    training_config = build_training_config(
        train_loader=train_loader,
        val_loader=val_loader,
        train_transform=train_transform,
        val_transform=val_transform,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        positive_class_index=positive_class_index,
    )
    save_training_config(training_config, TRAINING_CONFIG_PATH)

    parameter_counts = training_config["model"]
    print("Full Fine-Tuning ResNet-50")
    print(f"Train samples        : {len(train_loader.dataset)}")
    print(f"Validation samples   : {len(val_loader.dataset)}")
    print(f"Learning rate        : {LEARNING_RATE:.3e} (constant)")
    print(f"Optimizer            : SGD (momentum={MOMENTUM_OPTM:.1f})")
    print(f"Total parameters     : {parameter_counts['total_parameters']:,}")
    print(
        "Trainable parameters : "
        f"{parameter_counts['trainable_parameters']:,}"
    )
    print(f"Device               : {DEVICE}")
    print()

    best_metric = float("inf")
    training_started_at = time.perf_counter()
    for epoch in range(NUM_EPOCHS):
        _synchronize_device()
        epoch_started_at = time.perf_counter()
        train_phase_started_at = time.perf_counter()
        train_loss, train_accuracy, train_total_samples = train_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            classification_threshold=CLASSIFICATION_THRESHOLD,
            positive_class_index=positive_class_index,
        )
        _synchronize_device()
        train_time_sec = time.perf_counter() - train_phase_started_at

        val_phase_started_at = time.perf_counter()
        val_metrics, _, _, _ = validate_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            num_classes=num_classes,
            classification_threshold=CLASSIFICATION_THRESHOLD,
            positive_class_index=positive_class_index,
        )
        _synchronize_device()
        val_time_sec = time.perf_counter() - val_phase_started_at

        current_metric = val_metrics["loss"]
        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric
            save_best_model(model=model, save_path=BEST_MODEL_PATH)
            print(f"Best model updated (val_loss: {current_metric:.4f})")

        checkpoint_saved = False
        if SAVE_LATEST_CHECKPOINT:
            save_latest_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_metrics=val_metrics,
                best_metric=best_metric,
                num_classes=num_classes,
            )
            checkpoint_saved = True

        epoch_time = time.perf_counter() - epoch_started_at
        elapsed_time_sec = time.perf_counter() - training_started_at
        gpu_memory_allocated_mb = (
            torch.cuda.memory_allocated(DEVICE) / (1024**2)
            if DEVICE.type == "cuda"
            else 0.0
        )
        gpu_memory_reserved_mb = (
            torch.cuda.memory_reserved(DEVICE) / (1024**2)
            if DEVICE.type == "cuda"
            else 0.0
        )
        samples_per_sec = (
            train_total_samples / train_time_sec
            if train_time_sec > 0.0
            else 0.0
        )
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=epoch + 1,
            epoch_time=epoch_time,
            elapsed_time_sec=elapsed_time_sec,
            is_best=is_best,
            early_stop_counter=0,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            train_time_sec=train_time_sec,
            val_time_sec=val_time_sec,
            scheduler_updated=False,
            patience_counter=0,
            best_metric=best_metric,
            checkpoint_saved=checkpoint_saved,
            samples_per_sec=samples_per_sec,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            stopped_early=False,
            learning_rate=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
            precision=val_metrics["precision"],
            f1_score=val_metrics["f1_score"],
            auc_score=val_metrics["auc"],
        )

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Learning Rate        : {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Training Loss        : {train_loss:.4f}")
        print(f"Training Accuracy    : {train_accuracy:.2%}")
        print(f"Validation Loss      : {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy  : {val_metrics['accuracy']:.2%}")
        print(f"Sensitivity          : {val_metrics['sensitivity']:.4f}")
        print(f"Specificity          : {val_metrics['specificity']:.4f}")
        print(f"Precision            : {val_metrics['precision']:.4f}")
        print(f"F1-score             : {val_metrics['f1_score']:.4f}")
        print(f"ROC AUC              : {val_metrics['auc']:.4f}")
        print()

    history = pd.read_csv(TRAINING_LOG_PATH)
    best_model_state = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_model_state)
    (
        best_val_metrics,
        best_val_confusion_matrix,
        best_val_targets,
        best_val_probabilities,
    ) = validate_one_epoch(
        epoch=0,
        num_epochs=1,
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=num_classes,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        positive_class_index=positive_class_index,
    )

    print(f"Best Model Validation Loss     : {best_val_metrics['loss']:.4f}")
    print(
        "Best Model Validation Accuracy : "
        f"{best_val_metrics['accuracy']:.2%}"
    )
    print(f"Best Model ROC AUC             : {best_val_metrics['auc']:.4f}")
    plot_loss_curve(history, FIGURES_DIR)
    plot_accuracy_curve(history, FIGURES_DIR)
    plot_validation_metrics_curve(history, FIGURES_DIR)
    plot_confusion_matrix(
        confusion_matrix=best_val_confusion_matrix,
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )
    plot_roc_curve(
        targets=best_val_targets.numpy(),
        probabilities=best_val_probabilities.numpy(),
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
