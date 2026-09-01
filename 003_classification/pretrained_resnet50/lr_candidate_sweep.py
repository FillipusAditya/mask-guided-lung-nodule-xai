"""Compare fixed learning-rate candidates using equal-budget short training."""

from collections import Counter
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..utils.dataloader import create_dataloader
from ..utils.metrics import (
    compute_auc,
    compute_classification_metrics,
    update_confusion_matrix,
)
from ..utils.prediction import binary_probabilities_to_predictions
from ..utils.seed import set_seed


# ---------------------------------------------------------------------------
# Project and output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = (
    f"lr_candidate_sweep_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification"
    / "pretrained_resnet50"
    / RESULT_DIR_NAME
)
CANDIDATES_DIR = OUTPUT_DIR / "candidates"
FIGURES_DIR = OUTPUT_DIR / "figures"
CANDIDATE_SUMMARY_PATH = OUTPUT_DIR / "candidate_summary.csv"
SWEEP_CONFIG_PATH = OUTPUT_DIR / "sweep_config.json"
SWEEP_SUMMARY_PATH = OUTPUT_DIR / "sweep_summary.json"


# ---------------------------------------------------------------------------
# Dataset and preprocessing
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
# This is the metadata used by the first LR Range Test that established the
# candidate region evaluated by this sweep.
METADATA_PATH = DATASET_ROOT / "001_holdout_split_lidc_lndb.csv"
CT_PATH_COLUMN = "ct_windowed_path"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSIFICATION_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TRAIN_DROP_LAST = False
VAL_DROP_LAST = False
NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Model, optimizer, and short-training sweep
# ---------------------------------------------------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL_ARCHITECTURE = "ResNet50"
TRAINING_STRATEGY = "feature_extraction"
CLASSIFIER_DROPOUT = 0.3
WEIGHT_DECAY_OPTM = 1e-4

LR_CANDIDATE_START = 5e-3
LR_CANDIDATE_END = 1.2e-2
LR_CANDIDATE_STEP = 5e-4
SHORT_TRAINING_EPOCHS = 15
STABILITY_WINDOW = 5

SEED = 42
TRANSFORM_SEED = SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAINING_LOG_COLUMNS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_precision",
    "val_sensitivity",
    "val_specificity",
    "val_f1_score",
    "val_roc_auc",
    "epoch_time_seconds",
]

CANDIDATE_SUMMARY_COLUMNS = [
    "learning_rate",
    "status",
    "rank_by_val_loss",
    "completed_epochs",
    "best_epoch",
    "best_val_loss",
    "accuracy_at_best_val_loss",
    "precision_at_best_val_loss",
    "sensitivity_at_best_val_loss",
    "specificity_at_best_val_loss",
    "f1_at_best_val_loss",
    "roc_auc_at_best_val_loss",
    "final_train_loss",
    "final_val_loss",
    "minimum_train_loss",
    "val_loss_std_last_n_epochs",
    "total_training_seconds",
    "failure_message",
]


class NonFiniteLossError(RuntimeError):
    """Signal that an experiment produced a NaN or infinite loss."""


def validate_configuration() -> None:
    """Validate sweep constants before allocating training resources."""

    numeric_values = {
        "LR_CANDIDATE_START": LR_CANDIDATE_START,
        "LR_CANDIDATE_END": LR_CANDIDATE_END,
        "LR_CANDIDATE_STEP": LR_CANDIDATE_STEP,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if LR_CANDIDATE_START <= 0.0:
        raise ValueError("LR_CANDIDATE_START must be greater than zero.")
    if LR_CANDIDATE_END < LR_CANDIDATE_START:
        raise ValueError(
            "LR_CANDIDATE_END must be greater than or equal to the start."
        )
    if LR_CANDIDATE_STEP <= 0.0:
        raise ValueError("LR_CANDIDATE_STEP must be greater than zero.")

    integer_values = {
        "BATCH_SIZE": BATCH_SIZE,
        "SHORT_TRAINING_EPOCHS": SHORT_TRAINING_EPOCHS,
        "STABILITY_WINDOW": STABILITY_WINDOW,
        "NUM_WORKERS": NUM_WORKERS,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")
    if SHORT_TRAINING_EPOCHS <= 0:
        raise ValueError("SHORT_TRAINING_EPOCHS must be greater than zero.")
    if STABILITY_WINDOW <= 0:
        raise ValueError("STABILITY_WINDOW must be greater than zero.")
    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be non-negative.")
    if not 0.0 <= CLASSIFICATION_THRESHOLD <= 1.0:
        raise ValueError("CLASSIFICATION_THRESHOLD must be in [0, 1].")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Split metadata not found: {METADATA_PATH}")


def generate_lr_candidates(
    start: float = LR_CANDIDATE_START,
    end: float = LR_CANDIDATE_END,
    step: float = LR_CANDIDATE_STEP,
) -> list[float]:
    """Generate an inclusive, linearly spaced candidate list."""

    if not all(math.isfinite(value) for value in (start, end, step)):
        raise ValueError("Candidate start, end, and step must be finite.")
    if start <= 0.0:
        raise ValueError("Candidate start must be greater than zero.")
    if end < start:
        raise ValueError("Candidate end must not be less than start.")
    if step <= 0.0:
        raise ValueError("Candidate step must be greater than zero.")

    interval_count = int(round((end - start) / step))
    tolerance = max(abs(start), abs(end), abs(step), 1.0) * 1e-12
    reconstructed_end = start + interval_count * step
    if not math.isclose(
        reconstructed_end,
        end,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "Candidate range must be exactly divisible by the step so that "
            "the end value can be included."
        )

    candidates = [start + index * step for index in range(interval_count + 1)]
    candidates[-1] = end
    candidates = [float(f"{candidate:.12g}") for candidate in candidates]

    if not candidates or candidates[0] != start:
        raise RuntimeError("Failed to generate the candidate start value.")
    if not math.isclose(candidates[-1], end, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError("Failed to include the candidate end value.")
    if any(candidate > end + tolerance for candidate in candidates):
        raise RuntimeError("Generated candidate exceeds the configured end.")

    return candidates


def build_train_transform() -> A.Compose:
    """Build the stochastic transform used by normal training."""

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
    """Create fresh train and validation loaders for one candidate."""

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


def build_model() -> nn.Module:
    """Build ResNet-50 feature extraction with a fresh classifier head."""

    model = models.resnet50(weights=WEIGHTS)
    classifier_input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT),
        nn.Linear(
            in_features=classifier_input_features,
            out_features=len(CLASS_TO_IDX),
        ),
    )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    assert_feature_extraction_parameters(model)
    return model.to(DEVICE)


def assert_feature_extraction_parameters(model: nn.Module) -> None:
    """Verify that only classifier parameters are trainable."""

    trainable_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_names:
        raise RuntimeError("The model does not contain trainable parameters.")
    if any(not name.startswith("fc.") for name in trainable_names):
        raise RuntimeError(
            "Feature extraction expects only fc parameters to be trainable; "
            f"found {trainable_names}."
        )


def set_feature_extraction_mode(model: nn.Module) -> None:
    """Freeze backbone behavior while keeping classifier Dropout active."""

    model.eval()
    model.fc.train()

    training_batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.training
    ]
    if training_batch_norm_layers:
        raise RuntimeError("Frozen BatchNorm layers entered training mode.")


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.AdamW:
    """Build AdamW over trainable classifier parameters at a fixed LR."""

    trainable_parameters = [
        parameter
        for parameter in model.fc.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Classifier does not contain trainable parameters.")

    return torch.optim.AdamW(
        params=trainable_parameters,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY_OPTM,
    )


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    positive_class_index: int,
) -> tuple[float, float]:
    """Train the classifier for one epoch at a constant learning rate."""

    set_feature_extraction_mode(model)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{SHORT_TRAINING_EPOCHS} [Train]",
        unit="batch",
        leave=False,
    )
    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise NonFiniteLossError(
                f"Non-finite training loss at epoch {epoch}."
            )
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)
        predictions = binary_probabilities_to_predictions(
            probabilities=probabilities,
            threshold=CLASSIFICATION_THRESHOLD,
            positive_class_index=positive_class_index,
        )
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size
        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{correct_predictions / total_samples:.2%}",
        )

    if total_samples == 0:
        raise RuntimeError("Training epoch processed no samples.")
    return (
        running_loss / total_samples,
        correct_predictions / total_samples,
    )


def validate_one_epoch(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    positive_class_index: int,
) -> dict[str, float]:
    """Evaluate one epoch and return validation loss and metrics."""

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
        desc=f"Epoch {epoch:02d}/{SHORT_TRAINING_EPOCHS} [Validation]",
        unit="batch",
        leave=False,
    )
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss).item():
                raise NonFiniteLossError(
                    f"Non-finite validation loss at epoch {epoch}."
                )

            probabilities = torch.softmax(outputs, dim=1)
            predictions = binary_probabilities_to_predictions(
                probabilities=probabilities,
                threshold=CLASSIFICATION_THRESHOLD,
                positive_class_index=positive_class_index,
            )
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
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
                acc=f"{running_accuracy:.2%}",
            )

    if total_samples == 0:
        raise RuntimeError("Validation epoch processed no samples.")

    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = running_loss / total_samples
    metrics["auc"] = compute_auc(
        targets=torch.cat(all_targets).numpy(),
        probabilities=torch.cat(all_probabilities).numpy(),
    )
    return metrics


def _synchronize_device() -> None:
    """Synchronize CUDA before timing boundaries when applicable."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_candidate_experiment(
    learning_rate: float,
    candidate_index: int,
    num_candidates: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run one independent, equal-budget fixed-LR experiment."""

    set_seed(seed=SEED, deterministic=True)
    train_loader, val_loader, train_transform, val_transform = (
        build_dataloaders()
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]

    model = build_model()
    optimizer = build_optimizer(model, learning_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer_lr = float(optimizer.param_groups[0]["lr"])
    if not math.isclose(optimizer_lr, learning_rate, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("Optimizer learning rate differs from candidate LR.")

    print("=" * 68)
    print(f"Candidate {candidate_index}/{num_candidates}")
    print(f"Learning Rate: {learning_rate:.6f} ({learning_rate:.3e})")
    print("=" * 68)

    epoch_records: list[dict[str, float | int]] = []
    best_epoch_record: dict[str, float | int] | None = None
    _synchronize_device()
    candidate_started_at = time.perf_counter()

    for epoch in range(1, SHORT_TRAINING_EPOCHS + 1):
        _synchronize_device()
        epoch_started_at = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            positive_class_index=positive_class_index,
        )
        val_metrics = validate_one_epoch(
            epoch=epoch,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            positive_class_index=positive_class_index,
        )
        _synchronize_device()
        epoch_time_seconds = time.perf_counter() - epoch_started_at

        current_lr = float(optimizer.param_groups[0]["lr"])
        if current_lr != optimizer_lr:
            raise RuntimeError(
                "Learning rate changed during a fixed-LR candidate run."
            )

        record: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1_score": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["auc"],
            "epoch_time_seconds": epoch_time_seconds,
        }
        epoch_records.append(record)
        if (
            best_epoch_record is None
            or float(record["val_loss"])
            < float(best_epoch_record["val_loss"])
        ):
            best_epoch_record = dict(record)

        print(f"Epoch {epoch:02d}/{SHORT_TRAINING_EPOCHS}")
        print(f"  LR              : {current_lr:.3e}")
        print(f"  Train Loss      : {train_loss:.4f}")
        print(f"  Train Accuracy  : {train_accuracy:.2%}")
        print(f"  Val Loss        : {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy    : {val_metrics['accuracy']:.2%}")
        print(f"  Val ROC-AUC     : {val_metrics['auc']:.4f}")
        print(f"  Val F1          : {val_metrics['f1_score']:.4f}")

    _synchronize_device()
    total_training_seconds = time.perf_counter() - candidate_started_at
    history = pd.DataFrame(epoch_records, columns=TRAINING_LOG_COLUMNS)
    if best_epoch_record is None or history.empty:
        raise RuntimeError("Candidate experiment produced no epoch records.")

    stability_values = history["val_loss"].tail(STABILITY_WINDOW)
    summary = {
        "learning_rate": learning_rate,
        "status": "completed",
        "rank_by_val_loss": None,
        "completed_epochs": len(history),
        "best_epoch": int(best_epoch_record["epoch"]),
        "best_val_loss": float(best_epoch_record["val_loss"]),
        "accuracy_at_best_val_loss": float(
            best_epoch_record["val_accuracy"]
        ),
        "precision_at_best_val_loss": float(
            best_epoch_record["val_precision"]
        ),
        "sensitivity_at_best_val_loss": float(
            best_epoch_record["val_sensitivity"]
        ),
        "specificity_at_best_val_loss": float(
            best_epoch_record["val_specificity"]
        ),
        "f1_at_best_val_loss": float(
            best_epoch_record["val_f1_score"]
        ),
        "roc_auc_at_best_val_loss": float(
            best_epoch_record["val_roc_auc"]
        ),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_val_loss": float(history.iloc[-1]["val_loss"]),
        "minimum_train_loss": float(history["train_loss"].min()),
        "val_loss_std_last_n_epochs": float(
            stability_values.std(ddof=0)
        ),
        "total_training_seconds": total_training_seconds,
        "failure_message": None,
    }
    context = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_transform": train_transform,
        "val_transform": val_transform,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
    }
    return summary, history, context


def build_failed_summary(
    learning_rate: float,
    status: str,
    failure_message: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Create a summary row for a recoverable candidate failure."""

    return {
        column: (
            learning_rate
            if column == "learning_rate"
            else status
            if column == "status"
            else failure_message
            if column == "failure_message"
            else elapsed_seconds
            if column == "total_training_seconds"
            else 0
            if column == "completed_epochs"
            else None
        )
        for column in CANDIDATE_SUMMARY_COLUMNS
    }


def assign_ranks(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Rank completed candidates by ascending best validation loss."""

    ranked = summary_frame.copy().sort_values("learning_rate")
    ranked["rank_by_val_loss"] = pd.Series(
        pd.NA,
        index=ranked.index,
        dtype="Int64",
    )
    completed_mask = (
        (ranked["status"] == "completed")
        & np.isfinite(ranked["best_val_loss"].astype(float))
    )
    completed_indices = ranked.index[completed_mask]
    if len(completed_indices) > 0:
        ranks = (
            ranked.loc[completed_indices, "best_val_loss"]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )
        ranked.loc[completed_indices, "rank_by_val_loss"] = ranks
    return ranked[CANDIDATE_SUMMARY_COLUMNS].reset_index(drop=True)


def save_candidate_summary(summary_frame: pd.DataFrame) -> None:
    """Persist the latest cross-candidate results after each experiment."""

    assign_ranks(summary_frame).to_csv(
        CANDIDATE_SUMMARY_PATH,
        index=False,
        float_format="%.10g",
    )


def _json_safe(value: Any) -> Any:
    """Convert NumPy and non-finite values into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Write strict, human-readable JSON."""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(data), file, indent=2, allow_nan=False)
        file.write("\n")


def get_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen model parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_sweep_config(
    candidates: list[float],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a reproducibility snapshot for the complete sweep."""

    train_loader = context["train_loader"]
    val_loader = context["val_loader"]
    train_dataset = context["train_dataset"]
    val_dataset = context["val_dataset"]
    optimizer_parameters = context["optimizer"].param_groups[0]
    parameter_counts = get_parameter_counts(context["model"])
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    return {
        "experiment": {
            "type": "fixed_learning_rate_candidate_sweep",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
        },
        "candidates": {
            "start": LR_CANDIDATE_START,
            "end": LR_CANDIDATE_END,
            "step": LR_CANDIDATE_STEP,
            "values": candidates,
            "count": len(candidates),
            "learning_rate_behavior": "constant_per_candidate",
        },
        "training": {
            "epochs_per_candidate": SHORT_TRAINING_EPOCHS,
            "stability_window": STABILITY_WINDOW,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "transform_seed": TRANSFORM_SEED,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "selection_criterion": "minimum_validation_loss",
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "pretrained_weights": str(WEIGHTS),
            "training_strategy": TRAINING_STRATEGY,
            "backbone_frozen": True,
            "batch_norm_frozen": True,
            "trainable_component": "fc",
            "classifier": "Dropout -> Linear(2048, 2)",
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "parameters": parameter_counts,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "metadata_path": str(METADATA_PATH),
            "ct_path_column": CT_PATH_COLUMN,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "input_size": [INPUT_HEIGHT, INPUT_WIDTH, 3],
            "class_to_idx": train_dataset.class_to_idx,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "train_class_distribution": {
                class_name: train_counts[class_index]
                for class_name, class_index in train_dataset.class_to_idx.items()
            },
            "val_class_distribution": {
                class_name: val_counts[class_index]
                for class_name, class_index in val_dataset.class_to_idx.items()
            },
            "train_transforms": str(context["train_transform"]),
            "val_transforms": str(context["val_transform"]),
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "dataloader": {
            "train_shuffle": TRAIN_SHUFFLE,
            "val_shuffle": VAL_SHUFFLE,
            "train_drop_last": TRAIN_DROP_LAST,
            "val_drop_last": VAL_DROP_LAST,
            "num_workers": NUM_WORKERS,
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "pin_memory": PIN_MEMORY,
        },
        "optimizer": {
            "name": context["optimizer"].__class__.__name__,
            "weight_decay": optimizer_parameters["weight_decay"],
            "betas": optimizer_parameters["betas"],
            "epsilon": optimizer_parameters["eps"],
        },
        "loss": context["criterion"].__class__.__name__,
        "metrics": [
            "loss",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1_score",
            "roc_auc",
        ],
        "scheduler": None,
        "early_stopping": None,
        "checkpoint": None,
        "model_saving": None,
        "device": {
            "device": str(DEVICE),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "reproducibility": {
            "reset_seed_before_each_candidate": True,
            "fresh_model_per_candidate": True,
            "fresh_optimizer_per_candidate": True,
            "fresh_dataloaders_per_candidate": True,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


def plot_candidate_val_loss(histories: dict[float, pd.DataFrame]) -> None:
    """Plot validation-loss curves for all completed candidates."""

    if not histories:
        return
    figure, axis = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(histories)))
    for color, (learning_rate, history) in zip(colors, histories.items()):
        axis.plot(
            history["epoch"],
            history["val_loss"],
            color=color,
            linewidth=1.5,
            label=f"LR={learning_rate:.4f}",
        )
    axis.set_title("Fixed-LR Candidate Sweep: Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Loss")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "candidate_val_loss.png", dpi=300)
    plt.close(figure)


def _completed_by_lr(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Return finite, completed candidate rows sorted by learning rate."""

    completed = summary_frame[summary_frame["status"] == "completed"].copy()
    completed = completed[np.isfinite(completed["best_val_loss"].astype(float))]
    return completed.sort_values("learning_rate")


def plot_candidate_best_val_loss(summary_frame: pd.DataFrame) -> None:
    """Plot each candidate's best validation loss against learning rate."""

    completed = _completed_by_lr(summary_frame)
    if completed.empty:
        return
    best_row = completed.loc[completed["best_val_loss"].idxmin()]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        completed["learning_rate"],
        completed["best_val_loss"],
        marker="o",
        linewidth=1.8,
        color="tab:blue",
    )
    axis.scatter(
        [best_row["learning_rate"]],
        [best_row["best_val_loss"]],
        color="tab:red",
        zorder=3,
        label=f"Best LR={best_row['learning_rate']:.4f}",
    )
    axis.axvline(best_row["learning_rate"], color="tab:red", alpha=0.4)
    axis.set_title("Best Validation Loss vs Fixed Learning Rate")
    axis.set_xlabel("Learning Rate")
    axis.set_ylabel("Best Validation Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "candidate_best_val_loss.png", dpi=300)
    plt.close(figure)


def plot_candidate_roc_auc(summary_frame: pd.DataFrame) -> None:
    """Plot ROC-AUC at the best-validation-loss epoch for each LR."""

    completed = _completed_by_lr(summary_frame)
    if completed.empty:
        return
    best_row = completed.loc[completed["best_val_loss"].idxmin()]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        completed["learning_rate"],
        completed["roc_auc_at_best_val_loss"],
        marker="o",
        linewidth=1.8,
        color="tab:green",
    )
    axis.axvline(
        best_row["learning_rate"],
        color="tab:red",
        linestyle="--",
        alpha=0.65,
        label=f"Best-val-loss LR={best_row['learning_rate']:.4f}",
    )
    axis.set_title("ROC-AUC at Best-Val-Loss Epoch vs Learning Rate")
    axis.set_xlabel("Learning Rate")
    axis.set_ylabel("ROC-AUC")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "candidate_roc_auc.png", dpi=300)
    plt.close(figure)


def plot_candidate_metrics(summary_frame: pd.DataFrame) -> None:
    """Plot classification metrics at each candidate's selected epoch."""

    completed = _completed_by_lr(summary_frame)
    if completed.empty:
        return
    metric_columns = {
        "accuracy_at_best_val_loss": "Accuracy",
        "precision_at_best_val_loss": "Precision",
        "sensitivity_at_best_val_loss": "Sensitivity",
        "specificity_at_best_val_loss": "Specificity",
        "f1_at_best_val_loss": "F1",
    }
    figure, axis = plt.subplots(figsize=(11, 7))
    for column, label in metric_columns.items():
        axis.plot(
            completed["learning_rate"],
            completed[column],
            marker="o",
            linewidth=1.4,
            label=label,
        )
    axis.set_title("Validation Metrics at Best-Val-Loss Epoch")
    axis.set_xlabel("Learning Rate")
    axis.set_ylabel("Metric Value")
    axis.set_ylim(0.0, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "candidate_metrics.png", dpi=300)
    plt.close(figure)


def build_sweep_summary(summary_frame: pd.DataFrame) -> dict[str, Any]:
    """Build the machine-readable final candidate selection summary."""

    ranked = assign_ranks(summary_frame)
    completed = ranked[ranked["status"] == "completed"].sort_values(
        "rank_by_val_loss"
    )
    base: dict[str, Any] = {
        "num_candidates": len(ranked),
        "num_completed": len(completed),
        "num_failed": int((ranked["status"] != "completed").sum()),
        "candidate_start": LR_CANDIDATE_START,
        "candidate_end": LR_CANDIDATE_END,
        "candidate_step": LR_CANDIDATE_STEP,
        "selection_criterion": (
            "minimum validation loss during equal-budget short training"
        ),
        "best_candidate_lr": None,
        "best_candidate_rank": None,
        "best_val_loss": None,
        "best_epoch": None,
        "accuracy": None,
        "precision": None,
        "sensitivity": None,
        "specificity": None,
        "f1": None,
        "roc_auc": None,
        "runner_up_lr": None,
        "runner_up_val_loss": None,
    }
    if completed.empty:
        return base

    best = completed.iloc[0]
    base.update(
        {
            "best_candidate_lr": best["learning_rate"],
            "best_candidate_rank": best["rank_by_val_loss"],
            "best_val_loss": best["best_val_loss"],
            "best_epoch": best["best_epoch"],
            "accuracy": best["accuracy_at_best_val_loss"],
            "precision": best["precision_at_best_val_loss"],
            "sensitivity": best["sensitivity_at_best_val_loss"],
            "specificity": best["specificity_at_best_val_loss"],
            "f1": best["f1_at_best_val_loss"],
            "roc_auc": best["roc_auc_at_best_val_loss"],
        }
    )
    if len(completed) > 1:
        runner_up = completed.iloc[1]
        base["runner_up_lr"] = runner_up["learning_rate"]
        base["runner_up_val_loss"] = runner_up["best_val_loss"]
    return base


def print_sweep_header(candidates: list[float]) -> None:
    """Print the fixed configuration and candidate list."""

    print("Learning Rate Candidate Sweep")
    print()
    print(f"Candidates    : {len(candidates)}")
    print(f"Range         : {LR_CANDIDATE_START:.1e} -> {LR_CANDIDATE_END:.1e}")
    print(f"Step          : {LR_CANDIDATE_STEP:.1e}")
    print(f"Epochs        : {SHORT_TRAINING_EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print("Optimizer     : AdamW")
    print(f"Weight decay  : {WEIGHT_DECAY_OPTM:.1e}")
    print(f"Metadata      : {METADATA_PATH}")
    print(f"CT path column: {CT_PATH_COLUMN}")
    print("Candidate list:")
    print("  " + ", ".join(f"{candidate:.4f}" for candidate in candidates))
    print()


def print_final_ranking(
    summary_frame: pd.DataFrame,
    sweep_summary: dict[str, Any],
) -> None:
    """Print the top five completed candidates and output location."""

    ranked = assign_ranks(summary_frame)
    completed = ranked[ranked["status"] == "completed"].sort_values(
        "rank_by_val_loss"
    )
    print()
    print("Learning Rate Candidate Sweep Complete")
    print()
    if completed.empty:
        print("No candidate completed successfully.")
    else:
        print("Rank | LR       | Best Val Loss | Best Epoch | ROC-AUC | F1")
        print("-" * 66)
        for _, row in completed.head(5).iterrows():
            print(
                f"{int(row['rank_by_val_loss']):>4} | "
                f"{row['learning_rate']:.6f} | "
                f"{row['best_val_loss']:.6f}      | "
                f"{int(row['best_epoch']):>10} | "
                f"{row['roc_auc_at_best_val_loss']:.4f}  | "
                f"{row['f1_at_best_val_loss']:.4f}"
            )
        print()
        print("Best candidate LR based on validation loss:")
        print(f"  {sweep_summary['best_candidate_lr']:.6f}")
    print(f"Output directory: {OUTPUT_DIR}")


def release_candidate_resources(context: dict[str, Any] | None) -> None:
    """Release candidate-owned resources before constructing the next run."""

    if context is not None:
        context.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Execute the fixed-LR, equal-budget candidate comparison."""

    validate_configuration()
    candidates = generate_lr_candidates()
    print_sweep_header(candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)

    # Validate the complete construction path and persist configuration before
    # starting the expensive sweep. Candidate experiments still rebuild every
    # object after resetting the seed, so this probe cannot affect fairness.
    set_seed(seed=SEED, deterministic=True)
    (
        probe_train_loader,
        probe_val_loader,
        probe_train_transform,
        probe_val_transform,
    ) = build_dataloaders()
    probe_model = build_model()
    probe_optimizer = build_optimizer(probe_model, candidates[0])
    probe_context: dict[str, Any] = {
        "train_loader": probe_train_loader,
        "val_loader": probe_val_loader,
        "train_transform": probe_train_transform,
        "val_transform": probe_val_transform,
        "model": probe_model,
        "optimizer": probe_optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "train_dataset": probe_train_loader.dataset,
        "val_dataset": probe_val_loader.dataset,
    }
    save_json(build_sweep_config(candidates, probe_context), SWEEP_CONFIG_PATH)
    release_candidate_resources(probe_context)
    del probe_train_loader, probe_val_loader, probe_model, probe_optimizer

    candidate_summaries: list[dict[str, Any]] = []
    histories: dict[float, pd.DataFrame] = {}

    for candidate_index, learning_rate in enumerate(candidates, start=1):
        candidate_dir = CANDIDATES_DIR / f"lr_{learning_rate:.6f}"
        candidate_dir.mkdir(parents=False, exist_ok=False)
        context: dict[str, Any] | None = None
        candidate_started_at = time.perf_counter()

        try:
            summary, history, context = run_candidate_experiment(
                learning_rate=learning_rate,
                candidate_index=candidate_index,
                num_candidates=len(candidates),
            )
            history.to_csv(
                candidate_dir / "training_log.csv",
                index=False,
                float_format="%.10g",
            )
            histories[learning_rate] = history
        except NonFiniteLossError as error:
            summary = build_failed_summary(
                learning_rate=learning_rate,
                status="failed_non_finite_loss",
                failure_message=str(error),
                elapsed_seconds=time.perf_counter() - candidate_started_at,
            )
            print(f"Candidate failed: {error}")
            pd.DataFrame(columns=TRAINING_LOG_COLUMNS).to_csv(
                candidate_dir / "training_log.csv",
                index=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            summary = build_failed_summary(
                learning_rate=learning_rate,
                status="failed_cuda_oom",
                failure_message=str(error),
                elapsed_seconds=time.perf_counter() - candidate_started_at,
            )
            print(f"Candidate failed with CUDA OOM: {error}")
            pd.DataFrame(columns=TRAINING_LOG_COLUMNS).to_csv(
                candidate_dir / "training_log.csv",
                index=False,
            )
        except Exception:
            partial_frame = pd.DataFrame(
                candidate_summaries,
                columns=CANDIDATE_SUMMARY_COLUMNS,
            )
            if not partial_frame.empty:
                save_candidate_summary(partial_frame)
            raise
        finally:
            release_candidate_resources(context)

        candidate_summaries.append(summary)
        summary_frame = pd.DataFrame(
            candidate_summaries,
            columns=CANDIDATE_SUMMARY_COLUMNS,
        )
        save_candidate_summary(summary_frame)

    summary_frame = assign_ranks(
        pd.DataFrame(
            candidate_summaries,
            columns=CANDIDATE_SUMMARY_COLUMNS,
        )
    )
    save_candidate_summary(summary_frame)
    plot_candidate_val_loss(histories)
    plot_candidate_best_val_loss(summary_frame)
    plot_candidate_roc_auc(summary_frame)
    plot_candidate_metrics(summary_frame)

    sweep_summary = build_sweep_summary(summary_frame)
    save_json(sweep_summary, SWEEP_SUMMARY_PATH)
    print_final_ranking(summary_frame, sweep_summary)


if __name__ == "__main__":
    main()
