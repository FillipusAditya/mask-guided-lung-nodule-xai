from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4

import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm.auto import tqdm

from ..utils import (
    EarlyStopping,
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
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "003_classification/configs/cv_resnet50.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

EXPERIMENT_CONFIG = CONFIG["experiment"]
OUTPUT_CONFIG = CONFIG["output"]
DATA_CONFIG = CONFIG["data"]
CV_CONFIG = CONFIG["cross_validation"]
MODEL_CONFIG = CONFIG["model"]
TRAINING_CONFIG = CONFIG["training"]
OPTIMIZER_CONFIG = CONFIG["optimizer"]
DATALOADER_CONFIG = CONFIG["dataloader"]
EARLY_STOPPING_CONFIG = CONFIG["early_stopping"]
CHECKPOINT_CONFIG = CONFIG["checkpoint"]


def resolve_project_path(value: str | Path) -> Path:
    """Return an absolute path for one JSON path value."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]

EXPERIMENT_ID = str(EXPERIMENT_CONFIG["id"])
EXPERIMENT_COMPONENT = str(EXPERIMENT_CONFIG["component"])
OUTPUT_DIR = (
    resolve_project_path(OUTPUT_CONFIG["root_directory"])
    / EXPERIMENT_ID
    / EXPERIMENT_COMPONENT
)
CONFIG_SNAPSHOT_PATH = OUTPUT_DIR / str(OUTPUT_CONFIG["config_snapshot_filename"])
CV_CONFIG_PATH = OUTPUT_DIR / "cv_config.json"
CV_SUMMARY_PATH = OUTPUT_DIR / "cv_summary.csv"
CV_SUMMARY_JSON_PATH = OUTPUT_DIR / "cv_summary.json"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "out_of_fold_predictions.csv"
CV_FIGURES_DIR = OUTPUT_DIR / "figures"

DATASET_ROOT = resolve_project_path(DATA_CONFIG["dataset_root"])
METADATA_PATH = resolve_project_path(DATA_CONFIG["metadata_path"])
CT_PATH_COLUMN = str(DATA_CONFIG["ct_path_column"])
INPUT_HEIGHT = int(DATA_CONFIG["input_height"])
INPUT_WIDTH = int(DATA_CONFIG["input_width"])
CLASS_TO_IDX = {
    str(name): int(index) for name, index in DATA_CONFIG["class_to_idx"].items()
}
IMAGENET_MEAN = tuple(DATA_CONFIG["normalization_mean"])
IMAGENET_STD = tuple(DATA_CONFIG["normalization_std"])

NUM_FOLDS = int(CV_CONFIG["num_folds"])
CV_FOLDS = tuple(range(NUM_FOLDS))
DEVELOPMENT_ROLE = str(CV_CONFIG["development_role"])
HOLDOUT_ROLE = str(CV_CONFIG["holdout_role"])
HOLDOUT_FOLD = int(CV_CONFIG["holdout_fold"])
GROUP_COLUMN = str(CV_CONFIG["group_column"])
NODULE_COLUMN = str(CV_CONFIG["nodule_column"])
FOLD_COLUMN = str(CV_CONFIG["fold_column"])
ROLE_COLUMN = str(CV_CONFIG["role_column"])

MODEL_ARCHITECTURE = str(MODEL_CONFIG["architecture"])
TRAINING_STRATEGY = str(MODEL_CONFIG["training_strategy"])
TRAINABLE_COMPONENT = str(MODEL_CONFIG["trainable_component"])
CLASSIFIER_DROPOUT = float(MODEL_CONFIG["classifier_dropout"])

NUM_EPOCHS = int(TRAINING_CONFIG["num_epochs"])
BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
LEARNING_RATE = float(TRAINING_CONFIG["learning_rate"])
SEED = int(TRAINING_CONFIG["seed"])
TRANSFORM_SEED = int(TRAINING_CONFIG["transform_seed"])
CLASSIFICATION_THRESHOLD = float(TRAINING_CONFIG["classification_threshold"])

MOMENTUM = float(OPTIMIZER_CONFIG["momentum"])
WEIGHT_DECAY = float(OPTIMIZER_CONFIG["weight_decay"])
NESTEROV = bool(OPTIMIZER_CONFIG["nesterov"])

NUM_WORKERS = int(DATALOADER_CONFIG["num_workers"])
PERSISTENT_WORKERS = bool(DATALOADER_CONFIG["persistent_workers"])
PREFETCH_FACTOR = int(DATALOADER_CONFIG["prefetch_factor"])
PIN_MEMORY = bool(DATALOADER_CONFIG["pin_memory"]) and torch.cuda.is_available()
TRAIN_SHUFFLE = bool(DATALOADER_CONFIG["train_shuffle"])
VAL_SHUFFLE = bool(DATALOADER_CONFIG["val_shuffle"])
TRAIN_DROP_LAST = bool(DATALOADER_CONFIG["train_drop_last"])
VAL_DROP_LAST = bool(DATALOADER_CONFIG["val_drop_last"])

BEST_MODEL_MONITOR = str(EARLY_STOPPING_CONFIG["monitor"])
BEST_MODEL_MODE = str(EARLY_STOPPING_CONFIG["mode"])
EARLY_STOPPING_PATIENCE = int(EARLY_STOPPING_CONFIG["patience"])
EARLY_STOPPING_MIN_DELTA = float(EARLY_STOPPING_CONFIG["min_delta"])
EARLY_STOPPING_VERBOSE = bool(EARLY_STOPPING_CONFIG["verbose"])
SAVE_LATEST_CHECKPOINT = bool(CHECKPOINT_CONFIG["save_latest"])


def get_device() -> torch.device:
    """Select the configured device."""

    device_name = str(TRAINING_CONFIG["device"]).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but is not available.")
    return torch.device(device_name)


def get_pretrained_weights() -> ResNet50_Weights | None:
    """Convert the configured weight name to a torchvision value."""

    name = str(MODEL_CONFIG["pretrained_weights"]).upper()
    if name == "DEFAULT":
        return ResNet50_Weights.DEFAULT
    if name == "IMAGENET1K_V2":
        return ResNet50_Weights.IMAGENET1K_V2
    if name == "NONE":
        return None
    raise ValueError(f"Unsupported pretrained_weights: {name}")


DEVICE = get_device()
WEIGHTS = get_pretrained_weights()

SUMMARY_METRICS = (
    "best_val_loss",
    "best_val_accuracy",
    "best_sensitivity",
    "best_specificity",
    "best_precision",
    "best_f1_score",
    "best_auc",
)


def validate_configuration() -> None:
    """Validate values that are required by this baseline."""

    if MODEL_ARCHITECTURE != "ResNet50":
        raise ValueError("Only architecture='ResNet50' is supported.")
    if str(OPTIMIZER_CONFIG["name"]).upper() != "SGD":
        raise ValueError("Only the SGD optimizer is supported.")
    if ROLE_COLUMN != "cv_role" or FOLD_COLUMN != "cv_fold":
        raise ValueError("The dataset requires cv_role and cv_fold columns.")
    if not bool(EARLY_STOPPING_CONFIG["enabled"]):
        raise ValueError("Early stopping must be enabled.")
    if not bool(EARLY_STOPPING_CONFIG["restore_best_weights"]):
        raise ValueError("restore_best_weights must be true.")
    if BEST_MODEL_MONITOR != "val_loss" or BEST_MODEL_MODE != "min":
        raise ValueError("The baseline must select the lowest val_loss.")
    if NUM_FOLDS < 2:
        raise ValueError("num_folds must be at least 2.")
    if NUM_EPOCHS <= 0 or BATCH_SIZE <= 0:
        raise ValueError("num_epochs and batch_size must be positive.")


# ---------------------------------------------------------------------------
# Transforms, metadata, and DataLoaders
# ---------------------------------------------------------------------------
def build_train_transform() -> A.Compose:
    """Create preprocessing and augmentation for training."""

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
    """Create deterministic preprocessing for validation."""

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


def validate_cv_metadata() -> pd.DataFrame:
    """Load metadata and check patient and nodule isolation."""

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"CV metadata not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    required_columns = {
        "dataset",
        "patient_id",
        "filename",
        CT_PATH_COLUMN,
        "label",
        "split",
        GROUP_COLUMN,
        NODULE_COLUMN,
        ROLE_COLUMN,
        FOLD_COLUMN,
    }
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"CV metadata is missing columns: {sorted(missing_columns)}")
    if metadata.empty:
        raise ValueError("CV metadata must not be empty.")
    if metadata[list(required_columns)].isna().any().any():
        raise ValueError("Required CV columns contain missing values.")
    if metadata.duplicated(["dataset", "filename"]).any():
        raise ValueError("CV metadata contains duplicate image rows.")

    metadata = metadata.copy()
    metadata[ROLE_COLUMN] = metadata[ROLE_COLUMN].astype(str).str.strip().str.lower()
    numeric_folds = pd.to_numeric(metadata[FOLD_COLUMN], errors="coerce")
    if numeric_folds.isna().any():
        raise ValueError("cv_fold values must be integers.")
    if not (numeric_folds == numeric_folds.astype(int)).all():
        raise ValueError("cv_fold values must be integers.")
    metadata[FOLD_COLUMN] = numeric_folds.astype(int)

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE)]
    expected_roles = {DEVELOPMENT_ROLE, HOLDOUT_ROLE}
    if set(metadata[ROLE_COLUMN].unique()) != expected_roles:
        raise ValueError(f"Expected cv_role values: {expected_roles}.")
    if set(development[FOLD_COLUMN].unique()) != set(CV_FOLDS):
        raise ValueError("Development data does not contain every fold.")
    if not holdout[FOLD_COLUMN].eq(HOLDOUT_FOLD).all():
        raise ValueError("Every holdout row must use the holdout fold.")

    if set(development[GROUP_COLUMN]) & set(holdout[GROUP_COLUMN]):
        raise RuntimeError("Patient leakage exists between CV and holdout.")
    if development.groupby(GROUP_COLUMN)[FOLD_COLUMN].nunique().ne(1).any():
        raise RuntimeError("A development patient appears in multiple folds.")
    if development.groupby(NODULE_COLUMN)[FOLD_COLUMN].nunique().ne(1).any():
        raise RuntimeError("A development nodule appears in multiple folds.")

    expected_labels = {name.lower() for name in CLASS_TO_IDX}
    expected_datasets = set(development["dataset"].astype(str))
    for fold in CV_FOLDS:
        fold_metadata = development[development[FOLD_COLUMN].eq(fold)]
        labels = set(fold_metadata["label"].astype(str).str.lower())
        datasets = set(fold_metadata["dataset"].astype(str))
        if labels != expected_labels:
            raise ValueError(f"Fold {fold} does not contain every class.")
        if datasets != expected_datasets:
            raise ValueError(f"Fold {fold} does not contain every dataset.")
    return metadata


def build_fold_dataloaders(
    fold: int,
) -> tuple[DataLoader, DataLoader, A.Compose, A.Compose]:
    """Create train and validation DataLoaders for one fold."""

    train_transform = build_train_transform()
    val_transform = build_val_transform()
    common = {
        "root_dir": DATASET_ROOT,
        "metadata_path": METADATA_PATH,
        "cv_fold": fold,
        "class_to_idx": CLASS_TO_IDX,
        "ct_path_column": CT_PATH_COLUMN,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
    }
    train_loader = create_dataloader(
        split="train",
        transform=train_transform,
        shuffle=TRAIN_SHUFFLE,
        drop_last=TRAIN_DROP_LAST,
        **common,
    )
    val_loader = create_dataloader(
        split="val",
        transform=val_transform,
        shuffle=VAL_SHUFFLE,
        drop_last=VAL_DROP_LAST,
        **common,
    )
    assert_fold_isolation(train_loader, val_loader, fold)
    return train_loader, val_loader, train_transform, val_transform


def assert_fold_isolation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold: int,
) -> None:
    """Check that train and validation data do not overlap."""

    train_metadata = train_loader.dataset.metadata
    val_metadata = val_loader.dataset.metadata
    if train_metadata.empty or val_metadata.empty:
        raise ValueError(f"Fold {fold} has an empty dataset.")
    if not train_metadata[FOLD_COLUMN].ne(fold).all():
        raise RuntimeError(f"Fold {fold} leaked into training data.")
    if not val_metadata[FOLD_COLUMN].eq(fold).all():
        raise RuntimeError("Validation rows use the wrong fold.")
    for column in (GROUP_COLUMN, NODULE_COLUMN, "filename"):
        overlap = set(train_metadata[column]) & set(val_metadata[column])
        if overlap:
            raise RuntimeError(f"Train/validation overlap found in {column}.")
    if train_loader.dataset.class_to_idx != val_loader.dataset.class_to_idx:
        raise ValueError("Train and validation class mappings do not match.")


# ---------------------------------------------------------------------------
# Model and epoch loops
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    """Create a fully trainable pretrained ResNet-50."""

    model = models.resnet50(weights=WEIGHTS)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=CLASSIFIER_DROPOUT),
        nn.Linear(input_features, num_classes),
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    assert_full_model_trainable(model)
    return model.to(DEVICE)


def assert_full_model_trainable(model: nn.Module) -> None:
    """Verify that every parameter is trainable."""

    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        raise RuntimeError(f"Full fine-tuning found frozen parameters: {frozen}")


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    positive_class_index: int,
) -> tuple[float, float, int]:
    """Train the model over all batches once."""

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
        unit="batch",
        leave=True,
    )
    for images, labels in progress_bar:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
        loss.backward()
        optimizer.step()

        probabilities = torch.softmax(outputs, dim=1)
        predictions = binary_probabilities_to_predictions(
            probabilities,
            CLASSIFICATION_THRESHOLD,
            positive_class_index,
        )
        current_batch_size = labels.size(0)
        running_loss += loss.item() * current_batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += current_batch_size
        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            accuracy=f"{100 * correct_predictions / total_samples:.2f}%",
        )
    if total_samples == 0:
        raise RuntimeError("Training processed no samples.")
    return (
        running_loss / total_samples,
        correct_predictions / total_samples,
        total_samples,
    )


def validate_one_epoch(
    epoch: int,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    positive_class_index: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the model without updating parameters."""

    model.eval()
    running_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS} [Validation]",
        unit="batch",
        leave=True,
    )
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")
            probabilities = torch.softmax(outputs, dim=1)
            predictions = binary_probabilities_to_predictions(
                probabilities,
                CLASSIFICATION_THRESHOLD,
                positive_class_index,
            )
            current_batch_size = labels.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size
            confusion_matrix = update_confusion_matrix(
                confusion_matrix,
                predictions,
                labels,
                num_classes,
            )
            all_targets.append(labels.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
            accuracy = confusion_matrix.diag().sum().item() / total_samples
            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                accuracy=f"{100 * accuracy:.2f}%",
            )
    if total_samples == 0:
        raise RuntimeError("Validation processed no samples.")
    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = running_loss / total_samples
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    return metrics, confusion_matrix, targets, probabilities


# ---------------------------------------------------------------------------
# Saved files and configuration records
# ---------------------------------------------------------------------------
def create_fold_paths(fold: int) -> dict[str, Path]:
    """Create the output folders and filenames for one fold."""

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    figures_dir = fold_dir / "figures"
    fold_dir.mkdir(parents=False, exist_ok=False)
    figures_dir.mkdir(parents=False, exist_ok=False)
    return {
        "fold_dir": fold_dir,
        "figures_dir": figures_dir,
        "training_log": fold_dir / "training_log.csv",
        "training_config": fold_dir / "training_config.json",
        "best_model": fold_dir / "best_model.pth",
        "latest_checkpoint": fold_dir / "checkpoint_latest.pth",
        "validation_predictions": fold_dir / "validation_predictions.csv",
    }


def save_latest_checkpoint(
    path: Path,
    fold: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_metrics: dict[str, float],
    best_metric: float,
    best_epoch: int,
    num_classes: int,
    early_stopping: EarlyStopping,
) -> None:
    """Save the current fold state so the run can be inspected."""

    torch.save(
        {
            "fold": fold,
            "epoch": epoch,
            "architecture": MODEL_ARCHITECTURE,
            "training_strategy": TRAINING_STRATEGY,
            "num_classes": num_classes,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_size": BATCH_SIZE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "early_stopping_state_dict": early_stopping.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_metrics": val_metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "best_metric_name": BEST_MODEL_MONITOR,
            "best_metric_mode": BEST_MODEL_MODE,
        },
        path,
    )


def get_class_distribution(dataset) -> dict[str, int]:
    """Return the number of samples for each class."""

    counts = Counter(dataset.targets)
    return {name: counts[index] for name, index in dataset.class_to_idx.items()}


def build_fold_config(
    fold: int,
    paths: dict[str, Path],
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_transform: A.Compose,
    val_transform: A.Compose,
    model: nn.Module,
    optimizer: torch.optim.SGD,
    criterion: nn.Module,
    positive_class_index: int,
) -> dict[str, object]:
    """Create the training_config.json content for one fold."""

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    optimizer_group = optimizer.param_groups[0]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "experiment": {
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "result_directory": Path(EXPERIMENT_COMPONENT).name,
            "output_directory": str(paths["fold_dir"]),
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "type": "stratified_group_cross_validation",
            "experiment_id": EXPERIMENT_ID,
            "component": EXPERIMENT_COMPONENT,
            "fold": fold,
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
            "train_split": None,
            "val_split": None,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "val_class_to_idx": val_dataset.class_to_idx,
            "train_class_distribution": get_class_distribution(train_dataset),
            "val_class_distribution": get_class_distribution(val_dataset),
            "train_transforms": str(train_transform),
            "val_transforms": str(val_transform),
            "transform_seed": TRANSFORM_SEED,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
            "cv_role": DEVELOPMENT_ROLE,
            "validation_fold": fold,
            "train_filter": (
                f"{ROLE_COLUMN} == {DEVELOPMENT_ROLE} and " f"{FOLD_COLUMN} != {fold}"
            ),
            "validation_filter": (
                f"{ROLE_COLUMN} == {DEVELOPMENT_ROLE} and " f"{FOLD_COLUMN} == {fold}"
            ),
            "holdout_filter": (
                f"{ROLE_COLUMN} == {HOLDOUT_ROLE} and "
                f"{FOLD_COLUMN} == {HOLDOUT_FOLD}"
            ),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "seed": SEED,
            "learning_rate_behavior": "constant",
            "cross_validation": True,
            "num_folds": NUM_FOLDS,
            "seed_reset_before_each_fold": True,
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
            "binary_positive_class_name": train_dataset.classes[positive_class_index],
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
        "early_stopping": {
            "enabled": True,
            "monitor": BEST_MODEL_MONITOR,
            "mode": BEST_MODEL_MODE,
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
            "restore_best_weights": True,
        },
        "checkpoint": {
            "best_model_monitor": BEST_MODEL_MONITOR,
            "best_model_mode": BEST_MODEL_MODE,
            "save_latest_checkpoint": SAVE_LATEST_CHECKPOINT,
            "best_model_path": str(paths["best_model"]),
            "checkpoint_path": str(paths["latest_checkpoint"]),
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
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "cross_validation": {
            "method": "StratifiedGroupKFold",
            "fold": fold,
            "all_folds": list(CV_FOLDS),
            "group_column": GROUP_COLUMN,
            "nodule_column": NODULE_COLUMN,
            "role_column": ROLE_COLUMN,
            "fold_column": FOLD_COLUMN,
            "holdout_used_during_training": False,
        },
    }


def build_cv_config(metadata: pd.DataFrame) -> dict[str, object]:
    """Create the cv_config.json content shared by all folds."""

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE)]
    fold_distribution = {}
    for fold in CV_FOLDS:
        fold_metadata = development[development[FOLD_COLUMN].eq(fold)]
        fold_distribution[str(fold)] = {
            "slices": len(fold_metadata),
            "nodules": fold_metadata[NODULE_COLUMN].nunique(),
            "patients": fold_metadata[GROUP_COLUMN].nunique(),
            "labels": fold_metadata["label"].value_counts().to_dict(),
        }
    return {
        "experiment": {
            "type": "stratified_group_cross_validation",
            "experiment_id": EXPERIMENT_ID,
            "component": EXPERIMENT_COMPONENT,
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "output_directory": str(OUTPUT_DIR),
        },
        "cross_validation": {
            "num_folds": NUM_FOLDS,
            "folds": list(CV_FOLDS),
            "random_seed": SEED,
            "metadata_path": str(METADATA_PATH),
            "group_column": GROUP_COLUMN,
            "nodule_column": NODULE_COLUMN,
            "fold_column": FOLD_COLUMN,
            "role_column": ROLE_COLUMN,
            "train_rule": "development rows outside the validation fold",
            "validation_rule": "development rows in the validation fold",
            "holdout_rule": "holdout rows are not used during CV",
            "holdout_used_during_cv": False,
            "fold_distribution": fold_distribution,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "ct_path_column": CT_PATH_COLUMN,
            "input_size": [INPUT_HEIGHT, INPUT_WIDTH, 3],
            "class_to_idx": CLASS_TO_IDX,
            "development_slices": len(development),
            "development_nodules": development[NODULE_COLUMN].nunique(),
            "development_patients": development[GROUP_COLUMN].nunique(),
            "holdout_slices": len(holdout),
            "holdout_nodules": holdout[NODULE_COLUMN].nunique(),
            "holdout_patients": holdout[GROUP_COLUMN].nunique(),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "pretrained_weights": str(WEIGHTS),
            "training_strategy": TRAINING_STRATEGY,
            "trainable_component": TRAINABLE_COMPONENT,
            "classifier_dropout": CLASSIFIER_DROPOUT,
        },
        "training": {
            "epochs_per_fold": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "CrossEntropyLoss",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "seed": SEED,
            "fresh_model_per_fold": True,
            "fresh_optimizer_per_fold": True,
        },
        "optimizer": {
            "name": "SGD",
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": NESTEROV,
        },
        "scheduler": None,
        "early_stopping": {
            "enabled": True,
            "monitor": BEST_MODEL_MONITOR,
            "mode": BEST_MODEL_MODE,
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
            "restore_best_weights": True,
        },
        "device": str(DEVICE),
    }


# ---------------------------------------------------------------------------
# Train one complete fold
# ---------------------------------------------------------------------------
def build_validation_predictions(
    fold: int,
    val_loader: DataLoader,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    positive_class_index: int,
) -> pd.DataFrame:
    """Build one table of validation predictions."""

    dataset = val_loader.dataset
    if len(dataset) != len(targets) or len(dataset) != len(probabilities):
        raise RuntimeError("Validation predictions have the wrong size.")
    predictions = binary_probabilities_to_predictions(
        probabilities,
        CLASSIFICATION_THRESHOLD,
        positive_class_index,
    )
    columns = [
        "dataset",
        "patient_id",
        "filename",
        GROUP_COLUMN,
        NODULE_COLUMN,
        "label",
        ROLE_COLUMN,
        FOLD_COLUMN,
    ]
    frame = dataset.metadata[columns].reset_index(drop=True).copy()
    frame.insert(0, "validation_fold", fold)
    frame["target"] = targets.numpy()
    frame["prediction"] = predictions.cpu().numpy()
    for class_index, class_name in enumerate(dataset.classes):
        frame[f"probability_{class_name}"] = probabilities[:, class_index].numpy()
    return frame


def synchronize_device() -> None:
    """Wait for CUDA operations before recording a duration."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_fold(fold: int) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and evaluate one independent fold model."""

    set_seed(seed=SEED, deterministic=True)
    paths = create_fold_paths(fold)
    create_training_log(paths["training_log"])
    train_loader, val_loader, train_transform, val_transform = build_fold_dataloaders(
        fold
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]

    model = build_model(num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
    )
    criterion = nn.CrossEntropyLoss()
    fold_config = build_fold_config(
        fold,
        paths,
        train_loader,
        val_loader,
        train_transform,
        val_transform,
        model,
        optimizer,
        criterion,
        positive_class_index,
    )
    save_training_config(fold_config, paths["training_config"])

    print("=" * 72)
    print(f"Cross-validation fold {fold + 1}/{NUM_FOLDS}")
    print("=" * 72)
    print(f"Training samples   : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")
    print(f"Learning rate      : {LEARNING_RATE:.3e}")
    print(f"Device             : {DEVICE}")
    print()

    best_metric = float("inf")
    best_epoch = 0
    stopped_early = False
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        mode=BEST_MODEL_MODE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=EARLY_STOPPING_VERBOSE,
    )
    training_started_at = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        synchronize_device()
        epoch_started_at = time.perf_counter()
        train_started_at = time.perf_counter()
        train_loss, train_accuracy, train_samples = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            criterion,
            positive_class_index,
        )
        synchronize_device()
        train_time = time.perf_counter() - train_started_at

        val_started_at = time.perf_counter()
        val_metrics, _, _, _ = validate_one_epoch(
            epoch,
            model,
            val_loader,
            criterion,
            num_classes,
            positive_class_index,
        )
        synchronize_device()
        val_time = time.perf_counter() - val_started_at

        current_metric = val_metrics["loss"]
        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            save_best_model(model, paths["best_model"])
            print(f"Best model updated: val_loss={current_metric:.4f}")

        stopped_early = early_stopping(current_metric, epoch)
        checkpoint_saved = False
        if SAVE_LATEST_CHECKPOINT:
            save_latest_checkpoint(
                paths["latest_checkpoint"],
                fold,
                model,
                optimizer,
                epoch,
                train_loss,
                train_accuracy,
                val_metrics,
                best_metric,
                best_epoch,
                num_classes,
                early_stopping,
            )
            checkpoint_saved = True

        epoch_time = time.perf_counter() - epoch_started_at
        elapsed_time = time.perf_counter() - training_started_at
        allocated_memory = (
            torch.cuda.memory_allocated(DEVICE) / (1024**2)
            if DEVICE.type == "cuda"
            else 0.0
        )
        reserved_memory = (
            torch.cuda.memory_reserved(DEVICE) / (1024**2)
            if DEVICE.type == "cuda"
            else 0.0
        )
        samples_per_second = train_samples / train_time if train_time > 0.0 else 0.0
        append_training_log(
            log_path=paths["training_log"],
            epoch=epoch,
            epoch_time=epoch_time,
            elapsed_time_sec=elapsed_time,
            is_best=is_best,
            early_stop_counter=early_stopping.counter,
            gpu_memory_allocated_mb=allocated_memory,
            train_time_sec=train_time,
            val_time_sec=val_time,
            scheduler_updated=False,
            patience_counter=0,
            best_metric=best_metric,
            checkpoint_saved=checkpoint_saved,
            samples_per_sec=samples_per_second,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            gpu_memory_reserved_mb=reserved_memory,
            stopped_early=stopped_early,
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
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"Train loss         : {train_loss:.4f}")
        print(f"Validation loss    : {val_metrics['loss']:.4f}")
        print(f"Validation accuracy: {val_metrics['accuracy']:.2%}")
        print(f"Validation ROC-AUC : {val_metrics['auc']:.4f}")
        print()
        if stopped_early:
            print(f"Early stopping at epoch {epoch}.")
            break

    total_training_seconds = time.perf_counter() - training_started_at
    history = pd.read_csv(paths["training_log"])
    best_state = torch.load(
        paths["best_model"],
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    (
        best_metrics,
        best_confusion_matrix,
        best_targets,
        best_probabilities,
    ) = validate_one_epoch(
        best_epoch,
        model,
        val_loader,
        criterion,
        num_classes,
        positive_class_index,
    )

    predictions = build_validation_predictions(
        fold,
        val_loader,
        best_targets,
        best_probabilities,
        positive_class_index,
    )
    predictions.to_csv(
        paths["validation_predictions"],
        index=False,
        float_format="%.10g",
    )
    plot_loss_curve(history, paths["figures_dir"])
    plot_accuracy_curve(history, paths["figures_dir"])
    plot_validation_metrics_curve(history, paths["figures_dir"])
    plot_confusion_matrix(
        best_confusion_matrix,
        train_dataset.classes,
        paths["figures_dir"],
    )
    plot_roc_curve(
        best_targets.numpy(),
        best_probabilities.numpy(),
        train_dataset.classes,
        paths["figures_dir"],
    )

    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)
    summary = {
        "fold": fold,
        "status": "completed",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_patients": train_dataset.metadata[GROUP_COLUMN].nunique(),
        "val_patients": val_dataset.metadata[GROUP_COLUMN].nunique(),
        "train_nodules": train_dataset.metadata[NODULE_COLUMN].nunique(),
        "val_nodules": val_dataset.metadata[NODULE_COLUMN].nunique(),
        "train_benign": train_counts[CLASS_TO_IDX["benign"]],
        "train_malignant": train_counts[CLASS_TO_IDX["malignant"]],
        "val_benign": val_counts[CLASS_TO_IDX["benign"]],
        "val_malignant": val_counts[CLASS_TO_IDX["malignant"]],
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "stopped_early": stopped_early,
        "best_val_loss": best_metrics["loss"],
        "best_val_accuracy": best_metrics["accuracy"],
        "best_sensitivity": best_metrics["sensitivity"],
        "best_specificity": best_metrics["specificity"],
        "best_precision": best_metrics["precision"],
        "best_f1_score": best_metrics["f1_score"],
        "best_auc": best_metrics["auc"],
        "total_training_seconds": total_training_seconds,
        "best_model_path": str(paths["best_model"]),
    }
    return summary, predictions


# ---------------------------------------------------------------------------
# Final cross-validation summary
# ---------------------------------------------------------------------------
def build_cv_summary(summary_frame: pd.DataFrame) -> dict[str, object]:
    """Calculate aggregate statistics across completed folds."""

    aggregate_metrics = {}
    for metric in SUMMARY_METRICS:
        aggregate_metrics[metric] = {
            "mean": float(summary_frame[metric].mean()),
            "std": float(summary_frame[metric].std(ddof=1)),
            "minimum": float(summary_frame[metric].min()),
            "maximum": float(summary_frame[metric].max()),
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "component": EXPERIMENT_COMPONENT,
        "experiment_type": "stratified_group_cross_validation",
        "run_id": str(RUN_ID),
        "num_folds": NUM_FOLDS,
        "completed_folds": len(summary_frame),
        "selection_unit": "best validation-loss model from each fold",
        "holdout_test_used": False,
        "aggregate_metrics": aggregate_metrics,
        "out_of_fold_predictions_path": str(OOF_PREDICTIONS_PATH),
        "interpretation": ("Use the mean and standard deviation across all folds."),
    }


def plot_cv_metrics(summary_frame: pd.DataFrame) -> None:
    """Plot important validation metrics across folds."""

    columns = {
        "best_val_accuracy": "Accuracy",
        "best_sensitivity": "Sensitivity",
        "best_specificity": "Specificity",
        "best_f1_score": "F1",
        "best_auc": "ROC-AUC",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for column, label in columns.items():
        axis.plot(
            summary_frame["fold"],
            summary_frame[column],
            marker="o",
            label=label,
        )
    axis.set_title("ResNet-50 Cross-Validation Metrics")
    axis.set_xlabel("Validation fold")
    axis.set_ylabel("Metric")
    axis.set_xticks(list(CV_FOLDS))
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(CV_FIGURES_DIR / "cv_fold_metrics.png", dpi=300)
    plt.close(figure)


def validate_oof_predictions(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Verify that every development image has one OOF prediction."""

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    if len(predictions) != len(development):
        raise RuntimeError("OOF prediction count does not match development data.")
    if predictions.duplicated(["dataset", "filename"]).any():
        raise RuntimeError("OOF predictions contain duplicate images.")

    expected = set(
        zip(
            development["dataset"],
            development["filename"],
            strict=True,
        )
    )
    observed = set(
        zip(
            predictions["dataset"],
            predictions["filename"],
            strict=True,
        )
    )
    if expected != observed:
        raise RuntimeError("OOF predictions do not cover development data.")


def main() -> None:
    """Run all folds and save predictions, figures, and summaries."""

    validate_configuration()
    metadata = validate_cv_metadata()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    CV_FIGURES_DIR.mkdir(parents=False, exist_ok=False)
    shutil.copy2(CONFIG_PATH, CONFIG_SNAPSHOT_PATH)
    save_training_config(build_cv_config(metadata), CV_CONFIG_PATH)

    development_count = metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE).sum()
    holdout_count = metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE).sum()
    print("ResNet-50 Baseline — Patient-Grouped Cross-Validation")
    print(f"Metadata            : {METADATA_PATH}")
    print(f"Development samples : {development_count}")
    print(f"Holdout samples     : {holdout_count}")
    print(f"Folds               : {list(CV_FOLDS)}")
    print(f"Epochs per fold     : {NUM_EPOCHS}")
    print(f"Batch size          : {BATCH_SIZE}")
    print("Holdout test        : not used during cross-validation")
    print()

    fold_summaries: list[dict[str, object]] = []
    oof_predictions: list[pd.DataFrame] = []
    for fold in CV_FOLDS:
        fold_summary, fold_predictions = run_fold(fold)
        fold_summaries.append(fold_summary)
        oof_predictions.append(fold_predictions)

        # Save partial results after every fold. This makes interrupted runs
        # easier to inspect without marking them as complete.
        pd.DataFrame(fold_summaries).sort_values("fold").to_csv(
            CV_SUMMARY_PATH,
            index=False,
            float_format="%.10g",
        )
        pd.concat(oof_predictions, ignore_index=True).to_csv(
            OOF_PREDICTIONS_PATH,
            index=False,
            float_format="%.10g",
        )
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(fold_summaries).sort_values("fold")
    predictions_frame = pd.concat(oof_predictions, ignore_index=True)
    predictions_frame = predictions_frame.sort_values(
        ["validation_fold", "dataset", "patient_id", "filename"],
        kind="stable",
    ).reset_index(drop=True)
    validate_oof_predictions(metadata, predictions_frame)

    summary_frame.to_csv(
        CV_SUMMARY_PATH,
        index=False,
        float_format="%.10g",
    )
    predictions_frame.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
        float_format="%.10g",
    )
    cv_summary = build_cv_summary(summary_frame)
    with CV_SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(cv_summary, file, indent=4, allow_nan=False)
        file.write("\n")
    plot_cv_metrics(summary_frame)

    print("Cross-validation complete")
    print(summary_frame[["fold", *SUMMARY_METRICS]].to_string(index=False))
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
