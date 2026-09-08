"""Run five-fold full tuning of probability-guided ResNet-50."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from ..fulltuning_resnet50 import train as base_train
from ..utils import (
    EarlyStopping,
    append_training_log,
    binary_probabilities_to_predictions,
    create_training_log,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_roc_curve,
    plot_validation_metrics_curve,
    save_best_model,
    save_training_config,
    set_seed,
)
from .dataset import create_probability_dataloader
from .model import SegmentationGuidedResNet50
from .transforms import build_train_transform, build_val_transform

# ---------------------------------------------------------------------------
# JSON configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    CLASSIFICATION_ROOT / "configs" / "segmentation_guided_cv_resnet50.json"
)

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
    """Resolve configuration paths relative to the repository root."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# Project and output
# ---------------------------------------------------------------------------
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
EXPERIMENT_ID = str(EXPERIMENT_CONFIG["id"])
EXPERIMENT_COMPONENT = str(EXPERIMENT_CONFIG["component"])
RESULT_DIR_NAME = Path(EXPERIMENT_COMPONENT).name
OUTPUT_DIR = (
    resolve_project_path(OUTPUT_CONFIG["root_directory"])
    / EXPERIMENT_ID
    / EXPERIMENT_COMPONENT
)
CONFIG_SNAPSHOT_PATH = OUTPUT_DIR / str(
    OUTPUT_CONFIG["config_snapshot_filename"]
)
CV_CONFIG_PATH = OUTPUT_DIR / "cv_config.json"
CV_SUMMARY_PATH = OUTPUT_DIR / "cv_summary.csv"
CV_SUMMARY_JSON_PATH = OUTPUT_DIR / "cv_summary.json"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "out_of_fold_predictions.csv"
CV_FIGURES_DIR = OUTPUT_DIR / "figures"


# ---------------------------------------------------------------------------
# Dataset and cross-validation
# ---------------------------------------------------------------------------
DATASET_ROOT = resolve_project_path(DATA_CONFIG["dataset_root"])
METADATA_PATH = resolve_project_path(DATA_CONFIG["metadata_path"])
CT_PATH_COLUMN = str(DATA_CONFIG["ct_path_column"])
PROBABILITY_ROOT = resolve_project_path(DATA_CONFIG["probability_root"])
INPUT_HEIGHT = int(DATA_CONFIG["input_height"])
INPUT_WIDTH = int(DATA_CONFIG["input_width"])
CLASS_TO_IDX = {
    str(name): int(index)
    for name, index in DATA_CONFIG["class_to_idx"].items()
}
IMAGENET_MEAN = tuple(float(value) for value in DATA_CONFIG["normalization_mean"])
IMAGENET_STD = tuple(float(value) for value in DATA_CONFIG["normalization_std"])

N_SPLITS = int(CV_CONFIG["num_folds"])
CV_FOLDS = tuple(range(N_SPLITS))
DEVELOPMENT_ROLE = str(CV_CONFIG["development_role"])
HOLDOUT_ROLE = str(CV_CONFIG["holdout_role"])
HOLDOUT_FOLD = int(CV_CONFIG["holdout_fold"])
GROUP_COLUMN = str(CV_CONFIG["group_column"])
NODULE_COLUMN = str(CV_CONFIG["nodule_column"])
FOLD_COLUMN = str(CV_CONFIG["fold_column"])
ROLE_COLUMN = str(CV_CONFIG["role_column"])
if ROLE_COLUMN != "cv_role" or FOLD_COLUMN != "cv_fold":
    raise ValueError(
        "The classification dataset currently requires role_column='cv_role' "
        "and fold_column='cv_fold'."
    )

TRAIN_SHUFFLE = bool(DATALOADER_CONFIG["train_shuffle"])
VAL_SHUFFLE = bool(DATALOADER_CONFIG["val_shuffle"])
TRAIN_DROP_LAST = bool(DATALOADER_CONFIG["train_drop_last"])
VAL_DROP_LAST = bool(DATALOADER_CONFIG["val_drop_last"])


# ---------------------------------------------------------------------------
# Model and training
# ---------------------------------------------------------------------------
weights_name = str(MODEL_CONFIG["pretrained_weights"])
if weights_name == "DEFAULT":
    WEIGHTS = ResNet50_Weights.DEFAULT
elif weights_name == "IMAGENET1K_V2":
    WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
elif weights_name == "NONE":
    WEIGHTS = None
else:
    raise ValueError(f"Unsupported pretrained_weights: {weights_name}")

MODEL_ARCHITECTURE = str(MODEL_CONFIG["architecture"])
if MODEL_ARCHITECTURE != SegmentationGuidedResNet50.architecture_name:
    raise ValueError(f"Unsupported model architecture: {MODEL_ARCHITECTURE}")
TRAINING_STRATEGY = str(MODEL_CONFIG["training_strategy"])
TRAINABLE_COMPONENT = str(MODEL_CONFIG["trainable_component"])
CLASSIFIER_DROPOUT = float(MODEL_CONFIG["classifier_dropout"])
ATTENTION_FUSION_STAGE = str(MODEL_CONFIG["attention_fusion_stage"])
ATTENTION_FEATURE_CHANNELS = int(MODEL_CONFIG["attention_feature_channels"])
ATTENTION_HIDDEN_CHANNELS = int(MODEL_CONFIG["attention_hidden_channels"])
ATTENTION_ALPHA_INITIAL_VALUE = float(
    MODEL_CONFIG["attention_alpha_initial_value"]
)

NUM_WORKERS = int(DATALOADER_CONFIG["num_workers"])
PERSISTENT_WORKERS = bool(DATALOADER_CONFIG["persistent_workers"])
PREFETCH_FACTOR = int(DATALOADER_CONFIG["prefetch_factor"])
PIN_MEMORY = bool(DATALOADER_CONFIG["pin_memory"]) and torch.cuda.is_available()

SEED = int(TRAINING_CONFIG["seed"])
TRANSFORM_SEED = int(TRAINING_CONFIG["transform_seed"])
LEARNING_RATE = float(TRAINING_CONFIG["learning_rate"])
BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
NUM_EPOCHS = int(TRAINING_CONFIG["num_epochs"])
CLASSIFICATION_THRESHOLD = float(TRAINING_CONFIG["classification_threshold"])
WEIGHT_DECAY_OPTM = float(OPTIMIZER_CONFIG["weight_decay"])
MOMENTUM_OPTM = float(OPTIMIZER_CONFIG["momentum"])
NESTEROV_OPTM = bool(OPTIMIZER_CONFIG["nesterov"])
if str(OPTIMIZER_CONFIG["name"]).upper() != "SGD":
    raise ValueError("Only the SGD optimizer is supported.")

BEST_MODEL_MONITOR = str(EARLY_STOPPING_CONFIG["monitor"])
BEST_MODEL_MODE = str(EARLY_STOPPING_CONFIG["mode"])
SAVE_LATEST_CHECKPOINT = bool(CHECKPOINT_CONFIG["save_latest"])
EARLY_STOPPING_PATIENCE = int(EARLY_STOPPING_CONFIG["patience"])
EARLY_STOPPING_MIN_DELTA = float(EARLY_STOPPING_CONFIG["min_delta"])
EARLY_STOPPING_VERBOSE = bool(EARLY_STOPPING_CONFIG["verbose"])
if not bool(EARLY_STOPPING_CONFIG["enabled"]):
    raise ValueError("This training pipeline requires early stopping to be enabled.")
if not bool(EARLY_STOPPING_CONFIG["restore_best_weights"]):
    raise ValueError("restore_best_weights must be true for fold evaluation.")

device_name = str(TRAINING_CONFIG["device"])
if device_name == "auto":
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
if device_name == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA is configured but unavailable.")
DEVICE = torch.device(device_name)

REQUIRED_CV_COLUMNS = {
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

SUMMARY_METRICS = (
    "best_val_loss",
    "best_val_accuracy",
    "best_sensitivity",
    "best_specificity",
    "best_precision",
    "best_f1_score",
    "best_auc",
)


def validate_cv_metadata() -> pd.DataFrame:
    """Load and validate patient and nodule isolation in the CV metadata."""

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"CV metadata not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    if metadata.empty:
        raise ValueError("CV metadata must not be empty.")

    missing_columns = REQUIRED_CV_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"CV metadata is missing columns: {sorted(missing_columns)}")
    if metadata[list(REQUIRED_CV_COLUMNS)].isna().any().any():
        raise ValueError("Required CV metadata columns contain missing values.")
    if metadata.duplicated(["dataset", "filename"]).any():
        raise ValueError("CV metadata contains duplicate dataset/filename rows.")

    metadata = metadata.copy()
    metadata[ROLE_COLUMN] = (
        metadata[ROLE_COLUMN].astype(str).str.strip().str.lower()
    )
    numeric_folds = pd.to_numeric(metadata[FOLD_COLUMN], errors="coerce")
    if (
        numeric_folds.isna().any()
        or not (numeric_folds == numeric_folds.astype(int)).all()
    ):
        raise ValueError("cv_fold values must be integers.")
    metadata[FOLD_COLUMN] = numeric_folds.astype(int)

    expected_roles = {DEVELOPMENT_ROLE, HOLDOUT_ROLE}
    observed_roles = set(metadata[ROLE_COLUMN].unique())
    if observed_roles != expected_roles:
        raise ValueError(
            f"Expected cv roles {expected_roles}, observed {observed_roles}."
        )

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE)]
    observed_folds = set(development[FOLD_COLUMN].unique())
    if observed_folds != set(CV_FOLDS):
        raise ValueError(
            f"Expected development folds {set(CV_FOLDS)}, "
            f"observed {observed_folds}."
        )
    if not holdout[FOLD_COLUMN].eq(HOLDOUT_FOLD).all():
        raise ValueError(
            f"Every holdout row must have {FOLD_COLUMN} = {HOLDOUT_FOLD}."
        )

    development_groups = set(development[GROUP_COLUMN])
    holdout_groups = set(holdout[GROUP_COLUMN])
    group_overlap = development_groups & holdout_groups
    if group_overlap:
        raise RuntimeError(
            "Patient leakage exists between development and holdout data."
        )
    if development.groupby(GROUP_COLUMN)[FOLD_COLUMN].nunique().ne(1).any():
        raise RuntimeError("A development patient appears in multiple folds.")
    if development.groupby(NODULE_COLUMN)[FOLD_COLUMN].nunique().ne(1).any():
        raise RuntimeError("A development nodule appears in multiple folds.")

    expected_labels = {label.lower() for label in CLASS_TO_IDX}
    expected_datasets = set(development["dataset"].astype(str).unique())
    for fold in CV_FOLDS:
        fold_frame = development[development[FOLD_COLUMN].eq(fold)]
        fold_labels = set(fold_frame["label"].astype(str).str.lower())
        fold_datasets = set(fold_frame["dataset"].astype(str))
        if fold_labels != expected_labels:
            raise ValueError(f"Fold {fold} does not contain every class.")
        if fold_datasets != expected_datasets:
            raise ValueError(f"Fold {fold} does not contain every dataset.")

    return metadata


def build_fold_dataloaders(
    fold: int,
) -> tuple[DataLoader, DataLoader, Any, Any]:
    """Build train/validation loaders for one held-out CV fold."""

    if fold not in CV_FOLDS:
        raise ValueError(f"fold must be one of {CV_FOLDS}.")

    train_transform = build_train_transform(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        IMAGENET_MEAN,
        IMAGENET_STD,
        TRANSFORM_SEED,
    )
    val_transform = build_val_transform(
        INPUT_HEIGHT,
        INPUT_WIDTH,
        IMAGENET_MEAN,
        IMAGENET_STD,
        TRANSFORM_SEED,
    )
    common_arguments = {
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
    train_loader = create_probability_dataloader(
        split="train",
        probability_root=PROBABILITY_ROOT,
        transform=train_transform,
        shuffle=TRAIN_SHUFFLE,
        drop_last=TRAIN_DROP_LAST,
        **common_arguments,
    )
    val_loader = create_probability_dataloader(
        split="val",
        probability_root=PROBABILITY_ROOT,
        transform=val_transform,
        shuffle=VAL_SHUFFLE,
        drop_last=VAL_DROP_LAST,
        **common_arguments,
    )
    assert_fold_isolation(train_loader, val_loader, fold)
    return train_loader, val_loader, train_transform, val_transform


def assert_fold_isolation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold: int,
) -> None:
    """Ensure one fold has no patient, nodule, or slice leakage."""

    train_metadata = train_loader.dataset.metadata
    val_metadata = val_loader.dataset.metadata
    if train_metadata.empty or val_metadata.empty:
        raise ValueError(f"Fold {fold} has an empty train or validation set.")
    if not train_metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE).all():
        raise RuntimeError("Training data must only contain development rows.")
    if not val_metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE).all():
        raise RuntimeError("Validation data must only contain development rows.")
    if not train_metadata[FOLD_COLUMN].ne(fold).all():
        raise RuntimeError(f"Fold {fold} leaked into its training partition.")
    if not val_metadata[FOLD_COLUMN].eq(fold).all():
        raise RuntimeError("Validation rows do not match the selected fold.")

    isolation_columns = (GROUP_COLUMN, NODULE_COLUMN, "filename")
    for column in isolation_columns:
        overlap = set(train_metadata[column]) & set(val_metadata[column])
        if overlap:
            raise RuntimeError(f"Fold {fold} has train/validation overlap in {column}.")
    if train_loader.dataset.class_to_idx != val_loader.dataset.class_to_idx:
        raise ValueError("Train and validation class mappings do not match.")


def create_fold_paths(fold: int) -> dict[str, Path]:
    """Create and return the output paths for one fold."""

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
    """Save the latest independently resumable state for one fold."""

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


def build_fold_config(
    fold: int,
    paths: dict[str, Path],
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_transform: Any,
    val_transform: Any,
    model: nn.Module,
    optimizer: torch.optim.SGD,
    criterion: nn.Module,
    positive_class_index: int,
) -> dict[str, object]:
    """Adapt the normal full-tuning configuration for one CV fold."""

    config = base_train.build_training_config(
        train_loader=train_loader,
        val_loader=val_loader,
        train_transform=train_transform,
        val_transform=val_transform,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        positive_class_index=positive_class_index,
    )
    config["experiment"].update(
        {
            "type": "stratified_group_5fold_cross_validation",
            "experiment_id": EXPERIMENT_ID,
            "component": EXPERIMENT_COMPONENT,
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(paths["fold_dir"]),
            "fold": fold,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
        }
    )
    config["data"].update(
        {
            "metadata_path": str(METADATA_PATH),
            "probability_root": str(PROBABILITY_ROOT),
            "probability_input_channels": 1,
            "input_channels": 4,
            "train_split": None,
            "val_split": None,
            "cv_role": DEVELOPMENT_ROLE,
            "validation_fold": fold,
            "train_filter": (
                f"{ROLE_COLUMN} == {DEVELOPMENT_ROLE} and "
                f"{FOLD_COLUMN} != {fold}"
            ),
            "validation_filter": (
                f"{ROLE_COLUMN} == {DEVELOPMENT_ROLE} and "
                f"{FOLD_COLUMN} == {fold}"
            ),
            "holdout_filter": (
                f"{ROLE_COLUMN} == {HOLDOUT_ROLE} and "
                f"{FOLD_COLUMN} == {HOLDOUT_FOLD}"
            ),
        }
    )
    config["training"].update(
        {
            "cross_validation": True,
            "num_folds": N_SPLITS,
            "seed_reset_before_each_fold": True,
        }
    )
    config["early_stopping"] = {
        "enabled": True,
        "monitor": BEST_MODEL_MONITOR,
        "mode": BEST_MODEL_MODE,
        "patience": EARLY_STOPPING_PATIENCE,
        "min_delta": EARLY_STOPPING_MIN_DELTA,
        "restore_best_weights": True,
    }
    config["checkpoint"].update(
        {
            "best_model_path": str(paths["best_model"]),
            "checkpoint_path": str(paths["latest_checkpoint"]),
        }
    )
    config["cross_validation"] = {
        "method": "StratifiedGroupKFold",
        "fold": fold,
        "all_folds": list(CV_FOLDS),
        "group_column": GROUP_COLUMN,
        "nodule_column": NODULE_COLUMN,
        "role_column": ROLE_COLUMN,
        "fold_column": FOLD_COLUMN,
        "holdout_used_during_training": False,
    }
    config["model"].update(
        {
            "architecture": MODEL_ARCHITECTURE,
            "attention": {
                "type": "residual_multiplicative",
                "fusion_stage": ATTENTION_FUSION_STAGE,
                "feature_channels": ATTENTION_FEATURE_CHANNELS,
                "attention_hidden_channels": ATTENTION_HIDDEN_CHANNELS,
                "equation": "F_guided = F * (1 + alpha * A)",
                "alpha_initial_value": ATTENTION_ALPHA_INITIAL_VALUE,
            },
        }
    )
    return config


def build_cv_config(metadata: pd.DataFrame) -> dict[str, object]:
    """Build the root-level configuration shared by all five folds."""

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE)]
    fold_distribution = {}
    for fold in CV_FOLDS:
        fold_frame = development[development[FOLD_COLUMN].eq(fold)]
        fold_distribution[str(fold)] = {
            "slices": len(fold_frame),
            "nodules": fold_frame[NODULE_COLUMN].nunique(),
            "patients": fold_frame[GROUP_COLUMN].nunique(),
            "labels": fold_frame["label"].value_counts().to_dict(),
        }

    return {
        "experiment": {
            "type": "stratified_group_5fold_cross_validation",
            "experiment_id": EXPERIMENT_ID,
            "component": EXPERIMENT_COMPONENT,
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "output_directory": str(OUTPUT_DIR),
        },
        "cross_validation": {
            "num_folds": N_SPLITS,
            "folds": list(CV_FOLDS),
            "random_seed": SEED,
            "metadata_path": str(METADATA_PATH),
            "group_column": GROUP_COLUMN,
            "nodule_column": NODULE_COLUMN,
            "fold_column": FOLD_COLUMN,
            "role_column": ROLE_COLUMN,
            "train_rule": (
                f"{DEVELOPMENT_ROLE} rows whose {FOLD_COLUMN} != validation fold"
            ),
            "validation_rule": (
                f"{DEVELOPMENT_ROLE} rows whose {FOLD_COLUMN} == validation fold"
            ),
            "holdout_rule": (
                f"{HOLDOUT_ROLE} rows with {FOLD_COLUMN} == {HOLDOUT_FOLD}"
            ),
            "holdout_used_during_cv": False,
            "fold_distribution": fold_distribution,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "probability_root": str(PROBABILITY_ROOT),
            "ct_path_column": CT_PATH_COLUMN,
            "input_size": [INPUT_HEIGHT, INPUT_WIDTH, 4],
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
            "attention_type": "residual_multiplicative",
            "attention_fusion_stage": ATTENTION_FUSION_STAGE,
            "attention_feature_channels": ATTENTION_FEATURE_CHANNELS,
            "attention_hidden_channels": ATTENTION_HIDDEN_CHANNELS,
            "attention_alpha_initial_value": ATTENTION_ALPHA_INITIAL_VALUE,
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
            "momentum": MOMENTUM_OPTM,
            "weight_decay": WEIGHT_DECAY_OPTM,
            "nesterov": NESTEROV_OPTM,
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


def build_validation_predictions(
    fold: int,
    val_loader: DataLoader,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    positive_class_index: int,
) -> pd.DataFrame:
    """Build ordered out-of-fold predictions for one validation fold."""

    dataset = val_loader.dataset
    if len(dataset) != len(targets) or len(dataset) != len(probabilities):
        raise RuntimeError("Validation predictions do not match dataset size.")
    predictions = binary_probabilities_to_predictions(
        probabilities=probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
        positive_class_index=positive_class_index,
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


def _synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_fold(fold: int) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and evaluate one independent held-out validation fold."""

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

    model = SegmentationGuidedResNet50(
        num_classes=num_classes,
        dropout=CLASSIFIER_DROPOUT,
        weights=WEIGHTS,
        attention_hidden_channels=ATTENTION_HIDDEN_CHANNELS,
        attention_alpha_initial_value=ATTENTION_ALPHA_INITIAL_VALUE,
    ).to(DEVICE)
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM_OPTM,
        weight_decay=WEIGHT_DECAY_OPTM,
        nesterov=NESTEROV_OPTM,
    )
    criterion = nn.CrossEntropyLoss()
    base_train.assert_full_model_trainable(model)

    fold_config = build_fold_config(
        fold=fold,
        paths=paths,
        train_loader=train_loader,
        val_loader=val_loader,
        train_transform=train_transform,
        val_transform=val_transform,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        positive_class_index=positive_class_index,
    )
    save_training_config(fold_config, paths["training_config"])

    print("=" * 76)
    print(f"Cross-Validation Fold {fold}/{N_SPLITS - 1}")
    print("=" * 76)
    print(f"Train samples      : {len(train_dataset)}")
    print(f"Validation samples : {len(val_dataset)}")
    print(f"Train patients     : {train_dataset.metadata[GROUP_COLUMN].nunique()}")
    print(f"Validation patients: {val_dataset.metadata[GROUP_COLUMN].nunique()}")
    print(f"Learning rate      : {LEARNING_RATE:.3e} (constant)")
    print(f"Optimizer          : SGD (momentum={MOMENTUM_OPTM:.1f})")
    print(f"Device             : {DEVICE}")
    print()

    best_metric = float("inf")
    best_epoch = 0
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        mode=BEST_MODEL_MODE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        verbose=EARLY_STOPPING_VERBOSE,
    )
    training_started_at = time.perf_counter()
    for epoch in range(NUM_EPOCHS):
        _synchronize_device()
        epoch_started_at = time.perf_counter()
        train_phase_started_at = time.perf_counter()
        train_loss, train_accuracy, train_total_samples = base_train.train_one_epoch(
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
        val_metrics, _, _, _ = base_train.validate_one_epoch(
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
            best_epoch = epoch + 1
            save_best_model(model=model, save_path=paths["best_model"])
            print(
                f"Fold {fold} best model updated " f"(val_loss: {current_metric:.4f})"
            )

        stopped_early = early_stopping(
            metric=current_metric,
            epoch=epoch + 1,
        )

        checkpoint_saved = False
        if SAVE_LATEST_CHECKPOINT:
            save_latest_checkpoint(
                path=paths["latest_checkpoint"],
                fold=fold,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_metrics=val_metrics,
                best_metric=best_metric,
                best_epoch=best_epoch,
                num_classes=num_classes,
                early_stopping=early_stopping,
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
            train_total_samples / train_time_sec if train_time_sec > 0.0 else 0.0
        )
        append_training_log(
            log_path=paths["training_log"],
            epoch=epoch + 1,
            epoch_time=epoch_time,
            elapsed_time_sec=elapsed_time_sec,
            is_best=is_best,
            early_stop_counter=early_stopping.counter,
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

        print(f"Fold {fold} Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Training Loss       : {train_loss:.4f}")
        print(f"Validation Loss     : {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy : {val_metrics['accuracy']:.2%}")
        print(f"Validation ROC-AUC  : {val_metrics['auc']:.4f}")
        print()

        if stopped_early:
            print(
                f"Fold {fold} stopped early at epoch {epoch + 1}: "
                f"val_loss did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    total_training_seconds = time.perf_counter() - training_started_at
    history = pd.read_csv(paths["training_log"])
    best_model_state = torch.load(
        paths["best_model"],
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_model_state)
    (
        best_val_metrics,
        best_val_confusion_matrix,
        best_val_targets,
        best_val_probabilities,
    ) = base_train.validate_one_epoch(
        epoch=best_epoch - 1,
        num_epochs=NUM_EPOCHS,
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=num_classes,
        classification_threshold=CLASSIFICATION_THRESHOLD,
        positive_class_index=positive_class_index,
    )

    validation_predictions = build_validation_predictions(
        fold=fold,
        val_loader=val_loader,
        targets=best_val_targets,
        probabilities=best_val_probabilities,
        positive_class_index=positive_class_index,
    )
    validation_predictions.to_csv(
        paths["validation_predictions"],
        index=False,
        float_format="%.10g",
    )

    plot_loss_curve(history, paths["figures_dir"])
    plot_accuracy_curve(history, paths["figures_dir"])
    plot_validation_metrics_curve(history, paths["figures_dir"])
    plot_confusion_matrix(
        confusion_matrix=best_val_confusion_matrix,
        class_names=train_dataset.classes,
        output_dir=paths["figures_dir"],
    )
    plot_roc_curve(
        targets=best_val_targets.numpy(),
        probabilities=best_val_probabilities.numpy(),
        class_names=train_dataset.classes,
        output_dir=paths["figures_dir"],
    )

    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)
    summary: dict[str, object] = {
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
        "epochs_completed": epoch + 1,
        "stopped_early": stopped_early,
        "best_val_loss": best_val_metrics["loss"],
        "best_val_accuracy": best_val_metrics["accuracy"],
        "best_sensitivity": best_val_metrics["sensitivity"],
        "best_specificity": best_val_metrics["specificity"],
        "best_precision": best_val_metrics["precision"],
        "best_f1_score": best_val_metrics["f1_score"],
        "best_auc": best_val_metrics["auc"],
        "total_training_seconds": total_training_seconds,
        "best_model_path": str(paths["best_model"]),
    }
    print(
        f"Fold {fold} complete: best epoch={best_epoch}, "
        f"val_loss={best_val_metrics['loss']:.4f}, "
        f"ROC-AUC={best_val_metrics['auc']:.4f}"
    )
    return summary, validation_predictions


def build_cv_summary(summary_frame: pd.DataFrame) -> dict[str, object]:
    """Aggregate fold validation metrics as mean and sample std."""

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
        "experiment_type": "stratified_group_5fold_cross_validation",
        "run_id": str(RUN_ID),
        "num_folds": N_SPLITS,
        "completed_folds": len(summary_frame),
        "selection_unit": "best validation-loss model from each fold",
        "holdout_test_used": False,
        "aggregate_metrics": aggregate_metrics,
        "out_of_fold_predictions_path": str(OOF_PREDICTIONS_PATH),
        "interpretation": (
            "Use mean and standard deviation across all folds; do not select "
            "one fold model as the final model based only on its CV score."
        ),
    }


def plot_cv_metrics(summary_frame: pd.DataFrame) -> None:
    """Plot core best-model validation metrics across folds."""

    metric_columns = {
        "best_val_accuracy": "Accuracy",
        "best_sensitivity": "Sensitivity",
        "best_specificity": "Specificity",
        "best_f1_score": "F1",
        "best_auc": "ROC-AUC",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for column, label in metric_columns.items():
        axis.plot(
            summary_frame["fold"],
            summary_frame[column],
            marker="o",
            label=label,
        )
    axis.set_title("Segmentation-Guided ResNet-50 Five-Fold Metrics")
    axis.set_xlabel("Validation Fold")
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
    """Ensure every development slice has exactly one OOF prediction."""

    development = metadata[metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE)]
    if len(predictions) != len(development):
        raise RuntimeError(
            "Out-of-fold prediction count does not equal development rows."
        )
    if predictions.duplicated(["dataset", "filename"]).any():
        raise RuntimeError("Out-of-fold predictions contain duplicate slices.")
    expected_keys = set(
        zip(development["dataset"], development["filename"], strict=True)
    )
    observed_keys = set(
        zip(predictions["dataset"], predictions["filename"], strict=True)
    )
    if observed_keys != expected_keys:
        raise RuntimeError("Out-of-fold predictions do not cover development.")


def main() -> None:
    """Run five independent full-fine-tuning cross-validation folds."""

    metadata = validate_cv_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    shutil.copy2(CONFIG_PATH, CONFIG_SNAPSHOT_PATH)
    CV_FIGURES_DIR.mkdir(parents=False, exist_ok=False)
    save_training_config(build_cv_config(metadata), CV_CONFIG_PATH)

    print("Segmentation-Guided ResNet-50 — Stratified Group 5-Fold CV")
    print(f"Metadata            : {METADATA_PATH}")
    print(
        f"Development samples : "
        f"{metadata[ROLE_COLUMN].eq(DEVELOPMENT_ROLE).sum()}"
    )
    print(f"Holdout samples     : {metadata[ROLE_COLUMN].eq(HOLDOUT_ROLE).sum()}")
    print(f"Folds               : {list(CV_FOLDS)}")
    print(f"Epochs per fold     : {NUM_EPOCHS}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Optimizer           : SGD (momentum={MOMENTUM_OPTM:.1f})")
    print("Holdout test        : not used during cross-validation")
    print()

    fold_summaries: list[dict[str, object]] = []
    oof_predictions: list[pd.DataFrame] = []
    for fold in CV_FOLDS:
        summary, fold_predictions = run_fold(fold)
        fold_summaries.append(summary)
        oof_predictions.append(fold_predictions)
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
    summary_frame.to_csv(CV_SUMMARY_PATH, index=False, float_format="%.10g")

    predictions_frame = pd.concat(oof_predictions, ignore_index=True)
    predictions_frame = predictions_frame.sort_values(
        ["validation_fold", "dataset", "patient_id", "filename"],
        kind="stable",
    ).reset_index(drop=True)
    validate_oof_predictions(metadata, predictions_frame)
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

    print()
    print("Cross-validation complete")
    print(summary_frame[["fold", *SUMMARY_METRICS]].to_string(index=False))
    print()
    print("Aggregate metrics (mean ± std)")
    for metric, values in cv_summary["aggregate_metrics"].items():
        print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
