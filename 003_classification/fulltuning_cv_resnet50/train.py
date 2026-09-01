"""Run five-fold patient-grouped full fine-tuning of ResNet-50."""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..fulltuning_resnet50 import train as base_train
from ..utils import (
    append_training_log,
    binary_probabilities_to_predictions,
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
)


# ---------------------------------------------------------------------------
# Project and output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = (
    f"cv_result_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "fulltuning_cv_resnet50"
    / RESULT_DIR_NAME
)
CV_CONFIG_PATH = OUTPUT_DIR / "cv_config.json"
CV_SUMMARY_PATH = OUTPUT_DIR / "cv_summary.csv"
CV_SUMMARY_JSON_PATH = OUTPUT_DIR / "cv_summary.json"
OOF_PREDICTIONS_PATH = OUTPUT_DIR / "out_of_fold_predictions.csv"
CV_FIGURES_DIR = OUTPUT_DIR / "figures"


# ---------------------------------------------------------------------------
# Dataset and cross-validation
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
METADATA_PATH = DATASET_ROOT / "004_classification_cv_5fold_seed42.csv"
CT_PATH_COLUMN = base_train.CT_PATH_COLUMN
INPUT_HEIGHT = base_train.INPUT_HEIGHT
INPUT_WIDTH = base_train.INPUT_WIDTH
CLASS_TO_IDX = base_train.CLASS_TO_IDX
IMAGENET_MEAN = base_train.IMAGENET_MEAN
IMAGENET_STD = base_train.IMAGENET_STD

N_SPLITS = 5
CV_FOLDS = tuple(range(N_SPLITS))
DEVELOPMENT_ROLE = "development"
HOLDOUT_ROLE = "holdout_test"
HOLDOUT_FOLD = -1

TRAIN_SHUFFLE = base_train.TRAIN_SHUFFLE
VAL_SHUFFLE = base_train.VAL_SHUFFLE
TRAIN_DROP_LAST = base_train.TRAIN_DROP_LAST
VAL_DROP_LAST = base_train.VAL_DROP_LAST


# ---------------------------------------------------------------------------
# Model and training: intentionally equal to fulltuning_resnet50/train.py
# ---------------------------------------------------------------------------
WEIGHTS = base_train.WEIGHTS
MODEL_ARCHITECTURE = base_train.MODEL_ARCHITECTURE
TRAINING_STRATEGY = base_train.TRAINING_STRATEGY
TRAINABLE_COMPONENT = base_train.TRAINABLE_COMPONENT
CLASSIFIER_DROPOUT = base_train.CLASSIFIER_DROPOUT
CLASSIFICATION_THRESHOLD = base_train.CLASSIFICATION_THRESHOLD

NUM_WORKERS = base_train.NUM_WORKERS
PERSISTENT_WORKERS = base_train.PERSISTENT_WORKERS
PREFETCH_FACTOR = base_train.PREFETCH_FACTOR
PIN_MEMORY = base_train.PIN_MEMORY

SEED = base_train.SEED
TRANSFORM_SEED = base_train.TRANSFORM_SEED
LEARNING_RATE = base_train.LEARNING_RATE
BATCH_SIZE = base_train.BATCH_SIZE
NUM_EPOCHS = base_train.NUM_EPOCHS
WEIGHT_DECAY_OPTM = base_train.WEIGHT_DECAY_OPTM
MOMENTUM_OPTM = base_train.MOMENTUM_OPTM
NESTEROV_OPTM = base_train.NESTEROV_OPTM

BEST_MODEL_MONITOR = base_train.BEST_MODEL_MONITOR
BEST_MODEL_MODE = base_train.BEST_MODEL_MODE
SAVE_LATEST_CHECKPOINT = base_train.SAVE_LATEST_CHECKPOINT
DEVICE = base_train.DEVICE

REQUIRED_CV_COLUMNS = {
    "dataset",
    "patient_id",
    "filename",
    CT_PATH_COLUMN,
    "label",
    "split",
    "cv_group_id",
    "cv_nodule_id",
    "cv_role",
    "cv_fold",
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
        raise ValueError(
            f"CV metadata is missing columns: {sorted(missing_columns)}"
        )
    if metadata[list(REQUIRED_CV_COLUMNS)].isna().any().any():
        raise ValueError("Required CV metadata columns contain missing values.")
    if metadata.duplicated(["dataset", "filename"]).any():
        raise ValueError("CV metadata contains duplicate dataset/filename rows.")

    metadata = metadata.copy()
    metadata["cv_role"] = (
        metadata["cv_role"].astype(str).str.strip().str.lower()
    )
    numeric_folds = pd.to_numeric(metadata["cv_fold"], errors="coerce")
    if numeric_folds.isna().any() or not (
        numeric_folds == numeric_folds.astype(int)
    ).all():
        raise ValueError("cv_fold values must be integers.")
    metadata["cv_fold"] = numeric_folds.astype(int)

    expected_roles = {DEVELOPMENT_ROLE, HOLDOUT_ROLE}
    observed_roles = set(metadata["cv_role"].unique())
    if observed_roles != expected_roles:
        raise ValueError(
            f"Expected cv roles {expected_roles}, observed {observed_roles}."
        )

    development = metadata[metadata["cv_role"].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata["cv_role"].eq(HOLDOUT_ROLE)]
    observed_folds = set(development["cv_fold"].unique())
    if observed_folds != set(CV_FOLDS):
        raise ValueError(
            f"Expected development folds {set(CV_FOLDS)}, "
            f"observed {observed_folds}."
        )
    if not holdout["cv_fold"].eq(HOLDOUT_FOLD).all():
        raise ValueError("Every holdout row must have cv_fold = -1.")

    development_groups = set(development["cv_group_id"])
    holdout_groups = set(holdout["cv_group_id"])
    group_overlap = development_groups & holdout_groups
    if group_overlap:
        raise RuntimeError(
            "Patient leakage exists between development and holdout data."
        )
    if development.groupby("cv_group_id")["cv_fold"].nunique().ne(1).any():
        raise RuntimeError("A development patient appears in multiple folds.")
    if development.groupby("cv_nodule_id")["cv_fold"].nunique().ne(1).any():
        raise RuntimeError("A development nodule appears in multiple folds.")

    expected_labels = {label.lower() for label in CLASS_TO_IDX}
    expected_datasets = set(development["dataset"].astype(str).unique())
    for fold in CV_FOLDS:
        fold_frame = development[development["cv_fold"].eq(fold)]
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

    train_transform = base_train.build_train_transform()
    val_transform = base_train.build_val_transform()
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
    train_loader = create_dataloader(
        split="train",
        transform=train_transform,
        shuffle=TRAIN_SHUFFLE,
        drop_last=TRAIN_DROP_LAST,
        **common_arguments,
    )
    val_loader = create_dataloader(
        split="val",
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
    if not train_metadata["cv_role"].eq(DEVELOPMENT_ROLE).all():
        raise RuntimeError("Training data must only contain development rows.")
    if not val_metadata["cv_role"].eq(DEVELOPMENT_ROLE).all():
        raise RuntimeError("Validation data must only contain development rows.")
    if not train_metadata["cv_fold"].ne(fold).all():
        raise RuntimeError(f"Fold {fold} leaked into its training partition.")
    if not val_metadata["cv_fold"].eq(fold).all():
        raise RuntimeError("Validation rows do not match the selected fold.")

    isolation_columns = ("cv_group_id", "cv_nodule_id", "filename")
    for column in isolation_columns:
        overlap = set(train_metadata[column]) & set(val_metadata[column])
        if overlap:
            raise RuntimeError(
                f"Fold {fold} has train/validation overlap in {column}."
            )
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
            "early_stopping_state_dict": None,
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
            "train_split": None,
            "val_split": None,
            "cv_role": DEVELOPMENT_ROLE,
            "validation_fold": fold,
            "train_filter": f"cv_role == development and cv_fold != {fold}",
            "validation_filter": (
                f"cv_role == development and cv_fold == {fold}"
            ),
            "holdout_filter": "cv_role == holdout_test and cv_fold == -1",
        }
    )
    config["training"].update(
        {
            "cross_validation": True,
            "num_folds": N_SPLITS,
            "seed_reset_before_each_fold": True,
        }
    )
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
        "group_column": "cv_group_id",
        "nodule_column": "cv_nodule_id",
        "role_column": "cv_role",
        "fold_column": "cv_fold",
        "holdout_used_during_training": False,
    }
    return config


def build_cv_config(metadata: pd.DataFrame) -> dict[str, object]:
    """Build the root-level configuration shared by all five folds."""

    development = metadata[metadata["cv_role"].eq(DEVELOPMENT_ROLE)]
    holdout = metadata[metadata["cv_role"].eq(HOLDOUT_ROLE)]
    fold_distribution = {}
    for fold in CV_FOLDS:
        fold_frame = development[development["cv_fold"].eq(fold)]
        fold_distribution[str(fold)] = {
            "slices": len(fold_frame),
            "nodules": fold_frame["cv_nodule_id"].nunique(),
            "patients": fold_frame["cv_group_id"].nunique(),
            "labels": fold_frame["label"].value_counts().to_dict(),
        }

    return {
        "experiment": {
            "type": "stratified_group_5fold_cross_validation",
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
            "group_column": "cv_group_id",
            "nodule_column": "cv_nodule_id",
            "fold_column": "cv_fold",
            "role_column": "cv_role",
            "train_rule": "development rows whose cv_fold != validation fold",
            "validation_rule": (
                "development rows whose cv_fold == validation fold"
            ),
            "holdout_rule": "holdout_test rows with cv_fold == -1",
            "holdout_used_during_cv": False,
            "fold_distribution": fold_distribution,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "ct_path_column": CT_PATH_COLUMN,
            "input_size": [INPUT_HEIGHT, INPUT_WIDTH, 3],
            "class_to_idx": CLASS_TO_IDX,
            "development_slices": len(development),
            "development_nodules": development["cv_nodule_id"].nunique(),
            "development_patients": development["cv_group_id"].nunique(),
            "holdout_slices": len(holdout),
            "holdout_nodules": holdout["cv_nodule_id"].nunique(),
            "holdout_patients": holdout["cv_group_id"].nunique(),
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
            "momentum": MOMENTUM_OPTM,
            "weight_decay": WEIGHT_DECAY_OPTM,
            "nesterov": NESTEROV_OPTM,
        },
        "scheduler": None,
        "early_stopping": None,
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
        "cv_group_id",
        "cv_nodule_id",
        "label",
        "cv_role",
        "cv_fold",
    ]
    frame = dataset.metadata[columns].reset_index(drop=True).copy()
    frame.insert(0, "validation_fold", fold)
    frame["target"] = targets.numpy()
    frame["prediction"] = predictions.cpu().numpy()
    for class_index, class_name in enumerate(dataset.classes):
        frame[f"probability_{class_name}"] = probabilities[
            :, class_index
        ].numpy()
    return frame


def _synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_fold(fold: int) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and evaluate one independent held-out validation fold."""

    set_seed(seed=SEED, deterministic=True)
    paths = create_fold_paths(fold)
    create_training_log(paths["training_log"])
    train_loader, val_loader, train_transform, val_transform = (
        build_fold_dataloaders(fold)
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]

    model = base_train.build_model(num_classes)
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
    print(f"Train patients     : {train_dataset.metadata['cv_group_id'].nunique()}")
    print(f"Validation patients: {val_dataset.metadata['cv_group_id'].nunique()}")
    print(f"Learning rate      : {LEARNING_RATE:.3e} (constant)")
    print(f"Optimizer          : SGD (momentum={MOMENTUM_OPTM:.1f})")
    print(f"Device             : {DEVICE}")
    print()

    best_metric = float("inf")
    best_epoch = 0
    training_started_at = time.perf_counter()
    for epoch in range(NUM_EPOCHS):
        _synchronize_device()
        epoch_started_at = time.perf_counter()
        train_phase_started_at = time.perf_counter()
        train_loss, train_accuracy, train_total_samples = (
            base_train.train_one_epoch(
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
                f"Fold {fold} best model updated "
                f"(val_loss: {current_metric:.4f})"
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
            log_path=paths["training_log"],
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

        print(f"Fold {fold} Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Training Loss       : {train_loss:.4f}")
        print(f"Validation Loss     : {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy : {val_metrics['accuracy']:.2%}")
        print(f"Validation ROC-AUC  : {val_metrics['auc']:.4f}")
        print()

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
        "train_patients": train_dataset.metadata["cv_group_id"].nunique(),
        "val_patients": val_dataset.metadata["cv_group_id"].nunique(),
        "train_nodules": train_dataset.metadata["cv_nodule_id"].nunique(),
        "val_nodules": val_dataset.metadata["cv_nodule_id"].nunique(),
        "train_benign": train_counts[CLASS_TO_IDX["benign"]],
        "train_malignant": train_counts[CLASS_TO_IDX["malignant"]],
        "val_benign": val_counts[CLASS_TO_IDX["benign"]],
        "val_malignant": val_counts[CLASS_TO_IDX["malignant"]],
        "best_epoch": best_epoch,
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
    axis.set_title("ResNet-50 Five-Fold Validation Metrics")
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

    development = metadata[metadata["cv_role"].eq(DEVELOPMENT_ROLE)]
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
    CV_FIGURES_DIR.mkdir(parents=False, exist_ok=False)
    save_training_config(build_cv_config(metadata), CV_CONFIG_PATH)

    print("Full Fine-Tuning ResNet-50 — Stratified Group 5-Fold CV")
    print(f"Metadata            : {METADATA_PATH}")
    print(f"Development samples : {(metadata.cv_role == DEVELOPMENT_ROLE).sum()}")
    print(f"Holdout samples     : {(metadata.cv_role == HOLDOUT_ROLE).sum()}")
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
