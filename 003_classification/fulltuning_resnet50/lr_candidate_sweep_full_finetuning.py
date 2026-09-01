"""Compare fixed SGD learning rates while fine-tuning ResNet-50."""

from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..pretrained_resnet50 import lr_candidate_sweep as shared_sweep
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
    "lr_candidate_sweep_full_finetuning_"
    f"{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "full_tuning_resnet50"
    / "lr_candidate_sweep"
    / RESULT_DIR_NAME
)
CANDIDATES_DIR = OUTPUT_DIR / "candidates"
FIGURES_DIR = OUTPUT_DIR / "figures"
CANDIDATE_SUMMARY_PATH = OUTPUT_DIR / "candidate_summary.csv"
SWEEP_CONFIG_PATH = OUTPUT_DIR / "sweep_config.json"
SWEEP_SUMMARY_PATH = OUTPUT_DIR / "sweep_summary.json"


# ---------------------------------------------------------------------------
# Dataset and preprocessing: equal to the completed full-fine-tuning LR test
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
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
# Model and optimizer: every ResNet-50 parameter is trainable
# ---------------------------------------------------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL_ARCHITECTURE = "ResNet50"
TRAINING_STRATEGY = "full_fine_tuning"
TRAINABLE_COMPONENT = "entire_model"
CLASSIFIER_DROPOUT = 0.3
MOMENTUM_OPTM = 0.9
NESTEROV_OPTM = False


# ---------------------------------------------------------------------------
# Fixed-LR candidate sweep
# ---------------------------------------------------------------------------
# This interval is configurable. Update it from the SGD LR Range Test result
# before using this sweep to select the final learning-rate candidate.
LR_CANDIDATE_START = 0.001
LR_CANDIDATE_END = 0.010
LR_CANDIDATE_STEP = 0.001
# LR_CANDIDATE_START = 1.5e-3
# LR_CANDIDATE_END = 5e-3
# LR_CANDIDATE_STEP = 2.5e-4
SHORT_TRAINING_EPOCHS = 15
STABILITY_WINDOW = 5

SEED = 42
TRANSFORM_SEED = SEED
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_LOG_COLUMNS = shared_sweep.TRAINING_LOG_COLUMNS
CANDIDATE_SUMMARY_COLUMNS = shared_sweep.CANDIDATE_SUMMARY_COLUMNS
NonFiniteLossError = shared_sweep.NonFiniteLossError


def validate_configuration() -> None:
    """Validate the full-fine-tuning sweep and shared data behavior."""

    if not math.isfinite(MOMENTUM_OPTM) or MOMENTUM_OPTM < 0.0:
        raise ValueError("MOMENTUM_OPTM must be finite and non-negative.")
    if not isinstance(NESTEROV_OPTM, bool):
        raise TypeError("NESTEROV_OPTM must be a boolean.")
    if NESTEROV_OPTM and MOMENTUM_OPTM <= 0.0:
        raise ValueError("Nesterov SGD requires positive momentum.")
    if not all(
        math.isfinite(value)
        for value in (
            LR_CANDIDATE_START,
            LR_CANDIDATE_END,
            LR_CANDIDATE_STEP,
        )
    ):
        raise ValueError("Candidate start, end, and step must be finite.")
    if LR_CANDIDATE_START <= 0.0:
        raise ValueError("LR_CANDIDATE_START must be greater than zero.")
    if LR_CANDIDATE_END < LR_CANDIDATE_START:
        raise ValueError("LR_CANDIDATE_END must not be less than start.")
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

    shared_values = {
        "DATASET_ROOT": DATASET_ROOT,
        "METADATA_PATH": METADATA_PATH,
        "CT_PATH_COLUMN": CT_PATH_COLUMN,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "VAL_SPLIT": VAL_SPLIT,
        "INPUT_HEIGHT": INPUT_HEIGHT,
        "INPUT_WIDTH": INPUT_WIDTH,
        "CLASS_TO_IDX": CLASS_TO_IDX,
        "IMAGENET_MEAN": IMAGENET_MEAN,
        "IMAGENET_STD": IMAGENET_STD,
        "CLASSIFICATION_THRESHOLD": CLASSIFICATION_THRESHOLD,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAIN_SHUFFLE": TRAIN_SHUFFLE,
        "VAL_SHUFFLE": VAL_SHUFFLE,
        "TRAIN_DROP_LAST": TRAIN_DROP_LAST,
        "VAL_DROP_LAST": VAL_DROP_LAST,
        "NUM_WORKERS": NUM_WORKERS,
        "PERSISTENT_WORKERS": PERSISTENT_WORKERS,
        "PREFETCH_FACTOR": PREFETCH_FACTOR,
        "PIN_MEMORY": PIN_MEMORY,
        "SHORT_TRAINING_EPOCHS": SHORT_TRAINING_EPOCHS,
        "STABILITY_WINDOW": STABILITY_WINDOW,
        "SEED": SEED,
        "TRANSFORM_SEED": TRANSFORM_SEED,
        "DEVICE": DEVICE,
    }
    mismatches = [
        name
        for name, expected in shared_values.items()
        if getattr(shared_sweep, name) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Full fine-tuning sweep expects the same dataset and loop "
            "configuration as lr_candidate_sweep.py. Mismatched values: "
            f"{mismatches}"
        )


def generate_lr_candidates() -> list[float]:
    """Generate the inclusive dense full-fine-tuning LR interval."""

    return shared_sweep.generate_lr_candidates(
        start=LR_CANDIDATE_START,
        end=LR_CANDIDATE_END,
        step=LR_CANDIDATE_STEP,
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader, Any, Any]:
    """Reuse identical training and validation data construction."""

    return shared_sweep.build_dataloaders()


def build_model() -> nn.Module:
    """Build pretrained ResNet-50 with every parameter trainable."""

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
        parameter.requires_grad = True

    assert_full_model_trainable(model)
    return model.to(DEVICE)


def assert_full_model_trainable(model: nn.Module) -> None:
    """Verify that full fine-tuning has no frozen model parameter."""

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
            "Full fine-tuning requires all parameters to be trainable; "
            f"found frozen parameters: {frozen_names}"
        )


def set_full_fine_tuning_mode(model: nn.Module) -> None:
    """Train the backbone, BatchNorm layers, and classifier together."""

    model.train()
    assert_full_model_trainable(model)
    batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norm_layers:
        raise RuntimeError("ResNet-50 should contain BatchNorm layers.")
    if any(not module.training for module in batch_norm_layers):
        raise RuntimeError(
            "Every BatchNorm layer must be in train mode during full "
            "fine-tuning."
        )
    if not model.fc.training:
        raise RuntimeError("Classifier Dropout must be active during training.")


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.SGD:
    """Build fixed-LR SGD with momentum over the entire model."""

    all_parameters = list(model.parameters())
    trainable_parameters = [
        parameter
        for parameter in all_parameters
        if parameter.requires_grad
    ]
    if len(trainable_parameters) != len(all_parameters):
        raise RuntimeError("Optimizer received a partially frozen model.")

    optimizer = torch.optim.SGD(
        params=trainable_parameters,
        lr=learning_rate,
        momentum=MOMENTUM_OPTM,
        nesterov=NESTEROV_OPTM,
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_ids = {id(parameter) for parameter in all_parameters}
    if optimized_ids != model_ids:
        raise RuntimeError("SGD does not cover every model parameter.")
    return optimizer


def _synchronize_device() -> None:
    """Synchronize CUDA at timing boundaries when applicable."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    positive_class_index: int,
) -> tuple[float, float]:
    """Train the complete model for one epoch at a constant LR."""

    set_full_fine_tuning_mode(model)
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
        current_batch_size = labels.size(0)
        running_loss += loss.item() * current_batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += current_batch_size
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


def run_candidate_experiment(
    learning_rate: float,
    candidate_index: int,
    num_candidates: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run one independent full-fine-tuning candidate experiment."""

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
    fixed_lr = float(optimizer.param_groups[0]["lr"])
    if fixed_lr != learning_rate:
        raise RuntimeError("Optimizer learning rate differs from candidate LR.")

    print("=" * 72)
    print(f"Candidate {candidate_index}/{num_candidates}")
    print(f"Learning Rate: {learning_rate:.6f} ({learning_rate:.3e})")
    print("Training strategy: full fine-tuning")
    print("=" * 72)

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
        val_metrics = shared_sweep.validate_one_epoch(
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
        if current_lr != fixed_lr:
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


def save_candidate_summary(summary_frame: pd.DataFrame) -> None:
    """Persist candidate results sorted by learning rate."""

    shared_sweep.assign_ranks(summary_frame).to_csv(
        CANDIDATE_SUMMARY_PATH,
        index=False,
        float_format="%.10g",
    )


def build_sweep_config(
    candidates: list[float],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build an SGD-specific full-fine-tuning configuration snapshot."""

    train_loader = context["train_loader"]
    val_loader = context["val_loader"]
    train_dataset = context["train_dataset"]
    val_dataset = context["val_dataset"]
    optimizer_group = context["optimizer"].param_groups[0]
    parameter_counts = shared_sweep.get_parameter_counts(context["model"])
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    return {
        "experiment": {
            "type": "fixed_learning_rate_candidate_sweep_full_fine_tuning",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
            "source_lr_range_test": None,
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
            "backbone_frozen": False,
            "batch_norm_frozen": False,
            "batch_norm_mode_during_training": "train",
            "trainable_component": TRAINABLE_COMPONENT,
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
            "optimized_parameter_scope": "entire_model",
            "weight_decay": optimizer_group["weight_decay"],
            "weight_decay_enabled": False,
            "momentum": optimizer_group["momentum"],
            "dampening": optimizer_group["dampening"],
            "nesterov": optimizer_group["nesterov"],
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
        "memory_note": (
            "Batch size 64 matches the source LR Range Test, but full "
            "fine-tuning previously emitted recoverable CUDA allocation "
            "warnings on an 8 GB GPU."
        ),
    }


def build_sweep_summary(summary_frame: pd.DataFrame) -> dict[str, Any]:
    """Build the machine-readable final LR selection summary."""

    ranked = shared_sweep.assign_ranks(summary_frame)
    completed = ranked[ranked["status"] == "completed"].sort_values(
        "rank_by_val_loss"
    )
    summary: dict[str, Any] = {
        "experiment_type": (
            "fixed_learning_rate_candidate_sweep_full_fine_tuning"
        ),
        "training_strategy": TRAINING_STRATEGY,
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
        return summary

    best = completed.iloc[0]
    summary.update(
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
        summary["runner_up_lr"] = runner_up["learning_rate"]
        summary["runner_up_val_loss"] = runner_up["best_val_loss"]
    return summary


def save_figures(
    histories: dict[float, pd.DataFrame],
    summary_frame: pd.DataFrame,
) -> None:
    """Reuse shared plotting functions with this sweep's output directory."""

    previous_figures_dir = shared_sweep.FIGURES_DIR
    try:
        shared_sweep.FIGURES_DIR = FIGURES_DIR
        shared_sweep.plot_candidate_val_loss(histories)
        shared_sweep.plot_candidate_best_val_loss(summary_frame)
        shared_sweep.plot_candidate_roc_auc(summary_frame)
        shared_sweep.plot_candidate_metrics(summary_frame)
    finally:
        shared_sweep.FIGURES_DIR = previous_figures_dir


def print_sweep_header(candidates: list[float]) -> None:
    """Print the full-fine-tuning experiment configuration."""

    print("Full Fine-Tuning Learning Rate Candidate Sweep")
    print()
    print(f"Candidates     : {len(candidates)}")
    print(f"Range          : {LR_CANDIDATE_START:.2e} -> {LR_CANDIDATE_END:.2e}")
    print(f"Step           : {LR_CANDIDATE_STEP:.2e}")
    print(f"Epochs         : {SHORT_TRAINING_EPOCHS}")
    print(f"Batch size     : {BATCH_SIZE}")
    print("Optimizer      : SGD")
    print(f"Momentum       : {MOMENTUM_OPTM:.1f}")
    print(f"Nesterov       : {NESTEROV_OPTM}")
    print("Weight decay   : disabled (0.0)")
    print(f"Training mode  : {TRAINING_STRATEGY}")
    print(f"Metadata       : {METADATA_PATH}")
    print(f"CT path column : {CT_PATH_COLUMN}")
    print("Candidate list:")
    print("  " + ", ".join(f"{candidate:.5f}" for candidate in candidates))
    print()


def print_final_ranking(
    summary_frame: pd.DataFrame,
    sweep_summary: dict[str, Any],
) -> None:
    """Print the top five completed full-fine-tuning candidates."""

    ranked = shared_sweep.assign_ranks(summary_frame)
    completed = ranked[ranked["status"] == "completed"].sort_values(
        "rank_by_val_loss"
    )
    print()
    print("Full Fine-Tuning LR Candidate Sweep Complete")
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


def main() -> None:
    """Execute the equal-budget full-fine-tuning LR candidate sweep."""

    validate_configuration()
    candidates = generate_lr_candidates()
    print_sweep_header(candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)

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
    shared_sweep.save_json(
        build_sweep_config(candidates, probe_context),
        SWEEP_CONFIG_PATH,
    )
    shared_sweep.release_candidate_resources(probe_context)
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
            summary = shared_sweep.build_failed_summary(
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
            summary = shared_sweep.build_failed_summary(
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
            shared_sweep.release_candidate_resources(context)

        candidate_summaries.append(summary)
        summary_frame = pd.DataFrame(
            candidate_summaries,
            columns=CANDIDATE_SUMMARY_COLUMNS,
        )
        save_candidate_summary(summary_frame)

    summary_frame = shared_sweep.assign_ranks(
        pd.DataFrame(
            candidate_summaries,
            columns=CANDIDATE_SUMMARY_COLUMNS,
        )
    )
    save_candidate_summary(summary_frame)
    save_figures(histories, summary_frame)

    sweep_summary = build_sweep_summary(summary_frame)
    shared_sweep.save_json(sweep_summary, SWEEP_SUMMARY_PATH)
    print_final_ranking(summary_frame, sweep_summary)


if __name__ == "__main__":
    main()

