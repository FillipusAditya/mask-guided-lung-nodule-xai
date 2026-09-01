"""Run a short learning-rate range test for ResNet-50 feature extraction."""

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4
import warnings

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
from ..utils.seed import set_seed


# ---------------------------------------------------------------------------
# Project and output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = (
    f"lr_range_test_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification"
    / "pretrained_resnet50"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "lr_range_test.csv"
CONFIG_PATH = OUTPUT_DIR / "lr_range_config.json"
SUMMARY_PATH = OUTPUT_DIR / "lr_range_summary.json"
LR_LOSS_FIGURE_PATH = FIGURES_DIR / "lr_vs_loss.png"
LR_GRADIENT_FIGURE_PATH = FIGURES_DIR / "lr_loss_gradient.png"

# ---------------------------------------------------------------------------
# Dataset and preprocessing: kept equal to normal feature-extraction training
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
METADATA_PATH = (
    DATASET_ROOT
    / "001_holdout_split_lidc_lndb.csv"
)
CT_PATH_COLUMN = "ct_windowed_path"
TRAIN_SPLIT = "train"
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# DataLoader: kept equal to normal feature-extraction training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
TRAIN_SHUFFLE = True
TRAIN_DROP_LAST = False
NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Model and optimizer: only learning rate varies during this experiment
# ---------------------------------------------------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL_ARCHITECTURE = "ResNet50"
TRAINING_STRATEGY = "feature_extraction"
CLASSIFIER_DROPOUT = 0.3
WEIGHT_DECAY_OPTM = 1e-4


# ---------------------------------------------------------------------------
# LR range-test configuration
# ---------------------------------------------------------------------------
LR_START = 1e-7
LR_END = 1e-1
LR_FINDER_EPOCHS = 2
LR_FINDER_MAX_ITERATIONS: int | None = None
LOSS_SMOOTHING_BETA = 0.98
DIVERGENCE_THRESHOLD = 4.0
DIVERGENCE_WARMUP_ITERATIONS = 10
GRADIENT_IGNORE_INITIAL_ITERATIONS = 10
DIVERGENCE_PROXIMITY_ITERATIONS = 5
SEED = 42
TRANSFORM_SEED = SEED


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

RESULT_COLUMNS = [
    "iteration",
    "epoch",
    "batch_index",
    "learning_rate",
    "raw_loss",
    "smoothed_loss",
    "best_smoothed_loss",
    "loss_gradient",
]


def validate_configuration() -> None:
    """Validate LR finder constants before allocating training resources."""

    if not math.isfinite(LR_START) or LR_START <= 0.0:
        raise ValueError("LR_START must be a finite value greater than zero.")
    if not math.isfinite(LR_END) or LR_END <= LR_START:
        raise ValueError("LR_END must be finite and greater than LR_START.")
    if isinstance(BATCH_SIZE, bool) or not isinstance(BATCH_SIZE, int):
        raise TypeError("BATCH_SIZE must be an integer.")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be greater than zero.")
    if (
        isinstance(LR_FINDER_EPOCHS, bool)
        or not isinstance(LR_FINDER_EPOCHS, int)
    ):
        raise TypeError("LR_FINDER_EPOCHS must be an integer.")
    if LR_FINDER_EPOCHS <= 0:
        raise ValueError("LR_FINDER_EPOCHS must be greater than zero.")
    if LR_FINDER_MAX_ITERATIONS is not None:
        if (
            isinstance(LR_FINDER_MAX_ITERATIONS, bool)
            or not isinstance(LR_FINDER_MAX_ITERATIONS, int)
            or LR_FINDER_MAX_ITERATIONS < 2
        ):
            raise ValueError(
                "LR_FINDER_MAX_ITERATIONS must be None or at least 2."
            )
    if (
        not math.isfinite(LOSS_SMOOTHING_BETA)
        or not 0.0 <= LOSS_SMOOTHING_BETA < 1.0
    ):
        raise ValueError("LOSS_SMOOTHING_BETA must be in [0, 1).")
    if (
        not math.isfinite(DIVERGENCE_THRESHOLD)
        or DIVERGENCE_THRESHOLD <= 1.0
    ):
        raise ValueError("DIVERGENCE_THRESHOLD must be greater than 1.")
    if (
        isinstance(DIVERGENCE_WARMUP_ITERATIONS, bool)
        or not isinstance(DIVERGENCE_WARMUP_ITERATIONS, int)
        or DIVERGENCE_WARMUP_ITERATIONS < 0
    ):
        raise ValueError(
            "DIVERGENCE_WARMUP_ITERATIONS must be a non-negative integer."
        )
    if (
        isinstance(GRADIENT_IGNORE_INITIAL_ITERATIONS, bool)
        or not isinstance(GRADIENT_IGNORE_INITIAL_ITERATIONS, int)
        or GRADIENT_IGNORE_INITIAL_ITERATIONS < 0
    ):
        raise ValueError(
            "GRADIENT_IGNORE_INITIAL_ITERATIONS must be a non-negative "
            "integer."
        )
    if (
        isinstance(DIVERGENCE_PROXIMITY_ITERATIONS, bool)
        or not isinstance(DIVERGENCE_PROXIMITY_ITERATIONS, int)
        or DIVERGENCE_PROXIMITY_ITERATIONS < 0
    ):
        raise ValueError(
            "DIVERGENCE_PROXIMITY_ITERATIONS must be a non-negative integer."
        )


def build_train_transform() -> A.Compose:
    """Build the same stochastic preprocessing used by normal training."""

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
            A.GaussNoise(
                std_range=(0.01, 0.03),
                p=0.2,
            ),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=TRANSFORM_SEED,
    )


def build_train_loader(transform: A.Compose) -> DataLoader:
    """Create the normal training split DataLoader for the range test."""

    return create_dataloader(
        root_dir=DATASET_ROOT,
        metadata_path=METADATA_PATH,
        split=TRAIN_SPLIT,
        transform=transform,
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


def build_model() -> nn.Module:
    """Build pretrained ResNet-50 with only its classifier trainable."""

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

    return model.to(DEVICE)


def set_feature_extraction_mode(model: nn.Module) -> None:
    """Freeze backbone behavior while keeping classifier Dropout active."""

    model.eval()
    model.fc.train()

    trainable_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameter_names:
        raise RuntimeError("The model does not contain trainable parameters.")
    if any(
        not name.startswith("fc.")
        for name in trainable_parameter_names
    ):
        raise RuntimeError(
            "Feature-extraction range test expects only fc parameters to be "
            f"trainable, found: {trainable_parameter_names}"
        )

    training_batch_norm_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and module.training
    ]
    if training_batch_norm_layers:
        raise RuntimeError(
            "Backbone BatchNorm layers must remain in evaluation mode."
        )


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    """Create AdamW for classifier parameters at the starting LR."""

    trainable_parameters = [
        parameter
        for parameter in model.fc.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("Classifier does not contain trainable parameters.")

    return torch.optim.AdamW(
        params=trainable_parameters,
        lr=LR_START,
        weight_decay=WEIGHT_DECAY_OPTM,
    )


def resolve_iteration_counts(
    train_loader: DataLoader,
) -> tuple[int, int]:
    """Return full and planned iteration counts for this range test."""

    batches_per_epoch = len(train_loader)
    if batches_per_epoch <= 0:
        raise ValueError("The training DataLoader must not be empty.")

    full_iterations = batches_per_epoch * LR_FINDER_EPOCHS
    planned_iterations = (
        min(full_iterations, LR_FINDER_MAX_ITERATIONS)
        if LR_FINDER_MAX_ITERATIONS is not None
        else full_iterations
    )
    if planned_iterations < 2:
        raise ValueError(
            "LR Range Test requires at least two optimizer iterations."
        )

    return full_iterations, planned_iterations


def get_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen model parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def build_config(
    train_loader: DataLoader,
    train_transform: A.Compose,
    model: nn.Module,
    optimizer: torch.optim.AdamW,
    full_iterations: int,
    planned_iterations: int,
) -> dict[str, object]:
    """Build a reproducibility snapshot for the LR range experiment."""

    dataset = train_loader.dataset
    class_counts = Counter(dataset.targets)
    optimizer_group = optimizer.param_groups[0]
    parameter_counts = get_parameter_counts(model)

    return {
        "experiment": {
            "type": "learning_rate_range_test",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "purpose": (
                "Identify a reasonable learning-rate region; this is not "
                "final model training."
            ),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "pretrained_weights": str(WEIGHTS),
            "training_strategy": TRAINING_STRATEGY,
            "backbone_frozen": True,
            "batch_norm_frozen": True,
            "trainable_component": "fc",
            "classifier": {
                "architecture": "dropout_linear",
                "dropout_probability": CLASSIFIER_DROPOUT,
                "output_classes": len(CLASS_TO_IDX),
            },
            "total_parameters": parameter_counts["total"],
            "trainable_parameters": parameter_counts["trainable"],
            "frozen_parameters": parameter_counts["frozen"],
            "trainable_parameter_names": [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "metadata_path": str(METADATA_PATH),
            "split": TRAIN_SPLIT,
            "ct_path_column": CT_PATH_COLUMN,
            "image_height": INPUT_HEIGHT,
            "image_width": INPUT_WIDTH,
            "input_channels": 3,
            "classes": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
            "num_samples": len(dataset),
            "num_batches": len(train_loader),
            "class_distribution": {
                class_name: class_counts[class_index]
                for class_name, class_index in dataset.class_to_idx.items()
            },
            "train_transforms": str(train_transform),
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "dataloader": {
            "batch_size": BATCH_SIZE,
            "shuffle": TRAIN_SHUFFLE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
            "drop_last": TRAIN_DROP_LAST,
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "start_learning_rate": LR_START,
            "end_learning_rate": LR_END,
            "progression": "exponential_per_iteration",
            "weight_decay": optimizer_group["weight_decay"],
            "betas": optimizer_group["betas"],
            "eps": optimizer_group["eps"],
            "amsgrad": optimizer_group["amsgrad"],
        },
        "loss": {
            "name": "CrossEntropyLoss",
            "class_weighting": None,
            "label_smoothing": 0.0,
        },
        "lr_range_test": {
            "epochs": LR_FINDER_EPOCHS,
            "max_iterations": LR_FINDER_MAX_ITERATIONS,
            "full_iterations": full_iterations,
            "planned_iterations": planned_iterations,
            "lr_multiplier": (
                (LR_END / LR_START)
                ** (1.0 / (planned_iterations - 1))
            ),
            "loss_smoothing_beta": LOSS_SMOOTHING_BETA,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "divergence_warmup_iterations": (
                DIVERGENCE_WARMUP_ITERATIONS
            ),
            "gradient_ignore_initial_iterations": (
                GRADIENT_IGNORE_INITIAL_ITERATIONS
            ),
            "divergence_proximity_iterations": (
                DIVERGENCE_PROXIMITY_ITERATIONS
            ),
            "scheduler": None,
            "early_stopping": None,
            "validation": None,
            "model_checkpoint": None,
        },
        "reproducibility": {
            "seed": SEED,
            "transform_seed": TRANSFORM_SEED,
            "deterministic": True,
        },
        "runtime": {
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
    }


def save_json(data: dict[str, object], output_path: Path) -> None:
    """Save a standards-compliant JSON document."""

    with open(output_path, "w") as file:
        json.dump(
            data,
            file,
            indent=4,
            allow_nan=False,
        )


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Apply one learning rate to every optimizer parameter group."""

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def run_lr_range_test(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.AdamW,
    criterion: nn.Module,
    planned_iterations: int,
    full_iterations: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Increase LR per batch and record raw and bias-corrected EMA loss."""

    set_feature_extraction_mode(model)
    lr_multiplier = (
        (LR_END / LR_START)
        ** (1.0 / (planned_iterations - 1))
    )
    running_average_loss = 0.0
    best_smoothed_loss = float("inf")
    records: list[dict[str, float | int]] = []
    stop_reason = "completed_lr_range"
    stop_lr: float | None = None
    failed_iteration: int | None = None
    divergence_lr: float | None = None

    progress_bar = tqdm(
        total=planned_iterations,
        desc="LR Range Test",
        unit="batch",
    )

    should_stop = False
    for epoch_index in range(LR_FINDER_EPOCHS):
        for batch_index, (images, labels) in enumerate(train_loader):
            iteration = len(records) + 1
            if iteration > planned_iterations:
                should_stop = True
                break

            current_lr = (
                LR_START
                * lr_multiplier ** (iteration - 1)
            )
            if iteration == planned_iterations:
                # Avoid floating-point drift at the upper endpoint.
                current_lr = LR_END
            set_optimizer_learning_rate(optimizer, current_lr)

            images = images.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )
            labels = labels.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            raw_loss = float(loss.detach().item())

            if not math.isfinite(raw_loss):
                stop_reason = "non_finite_loss"
                stop_lr = current_lr
                failed_iteration = iteration
                should_stop = True
                warnings.warn(
                    "LR Range Test stopped because loss became non-finite "
                    f"at iteration {iteration}, LR={current_lr:.6e}."
                )
                break

            loss.backward()
            optimizer.step()

            running_average_loss = (
                LOSS_SMOOTHING_BETA * running_average_loss
                + (1.0 - LOSS_SMOOTHING_BETA) * raw_loss
            )
            smoothed_loss = (
                running_average_loss
                / (
                    1.0
                    - LOSS_SMOOTHING_BETA ** iteration
                )
            )

            previous_best = best_smoothed_loss
            divergence_detected = (
                iteration > DIVERGENCE_WARMUP_ITERATIONS
                and math.isfinite(previous_best)
                and smoothed_loss
                > previous_best * DIVERGENCE_THRESHOLD
            )
            best_smoothed_loss = min(
                best_smoothed_loss,
                smoothed_loss,
            )

            records.append({
                "iteration": iteration,
                "epoch": epoch_index + 1,
                "batch_index": batch_index + 1,
                "learning_rate": current_lr,
                "raw_loss": raw_loss,
                "smoothed_loss": smoothed_loss,
                "best_smoothed_loss": best_smoothed_loss,
            })
            progress_bar.update(1)
            progress_bar.set_postfix(
                lr=f"{current_lr:.3e}",
                loss=f"{raw_loss:.4f}",
                smooth=f"{smoothed_loss:.4f}",
                best=f"{best_smoothed_loss:.4f}",
            )

            if divergence_detected:
                stop_reason = "diverged"
                stop_lr = current_lr
                divergence_lr = current_lr
                should_stop = True
                break

            if iteration >= planned_iterations:
                should_stop = True
                if (
                    LR_FINDER_MAX_ITERATIONS is not None
                    and planned_iterations < full_iterations
                ):
                    stop_reason = "max_iterations_reached"
                else:
                    stop_reason = "completed_lr_range"
                break

        if should_stop:
            break

    progress_bar.close()

    results = pd.DataFrame(records)
    for column in RESULT_COLUMNS:
        if column not in results:
            results[column] = pd.Series(dtype=float)
    results = results[RESULT_COLUMNS]

    run_state: dict[str, object] = {
        "stop_reason": stop_reason,
        "stop_lr": stop_lr,
        "failed_iteration": failed_iteration,
        "divergence_lr": divergence_lr,
        "lr_multiplier": lr_multiplier,
    }
    return results, run_state


def _finite_float(value: float | np.floating | None) -> float | None:
    """Convert one finite numeric value to JSON-safe float or return None."""

    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def compute_lr_diagnostics(
    results: pd.DataFrame,
    run_state: dict[str, object],
    planned_iterations: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compute LR-loss gradient and transparent candidate-region heuristics."""

    analyzed = results.copy()
    analyzed["loss_gradient"] = np.nan
    executed_iterations = len(analyzed)
    warning_message: str | None = None
    minimum_loss_lr: float | None = None
    minimum_smoothed_loss: float | None = None
    minimum_index: int | None = None
    steepest_descent_lr: float | None = None
    steepest_gradient: float | None = None
    steepest_index: int | None = None

    if executed_iterations > 0:
        learning_rates = analyzed["learning_rate"].to_numpy(dtype=float)
        smoothed_losses = analyzed["smoothed_loss"].to_numpy(dtype=float)
        finite_mask = np.isfinite(learning_rates) & np.isfinite(smoothed_losses)

        if not finite_mask.all():
            raise ValueError(
                "Recorded LR and smoothed-loss values must all be finite."
            )

        analysis_start_index = min(
            GRADIENT_IGNORE_INITIAL_ITERATIONS,
            executed_iterations - 1,
        )
        eligible_indices = np.arange(
            analysis_start_index,
            executed_iterations,
        )
        eligible_losses = smoothed_losses[eligible_indices]
        minimum_index = int(
            eligible_indices[int(np.argmin(eligible_losses))]
        )
        minimum_loss_lr = float(learning_rates[minimum_index])
        minimum_smoothed_loss = float(smoothed_losses[minimum_index])

        if executed_iterations >= 3 and len(eligible_indices) >= 3:
            log_learning_rates = np.log10(learning_rates)
            gradients = np.gradient(
                smoothed_losses,
                log_learning_rates,
            )
            analyzed["loss_gradient"] = gradients
            eligible_gradients = gradients[eligible_indices]
            steepest_index = int(
                eligible_indices[int(np.argmin(eligible_gradients))]
            )
            steepest_descent_lr = float(
                learning_rates[steepest_index]
            )
            steepest_gradient = float(gradients[steepest_index])
        else:
            warning_message = (
                "Too few post-warmup observations to estimate the loss "
                "gradient reliably. Inspect lr_vs_loss.png manually."
            )
    else:
        analysis_start_index = None
        warning_message = (
            "No finite optimizer iteration was recorded; LR diagnostics "
            "could not be computed."
        )

    divergence_lr = _finite_float(run_state.get("divergence_lr"))
    lr_before_divergence: float | None = None
    if divergence_lr is not None and executed_iterations >= 2:
        lr_before_divergence = float(
            analyzed.iloc[-2]["learning_rate"]
        )

    minimum_near_divergence = False
    if divergence_lr is not None and minimum_index is not None:
        divergence_index = executed_iterations - 1
        minimum_near_divergence = (
            divergence_index - minimum_index
            <= DIVERGENCE_PROXIMITY_ITERATIONS
        )

    minimum_at_last_iteration = (
        minimum_index is not None
        and minimum_index == executed_iterations - 1
    )

    candidate_lower: float | None = None
    candidate_upper: float | None = None
    candidate_explanation = (
        "Candidate bounds are reported only when the post-warmup steepest-"
        "descent LR occurs before the post-warmup minimum-loss LR. The region "
        "is a diagnostic heuristic, not a guaranteed optimal interval."
    )
    if (
        steepest_index is not None
        and minimum_index is not None
        and steepest_index < minimum_index
        and not minimum_at_last_iteration
    ):
        candidate_lower = steepest_descent_lr
        candidate_upper = minimum_loss_lr

    if minimum_near_divergence:
        proximity_note = (
            " The minimum loss occurred within "
            f"{DIVERGENCE_PROXIMITY_ITERATIONS} iterations of detected "
            "divergence, so the upper bound should be interpreted cautiously."
        )
        candidate_explanation += proximity_note
    if minimum_at_last_iteration:
        candidate_explanation += (
            " The minimum occurred at the last recorded iteration, so this "
            "test did not bracket the minimum and no candidate upper bound "
            "is reported automatically."
        )

    initial_smoothed_loss = (
        _finite_float(analyzed.iloc[0]["smoothed_loss"])
        if executed_iterations > 0
        else None
    )
    last_lr = (
        _finite_float(analyzed.iloc[-1]["learning_rate"])
        if executed_iterations > 0
        else None
    )

    summary: dict[str, object] = {
        "executed_iterations": executed_iterations,
        "planned_iterations": planned_iterations,
        "stop_reason": run_state["stop_reason"],
        "failed_iteration": run_state["failed_iteration"],
        "start_lr": LR_START,
        "configured_end_lr": LR_END,
        "last_lr": last_lr,
        "stop_lr": _finite_float(run_state.get("stop_lr")),
        "initial_smoothed_loss": initial_smoothed_loss,
        "minimum_smoothed_loss": minimum_smoothed_loss,
        "minimum_loss_lr": minimum_loss_lr,
        "steepest_descent_lr": steepest_descent_lr,
        "steepest_loss_gradient": steepest_gradient,
        "divergence_detected": divergence_lr is not None,
        "divergence_lr": divergence_lr,
        "lr_before_divergence": lr_before_divergence,
        "candidate_lr_lower": candidate_lower,
        "candidate_lr_upper": candidate_upper,
        "candidate_region_available": (
            candidate_lower is not None
            and candidate_upper is not None
        ),
        "candidate_region_heuristic": candidate_explanation,
        "analysis_ignored_initial_iterations": (
            GRADIENT_IGNORE_INITIAL_ITERATIONS
        ),
        "analysis_start_iteration": (
            analysis_start_index + 1
            if analysis_start_index is not None
            else None
        ),
        "minimum_loss_near_divergence": minimum_near_divergence,
        "minimum_loss_at_last_iteration": minimum_at_last_iteration,
        "analysis_warning": warning_message,
        "model_artifact_saved": False,
    }
    return analyzed, summary


def plot_lr_vs_loss(
    results: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    """Plot raw and smoothed training loss against logarithmic LR."""

    figure, axis = plt.subplots(figsize=(10, 6))
    if results.empty:
        axis.text(
            0.5,
            0.5,
            "No finite LR-loss observations were recorded.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        axis.plot(
            results["learning_rate"],
            results["raw_loss"],
            color="tab:gray",
            alpha=0.30,
            linewidth=1.0,
            label="Raw batch loss",
        )
        axis.plot(
            results["learning_rate"],
            results["smoothed_loss"],
            color="tab:blue",
            linewidth=2.0,
            label="Bias-corrected smoothed loss",
        )

        candidate_lower = summary["candidate_lr_lower"]
        candidate_upper = summary["candidate_lr_upper"]
        if candidate_lower is not None and candidate_upper is not None:
            axis.axvspan(
                candidate_lower,
                candidate_upper,
                color="tab:green",
                alpha=0.12,
                label="Candidate reasonable LR region",
            )

        markers = [
            (
                summary["steepest_descent_lr"],
                "Steepest descent LR",
                "tab:green",
            ),
            (
                summary["minimum_loss_lr"],
                "Minimum-loss LR",
                "tab:orange",
            ),
            (
                summary["divergence_lr"],
                "Divergence LR",
                "tab:red",
            ),
        ]
        for learning_rate, label, color in markers:
            if learning_rate is not None:
                axis.axvline(
                    learning_rate,
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    label=label,
                )

    axis.set_xscale("log")
    axis.set_title("LR Range Test: Learning Rate vs Training Loss")
    axis.set_xlabel("Learning Rate (log scale)")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.grid(True, which="both", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_lr_loss_gradient(
    results: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    """Plot loss derivative with respect to log10 learning rate."""

    figure, axis = plt.subplots(figsize=(10, 6))
    valid_gradient = (
        results["loss_gradient"].notna()
        if "loss_gradient" in results
        else pd.Series(dtype=bool)
    )
    if not results.empty and valid_gradient.any():
        axis.plot(
            results.loc[valid_gradient, "learning_rate"],
            results.loc[valid_gradient, "loss_gradient"],
            color="tab:purple",
            linewidth=1.8,
            label="Loss gradient",
        )
        steepest_lr = summary["steepest_descent_lr"]
        if steepest_lr is not None:
            axis.axvline(
                steepest_lr,
                color="tab:green",
                linestyle="--",
                linewidth=1.5,
                label="Steepest descent LR",
            )
    else:
        axis.text(
            0.5,
            0.5,
            "Insufficient observations for reliable gradient analysis.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xscale("log")
    axis.set_title("LR Range Test: Smoothed-Loss Gradient")
    axis.set_xlabel("Learning Rate (log scale)")
    axis.set_ylabel("d(smoothed loss) / d(log10(LR))")
    axis.grid(True, which="both", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def print_experiment_header(
    parameter_counts: dict[str, int],
    train_loader: DataLoader,
    planned_iterations: int,
) -> None:
    """Print the configuration most relevant to interpreting LR results."""

    print()
    print("=" * 64)
    print("Learning Rate Range Test")
    print("=" * 64)
    print(f"Model                : {MODEL_ARCHITECTURE}")
    print(f"Training strategy    : {TRAINING_STRATEGY}")
    print(f"CT path column       : {CT_PATH_COLUMN}")
    print(f"Training samples     : {len(train_loader.dataset)}")
    print(f"Batches per epoch    : {len(train_loader)}")
    print(f"Planned iterations   : {planned_iterations}")
    print(f"LR range             : {LR_START:.2e} -> {LR_END:.2e}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Weight decay         : {WEIGHT_DECAY_OPTM:.2e}")
    print(f"Total parameters     : {parameter_counts['total']:,}")
    print(f"Trainable parameters : {parameter_counts['trainable']:,}")
    print(f"Frozen parameters    : {parameter_counts['frozen']:,}")
    print(f"Device               : {DEVICE}")
    print()


def print_summary(summary: dict[str, object]) -> None:
    """Print machine-summary highlights without claiming a final LR."""

    def format_lr(value: object) -> str:
        return (
            f"{float(value):.6e}"
            if value is not None
            else "Not available"
        )

    print()
    print("=" * 64)
    print("LR Range Test Complete")
    print("=" * 64)
    print(f"Executed iterations : {summary['executed_iterations']}")
    print(f"Stop reason         : {summary['stop_reason']}")
    print(
        "Steepest descent LR : "
        f"{format_lr(summary['steepest_descent_lr'])}"
    )
    print(
        "Minimum-loss LR     : "
        f"{format_lr(summary['minimum_loss_lr'])}"
    )
    print(
        "Divergence LR       : "
        f"{format_lr(summary['divergence_lr'])}"
    )
    print("Candidate reasonable LR region:")
    if summary["candidate_region_available"]:
        print(
            f"  {format_lr(summary['candidate_lr_lower'])} -> "
            f"{format_lr(summary['candidate_lr_upper'])}"
        )
    else:
        print("  Unable to determine reliably. Inspect lr_vs_loss.png.")

    if summary["minimum_loss_near_divergence"]:
        print(
            "Warning: minimum loss is close to detected divergence; "
            "interpret the upper candidate bound cautiously."
        )
    if summary["minimum_loss_at_last_iteration"]:
        print(
            "Warning: minimum loss is at the final recorded LR; the range "
            "did not bracket the minimum."
        )
    if summary["analysis_warning"] is not None:
        print(f"Warning: {summary['analysis_warning']}")
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}")


def main() -> None:
    """Run the complete short LR range experiment and save diagnostics."""

    validate_configuration()
    set_seed(seed=SEED, deterministic=True)
    train_transform = build_train_transform()
    train_loader = build_train_loader(train_transform)
    full_iterations, planned_iterations = resolve_iteration_counts(
        train_loader
    )
    model = build_model()
    set_feature_extraction_mode(model)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    parameter_counts = get_parameter_counts(model)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=False)
    config = build_config(
        train_loader=train_loader,
        train_transform=train_transform,
        model=model,
        optimizer=optimizer,
        full_iterations=full_iterations,
        planned_iterations=planned_iterations,
    )
    save_json(config, CONFIG_PATH)
    print_experiment_header(
        parameter_counts=parameter_counts,
        train_loader=train_loader,
        planned_iterations=planned_iterations,
    )

    results, run_state = run_lr_range_test(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        planned_iterations=planned_iterations,
        full_iterations=full_iterations,
    )
    results, summary = compute_lr_diagnostics(
        results=results,
        run_state=run_state,
        planned_iterations=planned_iterations,
    )
    results.to_csv(
        RESULTS_PATH,
        index=False,
        float_format="%.12e",
    )
    save_json(summary, SUMMARY_PATH)
    plot_lr_vs_loss(
        results=results,
        summary=summary,
        output_path=LR_LOSS_FIGURE_PATH,
    )
    plot_lr_loss_gradient(
        results=results,
        summary=summary,
        output_path=LR_GRADIENT_FIGURE_PATH,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
