"""Run an LR range test while fine-tuning the complete pretrained ResNet-50."""

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from uuid import uuid4
import warnings

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..pretrained_resnet50 import lr_range_test as shared_lr_finder
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
    "lr_range_test_full_finetuning_"
    f"{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "full_tuning_resnet50"
    / "lr_range_test"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_PATH = OUTPUT_DIR / "lr_range_test.csv"
CONFIG_PATH = OUTPUT_DIR / "lr_range_config.json"
SUMMARY_PATH = OUTPUT_DIR / "lr_range_summary.json"
LR_LOSS_FIGURE_PATH = FIGURES_DIR / "lr_vs_loss.png"
LR_GRADIENT_FIGURE_PATH = FIGURES_DIR / "lr_loss_gradient.png"


# ---------------------------------------------------------------------------
# Dataset and preprocessing: equal to the feature-extraction LR range test
# ---------------------------------------------------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)
METADATA_PATH = DATASET_ROOT / "001_holdout_split_lidc_lndb.csv"
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
# DataLoader: equal to the feature-extraction LR range test
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
TRAIN_SHUFFLE = True
TRAIN_DROP_LAST = False
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
# LR range-test configuration
# ---------------------------------------------------------------------------
LR_START = 1e-5
LR_END = 3e-1
LR_FINDER_EPOCHS = 2
LR_FINDER_MAX_ITERATIONS: int | None = None
LOSS_SMOOTHING_BETA = 0.98
DIVERGENCE_THRESHOLD = 4.0
DIVERGENCE_WARMUP_ITERATIONS = 10
GRADIENT_IGNORE_INITIAL_ITERATIONS = 10
DIVERGENCE_PROXIMITY_ITERATIONS = 5
SEED = 42
TRANSFORM_SEED = SEED


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_COLUMNS = shared_lr_finder.RESULT_COLUMNS


def validate_configuration() -> None:
    """Validate full-fine-tuning LR finder constants."""

    if not math.isfinite(MOMENTUM_OPTM) or MOMENTUM_OPTM < 0.0:
        raise ValueError("MOMENTUM_OPTM must be finite and non-negative.")
    if not isinstance(NESTEROV_OPTM, bool):
        raise TypeError("NESTEROV_OPTM must be a boolean.")
    if NESTEROV_OPTM and MOMENTUM_OPTM <= 0.0:
        raise ValueError("Nesterov SGD requires positive momentum.")
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
        or LR_FINDER_EPOCHS <= 0
    ):
        raise ValueError("LR_FINDER_EPOCHS must be a positive integer.")
    if LR_FINDER_MAX_ITERATIONS is not None and (
        isinstance(LR_FINDER_MAX_ITERATIONS, bool)
        or not isinstance(LR_FINDER_MAX_ITERATIONS, int)
        or LR_FINDER_MAX_ITERATIONS < 2
    ):
        raise ValueError(
            "LR_FINDER_MAX_ITERATIONS must be None or at least 2."
        )
    if not 0.0 <= LOSS_SMOOTHING_BETA < 1.0:
        raise ValueError("LOSS_SMOOTHING_BETA must be in [0, 1).")
    if not math.isfinite(LOSS_SMOOTHING_BETA):
        raise ValueError("LOSS_SMOOTHING_BETA must be finite.")
    if (
        not math.isfinite(DIVERGENCE_THRESHOLD)
        or DIVERGENCE_THRESHOLD <= 1.0
    ):
        raise ValueError("DIVERGENCE_THRESHOLD must be greater than 1.")
    for name, value in {
        "DIVERGENCE_WARMUP_ITERATIONS": DIVERGENCE_WARMUP_ITERATIONS,
        "GRADIENT_IGNORE_INITIAL_ITERATIONS": (
            GRADIENT_IGNORE_INITIAL_ITERATIONS
        ),
        "DIVERGENCE_PROXIMITY_ITERATIONS": (
            DIVERGENCE_PROXIMITY_ITERATIONS
        ),
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer.")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Split metadata not found: {METADATA_PATH}")


def build_train_transform() -> A.Compose:
    """Build the stochastic preprocessing used by the existing LR finder."""

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


def build_train_loader(transform: A.Compose) -> DataLoader:
    """Create the same training-split DataLoader as the existing LR finder."""

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
    """Reject a model containing parameters excluded from full fine-tuning."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("The model does not contain parameters.")

    frozen_parameter_names = [
        name
        for name, parameter in named_parameters
        if not parameter.requires_grad
    ]
    if frozen_parameter_names:
        raise RuntimeError(
            "Full fine-tuning requires every parameter to be trainable; "
            f"frozen parameters found: {frozen_parameter_names}"
        )


def set_full_fine_tuning_mode(model: nn.Module) -> None:
    """Enable training behavior for the backbone, BatchNorm, and classifier."""

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
            "Full fine-tuning requires every BatchNorm layer in train mode."
        )
    if not model.fc.training:
        raise RuntimeError("Classifier Dropout must remain active.")


def build_optimizer(model: nn.Module) -> torch.optim.SGD:
    """Create SGD with momentum over the entire model at the starting LR."""

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    all_parameters = list(model.parameters())
    if len(trainable_parameters) != len(all_parameters):
        raise RuntimeError(
            "Optimizer construction found parameters excluded from training."
        )

    optimizer = torch.optim.SGD(
        params=trainable_parameters,
        lr=LR_START,
        momentum=MOMENTUM_OPTM,
        nesterov=NESTEROV_OPTM,
    )
    optimized_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_parameter_ids = {id(parameter) for parameter in all_parameters}
    if optimized_parameter_ids != model_parameter_ids:
        raise RuntimeError(
            "SGD must optimize every full-fine-tuning model parameter."
        )
    return optimizer


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
        raise ValueError("LR Range Test requires at least two iterations.")
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
    optimizer: torch.optim.SGD,
    full_iterations: int,
    planned_iterations: int,
) -> dict[str, object]:
    """Build a reproducibility snapshot for full-network LR finding."""

    dataset = train_loader.dataset
    class_counts = Counter(dataset.targets)
    optimizer_group = optimizer.param_groups[0]
    parameter_counts = get_parameter_counts(model)

    return {
        "experiment": {
            "type": "learning_rate_range_test_full_fine_tuning",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "purpose": (
                "Identify a reasonable learning-rate region for full "
                "pretrained ResNet-50 fine-tuning; this is not final model "
                "training."
            ),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "pretrained_weights": str(WEIGHTS),
            "training_strategy": TRAINING_STRATEGY,
            "backbone_frozen": False,
            "batch_norm_frozen": False,
            "batch_norm_mode": "train",
            "trainable_component": TRAINABLE_COMPONENT,
            "classifier": {
                "architecture": "dropout_linear",
                "dropout_probability": CLASSIFIER_DROPOUT,
                "output_classes": len(CLASS_TO_IDX),
            },
            "total_parameters": parameter_counts["total"],
            "trainable_parameters": parameter_counts["trainable"],
            "frozen_parameters": parameter_counts["frozen"],
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
            "optimized_parameter_scope": "entire_model",
            "start_learning_rate": LR_START,
            "end_learning_rate": LR_END,
            "progression": "exponential_per_iteration",
            "weight_decay": optimizer_group["weight_decay"],
            "weight_decay_enabled": False,
            "momentum": optimizer_group["momentum"],
            "dampening": optimizer_group["dampening"],
            "nesterov": optimizer_group["nesterov"],
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
                (LR_END / LR_START) ** (1.0 / (planned_iterations - 1))
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

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, allow_nan=False)
        file.write("\n")


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
    optimizer: torch.optim.SGD,
    criterion: nn.Module,
    planned_iterations: int,
    full_iterations: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Increase LR per batch while updating the complete ResNet-50 model."""

    set_full_fine_tuning_mode(model)
    lr_multiplier = (
        (LR_END / LR_START) ** (1.0 / (planned_iterations - 1))
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
        desc="Full Fine-Tuning LR Range Test",
        unit="batch",
    )
    should_stop = False
    for epoch_index in range(LR_FINDER_EPOCHS):
        for batch_index, (images, labels) in enumerate(train_loader):
            iteration = len(records) + 1
            if iteration > planned_iterations:
                should_stop = True
                break

            current_lr = LR_START * lr_multiplier ** (iteration - 1)
            if iteration == planned_iterations:
                current_lr = LR_END
            set_optimizer_learning_rate(optimizer, current_lr)

            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
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
                    "Full fine-tuning LR Range Test stopped because loss "
                    "became non-finite at iteration "
                    f"{iteration}, LR={current_lr:.6e}."
                )
                break

            loss.backward()
            optimizer.step()

            running_average_loss = (
                LOSS_SMOOTHING_BETA * running_average_loss
                + (1.0 - LOSS_SMOOTHING_BETA) * raw_loss
            )
            smoothed_loss = running_average_loss / (
                1.0 - LOSS_SMOOTHING_BETA**iteration
            )
            previous_best = best_smoothed_loss
            divergence_detected = (
                iteration > DIVERGENCE_WARMUP_ITERATIONS
                and math.isfinite(previous_best)
                and smoothed_loss
                > previous_best * DIVERGENCE_THRESHOLD
            )
            best_smoothed_loss = min(best_smoothed_loss, smoothed_loss)

            records.append(
                {
                    "iteration": iteration,
                    "epoch": epoch_index + 1,
                    "batch_index": batch_index + 1,
                    "learning_rate": current_lr,
                    "raw_loss": raw_loss,
                    "smoothed_loss": smoothed_loss,
                    "best_smoothed_loss": best_smoothed_loss,
                }
            )
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


def compute_lr_diagnostics(
    results: pd.DataFrame,
    run_state: dict[str, object],
    planned_iterations: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reuse the existing diagnostics with this program's configuration."""

    diagnostic_names = (
        "LR_START",
        "LR_END",
        "GRADIENT_IGNORE_INITIAL_ITERATIONS",
        "DIVERGENCE_PROXIMITY_ITERATIONS",
    )
    previous_values = {
        name: getattr(shared_lr_finder, name)
        for name in diagnostic_names
    }
    replacement_values = {
        "LR_START": LR_START,
        "LR_END": LR_END,
        "GRADIENT_IGNORE_INITIAL_ITERATIONS": (
            GRADIENT_IGNORE_INITIAL_ITERATIONS
        ),
        "DIVERGENCE_PROXIMITY_ITERATIONS": (
            DIVERGENCE_PROXIMITY_ITERATIONS
        ),
    }
    try:
        for name, value in replacement_values.items():
            setattr(shared_lr_finder, name, value)
        return shared_lr_finder.compute_lr_diagnostics(
            results=results,
            run_state=run_state,
            planned_iterations=planned_iterations,
        )
    finally:
        for name, value in previous_values.items():
            setattr(shared_lr_finder, name, value)


def print_experiment_header(
    parameter_counts: dict[str, int],
    train_loader: DataLoader,
    planned_iterations: int,
) -> None:
    """Print the configuration relevant to interpreting the experiment."""

    print()
    print("=" * 72)
    print("Full Fine-Tuning Learning Rate Range Test")
    print("=" * 72)
    print(f"Model                : {MODEL_ARCHITECTURE}")
    print(f"Pretrained weights   : {WEIGHTS}")
    print(f"Training strategy    : {TRAINING_STRATEGY}")
    print(f"Trainable component  : {TRAINABLE_COMPONENT}")
    print(f"BatchNorm mode       : train")
    print(f"CT path column       : {CT_PATH_COLUMN}")
    print(f"Training samples     : {len(train_loader.dataset)}")
    print(f"Batches per epoch    : {len(train_loader)}")
    print(f"Planned iterations   : {planned_iterations}")
    print(f"LR range             : {LR_START:.2e} -> {LR_END:.2e}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Optimizer            : SGD")
    print(f"Momentum             : {MOMENTUM_OPTM:.1f}")
    print(f"Nesterov             : {NESTEROV_OPTM}")
    print("Weight decay         : disabled (0.0)")
    print(f"Total parameters     : {parameter_counts['total']:,}")
    print(f"Trainable parameters : {parameter_counts['trainable']:,}")
    print(f"Frozen parameters    : {parameter_counts['frozen']:,}")
    print(f"Device               : {DEVICE}")
    print()


def print_summary(summary: dict[str, object]) -> None:
    """Print summary highlights without claiming a final learning rate."""

    def format_lr(value: object) -> str:
        return f"{float(value):.6e}" if value is not None else "Not available"

    print()
    print("=" * 72)
    print("Full Fine-Tuning LR Range Test Complete")
    print("=" * 72)
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
    """Run the full-network LR range experiment and save diagnostics."""

    validate_configuration()
    set_seed(seed=SEED, deterministic=True)
    train_transform = build_train_transform()
    train_loader = build_train_loader(train_transform)
    full_iterations, planned_iterations = resolve_iteration_counts(
        train_loader
    )
    model = build_model()
    set_full_fine_tuning_mode(model)
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
    results.to_csv(RESULTS_PATH, index=False, float_format="%.12e")
    save_json(summary, SUMMARY_PATH)
    shared_lr_finder.plot_lr_vs_loss(
        results=results,
        summary=summary,
        output_path=LR_LOSS_FIGURE_PATH,
    )
    shared_lr_finder.plot_lr_loss_gradient(
        results=results,
        summary=summary,
        output_path=LR_GRADIENT_FIGURE_PATH,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()

