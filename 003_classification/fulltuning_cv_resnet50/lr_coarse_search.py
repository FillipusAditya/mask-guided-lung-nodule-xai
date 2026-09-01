"""Stage 1 coarse fixed-learning-rate search with five-fold CV."""

from datetime import datetime
import gc
import json
import math
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from . import train as cv_train
from ..utils import set_seed


# ---------------------------------------------------------------------------
# Experiment output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = (
    f"lr_coarse_search_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "fulltuning_cv_resnet50"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFIG_PATH = OUTPUT_DIR / "coarse_search_config.json"
EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_log.csv"
FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
LR_SUMMARY_PATH = OUTPUT_DIR / "lr_summary.csv"
LR_RANKING_PATH = OUTPUT_DIR / "lr_ranking.csv"
SEARCH_SUMMARY_PATH = OUTPUT_DIR / "search_summary.json"


# ---------------------------------------------------------------------------
# Fixed Stage 1 search configuration
# ---------------------------------------------------------------------------
LEARNING_RATES = [
    3e-05,
    0.0001,
    0.0003,
    0.001,
    0.003,
]
EPOCHS_PER_FOLD = 50
BATCH_SIZE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CLASSIFIER_DROPOUT = 0.2
NESTEROV = False
SEED = 42
CV_FOLDS = tuple(range(5))
PRIMARY_METRIC = "roc_auc"

DEVICE = cv_train.DEVICE
CLASSIFICATION_THRESHOLD = cv_train.CLASSIFICATION_THRESHOLD

VALIDATION_METRICS = (
    "val_loss",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "roc_auc",
)

FOLD_RESULT_COLUMNS = (
    "learning_rate",
    "fold",
    "status",
    "completed_epochs",
    "best_epoch",
    "roc_auc",
    "val_loss",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "final_train_loss",
    "final_val_loss",
    "train_samples",
    "val_samples",
    "train_patients",
    "val_patients",
    "training_seconds",
    "failure_message",
)

EPOCH_LOG_COLUMNS = (
    "learning_rate",
    "fold",
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "roc_auc",
    "epoch_time_seconds",
)


def validate_configuration() -> None:
    """Validate fixed search values and their agreement with CV training."""

    if not LEARNING_RATES:
        raise ValueError("LEARNING_RATES must not be empty.")
    if any(
        not math.isfinite(learning_rate) or learning_rate <= 0.0
        for learning_rate in LEARNING_RATES
    ):
        raise ValueError("Every learning rate must be finite and positive.")
    if LEARNING_RATES != sorted(set(LEARNING_RATES)):
        raise ValueError("LEARNING_RATES must be unique and ascending.")
    if EPOCHS_PER_FOLD <= 0:
        raise ValueError("EPOCHS_PER_FOLD must be positive.")
    if BATCH_SIZE != 64:
        raise ValueError("Stage 1 requires BATCH_SIZE = 64.")
    if not math.isclose(MOMENTUM, 0.9, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Stage 1 requires MOMENTUM = 0.9.")
    if not math.isclose(WEIGHT_DECAY, 1e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Stage 1 requires WEIGHT_DECAY = 1e-4.")
    if not math.isclose(
        CLASSIFIER_DROPOUT,
        0.2,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("Stage 1 requires CLASSIFIER_DROPOUT = 0.2.")
    if CV_FOLDS != tuple(range(5)):
        raise ValueError("Stage 1 requires folds 0 through 4.")
    if cv_train.BATCH_SIZE != BATCH_SIZE:
        raise RuntimeError("CV DataLoader batch size is not 64.")
    if cv_train.MOMENTUM_OPTM != MOMENTUM:
        raise RuntimeError("CV training momentum differs from Stage 1.")
    if cv_train.WEIGHT_DECAY_OPTM != WEIGHT_DECAY:
        raise RuntimeError("CV training weight decay differs from Stage 1.")
    if cv_train.CV_FOLDS != CV_FOLDS:
        raise RuntimeError("CV fold definitions differ from Stage 1.")


def build_model(num_classes: int) -> nn.Module:
    """Reuse the full-tuning model while applying Stage 1 dropout 0.2."""

    model = cv_train.base_train.build_model(num_classes)
    if (
        not isinstance(model.fc, nn.Sequential)
        or len(model.fc) != 2
        or not isinstance(model.fc[0], nn.Dropout)
        or not isinstance(model.fc[1], nn.Linear)
    ):
        raise RuntimeError("Unexpected ResNet-50 classifier structure.")
    model.fc[0] = nn.Dropout(p=CLASSIFIER_DROPOUT)
    cv_train.base_train.assert_full_model_trainable(model)
    return model


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.SGD:
    """Build constant-LR SGD over every trainable ResNet-50 parameter."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    all_parameters = list(model.parameters())
    if len(trainable_parameters) != len(all_parameters):
        raise RuntimeError("Stage 1 requires the entire model to be trainable.")
    optimizer = torch.optim.SGD(
        params=trainable_parameters,
        lr=learning_rate,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimized_ids != {id(parameter) for parameter in all_parameters}:
        raise RuntimeError("SGD does not cover every model parameter.")
    return optimizer


def _synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _is_better_epoch(
    candidate: dict[str, float | int],
    current_best: dict[str, float | int] | None,
) -> bool:
    """Select maximum ROC-AUC, breaking exact ties by lower val loss."""

    candidate_auc = float(candidate["roc_auc"])
    if not math.isfinite(candidate_auc):
        return False
    if current_best is None:
        return True
    best_auc = float(current_best["roc_auc"])
    if candidate_auc > best_auc:
        return True
    return math.isclose(
        candidate_auc,
        best_auc,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and float(candidate["val_loss"]) < float(current_best["val_loss"])


def run_fold(
    learning_rate: float,
    fold: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run one independent equal-budget LR/fold experiment."""

    set_seed(seed=SEED, deterministic=True)
    train_loader, val_loader, _, _ = cv_train.build_fold_dataloaders(fold)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]

    model = build_model(num_classes)
    optimizer = build_optimizer(model, learning_rate)
    criterion = nn.CrossEntropyLoss()
    fixed_lr = float(optimizer.param_groups[0]["lr"])
    if fixed_lr != learning_rate:
        raise RuntimeError("Optimizer LR differs from the candidate LR.")

    print("=" * 76)
    print(
        f"LR {learning_rate:.3e} | Fold {fold}/{len(CV_FOLDS) - 1} | "
        f"{EPOCHS_PER_FOLD} epochs"
    )
    print("=" * 76)

    epoch_records: list[dict[str, float | int]] = []
    best_record: dict[str, float | int] | None = None
    _synchronize_device()
    experiment_started_at = time.perf_counter()
    for epoch in range(EPOCHS_PER_FOLD):
        _synchronize_device()
        epoch_started_at = time.perf_counter()
        train_loss, train_accuracy, _ = cv_train.base_train.train_one_epoch(
            epoch=epoch,
            num_epochs=EPOCHS_PER_FOLD,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            classification_threshold=CLASSIFICATION_THRESHOLD,
            positive_class_index=positive_class_index,
        )
        val_metrics, _, _, _ = cv_train.base_train.validate_one_epoch(
            epoch=epoch,
            num_epochs=EPOCHS_PER_FOLD,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            num_classes=num_classes,
            classification_threshold=CLASSIFICATION_THRESHOLD,
            positive_class_index=positive_class_index,
        )
        _synchronize_device()
        epoch_time_seconds = time.perf_counter() - epoch_started_at

        if float(optimizer.param_groups[0]["lr"]) != fixed_lr:
            raise RuntimeError("Learning rate changed during a fixed-LR run.")
        record: dict[str, float | int] = {
            "learning_rate": learning_rate,
            "fold": fold,
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "f1_score": val_metrics["f1_score"],
            "sensitivity": val_metrics["sensitivity"],
            "specificity": val_metrics["specificity"],
            "precision": val_metrics["precision"],
            "roc_auc": val_metrics["auc"],
            "epoch_time_seconds": epoch_time_seconds,
        }
        epoch_records.append(record)
        if _is_better_epoch(record, best_record):
            best_record = dict(record)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS_PER_FOLD} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"ROC-AUC={val_metrics['auc']:.4f}"
        )

    training_seconds = time.perf_counter() - experiment_started_at
    history = pd.DataFrame(epoch_records, columns=EPOCH_LOG_COLUMNS)
    if best_record is None or history.empty:
        raise RuntimeError("No finite ROC-AUC was produced for this fold.")

    result: dict[str, object] = {
        "learning_rate": learning_rate,
        "fold": fold,
        "status": "completed",
        "completed_epochs": len(history),
        "best_epoch": int(best_record["epoch"]),
        "roc_auc": float(best_record["roc_auc"]),
        "val_loss": float(best_record["val_loss"]),
        "accuracy": float(best_record["accuracy"]),
        "f1_score": float(best_record["f1_score"]),
        "sensitivity": float(best_record["sensitivity"]),
        "specificity": float(best_record["specificity"]),
        "precision": float(best_record["precision"]),
        "final_train_loss": float(history.iloc[-1]["train_loss"]),
        "final_val_loss": float(history.iloc[-1]["val_loss"]),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_patients": train_dataset.metadata["cv_group_id"].nunique(),
        "val_patients": val_dataset.metadata["cv_group_id"].nunique(),
        "training_seconds": training_seconds,
        "failure_message": None,
    }
    return result, history


def build_failed_result(
    learning_rate: float,
    fold: int,
    status: str,
    message: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build a schema-compatible record for an expected numeric failure."""

    result: dict[str, object] = {
        column: None for column in FOLD_RESULT_COLUMNS
    }
    result.update(
        {
            "learning_rate": learning_rate,
            "fold": fold,
            "status": status,
            "completed_epochs": 0,
            "training_seconds": elapsed_seconds,
            "failure_message": message,
        }
    )
    return result


def summarize_learning_rates(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate mean and sample std across completed folds for each LR."""

    rows: list[dict[str, object]] = []
    for learning_rate in LEARNING_RATES:
        candidate = fold_results[
            fold_results["learning_rate"].eq(learning_rate)
        ]
        completed = candidate[candidate["status"].eq("completed")]
        row: dict[str, object] = {
            "learning_rate": learning_rate,
            "num_completed_folds": len(completed),
            "num_failed_folds": len(candidate) - len(completed),
            "eligible_for_ranking": len(completed) == len(CV_FOLDS),
            "rank_by_mean_roc_auc": None,
        }
        for metric in VALIDATION_METRICS:
            values = pd.to_numeric(completed[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = (
                float(values.mean()) if not values.empty else None
            )
            row[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("learning_rate")
    eligible = summary[summary["eligible_for_ranking"]].copy()
    eligible = eligible[
        pd.to_numeric(eligible["mean_roc_auc"], errors="coerce").notna()
    ]
    if not eligible.empty:
        ranked_indices = eligible.sort_values(
            ["mean_roc_auc", "mean_val_loss"],
            ascending=[False, True],
        ).index
        for rank, index in enumerate(ranked_indices, start=1):
            summary.loc[index, "rank_by_mean_roc_auc"] = rank
    summary["rank_by_mean_roc_auc"] = summary[
        "rank_by_mean_roc_auc"
    ].astype("Int64")
    return summary.reset_index(drop=True)


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    """Return fully completed candidates ordered by primary metric."""

    return summary[
        summary["eligible_for_ranking"]
        & summary["rank_by_mean_roc_auc"].notna()
    ].sort_values("rank_by_mean_roc_auc").reset_index(drop=True)


def derive_stage2_region(ranking: pd.DataFrame) -> dict[str, object]:
    """Describe a transparent error-bar-overlap region around the best LR."""

    if ranking.empty:
        return {
            "best_learning_rate": None,
            "immediate_lower_lr": None,
            "immediate_upper_lr": None,
            "competitive_learning_rates": [],
            "candidate_region_lower": None,
            "candidate_region_upper": None,
            "heuristic": (
                "Unavailable because no LR completed all five folds."
            ),
        }

    best = ranking.iloc[0]
    best_lr = float(best["learning_rate"])
    best_mean = float(best["mean_roc_auc"])
    best_std = float(best["std_roc_auc"])
    best_lower = best_mean - best_std
    best_upper = best_mean + best_std

    ordered = ranking.sort_values("learning_rate").reset_index(drop=True)
    ordered_lrs = ordered["learning_rate"].astype(float).tolist()
    best_index = ordered_lrs.index(best_lr)
    immediate_lower = ordered_lrs[best_index - 1] if best_index > 0 else None
    immediate_upper = (
        ordered_lrs[best_index + 1]
        if best_index + 1 < len(ordered_lrs)
        else None
    )

    competitive_flags = []
    for _, row in ordered.iterrows():
        mean_auc = float(row["mean_roc_auc"])
        std_auc = float(row["std_roc_auc"])
        candidate_lower = mean_auc - std_auc
        candidate_upper = mean_auc + std_auc
        competitive_flags.append(
            candidate_upper >= best_lower and candidate_lower <= best_upper
        )

    lower_index = best_index
    while lower_index > 0 and competitive_flags[lower_index - 1]:
        lower_index -= 1
    upper_index = best_index
    while (
        upper_index + 1 < len(ordered_lrs)
        and competitive_flags[upper_index + 1]
    ):
        upper_index += 1
    competitive_lrs = [
        learning_rate
        for learning_rate, is_competitive in zip(
            ordered_lrs,
            competitive_flags,
            strict=True,
        )
        if is_competitive
    ]
    return {
        "best_learning_rate": best_lr,
        "best_mean_roc_auc": best_mean,
        "best_std_roc_auc": best_std,
        "immediate_lower_lr": immediate_lower,
        "immediate_upper_lr": immediate_upper,
        "competitive_learning_rates": competitive_lrs,
        "candidate_region_lower": ordered_lrs[lower_index],
        "candidate_region_upper": ordered_lrs[upper_index],
        "heuristic": (
            "Contiguous LR values around the best mean ROC-AUC whose "
            "mean ± fold standard-deviation interval overlaps the best "
            "candidate interval. This is a Stage 2 search aid, not a "
            "statistical guarantee or final LR selection."
        ),
    }


def build_config(metadata: pd.DataFrame) -> dict[str, object]:
    """Build the complete reproducibility snapshot for Stage 1."""

    base_cv_config = cv_train.build_cv_config(metadata)
    return {
        "experiment": {
            "type": "stage_1_coarse_learning_rate_search",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "output_directory": str(OUTPUT_DIR),
            "primary_decision_metric": "mean_5fold_roc_auc",
            "purpose": (
                "Identify a reasonable LR region for a narrower Stage 2 "
                "search; this is not final model training."
            ),
        },
        "search": {
            "learning_rates": LEARNING_RATES,
            "scale": "logarithmic",
            "num_candidates": len(LEARNING_RATES),
            "folds_per_candidate": len(CV_FOLDS),
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "total_fold_experiments": len(LEARNING_RATES) * len(CV_FOLDS),
            "fold_selection_rule": (
                "maximum validation ROC-AUC; lower validation loss breaks ties"
            ),
            "scheduler": None,
            "early_stopping": None,
            "model_saving": None,
            "checkpoint": None,
        },
        "fixed_configuration": {
            "batch_size": BATCH_SIZE,
            "optimizer": "SGD",
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "nesterov": NESTEROV,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "loss": "CrossEntropyLoss",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "seed": SEED,
            "device": str(DEVICE),
        },
        "cross_validation": base_cv_config["cross_validation"],
        "data": base_cv_config["data"],
        "model": {
            **base_cv_config["model"],
            "classifier_dropout": CLASSIFIER_DROPOUT,
        },
        "runtime": {
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


def _json_safe(value: Any) -> Any:
    """Convert pandas/NumPy scalar values and non-finite floats for JSON."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def save_json(data: dict[str, object], path: Path) -> None:
    """Save strict human-readable JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(data), file, indent=4, allow_nan=False)
        file.write("\n")


def save_partial_results(
    fold_records: list[dict[str, object]],
    epoch_records: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist completed work after every fold experiment."""

    fold_frame = pd.DataFrame(
        fold_records,
        columns=FOLD_RESULT_COLUMNS,
    )
    fold_frame.to_csv(FOLD_RESULTS_PATH, index=False, float_format="%.12g")
    epoch_frame = (
        pd.concat(epoch_records, ignore_index=True)
        if epoch_records
        else pd.DataFrame(columns=EPOCH_LOG_COLUMNS)
    )
    epoch_frame.to_csv(EPOCH_LOG_PATH, index=False, float_format="%.12g")
    return fold_frame, epoch_frame


def plot_search_results(summary: pd.DataFrame) -> None:
    """Create primary, loss, and compact secondary-metric figures."""

    completed = summary[summary["eligible_for_ranking"]].copy()
    completed = completed.sort_values("learning_rate")
    if completed.empty:
        return

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        completed["learning_rate"],
        completed["mean_roc_auc"],
        yerr=completed["std_roc_auc"],
        marker="o",
        capsize=4,
        linewidth=1.5,
        label="Mean ROC-AUC ± fold std",
    )
    best = completed.loc[completed["mean_roc_auc"].idxmax()]
    axis.axvline(
        best["learning_rate"],
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=f"Best mean ROC-AUC LR={best['learning_rate']:.1e}",
    )
    axis.set_xscale("log")
    axis.set_title("Stage 1 Coarse LR Search — Five-Fold ROC-AUC")
    axis.set_xlabel("Learning Rate (log scale)")
    axis.set_ylabel("Validation ROC-AUC")
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lr_vs_mean_roc_auc.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        completed["learning_rate"],
        completed["mean_val_loss"],
        yerr=completed["std_val_loss"],
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
    axis.set_xscale("log")
    axis.set_title("Stage 1 Coarse LR Search — Validation Loss")
    axis.set_xlabel("Learning Rate (log scale)")
    axis.set_ylabel("Mean Validation Loss")
    axis.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lr_vs_mean_val_loss.png", dpi=300)
    plt.close(figure)

    secondary_metrics = {
        "mean_accuracy": "Accuracy",
        "mean_f1_score": "F1",
        "mean_sensitivity": "Sensitivity",
        "mean_specificity": "Specificity",
        "mean_precision": "Precision",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for column, label in secondary_metrics.items():
        axis.plot(
            completed["learning_rate"],
            completed[column],
            marker="o",
            linewidth=1.2,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Stage 1 Coarse LR Search — Secondary Metrics")
    axis.set_xlabel("Learning Rate (log scale)")
    axis.set_ylabel("Mean Validation Metric")
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="best", ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lr_vs_secondary_metrics.png", dpi=300)
    plt.close(figure)


def print_header() -> None:
    """Print the full fixed-budget search configuration."""

    print("Stage 1 — Coarse Learning Rate Search")
    print()
    print(f"Candidates        : {len(LEARNING_RATES)}")
    print("Learning rates    : " + ", ".join(f"{lr:.1e}" for lr in LEARNING_RATES))
    print(f"CV folds          : {list(CV_FOLDS)}")
    print(f"Epochs per fold   : {EPOCHS_PER_FOLD}")
    print(f"Total experiments : {len(LEARNING_RATES) * len(CV_FOLDS)}")
    print(f"Batch size        : {BATCH_SIZE}")
    print(f"Optimizer         : SGD (momentum={MOMENTUM})")
    print(f"Weight decay      : {WEIGHT_DECAY:.1e}")
    print(f"Dropout           : {CLASSIFIER_DROPOUT}")
    print("Primary metric    : mean five-fold ROC-AUC")
    print("Model saving      : disabled")
    print()


def print_ranking(
    ranking: pd.DataFrame,
    stage2_region: dict[str, object],
) -> None:
    """Print ranking and transparent Stage 2 neighborhood information."""

    print()
    print("Stage 1 Coarse Learning Rate Ranking")
    if ranking.empty:
        print("No learning rate completed all five folds.")
        return
    print("Rank | LR      | Mean ROC-AUC ± Std | Mean Val Loss | Mean F1")
    print("-" * 72)
    for _, row in ranking.iterrows():
        print(
            f"{int(row['rank_by_mean_roc_auc']):>4} | "
            f"{row['learning_rate']:.1e} | "
            f"{row['mean_roc_auc']:.4f} ± {row['std_roc_auc']:.4f} | "
            f"{row['mean_val_loss']:.4f}        | "
            f"{row['mean_f1_score']:.4f}"
        )
    print()
    print("Stage 2 search aid")
    print(f"Best coarse LR       : {stage2_region['best_learning_rate']:.1e}")
    print(f"Immediate lower LR   : {stage2_region['immediate_lower_lr']}")
    print(f"Immediate upper LR   : {stage2_region['immediate_upper_lr']}")
    print(
        "Competitive LR values: "
        f"{stage2_region['competitive_learning_rates']}"
    )
    print(
        "Contiguous candidate region around best: "
        f"{stage2_region['candidate_region_lower']:.1e} -> "
        f"{stage2_region['candidate_region_upper']:.1e}"
    )
    print("This region is heuristic evidence for Stage 2, not a final LR.")


def main() -> None:
    """Execute nine fixed learning rates across identical five CV folds."""

    validate_configuration()
    metadata = cv_train.validate_cv_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    FIGURES_DIR.mkdir(parents=False, exist_ok=False)
    save_json(build_config(metadata), CONFIG_PATH)
    print_header()

    fold_records: list[dict[str, object]] = []
    epoch_records: list[pd.DataFrame] = []
    for learning_rate in LEARNING_RATES:
        for fold in CV_FOLDS:
            experiment_started_at = time.perf_counter()
            try:
                result, history = run_fold(learning_rate, fold)
                epoch_records.append(history)
            except torch.cuda.OutOfMemoryError as error:
                result = build_failed_result(
                    learning_rate=learning_rate,
                    fold=fold,
                    status="failed_cuda_oom",
                    message=str(error),
                    elapsed_seconds=(
                        time.perf_counter() - experiment_started_at
                    ),
                )
                print(f"CUDA OOM for LR={learning_rate:.3e}, fold={fold}")
            except RuntimeError as error:
                if "Non-finite" not in str(error):
                    raise
                result = build_failed_result(
                    learning_rate=learning_rate,
                    fold=fold,
                    status="failed_non_finite_loss",
                    message=str(error),
                    elapsed_seconds=(
                        time.perf_counter() - experiment_started_at
                    ),
                )
                print(
                    f"Non-finite loss for LR={learning_rate:.3e}, "
                    f"fold={fold}: {error}"
                )
            finally:
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            fold_records.append(result)
            save_partial_results(fold_records, epoch_records)

    fold_frame, _ = save_partial_results(fold_records, epoch_records)
    lr_summary = summarize_learning_rates(fold_frame)
    lr_summary.to_csv(LR_SUMMARY_PATH, index=False, float_format="%.12g")
    ranking = build_ranking(lr_summary)
    ranking.to_csv(LR_RANKING_PATH, index=False, float_format="%.12g")

    stage2_region = derive_stage2_region(ranking)
    search_summary = {
        "primary_decision_metric": "mean five-fold validation ROC-AUC",
        "num_learning_rates": len(LEARNING_RATES),
        "num_fold_experiments": len(fold_frame),
        "num_completed_fold_experiments": int(
            fold_frame["status"].eq("completed").sum()
        ),
        "num_ranked_learning_rates": len(ranking),
        "stage2_region": stage2_region,
        "ranking": ranking.to_dict(orient="records"),
        "model_artifacts_saved": False,
        "holdout_test_used": False,
    }
    save_json(search_summary, SEARCH_SUMMARY_PATH)
    plot_search_results(lr_summary)
    print_ranking(ranking, stage2_region)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
