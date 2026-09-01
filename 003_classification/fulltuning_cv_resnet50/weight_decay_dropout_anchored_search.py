"""Stage 3 anchored random search over SGD weight decay and dropout."""

import argparse
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import lr_coarse_search as stage1


# ---------------------------------------------------------------------------
# Experiment output
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_STARTED_AT = datetime.now().astimezone()
RUN_ID = uuid4()
RUN_SHORT_ID = RUN_ID.hex[:8]
RESULT_DIR_NAME = (
    f"weight_decay_dropout_stage3_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_"
    f"{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "fulltuning_cv_resnet50"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFIG_PATH = OUTPUT_DIR / "stage3_config.json"
TRIALS_PATH = OUTPUT_DIR / "parameter_trials.csv"
EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_results.csv"
FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
TRIAL_SUMMARY_PATH = OUTPUT_DIR / "trial_summary.csv"
RANKING_PATH = OUTPUT_DIR / "trial_ranking.csv"
GPU_MEMORY_LOG_PATH = OUTPUT_DIR / "gpu_memory_log.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "stage3_summary.json"
SUMMARY_MARKDOWN_PATH = OUTPUT_DIR / "stage3_summary.md"


# ---------------------------------------------------------------------------
# Fixed Stage 3 configuration
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
MOMENTUM = 0.959
BATCH_SIZE = 64
NESTEROV = False

WEIGHT_DECAY_MIN = 1e-6
WEIGHT_DECAY_MAX = 1e-2
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.5
BASELINE_WEIGHT_DECAY = 1e-4
BASELINE_DROPOUT = 0.2
ANCHOR_WEIGHT_DECAYS = (3e-5, 1e-4, 3e-4)
ANCHOR_DROPOUTS = (0.1, 0.2, 0.3)
RANDOM_TRIALS = 9
RANDOM_SEARCH_SEED = 42
DROPOUT_DECIMAL_PLACES = 3

EPOCHS_PER_FOLD = stage1.EPOCHS_PER_FOLD
SEED = 42
CV_FOLDS = tuple(range(5))
PRIMARY_METRIC = "roc_auc"
EXPECTED_NUM_TRIALS = (
    len(ANCHOR_WEIGHT_DECAYS) * len(ANCHOR_DROPOUTS) + RANDOM_TRIALS
)

DEVICE = stage1.DEVICE
CLASSIFICATION_THRESHOLD = stage1.CLASSIFICATION_THRESHOLD

EPOCH_RESULT_COLUMNS = (
    "trial_id",
    "is_anchor",
    "sampling_method",
    "weight_decay",
    "dropout",
    "fold",
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "train_val_loss_gap",
    "positive_train_val_loss_gap",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "roc_auc",
    "epoch_time_seconds",
)

FOLD_RESULT_COLUMNS = (
    "trial_id",
    "is_anchor",
    "sampling_method",
    "weight_decay",
    "dropout",
    "fold",
    "status",
    "completed_epochs",
    "best_epoch",
    "roc_auc",
    "val_loss",
    "train_loss_at_best_epoch",
    "train_val_loss_gap",
    "positive_train_val_loss_gap",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "final_train_loss",
    "final_val_loss",
    "final_train_val_loss_gap",
    "train_samples",
    "val_samples",
    "train_patients",
    "val_patients",
    "training_seconds",
    "failure_message",
)

GPU_MEMORY_COLUMNS = (
    "trial_id",
    "phase",
    "device",
    "allocated_before_bytes",
    "reserved_before_bytes",
    "allocated_after_bytes",
    "reserved_after_bytes",
    "allocated_released_bytes",
    "reserved_released_bytes",
    "cleanup_time_seconds",
)

AGGREGATE_METRICS = (
    "roc_auc",
    "val_loss",
    "train_loss_at_best_epoch",
    "train_val_loss_gap",
    "positive_train_val_loss_gap",
    "accuracy",
    "f1_score",
    "sensitivity",
    "specificity",
    "precision",
    "best_epoch",
)


def configure_output_directory(output_directory: Path) -> None:
    """Point every output path to a fresh or resumed Stage 3 directory."""

    global OUTPUT_DIR
    global FIGURES_DIR
    global CONFIG_PATH
    global TRIALS_PATH
    global EPOCH_LOG_PATH
    global FOLD_RESULTS_PATH
    global TRIAL_SUMMARY_PATH
    global RANKING_PATH
    global GPU_MEMORY_LOG_PATH
    global SUMMARY_JSON_PATH
    global SUMMARY_MARKDOWN_PATH

    OUTPUT_DIR = output_directory.resolve()
    FIGURES_DIR = OUTPUT_DIR / "figures"
    CONFIG_PATH = OUTPUT_DIR / "stage3_config.json"
    TRIALS_PATH = OUTPUT_DIR / "parameter_trials.csv"
    EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_results.csv"
    FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
    TRIAL_SUMMARY_PATH = OUTPUT_DIR / "trial_summary.csv"
    RANKING_PATH = OUTPUT_DIR / "trial_ranking.csv"
    GPU_MEMORY_LOG_PATH = OUTPUT_DIR / "gpu_memory_log.csv"
    SUMMARY_JSON_PATH = OUTPUT_DIR / "stage3_summary.json"
    SUMMARY_MARKDOWN_PATH = OUTPUT_DIR / "stage3_summary.md"


def parse_arguments() -> argparse.Namespace:
    """Parse fresh-run and trial-level resume options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume Stage 3 weight-decay/dropout tuning. "
            "An incomplete trial always restarts from fold 0."
        )
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIRECTORY",
        help=(
            "Resume an existing weight_decay_dropout_stage3_* directory. "
            "Only trials with all five folds completed are skipped."
        ),
    )
    return parser.parse_args()


def validate_configuration() -> None:
    """Validate Stage 3 fixed values and search bounds."""

    if not math.isclose(LEARNING_RATE, 3e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Stage 3 requires learning rate 3e-4.")
    if not math.isclose(MOMENTUM, 0.959, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("Stage 3 requires momentum 0.959.")
    if BATCH_SIZE != 64:
        raise ValueError("Stage 3 requires batch size 64.")
    if NESTEROV:
        raise ValueError("Stage 3 requires Nesterov=False.")
    if not 0.0 < WEIGHT_DECAY_MIN < WEIGHT_DECAY_MAX:
        raise ValueError("Weight-decay bounds are invalid.")
    if not 0.0 <= DROPOUT_MIN < DROPOUT_MAX < 1.0:
        raise ValueError("Dropout bounds are invalid.")
    if not WEIGHT_DECAY_MIN <= BASELINE_WEIGHT_DECAY <= WEIGHT_DECAY_MAX:
        raise ValueError("Baseline weight decay is outside the search range.")
    if not DROPOUT_MIN <= BASELINE_DROPOUT <= DROPOUT_MAX:
        raise ValueError("Baseline dropout is outside the search range.")
    if RANDOM_TRIALS <= 0 or EPOCHS_PER_FOLD <= 0:
        raise ValueError("Trial and epoch counts must be positive.")
    if CV_FOLDS != tuple(range(5)):
        raise ValueError("Stage 3 requires folds 0 through 4.")
    if stage1.BATCH_SIZE != BATCH_SIZE:
        raise RuntimeError("Imported DataLoader batch size differs from Stage 3.")
    if stage1.CV_FOLDS != CV_FOLDS:
        raise RuntimeError("Imported CV folds differ from Stage 3.")


def generate_parameter_trials() -> list[dict[str, object]]:
    """Create a local anchor grid plus reproducible random combinations."""

    anchor_pairs = [
        (BASELINE_WEIGHT_DECAY, BASELINE_DROPOUT),
        *[
            (weight_decay, dropout)
            for weight_decay in ANCHOR_WEIGHT_DECAYS
            for dropout in ANCHOR_DROPOUTS
            if (weight_decay, dropout)
            != (BASELINE_WEIGHT_DECAY, BASELINE_DROPOUT)
        ],
    ]
    trials: list[dict[str, object]] = [
        {
            "trial_id": "",
            "is_anchor": True,
            "sampling_method": (
                "baseline_anchor"
                if pair == (BASELINE_WEIGHT_DECAY, BASELINE_DROPOUT)
                else "local_grid_anchor"
            ),
            "weight_decay": pair[0],
            "dropout": pair[1],
        }
        for pair in anchor_pairs
    ]

    generator = random.Random(RANDOM_SEARCH_SEED)
    used_pairs = set(anchor_pairs)
    while len(trials) < EXPECTED_NUM_TRIALS:
        log_weight_decay = generator.uniform(
            math.log10(WEIGHT_DECAY_MIN),
            math.log10(WEIGHT_DECAY_MAX),
        )
        weight_decay = float(f"{10 ** log_weight_decay:.8g}")
        dropout = round(
            generator.uniform(DROPOUT_MIN, DROPOUT_MAX),
            DROPOUT_DECIMAL_PLACES,
        )
        pair = (weight_decay, dropout)
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        trials.append(
            {
                "trial_id": "",
                "is_anchor": False,
                "sampling_method": "log_uniform_wd_uniform_dropout",
                "weight_decay": weight_decay,
                "dropout": dropout,
            }
        )

    for index, trial in enumerate(trials, start=1):
        trial["trial_id"] = f"trial_{index:02d}"

    frame = pd.DataFrame(trials)
    if len(frame) != EXPECTED_NUM_TRIALS:
        raise RuntimeError("Stage 3 generated an unexpected number of trials.")
    if frame.duplicated(["weight_decay", "dropout"]).any():
        raise RuntimeError("Duplicate Stage 3 combinations were generated.")
    if not frame["weight_decay"].between(
        WEIGHT_DECAY_MIN,
        WEIGHT_DECAY_MAX,
    ).all():
        raise RuntimeError("Generated weight decay lies outside search bounds.")
    if not frame["dropout"].between(DROPOUT_MIN, DROPOUT_MAX).all():
        raise RuntimeError("Generated dropout lies outside search bounds.")
    return trials


def build_model(num_classes: int, dropout: float) -> nn.Module:
    """Build a fresh full-tuning ResNet50 with trial-specific dropout."""

    model = stage1.cv_train.base_train.build_model(num_classes)
    if (
        not isinstance(model.fc, nn.Sequential)
        or len(model.fc) != 2
        or not isinstance(model.fc[0], nn.Dropout)
        or not isinstance(model.fc[1], nn.Linear)
    ):
        raise RuntimeError("Unexpected ResNet50 classifier structure.")
    model.fc[0] = nn.Dropout(p=dropout)
    stage1.cv_train.base_train.assert_full_model_trainable(model)
    return model


def build_optimizer(
    model: nn.Module,
    weight_decay: float,
) -> torch.optim.SGD:
    """Build fixed-LR/fixed-momentum SGD over the complete model."""

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    all_parameters = list(model.parameters())
    if len(parameters) != len(all_parameters):
        raise RuntimeError("Stage 3 requires full model fine-tuning.")
    optimizer = torch.optim.SGD(
        params=parameters,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=weight_decay,
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


def _gpu_memory_snapshot() -> tuple[int, int]:
    """Return allocated and reserved bytes for the active CUDA device."""

    if DEVICE.type != "cuda":
        return 0, 0
    return torch.cuda.memory_allocated(DEVICE), torch.cuda.memory_reserved(DEVICE)


def cleanup_gpu(trial_id: str, phase: str) -> dict[str, object]:
    """Collect Python/CUDA resources and report cache release for one trial."""

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    allocated_before, reserved_before = _gpu_memory_snapshot()
    cleanup_started_at = time.perf_counter()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError as error:
            print(f"CUDA IPC cleanup warning: {error}")
        torch.cuda.synchronize(DEVICE)
    allocated_after, reserved_after = _gpu_memory_snapshot()
    elapsed = time.perf_counter() - cleanup_started_at
    print(
        f"GPU cleanup [{phase}] {trial_id} | "
        f"allocated {allocated_before / 2**20:.1f} -> "
        f"{allocated_after / 2**20:.1f} MiB | "
        f"reserved {reserved_before / 2**20:.1f} -> "
        f"{reserved_after / 2**20:.1f} MiB"
    )
    return {
        "trial_id": trial_id,
        "phase": phase,
        "device": str(DEVICE),
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "allocated_after_bytes": allocated_after,
        "reserved_after_bytes": reserved_after,
        "allocated_released_bytes": allocated_before - allocated_after,
        "reserved_released_bytes": reserved_before - reserved_after,
        "cleanup_time_seconds": elapsed,
    }


def run_fold(
    trial: dict[str, object],
    fold: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train one fresh model for one Stage 3 trial/fold combination."""

    trial_id = str(trial["trial_id"])
    weight_decay = float(trial["weight_decay"])
    dropout = float(trial["dropout"])
    train_loader = None
    val_loader = None
    train_dataset = None
    val_dataset = None
    model = None
    optimizer = None
    criterion = None
    validation_confusion_matrix = None
    validation_targets = None
    validation_probabilities = None

    try:
        stage1.cv_train.set_seed(seed=SEED, deterministic=True)
        train_loader, val_loader, _, _ = (
            stage1.cv_train.build_fold_dataloaders(fold)
        )
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        num_classes = len(train_dataset.classes)
        positive_class_index = train_dataset.class_to_idx["malignant"]
        model = build_model(num_classes, dropout)
        optimizer = build_optimizer(model, weight_decay)
        criterion = nn.CrossEntropyLoss()

        print("=" * 80)
        print(
            f"{trial_id} | WD={weight_decay:.3e} | Dropout={dropout:.3f} | "
            f"Fold={fold}/{len(CV_FOLDS) - 1}"
        )
        print("=" * 80)

        epoch_records: list[dict[str, object]] = []
        best_record: dict[str, object] | None = None
        stage1._synchronize_device()
        fold_started_at = time.perf_counter()
        for epoch in range(EPOCHS_PER_FOLD):
            stage1._synchronize_device()
            epoch_started_at = time.perf_counter()
            train_loss, train_accuracy, _ = (
                stage1.cv_train.base_train.train_one_epoch(
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
            )
            (
                val_metrics,
                validation_confusion_matrix,
                validation_targets,
                validation_probabilities,
            ) = stage1.cv_train.base_train.validate_one_epoch(
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
            stage1._synchronize_device()
            epoch_time = time.perf_counter() - epoch_started_at
            if float(optimizer.param_groups[0]["lr"]) != LEARNING_RATE:
                raise RuntimeError("Learning rate changed during Stage 3.")
            if float(optimizer.param_groups[0]["momentum"]) != MOMENTUM:
                raise RuntimeError("Momentum changed during Stage 3.")
            if float(optimizer.param_groups[0]["weight_decay"]) != weight_decay:
                raise RuntimeError("Weight decay changed during a fixed trial.")

            loss_gap = float(val_metrics["loss"] - train_loss)
            record: dict[str, object] = {
                "trial_id": trial_id,
                "is_anchor": bool(trial["is_anchor"]),
                "sampling_method": trial["sampling_method"],
                "weight_decay": weight_decay,
                "dropout": dropout,
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_metrics["loss"],
                "train_val_loss_gap": loss_gap,
                "positive_train_val_loss_gap": max(loss_gap, 0.0),
                "accuracy": val_metrics["accuracy"],
                "f1_score": val_metrics["f1_score"],
                "sensitivity": val_metrics["sensitivity"],
                "specificity": val_metrics["specificity"],
                "precision": val_metrics["precision"],
                "roc_auc": val_metrics["auc"],
                "epoch_time_seconds": epoch_time,
            }
            epoch_records.append(record)
            if stage1._is_better_epoch(record, best_record):
                best_record = dict(record)
            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS_PER_FOLD} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"gap={loss_gap:+.4f} | ROC-AUC={val_metrics['auc']:.4f}"
            )

        training_seconds = time.perf_counter() - fold_started_at
        history = pd.DataFrame(epoch_records, columns=EPOCH_RESULT_COLUMNS)
        if history.empty or best_record is None:
            raise RuntimeError("No finite ROC-AUC was produced for this fold.")
        final_record = history.iloc[-1]
        result: dict[str, object] = {
            "trial_id": trial_id,
            "is_anchor": bool(trial["is_anchor"]),
            "sampling_method": trial["sampling_method"],
            "weight_decay": weight_decay,
            "dropout": dropout,
            "fold": fold,
            "status": "completed",
            "completed_epochs": len(history),
            "best_epoch": int(best_record["epoch"]),
            "roc_auc": float(best_record["roc_auc"]),
            "val_loss": float(best_record["val_loss"]),
            "train_loss_at_best_epoch": float(best_record["train_loss"]),
            "train_val_loss_gap": float(best_record["train_val_loss_gap"]),
            "positive_train_val_loss_gap": float(
                best_record["positive_train_val_loss_gap"]
            ),
            "accuracy": float(best_record["accuracy"]),
            "f1_score": float(best_record["f1_score"]),
            "sensitivity": float(best_record["sensitivity"]),
            "specificity": float(best_record["specificity"]),
            "precision": float(best_record["precision"]),
            "final_train_loss": float(final_record["train_loss"]),
            "final_val_loss": float(final_record["val_loss"]),
            "final_train_val_loss_gap": float(
                final_record["train_val_loss_gap"]
            ),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_patients": train_dataset.metadata[
                "cv_group_id"
            ].nunique(),
            "val_patients": val_dataset.metadata["cv_group_id"].nunique(),
            "training_seconds": training_seconds,
            "failure_message": None,
        }
        return result, history
    finally:
        validation_confusion_matrix = None
        validation_probabilities = None
        validation_targets = None
        criterion = None
        optimizer = None
        model = None
        val_dataset = None
        train_dataset = None
        val_loader = None
        train_loader = None
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def build_failed_result(
    trial: dict[str, object],
    fold: int,
    status: str,
    message: str,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build a schema-compatible result for an expected numeric failure."""

    result = {column: None for column in FOLD_RESULT_COLUMNS}
    result.update(
        {
            "trial_id": trial["trial_id"],
            "is_anchor": trial["is_anchor"],
            "sampling_method": trial["sampling_method"],
            "weight_decay": trial["weight_decay"],
            "dropout": trial["dropout"],
            "fold": fold,
            "status": status,
            "completed_epochs": 0,
            "training_seconds": elapsed_seconds,
            "failure_message": message,
        }
    )
    return result


def summarize_trials(
    trials: list[dict[str, object]],
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate validation evidence across five folds for every trial."""

    rows: list[dict[str, object]] = []
    for trial in trials:
        trial_frame = fold_results[
            fold_results["trial_id"].eq(trial["trial_id"])
        ]
        completed = trial_frame[trial_frame["status"].eq("completed")]
        row: dict[str, object] = {
            **trial,
            "num_completed_folds": len(completed),
            "num_failed_folds": len(trial_frame) - len(completed),
            "eligible_for_ranking": len(completed) == len(CV_FOLDS),
            "rank_by_mean_roc_auc": None,
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(completed[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = (
                float(values.mean()) if not values.empty else None
            )
            row[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
            row[f"min_{metric}"] = (
                float(values.min()) if not values.empty else None
            )
            row[f"max_{metric}"] = (
                float(values.max()) if not values.empty else None
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    eligible = summary[
        summary["eligible_for_ranking"]
        & pd.to_numeric(summary["mean_roc_auc"], errors="coerce").notna()
    ]
    ranked_indices = eligible.sort_values(
        [
            "mean_roc_auc",
            "std_roc_auc",
            "mean_val_loss",
            "mean_positive_train_val_loss_gap",
        ],
        ascending=[False, True, True, True],
    ).index
    for rank, index in enumerate(ranked_indices, start=1):
        summary.loc[index, "rank_by_mean_roc_auc"] = rank
    summary["rank_by_mean_roc_auc"] = summary[
        "rank_by_mean_roc_auc"
    ].astype("Int64")
    return summary.sort_values("trial_id").reset_index(drop=True)


def build_ranking(trial_summary: pd.DataFrame) -> pd.DataFrame:
    """Rank trials primarily by mean five-fold validation ROC-AUC."""

    return trial_summary[
        trial_summary["eligible_for_ranking"]
        & trial_summary["rank_by_mean_roc_auc"].notna()
    ].sort_values("rank_by_mean_roc_auc").reset_index(drop=True)


def _dominates(candidate: pd.Series, comparison: pd.Series) -> bool:
    """Return whether candidate Pareto-dominates comparison."""

    candidate_values = (
        -float(candidate["mean_roc_auc"]),
        float(candidate["std_roc_auc"]),
        float(candidate["mean_val_loss"]),
        float(candidate["mean_positive_train_val_loss_gap"]),
    )
    comparison_values = (
        -float(comparison["mean_roc_auc"]),
        float(comparison["std_roc_auc"]),
        float(comparison["mean_val_loss"]),
        float(comparison["mean_positive_train_val_loss_gap"]),
    )
    no_worse = all(
        candidate_value <= comparison_value
        for candidate_value, comparison_value in zip(
            candidate_values,
            comparison_values,
            strict=True,
        )
    )
    strictly_better = any(
        candidate_value < comparison_value
        for candidate_value, comparison_value in zip(
            candidate_values,
            comparison_values,
            strict=True,
        )
    )
    return no_worse and strictly_better


def derive_reasonable_region(ranking: pd.DataFrame) -> dict[str, object]:
    """Find competitive Pareto candidates without a weighted score."""

    if ranking.empty:
        return {
            "best_trial_id": None,
            "competitive_trial_ids": [],
            "pareto_candidate_trial_ids": [],
            "weight_decay_lower": None,
            "weight_decay_upper": None,
            "dropout_lower": None,
            "dropout_upper": None,
            "heuristic": "Unavailable because no trial completed all folds.",
        }

    best = ranking.iloc[0]
    best_lower = float(best["mean_roc_auc"] - best["std_roc_auc"])
    best_upper = float(best["mean_roc_auc"] + best["std_roc_auc"])
    competitive = ranking[
        (ranking["mean_roc_auc"] + ranking["std_roc_auc"] >= best_lower)
        & (ranking["mean_roc_auc"] - ranking["std_roc_auc"] <= best_upper)
    ].copy()

    pareto_indices: list[int] = []
    for candidate_index, candidate in competitive.iterrows():
        dominated = any(
            _dominates(comparison, candidate)
            for comparison_index, comparison in competitive.iterrows()
            if comparison_index != candidate_index
        )
        if not dominated:
            pareto_indices.append(candidate_index)
    pareto = competitive.loc[pareto_indices].sort_values(
        ["mean_roc_auc", "std_roc_auc"],
        ascending=[False, True],
    )
    recommended = pareto if not pareto.empty else competitive
    return {
        "best_trial_id": str(best["trial_id"]),
        "best_weight_decay": float(best["weight_decay"]),
        "best_dropout": float(best["dropout"]),
        "best_mean_roc_auc": float(best["mean_roc_auc"]),
        "best_std_roc_auc": float(best["std_roc_auc"]),
        "competitive_trial_ids": competitive["trial_id"].tolist(),
        "pareto_candidate_trial_ids": recommended["trial_id"].tolist(),
        "weight_decay_lower": float(recommended["weight_decay"].min()),
        "weight_decay_upper": float(recommended["weight_decay"].max()),
        "dropout_lower": float(recommended["dropout"].min()),
        "dropout_upper": float(recommended["dropout"].max()),
        "heuristic": (
            "Retain trials whose mean ROC-AUC ± fold std overlaps the best "
            "trial interval, then retain the Pareto frontier that maximizes "
            "mean ROC-AUC while minimizing ROC-AUC std, validation loss, and "
            "positive train-validation loss gap. No weighted score is used."
        ),
    }


def _describe_trial(row: pd.Series) -> dict[str, object]:
    """Convert one ranked row into a concise diagnostic record."""

    return {
        "trial_id": str(row["trial_id"]),
        "weight_decay": float(row["weight_decay"]),
        "dropout": float(row["dropout"]),
        "mean_roc_auc": float(row["mean_roc_auc"]),
        "std_roc_auc": float(row["std_roc_auc"]),
        "mean_val_loss": float(row["mean_val_loss"]),
        "mean_train_val_loss_gap": float(row["mean_train_val_loss_gap"]),
        "mean_positive_train_val_loss_gap": float(
            row["mean_positive_train_val_loss_gap"]
        ),
        "mean_best_epoch": float(row["mean_best_epoch"]),
        "std_best_epoch": float(row["std_best_epoch"]),
    }


def build_diagnostics(
    ranking: pd.DataFrame,
    reasonable_region: dict[str, object],
) -> dict[str, object]:
    """Identify discrimination, stability, loss, and overfitting trade-offs."""

    if ranking.empty:
        return {
            "best_trial": None,
            "most_stable_competitive_trial": None,
            "strongest_overfitting_indication": None,
            "lowest_validation_loss_trial": None,
            "reasonable_region": reasonable_region,
        }
    competitive = ranking[
        ranking["trial_id"].isin(reasonable_region["competitive_trial_ids"])
    ]
    most_stable = competitive.sort_values(
        ["std_roc_auc", "mean_roc_auc"],
        ascending=[True, False],
    ).iloc[0]
    overfitting = ranking.sort_values(
        "mean_positive_train_val_loss_gap",
        ascending=False,
    ).iloc[0]
    lowest_loss = ranking.sort_values(
        ["mean_val_loss", "mean_roc_auc"],
        ascending=[True, False],
    ).iloc[0]
    return {
        "best_trial": _describe_trial(ranking.iloc[0]),
        "most_stable_competitive_trial": _describe_trial(most_stable),
        "strongest_overfitting_indication": _describe_trial(overfitting),
        "lowest_validation_loss_trial": _describe_trial(lowest_loss),
        "reasonable_region": reasonable_region,
    }


def summarize_by_parameter(ranking: pd.DataFrame) -> dict[str, object]:
    """Summarize anchor-grid tendencies separately by WD and dropout."""

    if ranking.empty:
        return {"by_weight_decay": [], "by_dropout": []}
    anchors = ranking[ranking["is_anchor"].astype(bool)]
    by_weight_decay = (
        anchors.groupby("weight_decay", as_index=False)
        .agg(
            mean_trial_roc_auc=("mean_roc_auc", "mean"),
            std_across_trial_means=("mean_roc_auc", "std"),
            mean_validation_loss=("mean_val_loss", "mean"),
            mean_positive_loss_gap=(
                "mean_positive_train_val_loss_gap",
                "mean",
            ),
            best_trial_mean_roc_auc=("mean_roc_auc", "max"),
            num_trials=("trial_id", "size"),
        )
        .sort_values("weight_decay")
    )
    by_dropout = (
        anchors.groupby("dropout", as_index=False)
        .agg(
            mean_trial_roc_auc=("mean_roc_auc", "mean"),
            std_across_trial_means=("mean_roc_auc", "std"),
            mean_validation_loss=("mean_val_loss", "mean"),
            mean_positive_loss_gap=(
                "mean_positive_train_val_loss_gap",
                "mean",
            ),
            best_trial_mean_roc_auc=("mean_roc_auc", "max"),
            num_trials=("trial_id", "size"),
        )
        .sort_values("dropout")
    )
    return {
        "by_weight_decay": by_weight_decay.to_dict(orient="records"),
        "by_dropout": by_dropout.to_dict(orient="records"),
    }


def build_config(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Build a complete Stage 3 reproducibility snapshot."""

    stage1_config = stage1.build_config(metadata)
    return {
        "experiment": {
            "type": "stage_3_anchored_random_weight_decay_dropout_search",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "output_directory": str(OUTPUT_DIR),
            "purpose": (
                "Identify a reasonable weight-decay/dropout region for final "
                "joint tuning, not final hyperparameters or a final model."
            ),
            "primary_decision_metric": "mean_5fold_validation_roc_auc",
        },
        "search": {
            "method": "anchored_random_search",
            "weight_decay_distribution": "log_uniform",
            "weight_decay_min": WEIGHT_DECAY_MIN,
            "weight_decay_max": WEIGHT_DECAY_MAX,
            "dropout_distribution": "uniform_linear",
            "dropout_min": DROPOUT_MIN,
            "dropout_max": DROPOUT_MAX,
            "baseline_weight_decay": BASELINE_WEIGHT_DECAY,
            "baseline_dropout": BASELINE_DROPOUT,
            "anchor_weight_decays": list(ANCHOR_WEIGHT_DECAYS),
            "anchor_dropouts": list(ANCHOR_DROPOUTS),
            "random_trials": RANDOM_TRIALS,
            "random_search_seed": RANDOM_SEARCH_SEED,
            "trials": trials,
            "num_parameter_combinations": len(trials),
            "folds_per_combination": len(CV_FOLDS),
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "total_fold_experiments": len(trials) * len(CV_FOLDS),
            "fold_selection_rule": (
                "maximum validation ROC-AUC; lower validation loss breaks ties"
            ),
        },
        "fixed_configuration": {
            "batch_size": BATCH_SIZE,
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "nesterov": NESTEROV,
            "loss": "CrossEntropyLoss",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "seed_reset_before_each_fold": SEED,
        },
        "cross_validation": stage1_config["cross_validation"],
        "data": stage1_config["data"],
        "model": {
            **stage1_config["model"],
            "classifier_dropout": "trial_specific",
        },
        "gpu_cleanup": {
            "between_trials": True,
            "python_garbage_collection": True,
            "cuda_empty_cache": True,
            "cuda_ipc_collect": True,
            "memory_logging": True,
            "process_isolation": False,
            "note": (
                "Each fold uses a fresh model/optimizer/DataLoader. Explicit "
                "cleanup runs per fold and before/after every trial."
            ),
        },
        "scheduler": None,
        "early_stopping": None,
        "checkpoint": None,
        "model_saving": None,
        "holdout_test_used": False,
        "runtime": stage1_config["runtime"],
    }


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so interruption cannot truncate the last state."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False, float_format="%.12g")
    temporary_path.replace(path)


def save_partial_results(
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
    gpu_memory_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Persist recoverable experiment state after every fold/trial."""

    fold_frame = fold_frame.loc[:, list(FOLD_RESULT_COLUMNS)].copy()
    epoch_frame = epoch_frame.loc[:, list(EPOCH_RESULT_COLUMNS)].copy()
    gpu_memory_frame = gpu_memory_frame.loc[:, list(GPU_MEMORY_COLUMNS)].copy()
    _atomic_write_csv(epoch_frame, EPOCH_LOG_PATH)
    _atomic_write_csv(fold_frame, FOLD_RESULTS_PATH)
    _atomic_write_csv(gpu_memory_frame, GPU_MEMORY_LOG_PATH)
    return fold_frame, epoch_frame, gpu_memory_frame


def _read_result_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a persisted CSV while enforcing its expected schema."""

    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(f"Resume file is empty: {path}") from error
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Resume file {path} is missing columns: {missing}")
    return frame.loc[:, list(columns)].copy()


def _validate_saved_trials(trials: list[dict[str, object]]) -> None:
    """Ensure a resumed directory uses the exact current search space."""

    if not TRIALS_PATH.is_file():
        raise FileNotFoundError(f"Trial definition not found: {TRIALS_PATH}")
    saved = pd.read_csv(TRIALS_PATH)
    current = pd.DataFrame(trials)
    missing = sorted(set(current.columns) - set(saved.columns))
    if missing:
        raise RuntimeError(f"Saved trial definition lacks columns: {missing}")
    saved = saved.loc[:, current.columns].reset_index(drop=True)
    if len(saved) != len(current):
        raise RuntimeError("Resume refused: trial counts differ.")
    for column in ("weight_decay", "dropout"):
        if not np.allclose(
            pd.to_numeric(saved[column]),
            pd.to_numeric(current[column]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"Resume refused: saved {column} values have changed."
            )
    for column in ("trial_id", "is_anchor", "sampling_method"):
        if saved[column].astype(str).tolist() != current[column].astype(str).tolist():
            raise RuntimeError(
                f"Resume refused: saved {column} values have changed."
            )


def _validate_resume_config(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> None:
    """Reject mixing results from different data/model/search settings."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    current = build_config(trials, metadata)
    sections = (
        "search",
        "fixed_configuration",
        "cross_validation",
        "data",
        "model",
        "scheduler",
        "early_stopping",
        "checkpoint",
        "model_saving",
        "holdout_test_used",
    )
    for section in sections:
        if saved.get(section) != current.get(section):
            raise RuntimeError(
                "Resume refused because saved and current "
                f"'{section}' configurations differ."
            )


def _validate_result_keys(
    trials: list[dict[str, object]],
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate keys and discard epoch rows without a committed fold."""

    valid_ids = {str(trial["trial_id"]) for trial in trials}
    for frame, name in (
        (fold_frame, "fold results"),
        (epoch_frame, "epoch results"),
    ):
        if not set(frame["trial_id"].astype(str)).issubset(valid_ids):
            raise RuntimeError(f"Resume {name} contain an unknown trial_id.")
        folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
        if not folds.issubset(set(CV_FOLDS)):
            raise RuntimeError(f"Resume {name} contain an invalid fold.")
    if fold_frame.duplicated(["trial_id", "fold"]).any():
        raise RuntimeError("Resume fold results contain duplicate keys.")
    if epoch_frame.duplicated(["trial_id", "fold", "epoch"]).any():
        raise RuntimeError("Resume epoch results contain duplicate keys.")

    committed_keys = {
        (str(row.trial_id), int(row.fold))
        for row in fold_frame.itertuples(index=False)
    }
    if not epoch_frame.empty:
        epoch_keys = pd.Series(
            list(
                zip(
                    epoch_frame["trial_id"].astype(str),
                    pd.to_numeric(epoch_frame["fold"]).astype(int),
                )
            ),
            index=epoch_frame.index,
        )
        orphan_mask = ~epoch_keys.isin(committed_keys)
        if orphan_mask.any():
            print(
                "Resume recovery: discarding "
                f"{int(orphan_mask.sum())} uncommitted epoch rows."
            )
            epoch_frame = epoch_frame.loc[~orphan_mask].reset_index(drop=True)
    return fold_frame, epoch_frame


def _find_completed_trial_ids(
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
) -> set[str]:
    """Return trials with all folds and all expected epoch histories."""

    completed_ids: set[str] = set()
    expected_folds = set(CV_FOLDS)
    for trial_id, trial_folds in fold_frame.groupby("trial_id", sort=False):
        recorded_folds = set(
            pd.to_numeric(trial_folds["fold"], errors="raise").astype(int)
        )
        folds_complete = (
            len(trial_folds) == len(CV_FOLDS)
            and recorded_folds == expected_folds
            and trial_folds["status"].eq("completed").all()
            and pd.to_numeric(
                trial_folds["completed_epochs"],
                errors="coerce",
            ).eq(EPOCHS_PER_FOLD).all()
        )
        if not folds_complete:
            continue
        history = epoch_frame[epoch_frame["trial_id"].eq(trial_id)]
        per_fold_counts = history.groupby("fold").size().to_dict()
        if all(per_fold_counts.get(fold) == EPOCHS_PER_FOLD for fold in CV_FOLDS):
            completed_ids.add(str(trial_id))
    return completed_ids


def load_resume_state(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    """Keep complete trials and reset every partial trial to fold 0."""

    _validate_saved_trials(trials)
    _validate_resume_config(trials, metadata)
    fold_frame = _read_result_csv(FOLD_RESULTS_PATH, FOLD_RESULT_COLUMNS)
    epoch_frame = _read_result_csv(EPOCH_LOG_PATH, EPOCH_RESULT_COLUMNS)
    gpu_frame = _read_result_csv(GPU_MEMORY_LOG_PATH, GPU_MEMORY_COLUMNS)
    valid_trial_ids = {str(trial["trial_id"]) for trial in trials}
    if not set(gpu_frame["trial_id"].astype(str)).issubset(valid_trial_ids):
        raise RuntimeError("GPU memory log contains an unknown trial_id.")
    fold_frame, epoch_frame = _validate_result_keys(
        trials,
        fold_frame,
        epoch_frame,
    )
    completed_ids = _find_completed_trial_ids(fold_frame, epoch_frame)
    recorded_ids = set(fold_frame["trial_id"].astype(str)) | set(
        epoch_frame["trial_id"].astype(str)
    ) | set(
        gpu_frame["trial_id"].astype(str)
    )
    incomplete_ids = recorded_ids - completed_ids
    if incomplete_ids:
        print(
            "Resume recovery: restarting incomplete trial(s) from fold 0: "
            f"{sorted(incomplete_ids)}"
        )
        fold_frame = fold_frame.loc[
            ~fold_frame["trial_id"].astype(str).isin(incomplete_ids)
        ].reset_index(drop=True)
        epoch_frame = epoch_frame.loc[
            ~epoch_frame["trial_id"].astype(str).isin(incomplete_ids)
        ].reset_index(drop=True)
        gpu_frame = gpu_frame.loc[
            ~gpu_frame["trial_id"].astype(str).isin(incomplete_ids)
        ].reset_index(drop=True)
    return fold_frame, epoch_frame, gpu_frame, completed_ids


def _highlight_best_parameter_point(
    axis: plt.Axes,
    best: pd.Series,
) -> None:
    """Highlight the primary best trial on WD/dropout parameter plots."""

    axis.scatter(
        [best["weight_decay"]],
        [best["dropout"]],
        marker="*",
        s=280,
        color="gold",
        edgecolor="black",
        linewidth=1.0,
        zorder=10,
        label=f"Best: {best['trial_id']}",
    )


def plot_results(
    trial_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    reasonable_region: dict[str, object],
) -> None:
    """Create Stage 3 discrimination, stability, and trade-off figures."""

    completed = trial_summary[trial_summary["eligible_for_ranking"]].copy()
    if completed.empty or ranking.empty:
        return
    completed = completed.sort_values("trial_id").reset_index(drop=True)
    best = ranking.iloc[0]
    pareto_ids = set(reasonable_region["pareto_candidate_trial_ids"])
    trial_positions = np.arange(len(completed))
    trial_labels = completed["trial_id"].tolist()

    figure, axis = plt.subplots(figsize=(10, 7))
    scatter = axis.scatter(
        completed["weight_decay"],
        completed["dropout"],
        c=completed["mean_roc_auc"],
        cmap="viridis",
        s=110,
        edgecolor="black",
        linewidth=0.5,
    )
    _highlight_best_parameter_point(axis, best)
    pareto_points = completed[completed["trial_id"].isin(pareto_ids)]
    if not pareto_points.empty:
        axis.scatter(
            pareto_points["weight_decay"],
            pareto_points["dropout"],
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            s=190,
            label="Best trade-off candidates",
        )
    for _, row in completed.iterrows():
        axis.annotate(
            row["trial_id"],
            (row["weight_decay"], row["dropout"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Weight Decay (log scale)")
    axis.set_ylabel("Classifier Dropout")
    axis.set_title("Stage 3: Weight Decay × Dropout vs Mean 5-Fold ROC-AUC")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="Mean validation ROC-AUC")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "weight_decay_dropout_vs_mean_roc_auc.png",
        dpi=300,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 6))
    axis.errorbar(
        trial_positions,
        completed["mean_roc_auc"],
        yerr=completed["std_roc_auc"],
        fmt="o",
        capsize=4,
        color="tab:blue",
        ecolor="tab:gray",
    )
    best_position = completed.index[completed["trial_id"].eq(best["trial_id"])][0]
    axis.scatter(
        [best_position],
        [best["mean_roc_auc"]],
        marker="*",
        s=260,
        color="gold",
        edgecolor="black",
        zorder=10,
        label=f"Best: {best['trial_id']}",
    )
    axis.set_xticks(trial_positions, trial_labels, rotation=45, ha="right")
    axis.set_xlabel("Stage 3 Trial")
    axis.set_ylabel("Mean Validation ROC-AUC ± Fold Std")
    axis.set_title("Stage 3 ROC-AUC and Cross-Fold Stability")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "roc_auc_with_fold_std.png", dpi=300)
    plt.close(figure)

    for column, ylabel, title, filename, color in (
        (
            "mean_val_loss",
            "Mean Validation Loss",
            "Stage 3 Validation Loss by Trial",
            "validation_loss.png",
            "tab:orange",
        ),
        (
            "mean_train_val_loss_gap",
            "Mean Validation Loss - Training Loss",
            "Stage 3 Train-Validation Loss Gap",
            "train_validation_loss_gap.png",
            "tab:red",
        ),
    ):
        figure, axis = plt.subplots(figsize=(13, 6))
        axis.bar(trial_positions, completed[column], color=color, alpha=0.8)
        axis.scatter(
            [best_position],
            [completed.loc[best_position, column]],
            marker="*",
            s=220,
            color="gold",
            edgecolor="black",
            zorder=10,
            label=f"Best ROC-AUC: {best['trial_id']}",
        )
        axis.set_xticks(trial_positions, trial_labels, rotation=45, ha="right")
        axis.set_xlabel("Stage 3 Trial")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    point_sizes = 70 + 55 * (
        np.log10(completed["weight_decay"]) - math.log10(WEIGHT_DECAY_MIN)
    )
    scatter = axis.scatter(
        completed["mean_val_loss"],
        completed["mean_roc_auc"],
        c=completed["dropout"],
        s=point_sizes,
        cmap="plasma",
        edgecolor="black",
        alpha=0.85,
    )
    axis.scatter(
        [best["mean_val_loss"]],
        [best["mean_roc_auc"]],
        marker="*",
        s=280,
        color="gold",
        edgecolor="black",
        label=f"Best: {best['trial_id']}",
        zorder=10,
    )
    for _, row in completed[completed["trial_id"].isin(pareto_ids)].iterrows():
        axis.annotate(
            row["trial_id"],
            (row["mean_val_loss"], row["mean_roc_auc"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean Validation Loss (lower is better)")
    axis.set_ylabel("Mean Validation ROC-AUC (higher is better)")
    axis.set_title("Stage 3 Discrimination vs Validation-Loss Trade-off")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.colorbar(scatter, ax=axis, label="Dropout")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "roc_auc_vs_validation_loss.png", dpi=300)
    plt.close(figure)

    anchors = completed[completed["is_anchor"].astype(bool)].copy()
    heatmap = anchors.pivot(
        index="dropout",
        columns="weight_decay",
        values="mean_roc_auc",
    ).sort_index(ascending=False)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(heatmap.to_numpy(), cmap="viridis", aspect="auto")
    axis.set_xticks(
        np.arange(len(heatmap.columns)),
        [f"{value:.0e}" for value in heatmap.columns],
    )
    axis.set_yticks(
        np.arange(len(heatmap.index)),
        [f"{value:.1f}" for value in heatmap.index],
    )
    for row_index in range(len(heatmap.index)):
        for column_index in range(len(heatmap.columns)):
            value = heatmap.iloc[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value < heatmap.stack().median() else "black",
                fontsize=9,
            )
    axis.set_xlabel("Weight Decay")
    axis.set_ylabel("Dropout")
    axis.set_title("Anchor Grid Heatmap: Mean 5-Fold ROC-AUC")
    figure.colorbar(image, ax=axis, label="Mean validation ROC-AUC")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "weight_decay_dropout_roc_auc_heatmap.png",
        dpi=300,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 7))
    for metric, label in (
        ("mean_accuracy", "Accuracy"),
        ("mean_f1_score", "F1"),
        ("mean_sensitivity", "Sensitivity"),
        ("mean_specificity", "Specificity"),
        ("mean_precision", "Precision"),
    ):
        axis.plot(
            trial_positions,
            completed[metric],
            marker="o",
            linewidth=1.4,
            label=label,
        )
    axis.axvline(best_position, color="goldenrod", linestyle="--", alpha=0.8)
    axis.set_xticks(trial_positions, trial_labels, rotation=45, ha="right")
    axis.set_xlabel("Stage 3 Trial")
    axis.set_ylabel("Mean Validation Metric")
    axis.set_title("Stage 3 Secondary Validation Metrics")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "secondary_metrics.png", dpi=300)
    plt.close(figure)


def build_summary_markdown(
    ranking: pd.DataFrame,
    diagnostics: dict[str, object],
    parameter_comparison: dict[str, object],
) -> str:
    """Build a human-readable Stage 3 decision report."""

    lines = [
        "# Stage 3 Weight Decay + Dropout Search",
        "",
        "## Decision principle",
        "",
        "Primary ranking uses mean five-fold validation ROC-AUC. ROC-AUC ",
        "fold standard deviation, validation loss, and positive train-validation ",
        "loss gap are diagnostic guardrails. This search identifies a reasonable ",
        "region for final joint tuning; it does not prove final hyperparameters.",
        "",
        "## Final ranking",
        "",
        "| Rank | Trial | Weight Decay | Dropout | ROC-AUC mean ± std | "
        "Val Loss | Positive Gap | Mean Best Epoch |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row['rank_by_mean_roc_auc'])} | {row['trial_id']} | "
            f"{row['weight_decay']:.3e} | {row['dropout']:.3f} | "
            f"{row['mean_roc_auc']:.4f} ± {row['std_roc_auc']:.4f} | "
            f"{row['mean_val_loss']:.4f} | "
            f"{row['mean_positive_train_val_loss_gap']:.4f} | "
            f"{row['mean_best_epoch']:.1f} |"
        )

    lines.extend(["", "## Diagnostic configurations", ""])
    diagnostic_labels = (
        ("best_trial", "Best mean ROC-AUC"),
        ("most_stable_competitive_trial", "Most stable competitive trial"),
        ("lowest_validation_loss_trial", "Lowest validation loss"),
        (
            "strongest_overfitting_indication",
            "Strongest overfitting indication",
        ),
    )
    for key, label in diagnostic_labels:
        value = diagnostics.get(key)
        if value is None:
            lines.append(f"- {label}: unavailable")
            continue
        lines.append(
            f"- {label}: {value['trial_id']} "
            f"(WD={value['weight_decay']:.3e}, "
            f"dropout={value['dropout']:.3f}, "
            f"ROC-AUC={value['mean_roc_auc']:.4f} ± "
            f"{value['std_roc_auc']:.4f}, "
            f"val loss={value['mean_val_loss']:.4f}, "
            f"positive gap={value['mean_positive_train_val_loss_gap']:.4f})"
        )

    region = diagnostics["reasonable_region"]
    lines.extend(
        [
            "",
            "## Reasonable region for final joint tuning",
            "",
            f"- Candidate trials: {region['pareto_candidate_trial_ids']}",
            (
                "- Weight decay range: "
                f"{region['weight_decay_lower']} to "
                f"{region['weight_decay_upper']}"
            ),
            (
                "- Dropout range: "
                f"{region['dropout_lower']} to {region['dropout_upper']}"
            ),
            f"- Heuristic: {region['heuristic']}",
            "",
            "## Anchor-grid parameter tendencies",
            "",
            "```json",
            json.dumps(
                stage1._json_safe(parameter_comparison),
                indent=2,
                allow_nan=False,
            ),
            "```",
            "",
            "No model/checkpoint was saved and the holdout test set was not used.",
        ]
    )
    return "\n".join(lines) + "\n"


def print_header(trials: list[dict[str, object]]) -> None:
    """Print fixed conditions and every Stage 3 combination."""

    print()
    print("=" * 80)
    print("Stage 3 Weight Decay + Dropout Anchored Random Search")
    print("=" * 80)
    print(f"Trials              : {len(trials)}")
    print(f"Folds per trial     : {len(CV_FOLDS)}")
    print(f"Epochs per fold     : {EPOCHS_PER_FOLD}")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Learning rate       : {LEARNING_RATE:.3e}")
    print(f"Momentum            : {MOMENTUM:.3f}")
    print("Optimizer           : SGD")
    print(f"Nesterov            : {NESTEROV}")
    print(
        f"Weight decay range  : {WEIGHT_DECAY_MIN:.1e} -> "
        f"{WEIGHT_DECAY_MAX:.1e} (log-uniform random)"
    )
    print(
        f"Dropout range       : {DROPOUT_MIN:.1f} -> "
        f"{DROPOUT_MAX:.1f} (uniform random)"
    )
    print("Primary metric      : mean five-fold validation ROC-AUC")
    print(f"Device              : {DEVICE}")
    print("Parameter combinations:")
    for trial in trials:
        print(
            f"  {trial['trial_id']}: WD={trial['weight_decay']:.3e}, "
            f"dropout={trial['dropout']:.3f}, "
            f"method={trial['sampling_method']}"
        )
    print()


def print_ranking(
    ranking: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    """Print the primary ranking and recommended diagnostic region."""

    print()
    print("=" * 100)
    print("Stage 3 Weight Decay + Dropout Search Complete")
    print("=" * 100)
    if ranking.empty:
        print("No trial completed all five folds; ranking is unavailable.")
        return
    print(
        "Rank | Trial    | Weight Decay | Dropout | ROC-AUC mean±std | "
        "Val Loss | Gap"
    )
    print("-" * 100)
    for _, row in ranking.head(10).iterrows():
        print(
            f"{int(row['rank_by_mean_roc_auc']):>4} | "
            f"{row['trial_id']:<8} | {row['weight_decay']:.3e} | "
            f"{row['dropout']:.3f}   | {row['mean_roc_auc']:.4f}±"
            f"{row['std_roc_auc']:.4f} | {row['mean_val_loss']:.4f}   | "
            f"{row['mean_train_val_loss_gap']:+.4f}"
        )
    region = diagnostics["reasonable_region"]
    print()
    print("Reasonable Stage 3 region for final joint tuning:")
    print(f"  Trials       : {region['pareto_candidate_trial_ids']}")
    print(
        f"  Weight decay : {region['weight_decay_lower']} -> "
        f"{region['weight_decay_upper']}"
    )
    print(
        f"  Dropout      : {region['dropout_lower']} -> "
        f"{region['dropout_upper']}"
    )
    print("  This is a diagnostic region, not final hyperparameters.")


def main() -> None:
    """Execute or resume all Stage 3 trials across five-fold CV."""

    arguments = parse_arguments()
    validate_configuration()
    trials = generate_parameter_trials()
    metadata = stage1.cv_train.validate_cv_metadata()

    if arguments.resume is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        FIGURES_DIR.mkdir(parents=False, exist_ok=False)
        _atomic_write_csv(pd.DataFrame(trials), TRIALS_PATH)
        stage1.save_json(build_config(trials, metadata), CONFIG_PATH)
        fold_frame = pd.DataFrame(columns=FOLD_RESULT_COLUMNS)
        epoch_frame = pd.DataFrame(columns=EPOCH_RESULT_COLUMNS)
        gpu_frame = pd.DataFrame(columns=GPU_MEMORY_COLUMNS)
        completed_trial_ids: set[str] = set()
    else:
        resume_directory = arguments.resume
        if not resume_directory.is_absolute():
            resume_directory = Path.cwd() / resume_directory
        if not resume_directory.is_dir():
            raise NotADirectoryError(
                f"Resume run directory not found: {resume_directory}"
            )
        configure_output_directory(resume_directory)
        FIGURES_DIR.mkdir(parents=False, exist_ok=True)
        (
            fold_frame,
            epoch_frame,
            gpu_frame,
            completed_trial_ids,
        ) = load_resume_state(trials, metadata)
        fold_frame, epoch_frame, gpu_frame = save_partial_results(
            fold_frame,
            epoch_frame,
            gpu_frame,
        )

    print_header(trials)
    if arguments.resume is not None:
        print("Resume mode")
        print(f"  Directory       : {OUTPUT_DIR}")
        print(
            f"  Completed trials: {len(completed_trial_ids)}/{len(trials)}"
        )
        print(f"  Remaining trials: {len(trials) - len(completed_trial_ids)}")
        print("  Incomplete trial: restarted from fold 0")
        print()

    for trial in trials:
        trial_id = str(trial["trial_id"])
        if trial_id in completed_trial_ids:
            print(f"Skipping {trial_id}: all {len(CV_FOLDS)} folds completed.")
            continue

        pre_cleanup = cleanup_gpu(trial_id, "before_trial")
        gpu_frame = pd.concat(
            [
                gpu_frame,
                pd.DataFrame([pre_cleanup], columns=GPU_MEMORY_COLUMNS),
            ],
            ignore_index=True,
        )
        print("#" * 80)
        print(
            f"Starting {trial_id}: WD={trial['weight_decay']:.3e}, "
            f"dropout={trial['dropout']:.3f}"
        )
        print("#" * 80)

        for fold in CV_FOLDS:
            experiment_started_at = time.perf_counter()
            try:
                result, history = run_fold(trial, fold)
            except torch.cuda.OutOfMemoryError as error:
                result = build_failed_result(
                    trial=trial,
                    fold=fold,
                    status="failed_cuda_oom",
                    message=str(error),
                    elapsed_seconds=time.perf_counter() - experiment_started_at,
                )
                print(f"CUDA OOM: {trial_id}, fold={fold}: {error}")
            except RuntimeError as error:
                if "Non-finite" not in str(error):
                    raise
                result = build_failed_result(
                    trial=trial,
                    fold=fold,
                    status="failed_non_finite_loss",
                    message=str(error),
                    elapsed_seconds=time.perf_counter() - experiment_started_at,
                )
                print(f"Non-finite loss: {trial_id}, fold={fold}: {error}")

            if result["status"] == "completed":
                epoch_frame = pd.concat(
                    [epoch_frame, history],
                    ignore_index=True,
                )
            fold_frame = pd.concat(
                [
                    fold_frame,
                    pd.DataFrame([result], columns=FOLD_RESULT_COLUMNS),
                ],
                ignore_index=True,
            )
            fold_frame, epoch_frame, gpu_frame = save_partial_results(
                fold_frame,
                epoch_frame,
                gpu_frame,
            )

        post_cleanup = cleanup_gpu(trial_id, "after_trial")
        gpu_frame = pd.concat(
            [
                gpu_frame,
                pd.DataFrame([post_cleanup], columns=GPU_MEMORY_COLUMNS),
            ],
            ignore_index=True,
        )
        fold_frame, epoch_frame, gpu_frame = save_partial_results(
            fold_frame,
            epoch_frame,
            gpu_frame,
        )

    trial_summary = summarize_trials(trials, fold_frame)
    _atomic_write_csv(trial_summary, TRIAL_SUMMARY_PATH)
    ranking = build_ranking(trial_summary)
    _atomic_write_csv(ranking, RANKING_PATH)

    reasonable_region = derive_reasonable_region(ranking)
    diagnostics = build_diagnostics(ranking, reasonable_region)
    parameter_comparison = summarize_by_parameter(ranking)
    summary = {
        "primary_decision_metric": "mean five-fold validation ROC-AUC",
        "selection_guardrails": [
            "ROC-AUC fold standard deviation",
            "validation loss",
            "positive train-validation loss gap",
        ],
        "num_trials": len(trials),
        "num_fold_experiments": len(fold_frame),
        "num_completed_fold_experiments": int(
            fold_frame["status"].eq("completed").sum()
        ),
        "num_ranked_trials": len(ranking),
        "diagnostics": diagnostics,
        "parameter_comparison": parameter_comparison,
        "ranking": ranking.to_dict(orient="records"),
        "model_artifacts_saved": False,
        "holdout_test_used": False,
    }
    stage1.save_json(summary, SUMMARY_JSON_PATH)
    SUMMARY_MARKDOWN_PATH.write_text(
        build_summary_markdown(ranking, diagnostics, parameter_comparison),
        encoding="utf-8",
    )
    plot_results(trial_summary, ranking, reasonable_region)
    print_ranking(ranking, diagnostics)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
