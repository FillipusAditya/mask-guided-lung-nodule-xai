"""Stage 2 anchored random search over SGD learning rate and momentum."""

import argparse
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any
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
    f"lr_momentum_stage2_{RUN_STARTED_AT:%Y%m%d_%H%M%S}_{RUN_SHORT_ID}"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "classification_results"
    / "fulltuning_cv_resnet50"
    / RESULT_DIR_NAME
)
FIGURES_DIR = OUTPUT_DIR / "figures"
CONFIG_PATH = OUTPUT_DIR / "stage2_config.json"
TRIALS_PATH = OUTPUT_DIR / "parameter_trials.csv"
EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_results.csv"
FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
TRIAL_SUMMARY_PATH = OUTPUT_DIR / "trial_summary.csv"
RANKING_PATH = OUTPUT_DIR / "trial_ranking.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "stage2_summary.json"
SUMMARY_MARKDOWN_PATH = OUTPUT_DIR / "stage2_summary.md"


# ---------------------------------------------------------------------------
# Fixed Stage 2 configuration
# ---------------------------------------------------------------------------
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
ANCHOR_MOMENTUM = 0.90
MOMENTUM_MIN = 0.70
MOMENTUM_MAX = 0.99
RANDOM_MOMENTUM_TRIALS_PER_LR = 5
RANDOM_SEARCH_SEED = 42
MOMENTUM_DECIMAL_PLACES = 3

EPOCHS_PER_FOLD = stage1.EPOCHS_PER_FOLD
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
CLASSIFIER_DROPOUT = 0.2
NESTEROV = False
SEED = 42
CV_FOLDS = tuple(range(5))
PRIMARY_METRIC = "roc_auc"
EXPECTED_NUM_TRIALS = len(LEARNING_RATES) * (
    1 + RANDOM_MOMENTUM_TRIALS_PER_LR
)

DEVICE = stage1.DEVICE
CLASSIFICATION_THRESHOLD = stage1.CLASSIFICATION_THRESHOLD

EPOCH_RESULT_COLUMNS = (
    "trial_id",
    "is_anchor",
    "learning_rate",
    "momentum",
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
    "learning_rate",
    "momentum",
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
    """Point every output path to a fresh or resumed run directory."""

    global OUTPUT_DIR
    global FIGURES_DIR
    global CONFIG_PATH
    global TRIALS_PATH
    global EPOCH_LOG_PATH
    global FOLD_RESULTS_PATH
    global TRIAL_SUMMARY_PATH
    global RANKING_PATH
    global SUMMARY_JSON_PATH
    global SUMMARY_MARKDOWN_PATH

    OUTPUT_DIR = output_directory.resolve()
    FIGURES_DIR = OUTPUT_DIR / "figures"
    CONFIG_PATH = OUTPUT_DIR / "stage2_config.json"
    TRIALS_PATH = OUTPUT_DIR / "parameter_trials.csv"
    EPOCH_LOG_PATH = OUTPUT_DIR / "epoch_results.csv"
    FOLD_RESULTS_PATH = OUTPUT_DIR / "fold_results.csv"
    TRIAL_SUMMARY_PATH = OUTPUT_DIR / "trial_summary.csv"
    RANKING_PATH = OUTPUT_DIR / "trial_ranking.csv"
    SUMMARY_JSON_PATH = OUTPUT_DIR / "stage2_summary.json"
    SUMMARY_MARKDOWN_PATH = OUTPUT_DIR / "stage2_summary.md"


def parse_arguments() -> argparse.Namespace:
    """Parse fresh-run and explicit resume options."""

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the Stage 2 LR-momentum anchored search. "
            "Resume granularity is one complete parameter trial."
        )
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIRECTORY",
        help=(
            "Resume an existing lr_momentum_stage2_* output directory. "
            "Completed trials are skipped; an incomplete trial restarts "
            "from fold 0."
        ),
    )
    return parser.parse_args()


def validate_configuration() -> None:
    """Validate fixed search constraints and imported Stage 1 behavior."""

    if LEARNING_RATES != [1e-4, 3e-4, 1e-3]:
        raise ValueError("Stage 2 learning rates must match the specification.")
    if not 0.0 <= MOMENTUM_MIN < MOMENTUM_MAX:
        raise ValueError("Momentum bounds are invalid.")
    if MOMENTUM_MAX >= 1.0:
        raise ValueError("MOMENTUM_MAX must be less than 1.0.")
    if not MOMENTUM_MIN <= ANCHOR_MOMENTUM <= MOMENTUM_MAX:
        raise ValueError("Anchor momentum must be inside the search range.")
    if RANDOM_MOMENTUM_TRIALS_PER_LR <= 0:
        raise ValueError("Random momentum trials per LR must be positive.")
    if BATCH_SIZE != 64:
        raise ValueError("Stage 2 requires batch size 64.")
    if WEIGHT_DECAY != 1e-4:
        raise ValueError("Stage 2 requires weight decay 1e-4.")
    if CLASSIFIER_DROPOUT != 0.2:
        raise ValueError("Stage 2 requires dropout 0.2.")
    if NESTEROV:
        raise ValueError("Stage 2 requires Nesterov=False.")
    if CV_FOLDS != tuple(range(5)):
        raise ValueError("Stage 2 requires folds 0 through 4.")
    if EPOCHS_PER_FOLD <= 0:
        raise ValueError("EPOCHS_PER_FOLD must be positive.")
    if stage1.BATCH_SIZE != BATCH_SIZE:
        raise RuntimeError("Imported Stage 1 DataLoader batch size differs.")
    if stage1.WEIGHT_DECAY != WEIGHT_DECAY:
        raise RuntimeError("Imported Stage 1 weight decay differs.")
    if stage1.CLASSIFIER_DROPOUT != CLASSIFIER_DROPOUT:
        raise RuntimeError("Imported Stage 1 dropout differs.")
    if stage1.CV_FOLDS != CV_FOLDS:
        raise RuntimeError("Imported Stage 1 CV folds differ.")


def generate_parameter_trials() -> list[dict[str, object]]:
    """Generate three anchors and balanced reproducible random trials."""

    generator = random.Random(RANDOM_SEARCH_SEED)
    trials: list[dict[str, object]] = []
    for learning_rate in LEARNING_RATES:
        trials.append(
            {
                "trial_id": "",
                "is_anchor": True,
                "learning_rate": learning_rate,
                "momentum": ANCHOR_MOMENTUM,
                "sampling_method": "mandatory_anchor",
            }
        )

    for learning_rate in LEARNING_RATES:
        sampled_momenta: list[float] = []
        while len(sampled_momenta) < RANDOM_MOMENTUM_TRIALS_PER_LR:
            momentum = round(
                generator.uniform(MOMENTUM_MIN, MOMENTUM_MAX),
                MOMENTUM_DECIMAL_PLACES,
            )
            if momentum == ANCHOR_MOMENTUM or momentum in sampled_momenta:
                continue
            sampled_momenta.append(momentum)
        for momentum in sampled_momenta:
            trials.append(
                {
                    "trial_id": "",
                    "is_anchor": False,
                    "learning_rate": learning_rate,
                    "momentum": momentum,
                    "sampling_method": "uniform_random",
                }
            )

    for index, trial in enumerate(trials, start=1):
        trial["trial_id"] = f"trial_{index:02d}"

    if len(trials) != EXPECTED_NUM_TRIALS:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_TRIALS} trials, generated {len(trials)}."
        )
    trial_frame = pd.DataFrame(trials)
    counts = trial_frame.groupby("learning_rate").size().to_dict()
    expected_per_lr = 1 + RANDOM_MOMENTUM_TRIALS_PER_LR
    if any(count != expected_per_lr for count in counts.values()):
        raise RuntimeError("Learning rates do not have balanced trial counts.")
    if trial_frame.duplicated(["learning_rate", "momentum"]).any():
        raise RuntimeError("Duplicate LR-momentum combinations were created.")
    return trials


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file so interruption cannot truncate it."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary_path, index=False, float_format="%.12g")
    temporary_path.replace(path)


def _read_result_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a persisted result CSV and enforce its expected schema."""

    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(f"Resume file is empty: {path}") from error
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(
            f"Resume file {path} is missing columns: {missing_columns}"
        )
    return frame.loc[:, list(columns)].copy()


def _validate_saved_trials(
    current_trials: list[dict[str, object]],
) -> None:
    """Ensure the resumed directory belongs to the exact current search space."""

    if not TRIALS_PATH.is_file():
        raise FileNotFoundError(
            f"Resume trial definition not found: {TRIALS_PATH}"
        )
    saved = pd.read_csv(TRIALS_PATH)
    current = pd.DataFrame(current_trials)
    required_columns = list(current.columns)
    missing_columns = sorted(set(required_columns) - set(saved.columns))
    if missing_columns:
        raise RuntimeError(
            f"Saved trial definition is missing columns: {missing_columns}"
        )
    saved = saved.loc[:, required_columns].reset_index(drop=True)
    current = current.loc[:, required_columns].reset_index(drop=True)
    if len(saved) != len(current):
        raise RuntimeError(
            "Resume refused: saved and current trial counts differ "
            f"({len(saved)} != {len(current)})."
        )
    for column in ("learning_rate", "momentum"):
        if not np.allclose(
            pd.to_numeric(saved[column]),
            pd.to_numeric(current[column]),
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                f"Resume refused: saved {column} values differ from the "
                "current search configuration."
            )
    for column in ("trial_id", "is_anchor", "sampling_method"):
        if saved[column].astype(str).tolist() != current[column].astype(str).tolist():
            raise RuntimeError(
                f"Resume refused: saved {column} values differ from the "
                "current search configuration."
            )


def _validate_resume_config(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> None:
    """Reject a resume when data, model, CV, or search settings changed."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Resume configuration not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        saved_config = json.load(file)
    current_config = build_config(trials, metadata)
    comparable_sections = (
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
    for section in comparable_sections:
        if saved_config.get(section) != current_config.get(section):
            raise RuntimeError(
                "Resume refused because the saved and current "
                f"'{section}' configurations differ. Resume the run with "
                "the same source configuration used to create it."
            )


def _validate_resume_results(
    trials: list[dict[str, object]],
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate persisted work and discard only uncommitted epoch rows."""

    valid_trial_ids = {str(trial["trial_id"]) for trial in trials}
    if not set(fold_frame["trial_id"].astype(str)).issubset(valid_trial_ids):
        raise RuntimeError("Fold results contain an unknown trial_id.")
    if not set(epoch_frame["trial_id"].astype(str)).issubset(valid_trial_ids):
        raise RuntimeError("Epoch results contain an unknown trial_id.")

    for frame, key_columns, name in (
        (fold_frame, ["trial_id", "fold"], "fold results"),
        (epoch_frame, ["trial_id", "fold", "epoch"], "epoch results"),
    ):
        if frame.duplicated(key_columns).any():
            raise RuntimeError(f"Resume {name} contain duplicate keys.")

    valid_folds = set(CV_FOLDS)
    for frame, name in (
        (fold_frame, "fold results"),
        (epoch_frame, "epoch results"),
    ):
        folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
        if not folds.issubset(valid_folds):
            raise RuntimeError(f"Resume {name} contain an invalid CV fold.")

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
                f"{int(orphan_mask.sum())} uncommitted epoch rows from an "
                "interrupted fold."
            )
            epoch_frame = epoch_frame.loc[~orphan_mask].reset_index(drop=True)

    incomplete_commits: set[tuple[str, int]] = set()
    completed = fold_frame[fold_frame["status"].eq("completed")]
    for row in completed.itertuples(index=False):
        history = epoch_frame[
            epoch_frame["trial_id"].astype(str).eq(str(row.trial_id))
            & pd.to_numeric(epoch_frame["fold"]).eq(int(row.fold))
        ]
        if len(history) != EPOCHS_PER_FOLD:
            incomplete_commits.add((str(row.trial_id), int(row.fold)))

    if incomplete_commits:
        print(
            "Resume recovery: restarting "
            f"{len(incomplete_commits)} fold(s) whose persisted epoch history "
            "was incomplete."
        )
        fold_keys = pd.Series(
            list(
                zip(
                    fold_frame["trial_id"].astype(str),
                    pd.to_numeric(fold_frame["fold"]).astype(int),
                )
            ),
            index=fold_frame.index,
        )
        epoch_keys = pd.Series(
            list(
                zip(
                    epoch_frame["trial_id"].astype(str),
                    pd.to_numeric(epoch_frame["fold"]).astype(int),
                )
            ),
            index=epoch_frame.index,
        )
        fold_frame = fold_frame.loc[
            ~fold_keys.isin(incomplete_commits)
        ].reset_index(drop=True)
        epoch_frame = epoch_frame.loc[
            ~epoch_keys.isin(incomplete_commits)
        ].reset_index(drop=True)

    invalid_completed_epoch_counts = fold_frame[
        fold_frame["status"].eq("completed")
        & pd.to_numeric(
            fold_frame["completed_epochs"],
            errors="coerce",
        ).ne(EPOCHS_PER_FOLD)
    ]
    if not invalid_completed_epoch_counts.empty:
        raise RuntimeError(
            "Resume data contain a completed fold whose completed_epochs "
            f"value is not {EPOCHS_PER_FOLD}."
        )
    return fold_frame, epoch_frame


def _find_completed_trial_ids(fold_frame: pd.DataFrame) -> set[str]:
    """Return trials whose five folds all completed successfully."""

    completed_trial_ids: set[str] = set()
    expected_folds = set(CV_FOLDS)
    for trial_id, trial_frame in fold_frame.groupby("trial_id", sort=False):
        recorded_folds = set(
            pd.to_numeric(trial_frame["fold"], errors="raise").astype(int)
        )
        if (
            len(trial_frame) == len(CV_FOLDS)
            and recorded_folds == expected_folds
            and trial_frame["status"].eq("completed").all()
        ):
            completed_trial_ids.add(str(trial_id))
    return completed_trial_ids


def _reset_incomplete_trials(
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Discard every fold of a partial trial so it can restart at fold 0."""

    completed_trial_ids = _find_completed_trial_ids(fold_frame)
    recorded_trial_ids = set(fold_frame["trial_id"].astype(str)) | set(
        epoch_frame["trial_id"].astype(str)
    )
    incomplete_trial_ids = recorded_trial_ids - completed_trial_ids
    if incomplete_trial_ids:
        ordered_ids = sorted(incomplete_trial_ids)
        print(
            "Resume recovery: restarting incomplete trial(s) from fold 0: "
            f"{ordered_ids}"
        )
        fold_frame = fold_frame.loc[
            ~fold_frame["trial_id"].astype(str).isin(incomplete_trial_ids)
        ].reset_index(drop=True)
        epoch_frame = epoch_frame.loc[
            ~epoch_frame["trial_id"].astype(str).isin(incomplete_trial_ids)
        ].reset_index(drop=True)
    return fold_frame, epoch_frame, completed_trial_ids


def load_resume_state(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Load complete trials and reset every incomplete trial to fold 0."""

    _validate_saved_trials(trials)
    _validate_resume_config(trials, metadata)
    fold_frame = _read_result_csv(FOLD_RESULTS_PATH, FOLD_RESULT_COLUMNS)
    epoch_frame = _read_result_csv(EPOCH_LOG_PATH, EPOCH_RESULT_COLUMNS)
    fold_frame, epoch_frame = _validate_resume_results(
        trials,
        fold_frame,
        epoch_frame,
    )
    return _reset_incomplete_trials(fold_frame, epoch_frame)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    momentum: float,
) -> torch.optim.SGD:
    """Build constant-parameter SGD for one Stage 2 trial."""

    if not MOMENTUM_MIN <= momentum <= MOMENTUM_MAX:
        raise ValueError("Trial momentum lies outside the configured range.")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    all_parameters = list(model.parameters())
    if len(parameters) != len(all_parameters):
        raise RuntimeError("Stage 2 requires full model fine-tuning.")
    optimizer = torch.optim.SGD(
        params=parameters,
        lr=learning_rate,
        momentum=momentum,
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


def run_fold(
    trial: dict[str, object],
    fold: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run one independent fixed LR-momentum experiment on one fold."""

    learning_rate = float(trial["learning_rate"])
    momentum = float(trial["momentum"])
    trial_id = str(trial["trial_id"])
    is_anchor = bool(trial["is_anchor"])

    stage1.cv_train.set_seed(seed=SEED, deterministic=True)
    train_loader, val_loader, _, _ = (
        stage1.cv_train.build_fold_dataloaders(fold)
    )
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    num_classes = len(train_dataset.classes)
    positive_class_index = train_dataset.class_to_idx["malignant"]
    model = stage1.build_model(num_classes)
    optimizer = build_optimizer(model, learning_rate, momentum)
    criterion = nn.CrossEntropyLoss()

    print("=" * 80)
    print(
        f"{trial_id} | LR={learning_rate:.3e} | Momentum={momentum:.3f} | "
        f"Fold={fold}/{len(CV_FOLDS) - 1}"
    )
    print("=" * 80)

    epoch_records: list[dict[str, float | int | bool | str]] = []
    best_record: dict[str, float | int | bool | str] | None = None
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
        val_metrics, _, _, _ = (
            stage1.cv_train.base_train.validate_one_epoch(
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
        )
        stage1._synchronize_device()
        epoch_time_seconds = time.perf_counter() - epoch_started_at
        if float(optimizer.param_groups[0]["lr"]) != learning_rate:
            raise RuntimeError("Learning rate changed during a fixed trial.")
        if float(optimizer.param_groups[0]["momentum"]) != momentum:
            raise RuntimeError("Momentum changed during a fixed trial.")

        loss_gap = float(val_metrics["loss"] - train_loss)
        record: dict[str, float | int | bool | str] = {
            "trial_id": trial_id,
            "is_anchor": is_anchor,
            "learning_rate": learning_rate,
            "momentum": momentum,
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
            "epoch_time_seconds": epoch_time_seconds,
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
        "is_anchor": is_anchor,
        "learning_rate": learning_rate,
        "momentum": momentum,
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
        "train_patients": train_dataset.metadata["cv_group_id"].nunique(),
        "val_patients": val_dataset.metadata["cv_group_id"].nunique(),
        "training_seconds": training_seconds,
        "failure_message": None,
    }
    return result, history


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
            "learning_rate": trial["learning_rate"],
            "momentum": trial["momentum"],
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
    """Aggregate every trial across its five completed validation folds."""

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
    """Rank only trials that completed all five folds."""

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
    """Find competitive non-dominated trials without a weighted score."""

    if ranking.empty:
        return {
            "best_trial_id": None,
            "competitive_trial_ids": [],
            "pareto_candidate_trial_ids": [],
            "learning_rate_lower": None,
            "learning_rate_upper": None,
            "momentum_lower": None,
            "momentum_upper": None,
            "heuristic": "Unavailable because no trial completed all folds.",
        }

    best = ranking.iloc[0]
    best_lower = float(best["mean_roc_auc"] - best["std_roc_auc"])
    best_upper = float(best["mean_roc_auc"] + best["std_roc_auc"])
    competitive = ranking[
        (ranking["mean_roc_auc"] + ranking["std_roc_auc"] >= best_lower)
        & (ranking["mean_roc_auc"] - ranking["std_roc_auc"] <= best_upper)
    ].copy()

    pareto_indices = []
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
        "best_learning_rate": float(best["learning_rate"]),
        "best_momentum": float(best["momentum"]),
        "best_mean_roc_auc": float(best["mean_roc_auc"]),
        "best_std_roc_auc": float(best["std_roc_auc"]),
        "competitive_trial_ids": competitive["trial_id"].tolist(),
        "pareto_candidate_trial_ids": recommended["trial_id"].tolist(),
        "learning_rate_lower": float(recommended["learning_rate"].min()),
        "learning_rate_upper": float(recommended["learning_rate"].max()),
        "momentum_lower": float(recommended["momentum"].min()),
        "momentum_upper": float(recommended["momentum"].max()),
        "heuristic": (
            "First retain trials whose mean ROC-AUC ± fold std overlaps the "
            "best trial interval. Then retain the Pareto frontier that "
            "maximizes mean ROC-AUC while minimizing ROC-AUC std, validation "
            "loss, and positive train-validation loss gap. No weighted score "
            "is used; this region is evidence for the next stage, not final "
            "hyperparameter selection."
        ),
    }


def build_diagnostics(
    ranking: pd.DataFrame,
    reasonable_region: dict[str, object],
) -> dict[str, object]:
    """Identify best, stable, and overfitting-indicated configurations."""

    if ranking.empty:
        return {
            "best_trial": None,
            "most_stable_competitive_trial": None,
            "strongest_overfitting_indication": None,
            "lowest_validation_loss_trial": None,
            "reasonable_region": reasonable_region,
        }

    competitive_ids = reasonable_region["competitive_trial_ids"]
    competitive = ranking[ranking["trial_id"].isin(competitive_ids)]
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

    def describe(row: pd.Series) -> dict[str, object]:
        return {
            "trial_id": str(row["trial_id"]),
            "learning_rate": float(row["learning_rate"]),
            "momentum": float(row["momentum"]),
            "mean_roc_auc": float(row["mean_roc_auc"]),
            "std_roc_auc": float(row["std_roc_auc"]),
            "mean_val_loss": float(row["mean_val_loss"]),
            "mean_train_val_loss_gap": float(
                row["mean_train_val_loss_gap"]
            ),
            "mean_best_epoch": float(row["mean_best_epoch"]),
            "std_best_epoch": float(row["std_best_epoch"]),
        }

    return {
        "best_trial": describe(ranking.iloc[0]),
        "most_stable_competitive_trial": describe(most_stable),
        "strongest_overfitting_indication": describe(overfitting),
        "lowest_validation_loss_trial": describe(lowest_loss),
        "reasonable_region": reasonable_region,
    }


def summarize_by_parameter(ranking: pd.DataFrame) -> dict[str, object]:
    """Summarize broad LR and momentum tendencies across completed trials."""

    if ranking.empty:
        return {"by_learning_rate": [], "by_momentum": []}
    by_learning_rate = (
        ranking.groupby("learning_rate", as_index=False)
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
        .sort_values("learning_rate")
    )
    by_momentum = (
        ranking.groupby("momentum", as_index=False)
        .agg(
            mean_trial_roc_auc=("mean_roc_auc", "mean"),
            mean_validation_loss=("mean_val_loss", "mean"),
            mean_positive_loss_gap=(
                "mean_positive_train_val_loss_gap",
                "mean",
            ),
            best_trial_mean_roc_auc=("mean_roc_auc", "max"),
            num_trials=("trial_id", "size"),
        )
        .sort_values("momentum")
    )
    return {
        "by_learning_rate": by_learning_rate.to_dict(orient="records"),
        "by_momentum": by_momentum.to_dict(orient="records"),
    }


def build_config(
    trials: list[dict[str, object]],
    metadata: pd.DataFrame,
) -> dict[str, object]:
    """Build the complete Stage 2 reproducibility snapshot."""

    stage1_config = stage1.build_config(metadata)
    return {
        "experiment": {
            "type": "stage_2_anchored_random_lr_momentum_search",
            "run_id": str(RUN_ID),
            "short_run_id": RUN_SHORT_ID,
            "created_at": RUN_STARTED_AT.isoformat(timespec="seconds"),
            "output_directory": str(OUTPUT_DIR),
            "purpose": (
                "Identify a reasonable LR-momentum region, not final "
                "hyperparameters or a final model."
            ),
            "primary_decision_metric": "mean_5fold_validation_roc_auc",
        },
        "search": {
            "method": "anchored_random_search",
            "learning_rates": LEARNING_RATES,
            "momentum_min": MOMENTUM_MIN,
            "momentum_max": MOMENTUM_MAX,
            "anchor_momentum": ANCHOR_MOMENTUM,
            "random_trials_per_learning_rate": (
                RANDOM_MOMENTUM_TRIALS_PER_LR
            ),
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
            "weight_decay": WEIGHT_DECAY,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "nesterov": NESTEROV,
            "loss": "CrossEntropyLoss",
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "seed_reset_before_each_fold": SEED,
        },
        "cross_validation": stage1_config["cross_validation"],
        "data": stage1_config["data"],
        "model": stage1_config["model"],
        "scheduler": None,
        "early_stopping": None,
        "checkpoint": None,
        "model_saving": None,
        "holdout_test_used": False,
        "runtime": stage1_config["runtime"],
    }


def save_partial_results(
    fold_frame: pd.DataFrame,
    epoch_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atomically persist completed work after every fold experiment."""

    fold_frame = fold_frame.loc[:, list(FOLD_RESULT_COLUMNS)].copy()
    epoch_frame = epoch_frame.loc[:, list(EPOCH_RESULT_COLUMNS)].copy()
    # Epoch rows are written first; fold_results.csv acts as the commit marker.
    # If interruption occurs between these writes, resume safely removes the
    # uncommitted epoch rows and repeats that fold from its deterministic start.
    _atomic_write_csv(epoch_frame, EPOCH_LOG_PATH)
    _atomic_write_csv(fold_frame, FOLD_RESULTS_PATH)
    return fold_frame, epoch_frame


def _highlight_best(
    axis: plt.Axes,
    best: pd.Series,
    y_column: str = "mean_roc_auc",
) -> None:
    """Mark the primary best trial consistently across plots."""

    axis.scatter(
        [best["momentum"]],
        [best[y_column]],
        marker="*",
        s=240,
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
    """Create decision-oriented Stage 2 visual diagnostics."""

    completed = trial_summary[trial_summary["eligible_for_ranking"]].copy()
    if completed.empty:
        return
    best = ranking.iloc[0]
    lr_colors = {
        learning_rate: color
        for learning_rate, color in zip(
            LEARNING_RATES,
            ["tab:blue", "tab:orange", "tab:green"],
            strict=True,
        )
    }
    pareto_ids = set(reasonable_region["pareto_candidate_trial_ids"])

    figure, axis = plt.subplots(figsize=(10, 6))
    for learning_rate in LEARNING_RATES:
        subset = completed[completed["learning_rate"].eq(learning_rate)]
        axis.scatter(
            subset["momentum"],
            subset["mean_roc_auc"],
            s=90,
            color=lr_colors[learning_rate],
            label=f"LR={learning_rate:.1e}",
        )
        for _, row in subset.iterrows():
            axis.annotate(
                row["trial_id"],
                (row["momentum"], row["mean_roc_auc"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    _highlight_best(axis, best)
    axis.set_title("Stage 2: LR + Momentum vs Mean Five-Fold ROC-AUC")
    axis.set_xlabel("SGD Momentum")
    axis.set_ylabel("Mean Validation ROC-AUC")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lr_momentum_vs_mean_roc_auc.png", dpi=300)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    for learning_rate in LEARNING_RATES:
        subset = completed[
            completed["learning_rate"].eq(learning_rate)
        ].sort_values("momentum")
        axis.errorbar(
            subset["momentum"],
            subset["mean_roc_auc"],
            yerr=subset["std_roc_auc"],
            marker="o",
            capsize=4,
            color=lr_colors[learning_rate],
            label=f"LR={learning_rate:.1e}",
        )
    _highlight_best(axis, best)
    axis.set_title("Stage 2: Mean ROC-AUC ± Fold Standard Deviation")
    axis.set_xlabel("SGD Momentum")
    axis.set_ylabel("Validation ROC-AUC")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "roc_auc_with_fold_std.png", dpi=300)
    plt.close(figure)

    for column, ylabel, filename, title in (
        (
            "mean_val_loss",
            "Mean Validation Loss",
            "validation_loss.png",
            "Stage 2: Validation Loss by LR and Momentum",
        ),
        (
            "mean_train_val_loss_gap",
            "Mean Validation Loss − Training Loss",
            "train_validation_loss_gap.png",
            "Stage 2: Train–Validation Loss Gap",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        for learning_rate in LEARNING_RATES:
            subset = completed[
                completed["learning_rate"].eq(learning_rate)
            ].sort_values("momentum")
            axis.plot(
                subset["momentum"],
                subset[column],
                marker="o",
                color=lr_colors[learning_rate],
                label=f"LR={learning_rate:.1e}",
            )
        _highlight_best(axis, best, y_column=column)
        if column == "mean_train_val_loss_gap":
            axis.axhline(0.0, color="black", linestyle="--", alpha=0.6)
        axis.set_title(title)
        axis.set_xlabel("SGD Momentum")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")
        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    momentum_colors = plt.cm.viridis(
        (completed["momentum"] - MOMENTUM_MIN)
        / (MOMENTUM_MAX - MOMENTUM_MIN)
    )
    for learning_rate, marker in zip(
        LEARNING_RATES,
        ["o", "s", "^"],
        strict=True,
    ):
        subset = completed[completed["learning_rate"].eq(learning_rate)]
        indices = subset.index
        axis.scatter(
            subset["mean_val_loss"],
            subset["mean_roc_auc"],
            c=momentum_colors[completed.index.get_indexer(indices)],
            marker=marker,
            s=100,
            edgecolor="black",
            linewidth=0.5,
            label=f"LR={learning_rate:.1e}",
        )
    axis.scatter(
        [best["mean_val_loss"]],
        [best["mean_roc_auc"]],
        marker="*",
        s=260,
        color="gold",
        edgecolor="black",
        label=f"Best: {best['trial_id']}",
        zorder=10,
    )
    for _, row in completed[completed["trial_id"].isin(pareto_ids)].iterrows():
        axis.annotate(
            row["trial_id"],
            (row["mean_val_loss"], row["mean_roc_auc"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    colorbar_source = plt.cm.ScalarMappable(
        norm=plt.Normalize(MOMENTUM_MIN, MOMENTUM_MAX),
        cmap="viridis",
    )
    figure.colorbar(colorbar_source, ax=axis, label="SGD Momentum")
    axis.set_title("Stage 2 Trade-off: ROC-AUC vs Validation Loss")
    axis.set_xlabel("Mean Validation Loss (lower is better)")
    axis.set_ylabel("Mean Validation ROC-AUC (higher is better)")
    axis.grid(alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "roc_auc_vs_validation_loss.png", dpi=300)
    plt.close(figure)

    pivot = completed.pivot(
        index="learning_rate",
        columns="momentum",
        values="mean_roc_auc",
    ).sort_index()
    figure_width = max(10, 0.75 * len(pivot.columns))
    figure, axis = plt.subplots(figsize=(figure_width, 4.5))
    masked_values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    image = axis.imshow(masked_values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(
        [f"{momentum:.3f}" for momentum in pivot.columns],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([f"{lr:.1e}" for lr in pivot.index])
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            value = pivot.iloc[row_index, column_index]
            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < masked_values.mean() else "black",
                    fontsize=8,
                )
    best_row = list(pivot.index).index(float(best["learning_rate"]))
    best_column = list(pivot.columns).index(float(best["momentum"]))
    best_outline = plt.Rectangle(
        (best_column - 0.5, best_row - 0.5),
        1,
        1,
        fill=False,
        edgecolor="gold",
        linewidth=3,
        label=f"Best: {best['trial_id']}",
    )
    axis.add_patch(best_outline)
    figure.colorbar(image, ax=axis, label="Mean Five-Fold ROC-AUC")
    axis.set_title(
        "Stage 2 Sampled LR × Momentum ROC-AUC Heatmap\n"
        "Blank cells were not sampled by anchored random search"
    )
    axis.set_xlabel("SGD Momentum")
    axis.set_ylabel("Learning Rate")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "lr_momentum_roc_auc_heatmap.png", dpi=300)
    plt.close(figure)

    secondary_columns = {
        "mean_accuracy": "Accuracy",
        "mean_f1_score": "F1",
        "mean_sensitivity": "Sensitivity",
        "mean_specificity": "Specificity",
        "mean_precision": "Precision",
    }
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for axis, (column, label) in zip(
        axes.flat,
        secondary_columns.items(),
        strict=False,
    ):
        for learning_rate in LEARNING_RATES:
            subset = completed[
                completed["learning_rate"].eq(learning_rate)
            ].sort_values("momentum")
            axis.plot(
                subset["momentum"],
                subset[column],
                marker="o",
                color=lr_colors[learning_rate],
                label=f"LR={learning_rate:.1e}",
            )
        _highlight_best(axis, best, y_column=column)
        axis.set_title(label)
        axis.set_xlabel("Momentum")
        axis.set_ylabel(f"Mean {label}")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle("Stage 2 Secondary Validation Metrics", y=0.98)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(FIGURES_DIR / "secondary_metrics.png", dpi=300)
    plt.close(figure)


def build_summary_markdown(
    ranking: pd.DataFrame,
    diagnostics: dict[str, object],
    parameter_comparison: dict[str, object],
) -> str:
    """Build a human-readable decision report without claiming a final pair."""

    lines = [
        "# Stage 2 Learning Rate + Momentum Search",
        "",
        "Primary ranking: mean five-fold validation ROC-AUC.",
        "Validation loss, fold ROC-AUC standard deviation, and positive "
        "train-validation loss gap are diagnostic guardrails.",
        "",
        "## Ranking",
        "",
        "| Rank | Trial | LR | Momentum | Mean ROC-AUC ± std | Val loss | "
        "Loss gap | Best epoch mean ± std |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row['rank_by_mean_roc_auc'])} | {row['trial_id']} | "
            f"{row['learning_rate']:.1e} | {row['momentum']:.3f} | "
            f"{row['mean_roc_auc']:.4f} ± {row['std_roc_auc']:.4f} | "
            f"{row['mean_val_loss']:.4f} | "
            f"{row['mean_train_val_loss_gap']:+.4f} | "
            f"{row['mean_best_epoch']:.2f} ± {row['std_best_epoch']:.2f} |"
        )

    lines.extend(["", "## Diagnostics", ""])
    diagnostic_labels = {
        "best_trial": "Best mean ROC-AUC",
        "most_stable_competitive_trial": "Most stable competitive trial",
        "lowest_validation_loss_trial": "Lowest validation-loss trial",
        "strongest_overfitting_indication": (
            "Strongest observed overfitting indication"
        ),
    }
    for key, label in diagnostic_labels.items():
        value = diagnostics.get(key)
        if value is None:
            lines.append(f"- {label}: unavailable")
            continue
        lines.append(
            f"- {label}: {value['trial_id']} "
            f"(LR={value['learning_rate']:.1e}, "
            f"momentum={value['momentum']:.3f}, "
            f"ROC-AUC={value['mean_roc_auc']:.4f} ± "
            f"{value['std_roc_auc']:.4f}, "
            f"val_loss={value['mean_val_loss']:.4f}, "
            f"gap={value['mean_train_val_loss_gap']:+.4f})"
        )

    region = diagnostics["reasonable_region"]
    lines.extend(
        [
            "",
            "## Reasonable Region for the Next Stage",
            "",
            f"- Candidate trials: {region['pareto_candidate_trial_ids']}",
            f"- Learning-rate bounds: {region['learning_rate_lower']} to "
            f"{region['learning_rate_upper']}",
            f"- Momentum bounds: {region['momentum_lower']} to "
            f"{region['momentum_upper']}",
            f"- Rationale: {region['heuristic']}",
            "",
            "The region is intended for narrower follow-up tuning and is not "
            "a final hyperparameter decision.",
            "",
            "## Broad Parameter Comparison",
            "",
            "### Learning Rate",
            "",
        ]
    )
    for row in parameter_comparison["by_learning_rate"]:
        lines.append(
            f"- LR={row['learning_rate']:.1e}: average trial ROC-AUC="
            f"{row['mean_trial_roc_auc']:.4f}, best trial ROC-AUC="
            f"{row['best_trial_mean_roc_auc']:.4f}, mean val loss="
            f"{row['mean_validation_loss']:.4f}."
        )
    lines.extend(["", "### Momentum", ""])
    for row in parameter_comparison["by_momentum"]:
        lines.append(
            f"- Momentum={row['momentum']:.3f}: average trial ROC-AUC="
            f"{row['mean_trial_roc_auc']:.4f}, best trial ROC-AUC="
            f"{row['best_trial_mean_roc_auc']:.4f}, mean val loss="
            f"{row['mean_validation_loss']:.4f}."
        )
    lines.append("")
    return "\n".join(lines)


def print_header(trials: list[dict[str, object]]) -> None:
    """Print the sampled search space before any training starts."""

    print("Stage 2 — Anchored Random LR + Momentum Search")
    print()
    print(f"Parameter combinations : {len(trials)}")
    print(f"Folds per combination  : {len(CV_FOLDS)}")
    print(f"Epochs per fold        : {EPOCHS_PER_FOLD}")
    print(f"Total fold experiments : {len(trials) * len(CV_FOLDS)}")
    print(f"Batch size             : {BATCH_SIZE}")
    print(f"Weight decay           : {WEIGHT_DECAY:.1e}")
    print(f"Dropout                : {CLASSIFIER_DROPOUT}")
    print("Optimizer              : SGD, Nesterov=False")
    print("Primary metric         : mean five-fold validation ROC-AUC")
    print("Model/checkpoint saving: disabled")
    print()
    print("Trials:")
    for trial in trials:
        trial_type = "anchor" if trial["is_anchor"] else "random"
        print(
            f"  {trial['trial_id']}: LR={trial['learning_rate']:.1e}, "
            f"momentum={trial['momentum']:.3f} ({trial_type})"
        )
    print()


def print_ranking(
    ranking: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    """Print primary ranking and diagnostic guardrails."""

    print()
    print("Stage 2 Ranking by Mean Five-Fold ROC-AUC")
    if ranking.empty:
        print("No parameter combination completed all five folds.")
        return
    print("Rank | Trial    | LR      | Mom.  | ROC-AUC ± Std | Val Loss | Gap")
    print("-" * 78)
    for _, row in ranking.iterrows():
        print(
            f"{int(row['rank_by_mean_roc_auc']):>4} | "
            f"{row['trial_id']:<8} | {row['learning_rate']:.1e} | "
            f"{row['momentum']:.3f} | "
            f"{row['mean_roc_auc']:.4f} ± {row['std_roc_auc']:.4f} | "
            f"{row['mean_val_loss']:.4f}   | "
            f"{row['mean_train_val_loss_gap']:+.4f}"
        )
    region = diagnostics["reasonable_region"]
    print()
    print("Reasonable Stage 2 region for follow-up:")
    print(f"  Trials   : {region['pareto_candidate_trial_ids']}")
    print(
        f"  LR       : {region['learning_rate_lower']} -> "
        f"{region['learning_rate_upper']}"
    )
    print(
        f"  Momentum : {region['momentum_lower']} -> "
        f"{region['momentum_upper']}"
    )
    print("  This is a diagnostic region, not a final parameter pair.")


def main() -> None:
    """Execute all anchored/random LR-momentum combinations across CV."""

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
        fold_frame, epoch_frame, completed_trial_ids = load_resume_state(
            trials=trials,
            metadata=metadata,
        )
        # Commit the trial-level cleanup before starting new GPU work. If the
        # process stops again, the incomplete trial still restarts at fold 0.
        fold_frame, epoch_frame = save_partial_results(
            fold_frame,
            epoch_frame,
        )

    print_header(trials)
    total_fold_experiments = len(trials) * len(CV_FOLDS)
    if arguments.resume is not None:
        print("Resume mode")
        print(f"  Directory       : {OUTPUT_DIR}")
        print(
            "  Completed trials: "
            f"{len(completed_trial_ids)}/{len(trials)}"
        )
        print(
            "  Remaining trials: "
            f"{len(trials) - len(completed_trial_ids)}"
        )
        print(
            "  Remaining folds : "
            f"{total_fold_experiments - len(fold_frame)}"
        )
        print(
            "  Granularity     : completed trials are skipped; an incomplete "
            "trial restarts from fold 0"
        )
        print()

    for trial in trials:
        trial_id = str(trial["trial_id"])
        if trial_id in completed_trial_ids:
            print(f"Skipping {trial_id}: all {len(CV_FOLDS)} folds completed.")
            continue

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
                    elapsed_seconds=(
                        time.perf_counter() - experiment_started_at
                    ),
                )
                print(
                    f"CUDA OOM: {trial['trial_id']}, fold={fold}, "
                    f"LR={trial['learning_rate']}, momentum={trial['momentum']}"
                )
            except RuntimeError as error:
                if "Non-finite" not in str(error):
                    raise
                result = build_failed_result(
                    trial=trial,
                    fold=fold,
                    status="failed_non_finite_loss",
                    message=str(error),
                    elapsed_seconds=(
                        time.perf_counter() - experiment_started_at
                    ),
                )
                print(
                    f"Non-finite loss: {trial['trial_id']}, fold={fold}: "
                    f"{error}"
                )
            finally:
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            epoch_frame = pd.concat(
                [epoch_frame, history],
                ignore_index=True,
            ) if result["status"] == "completed" else epoch_frame
            fold_frame = pd.concat(
                [
                    fold_frame,
                    pd.DataFrame([result], columns=FOLD_RESULT_COLUMNS),
                ],
                ignore_index=True,
            )
            fold_frame, epoch_frame = save_partial_results(
                fold_frame,
                epoch_frame,
            )

    fold_frame, _ = save_partial_results(fold_frame, epoch_frame)
    trial_summary = summarize_trials(trials, fold_frame)
    trial_summary.to_csv(
        TRIAL_SUMMARY_PATH,
        index=False,
        float_format="%.12g",
    )
    ranking = build_ranking(trial_summary)
    ranking.to_csv(RANKING_PATH, index=False, float_format="%.12g")

    reasonable_region = derive_reasonable_region(ranking)
    diagnostics = build_diagnostics(ranking, reasonable_region)
    parameter_comparison = summarize_by_parameter(ranking)
    summary = {
        "primary_decision_metric": "mean five-fold validation ROC-AUC",
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
