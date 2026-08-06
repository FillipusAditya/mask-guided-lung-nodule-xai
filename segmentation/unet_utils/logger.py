"""Helpers for persisting training metrics and configuration."""

import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Training Log
# ---------------------------------------------------------------------
def create_training_log(
    log_path: Path,
) -> None:
    """
    Create the training log CSV file.

    Parameters
    ----------
    log_path : Path
        Output CSV file path.
    """

    with open(log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "epoch_time",
                "elapsed_time_sec",
                "is_best",
                "early_stop_counter",
                "gpu_memory_allocated_mb",
                "train_time_sec",
                "val_time_sec",
                "scheduler_updated",
                "patience_counter",
                "best_metric",
                "checkpoint_saved",
                "samples_per_sec",
                "train_batches",
                "val_batches",
                "gpu_memory_reserved_mb",
                "stopped_early",
                "learning_rate",
                "train_loss",
                "val_loss",
                "dice_score",
                "iou",
                "precision",
                "sensitivity",
                "specificity",
            ]
        )


def append_training_log(
    log_path: Path,
    epoch: int,
    epoch_time: float,
    elapsed_time_sec: float,
    is_best: bool,
    early_stop_counter: int,
    gpu_memory_allocated_mb: float,
    train_time_sec: float,
    val_time_sec: float,
    scheduler_updated: bool,
    patience_counter: int,
    best_metric: float,
    checkpoint_saved: bool,
    samples_per_sec: float,
    train_batches: int,
    val_batches: int,
    gpu_memory_reserved_mb: float,
    stopped_early: bool,
    learning_rate: float,
    train_loss: float,
    val_loss: float,
    dice_score: float,
    iou: float,
    precision: float,
    sensitivity: float,
    specificity: float,
) -> None:
    """
    Append one epoch training result.

    Parameters
    ----------
    log_path : Path
        Output CSV file path.

    epoch : int
        Current epoch.

    epoch_time : float
        Duration of the epoch in seconds.

    elapsed_time_sec : float
        Cumulative training duration through the end of this epoch, in
        seconds.

    is_best : bool
        Whether this epoch produced the best model observed so far.

    early_stop_counter : int
        Number of consecutive epochs without sufficient improvement according
        to the early-stopping criterion.

    gpu_memory_allocated_mb : float
        GPU memory currently allocated by PyTorch at the end of the epoch, in
        megabytes. This is zero when CUDA is unavailable.

    train_time_sec : float
        Duration of the training phase for this epoch, in seconds.

    val_time_sec : float
        Duration of the validation phase for this epoch, in seconds.

    scheduler_updated : bool
        Whether the scheduler changed the learning rate during this epoch.

    patience_counter : int
        Current learning-rate scheduler count of consecutive epochs without
        sufficient improvement.

    best_metric : float
        Best value of the model-selection metric observed through this epoch.

    checkpoint_saved : bool
        Whether the latest training checkpoint was saved for this epoch.

    samples_per_sec : float
        Training throughput in samples per second.

    train_batches : int
        Number of training batches processed during this epoch.

    val_batches : int
        Number of validation batches processed during this epoch.

    gpu_memory_reserved_mb : float
        GPU memory reserved by PyTorch at the end of the epoch, in megabytes.
        This is zero when CUDA is unavailable.

    stopped_early : bool
        Whether early stopping ended training at this epoch.

    learning_rate : float
        Learning rate used during the epoch.

    train_loss : float
        Average training loss.

    val_loss : float
        Average validation loss.

    dice_score : float
        Dice score.

    iou : float
        Intersection-over-Union.

    precision : float
        Precision.

    sensitivity : float
        Recall / Sensitivity.

    specificity : float
        Specificity.
    """

    with open(log_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                epoch_time,
                elapsed_time_sec,
                is_best,
                early_stop_counter,
                gpu_memory_allocated_mb,
                train_time_sec,
                val_time_sec,
                scheduler_updated,
                patience_counter,
                best_metric,
                checkpoint_saved,
                samples_per_sec,
                train_batches,
                val_batches,
                gpu_memory_reserved_mb,
                stopped_early,
                learning_rate,
                train_loss,
                val_loss,
                dice_score,
                iou,
                precision,
                sensitivity,
                specificity,
            ]
        )


# ---------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------
def save_training_config(
    config: dict,
    save_path: Path,
) -> None:
    """
    Save training configuration as JSON.

    Parameters
    ----------
    config : dict
        Training configuration.

    save_path : Path
        Output JSON file path.
    """

    with open(save_path, "w") as file:
        json.dump(
            config,
            file,
            indent=4,
        )
