"""Helpers for persisting classification metrics and configuration."""

import csv
import json
from pathlib import Path


#---------------------------------
# TRAINING LOG
#---------------------------------
def create_training_log(
    log_path: Path,
) -> None:
    """
    Create the training log CSV file.

    Parameters
    ----------
    log_path : Path
        Path to the training log CSV file.
    """

    with open(log_path, "w", newline="") as file:

        writer = csv.writer(file)

        # Write the CSV header
        writer.writerow([
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
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1_score",
            "auc",
        ])


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
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    sensitivity: float,
    specificity: float,
    precision: float,
    f1_score: float,
    auc_score: float,
) -> None:
    """
    Append one epoch of training results to the CSV file.

    Parameters
    ----------
    log_path : Path
        Path to the training log CSV file.

    epoch : int
        Current epoch number.

    epoch_time : float
        Total duration of this epoch in seconds.

    elapsed_time_sec : float
        Cumulative training duration in seconds.

    is_best : bool
        Whether this epoch produced the best model so far.

    gpu_memory_allocated_mb : float
        GPU memory allocated at the end of the epoch.

    train_time_sec, val_time_sec : float
        Durations of the training and validation phases.

    best_metric : float
        Best model-selection metric observed so far.

    checkpoint_saved : bool
        Whether the latest checkpoint was saved.

    samples_per_sec : float
        Training throughput in samples per second.

    train_batches, val_batches : int
        Numbers of training and validation batches.

    gpu_memory_reserved_mb : float
        GPU memory reserved at the end of the epoch.

    learning_rate : float
        Learning rate after the scheduler update.

    scheduler_updated : bool
        Whether the scheduler changed the learning rate in this epoch.

    patience_counter : int
        Number of consecutive epochs without sufficient improvement tracked
        by the scheduler.

    early_stop_counter : int
        Number of consecutive epochs without sufficient improvement tracked
        by early stopping.

    stopped_early : bool
        Whether early stopping ended training in this epoch.

    train_loss : float
        Training loss.

    train_accuracy : float
        Training accuracy.

    val_loss : float
        Validation loss.

    val_accuracy : float
        Validation accuracy.

    sensitivity : float
        Validation sensitivity or recall.

    specificity : float
        Validation specificity.

    precision : float
        Validation precision.

    f1_score : float
        Validation F1-score.

    auc_score : float
        Validation ROC AUC.
    """

    with open(log_path, "a", newline="") as file:

        writer = csv.writer(file)

        # Write one epoch of training results
        writer.writerow([
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
            train_accuracy,
            val_loss,
            val_accuracy,
            sensitivity,
            specificity,
            precision,
            f1_score,
            auc_score,
        ])


#---------------------------------
# TRAINING CONFIGURATION
#---------------------------------
def save_training_config(
    config: dict[str, object],
    save_path: Path,
) -> None:
    """
    Save the training configuration as a JSON file.

    Parameters
    ----------
    config : dict[str, object]
        Dictionary containing the training configuration.

    save_path : Path
        Path to the output JSON file.
    """

    with open(save_path, "w") as file:

        json.dump(
            config,
            file,
            indent=4,
        )
