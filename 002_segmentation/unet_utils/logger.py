"""Helpers for persisting training metrics."""

import csv
from pathlib import Path


def create_training_log(log_path: Path) -> None:
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
                train_loss,
                val_loss,
                dice_score,
                iou,
                precision,
                sensitivity,
                specificity,
            ]
        )


def synchronize_training_log(log_path: Path, completed_epochs: int) -> None:
    """
    Synchronize the training log with the latest completed checkpoint.

    Rows newer than the checkpoint are removed, and duplicate epoch rows are
    reduced to their latest occurrence. The checkpoint remains the source of
    truth when training is resumed after an interrupted write.

    Parameters
    ----------
    log_path : Path
        Training log CSV path.
    completed_epochs : int
        Number of completed epochs stored in the checkpoint.
    """

    with open(log_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows_by_epoch = {}

        for row in reader:
            try:
                epoch = int(row["epoch"])
            except (KeyError, TypeError, ValueError):
                continue

            if epoch <= completed_epochs:
                rows_by_epoch[epoch] = row

    expected_epochs = list(range(1, completed_epochs + 1))
    logged_epochs = sorted(rows_by_epoch)
    if logged_epochs != expected_epochs:
        raise RuntimeError(
            "Training log cannot be synchronized with the checkpoint: "
            f"expected epochs {expected_epochs}, found {logged_epochs}."
        )

    temporary_path = log_path.with_suffix(f"{log_path.suffix}.tmp")
    with open(temporary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_epoch[epoch] for epoch in logged_epochs)

    temporary_path.replace(log_path)
