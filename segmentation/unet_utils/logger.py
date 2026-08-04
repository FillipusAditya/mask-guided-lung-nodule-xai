import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Training Log
# ---------------------------------------------------------------------
def create_training_log(
    log_path: Path,
) ->None:
    """
    Create the training log CSV file.

    Parameters
    ----------
    log_path : Path
        Output CSV file path.
    """

    with open(log_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "dice_score",
            "iou",
            "precision",
            "sensitivity",
            "specificity",
        ])


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

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            dice_score,
            iou,
            precision,
            sensitivity,
            specificity,
        ])


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