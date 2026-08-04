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

    Args:
        log_path (Path):
            Path to the training log CSV file.
    """

    with open(log_path, "w", newline="") as file:

        writer = csv.writer(file)

        # Write the CSV header
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ])


def append_training_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """
    Append one epoch of training results to the CSV file.

    Args:
        log_path (Path):
            Path to the training log CSV file.

        epoch (int):
            Current epoch number.

        train_loss (float):
            Training loss.

        train_accuracy (float):
            Training accuracy.

        val_loss (float):
            Validation loss.

        val_accuracy (float):
            Validation accuracy.
    """

    with open(log_path, "a", newline="") as file:

        writer = csv.writer(file)

        # Write one epoch of training results
        writer.writerow([
            epoch,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        ])


#---------------------------------
# TRAINING CONFIGURATION
#---------------------------------
def save_training_config(
    config: dict,
    save_path: Path,
) -> None:
    """
    Save the training configuration as a JSON file.

    Args:
        config (dict):
            Dictionary containing the training configuration.

        save_path (Path):
            Path to the output JSON file.
    """

    with open(save_path, "w") as file:

        json.dump(
            config,
            file,
            indent=4,
        )