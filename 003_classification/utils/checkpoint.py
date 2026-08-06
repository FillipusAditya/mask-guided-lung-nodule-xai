"""Utilities for saving and restoring classification checkpoints."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer


#---------------------------------
# MODEL CHECKPOINT
#---------------------------------
def save_best_model(
    model: nn.Module,
    save_path: Path,
) -> None:
    """
    Save model parameters associated with the best validation result.

    Parameters
    ----------
    model : nn.Module
        Model whose weights will be saved.

    save_path : Path
        Output path for the model parameters.
    """

    torch.save(
        model.state_dict(),
        save_path,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    best_val_loss: float,
    num_classes: int,
    learning_rate: float,
    batch_size: int,
    save_path: Path,
    architecture: str = "Transfer Learning ResNet50",
) -> None:
    """
    Save the latest classification training checkpoint.

    The checkpoint contains the model and optimizer states together with the
    current training progress, configuration, and metrics required to inspect
    or continue the run.

    Parameters
    ----------
    model : nn.Module
        Model being trained.

    optimizer : Optimizer
        Optimizer used during training.

    epoch : int
        Current epoch number.

    train_loss : float
        Training loss.

    train_accuracy : float
        Training accuracy.

    val_loss : float
        Validation loss.

    val_accuracy : float
        Validation accuracy.

    best_val_loss : float
        Best validation loss observed so far.

    num_classes : int
        Number of output classes.

    learning_rate : float
        Learning rate.

    batch_size : int
        Mini-batch size.

    save_path : Path
        Output path for the checkpoint.

    architecture : str, default="Transfer Learning ResNet50"
        Model architecture name.
    """

    # Save the training checkpoint
    torch.save(
        {
            "epoch": epoch,
            "architecture": architecture,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, object]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file.

    model : nn.Module
        Model into which the weights will be loaded.

    optimizer : Optimizer, optional
        Optimizer into which the optimizer state will be loaded. If omitted,
        only the model parameters are restored.

    Returns
    -------
    dict[str, object]
        Dictionary containing the checkpoint contents.
    """

    # Load the checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Restore the model weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore the optimizer state if provided
    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
