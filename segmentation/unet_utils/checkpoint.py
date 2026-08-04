from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer


# ---------------------------------------------------------------------
# Best Model
# ---------------------------------------------------------------------
def save_best_model(
    model: nn.Module,
    save_path: Path,
) -> None:
    """
    Save the best model weights.

    Parameters
    ----------
    model : nn.Module
        Model whose weights will be saved.

    save_path : Path
        Output path.
    """

    torch.save(
        model.state_dict(),
        save_path,
    )


# ---------------------------------------------------------------------
# Training Checkpoint
# ---------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    dice_score: float,
    iou: float,
    precision: float,
    sensitivity: float,
    specificity: float,
    best_dice: float,
    learning_rate: float,
    batch_size: int,
    save_path: Path,
    architecture: str = "UNet",
) -> None:
    """
    Save the latest training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model being trained.

    optimizer : Optimizer
        Training optimizer.

    epoch : int
        Current epoch.

    train_loss : float
        Training loss.

    val_loss : float
        Validation loss.

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

    best_dice : float
        Best Dice score observed.

    learning_rate : float
        Learning rate.

    batch_size : int
        Mini-batch size.

    save_path : Path
        Output checkpoint path.

    architecture : str, default="UNet"
        Model architecture.
    """

    torch.save(
        {
            "epoch": epoch,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice_score": dice_score,
            "iou": iou,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "best_dice": best_dice,
        },
        save_path,
    )


# ---------------------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict:
    """
    Load a training checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Checkpoint file.

    model : nn.Module
        Model to restore.

    optimizer : Optimizer, optional
        Optimizer to restore.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint