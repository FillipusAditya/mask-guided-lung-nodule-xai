"""Utilities for saving and restoring U-Net training checkpoints."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .early_stopping import EarlyStopping


# ---------------------------------------------------------------------
# Best Model
# ---------------------------------------------------------------------
def save_best_model(
    model: nn.Module,
    save_path: Path,
) -> None:
    """
    Save the model parameters associated with the best validation result.

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
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler,
    early_stopping: EarlyStopping,
    epoch: int,
    train_loss: float,
    val_loss: float,
    dice_score: float,
    iou: float,
    precision: float,
    sensitivity: float,
    specificity: float,
    best_metric: float,
    best_metric_name: str,
    best_metric_mode: str,
    learning_rate: float,
    batch_size: int,
    save_path: Path,
    architecture: str = "UNet",
) -> None:
    """
    Save all state required to resume training from the current epoch.

    The checkpoint contains the model, optimizer, scheduler, gradient-scaler,
    and early-stopping states, along with training configuration and metrics.

    Parameters
    ----------
    model : nn.Module
        Model being trained.

    optimizer : Optimizer
        Training optimizer.

    scheduler : LRScheduler or None
        Learning-rate scheduler. Its state is saved when provided.

    scaler : torch.amp.GradScaler
        Gradient scaler used for automatic mixed-precision training.

    early_stopping : EarlyStopping
        Early-stopping controller whose state is stored in the checkpoint.

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

    best_metric : float
        Best value of the monitored model-selection metric observed.

    best_metric_name : str
        Name of the metric used to select the best model.

    best_metric_mode : str
        Optimization mode used for the best metric (``"min"`` or ``"max"``).

    learning_rate : float
        Learning rate.

    batch_size : int
        Mini-batch size.

    save_path : Path
        Output checkpoint path.

    architecture : str, default="UNet"
        Model architecture.
    """

    checkpoint = {
        # Training progress
        "epoch": epoch,
        "architecture": architecture,

        # Hyperparameters
        "learning_rate": learning_rate,
        "batch_size": batch_size,

        # Model state
        "model_state_dict": model.state_dict(),

        # Optimizer state
        "optimizer_state_dict": optimizer.state_dict(),

        # Scheduler state
        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        # Gradient scaler state
        "scaler_state_dict": scaler.state_dict(),

        # Early stopping
        "early_stopping_state_dict":
            early_stopping.state_dict(),

        # Metrics
        "train_loss": train_loss,
        "val_loss": val_loss,
        "dice_score": dice_score,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,

        # Best metric
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
        "best_metric_mode": best_metric_mode,
    }

    torch.save(
        checkpoint,
        save_path,
    )


# ---------------------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------------------
def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
) -> dict:
    """
    Load a training checkpoint and restore the supplied training components.

    Parameters
    ----------
    checkpoint_path : Path
        Checkpoint file.

    model : nn.Module
        Model to restore.

    optimizer : Optimizer, optional
        Optimizer to restore. If omitted, its state is not loaded.

    scheduler : LRScheduler, optional
        Learning-rate scheduler to restore. If omitted, its state is not
        loaded.

    scaler : torch.amp.GradScaler, optional
        Gradient scaler to restore. If omitted, its state is not loaded.

    early_stopping : EarlyStopping, optional
        Early-stopping controller to restore. If omitted, its state is not
        loaded.

    Returns
    -------
    dict
        Checkpoint dictionary containing the saved states, configuration, and
        metrics.
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

    if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    if scaler is not None and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    if (
        early_stopping is not None
        and checkpoint["early_stopping_state_dict"] is not None
    ):
        early_stopping.load_state_dict(
            checkpoint["early_stopping_state_dict"]
        )

    return checkpoint
