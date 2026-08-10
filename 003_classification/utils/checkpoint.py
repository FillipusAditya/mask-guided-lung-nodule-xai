"""Utilities for saving and restoring classification checkpoints."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .early_stopping import EarlyStopping


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
    scheduler: LRScheduler,
    early_stopping: EarlyStopping,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    sensitivity: float,
    specificity: float,
    precision: float,
    f1_score: float,
    auc_score: float,
    best_metric: float,
    best_metric_name: str,
    best_metric_mode: str,
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

    scheduler : LRScheduler
        Learning-rate scheduler used during training.

    early_stopping : EarlyStopping
        Early-stopping controller used during training.

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

    best_metric : float
        Best model-selection metric observed so far.

    best_metric_name : str
        Name of the metric used to select the best model.

    best_metric_mode : str
        Optimization direction, either ``"min"`` or ``"max"``.

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
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping_state_dict": early_stopping.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
            "auc": auc_score,
            "best_metric": best_metric,
            "best_metric_name": best_metric_name,
            "best_metric_mode": best_metric_mode,
        },
        save_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    early_stopping: EarlyStopping | None = None,
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

    scheduler : LRScheduler, optional
        Scheduler into which the scheduler state will be loaded. If omitted,
        its state is not restored.

    early_stopping : EarlyStopping, optional
        Controller into which the early-stopping state will be loaded. If
        omitted, its state is not restored.

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

    # Restore the scheduler state if provided and available
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    # Restore the early-stopping state if provided and available
    if (
        early_stopping is not None
        and checkpoint.get("early_stopping_state_dict") is not None
    ):

        early_stopping.load_state_dict(
            checkpoint["early_stopping_state_dict"]
        )

    return checkpoint
