"""Utilities for saving and loading U-Net training checkpoints."""

import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_config: dict[str, object],
    scheduler_config: dict[str, object],
    epoch: int,
    best_val_loss: float,
    best_loss_epoch: int,
    best_val_dice: float,
    best_dice_epoch: int,
    epochs_without_improvement: int,
    save_path: Path,
) -> None:
    """
    Save the latest training state required to resume training.

    Parameters
    ----------
    model : nn.Module
        Model being trained.
    optimizer : Optimizer
        Training optimizer.
    scaler : torch.amp.GradScaler
        Gradient scaler used for automatic mixed-precision training.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning-rate scheduler whose state will be saved when enabled.
    loss_config : dict[str, object]
        Loss function configuration used by the experiment.
    scheduler_config : dict[str, object]
        Learning-rate scheduler configuration used by the experiment.
    epoch : int
        Number of completed epochs.
    best_val_loss : float
        Lowest validation loss observed so far.
    best_loss_epoch : int
        Epoch associated with the lowest validation loss.
    best_val_dice : float
        Highest validation Dice score observed so far.
    best_dice_epoch : int
        Epoch associated with the highest validation Dice score.
    epochs_without_improvement : int
        Consecutive epochs completed without a lower validation loss.
    save_path : Path
        Output checkpoint path.
    """

    numpy_rng_state = np.random.get_state()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "loss_config": dict(loss_config),
        "scheduler_config": dict(scheduler_config),
        "best_val_loss": best_val_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_val_dice": best_val_dice,
        "best_dice_epoch": best_dice_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "bit_generator": numpy_rng_state[0],
            "state": torch.from_numpy(numpy_rng_state[1].copy()),
            "position": numpy_rng_state[2],
            "has_gauss": numpy_rng_state[3],
            "cached_gaussian": numpy_rng_state[4],
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    # Replace the current run's checkpoint only after the new file is complete.
    temporary_path = save_path.with_suffix(f"{save_path.suffix}.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(save_path)


def save_best_model(model: nn.Module, save_path: Path) -> None:
    """
    Save model parameters associated with the selected validation metric.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be saved.
    save_path : Path
        Output model path.
    """

    temporary_path = save_path.with_suffix(f"{save_path.suffix}.tmp")
    torch.save(model.state_dict(), temporary_path)
    temporary_path.replace(save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    expected_loss_config: dict[str, object],
    expected_scheduler_config: dict[str, object],
) -> tuple[int, float, int, float, int, int]:
    """
    Restore training state and return the saved training progress.

    Parameters
    ----------
    checkpoint_path : Path
        Checkpoint to load.
    model : nn.Module
        Model whose parameters will be restored.
    optimizer : Optimizer
        Optimizer whose state will be restored.
    scaler : torch.amp.GradScaler
        Gradient scaler whose state will be restored.
    scheduler : torch.optim.lr_scheduler.LRScheduler, optional
        Learning-rate scheduler whose state will be restored when available.
    expected_loss_config : dict[str, object]
        Loss configuration required for the resumed experiment.
    expected_scheduler_config : dict[str, object]
        Scheduler configuration required for the resumed experiment.

    Returns
    -------
    tuple[int, float, int, float, int, int]
        Completed epochs, lowest validation loss and its epoch, highest
        validation Dice score and its epoch, and consecutive epochs without
        validation-loss improvement.

    Raises
    ------
    RuntimeError
        If the checkpoint does not contain a loss configuration.
    ValueError
        If the checkpoint loss configuration differs from the current one.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    checkpoint_loss_config = checkpoint.get("loss_config")
    if checkpoint_loss_config is None:
        raise RuntimeError(
            "Checkpoint does not contain a loss configuration and cannot be "
            "safely resumed. Start a new experiment or use a checkpoint created "
            "by the current training code."
        )

    if checkpoint_loss_config != expected_loss_config:
        raise ValueError(
            "Loss configuration mismatch when resuming training: "
            f"checkpoint uses {checkpoint_loss_config!r}, but the current "
            f"configuration uses {expected_loss_config!r}."
        )

    checkpoint_scheduler_config = checkpoint.get("scheduler_config")
    if (
        checkpoint_scheduler_config is not None
        and checkpoint_scheduler_config != expected_scheduler_config
    ):
        raise ValueError(
            "Scheduler configuration mismatch when resuming training: "
            f"checkpoint uses {checkpoint_scheduler_config!r}, but the current "
            f"configuration uses {expected_scheduler_config!r}."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    scheduler_state_dict = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)
    elif scheduler is not None and scheduler_state_dict is None:
        warnings.warn(
            "Checkpoint has no scheduler state. The configured scheduler will "
            "start with a new state after resume.",
            stacklevel=2,
        )

    random.setstate(checkpoint["python_rng_state"])

    numpy_rng_state = checkpoint["numpy_rng_state"]
    np.random.set_state(
        (
            numpy_rng_state["bit_generator"],
            numpy_rng_state["state"].numpy(),
            numpy_rng_state["position"],
            numpy_rng_state["has_gauss"],
            numpy_rng_state["cached_gaussian"],
        )
    )

    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_rng_state_all = checkpoint["cuda_rng_state_all"]
    if torch.cuda.is_available() and cuda_rng_state_all is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    completed_epochs = int(checkpoint["epoch"])
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_loss_epoch = int(
        checkpoint.get("best_loss_epoch", checkpoint.get("best_epoch", 0))
    )
    best_val_dice = float(checkpoint.get("best_val_dice", float("-inf")))
    best_dice_epoch = int(checkpoint.get("best_dice_epoch", 0))
    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            completed_epochs - best_loss_epoch if best_loss_epoch > 0 else 0,
        )
    )

    return (
        completed_epochs,
        best_val_loss,
        best_loss_epoch,
        best_val_dice,
        best_dice_epoch,
        epochs_without_improvement,
    )
