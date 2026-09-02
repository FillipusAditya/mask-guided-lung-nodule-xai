"""Utilities for saving and loading U-Net training checkpoints."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    best_epoch: int,
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
    epoch : int
        Number of completed epochs.
    best_val_loss : float
        Lowest validation loss observed so far.
    best_epoch : int
        Epoch associated with the lowest validation loss.
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
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
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
    Save model parameters associated with the lowest validation loss.

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
) -> tuple[int, float, int, int]:
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

    Returns
    -------
    tuple[int, float, int, int]
        Completed epochs, lowest validation loss, its epoch, and consecutive
        epochs without improvement.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

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
    best_epoch = int(checkpoint.get("best_epoch", 0))
    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            completed_epochs - best_epoch if best_epoch > 0 else 0,
        )
    )

    return completed_epochs, best_val_loss, best_epoch, epochs_without_improvement
