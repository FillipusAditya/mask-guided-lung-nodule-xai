"""Training and validation routines for one segmentation epoch."""

import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader

from tqdm import tqdm

from .metrics import compute_segmentation_metrics, update_confusion_matrix
from .tiles import merge_tiles


def _check_finite(tensor: torch.Tensor, name: str, context: str) -> None:
    """Raise an error when a tensor contains NaN or infinity."""

    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite {name} detected in {context}.")


def train_one_epoch(
    epoch: int,
    num_epochs: int,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    tile_grid_size: int = 4,
) -> float:
    """
    Train the segmentation model for one epoch.

    This function performs one complete pass over the training dataset.
    For each mini-batch, it processes every tile with activation
    checkpointing, reconstructs the full-size logits, computes one Dice loss
    per image, performs backpropagation, and updates the model parameters.

    Parameters
    ----------
    epoch : int
        Current training epoch (zero-based index).
    num_epochs : int
        Total number of training epochs.
    model : torch.nn.Module
        Segmentation model to be trained.
    train_loader : DataLoader
        DataLoader providing training image-mask pairs.
    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters.
    criterion : torch.nn.Module
        Loss function used to optimize the model.
    device : str
        Device on which the model and data are stored (e.g., CPU or CUDA).
    scaler : torch.amp.GradScaler
        Gradient scaler used by automatic mixed precision.
    amp_enabled : bool
        Whether automatic mixed precision is enabled.
    tile_grid_size : int, default=4
        Number of tile rows and columns used to reconstruct each image.

    Returns
    -------
    float
        Average full-image Dice loss for the epoch.
    """

    # Enable training mode
    model.train()

    # Initialize training statistics
    running_loss = 0.0
    total_samples = 0

    # Create the training progress bar
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        unit="batch",
        leave=True,
    )

    # Iterate over all training mini-batches.
    for batch_index, (images, masks) in enumerate(progress_bar):
        batch_size = images.size(0)
        num_tiles = images.size(1)
        prediction_tiles = []

        # Clear gradients before processing the current image batch.
        optimizer.zero_grad(set_to_none=True)

        # Checkpoint each tile so backward only stores one tile's activations.
        for tile_index in range(num_tiles):
            image_tiles = images[:, tile_index].to(device)

            with torch.amp.autocast(device_type=device, enabled=amp_enabled):
                predictions = activation_checkpoint(
                    model, image_tiles, use_reentrant=False
                )

            prediction_tiles.append(predictions)

        # Reconstruct full-size logits and masks before computing Dice loss.
        predictions = merge_tiles(
            torch.stack(prediction_tiles, dim=1),
            grid_size=tile_grid_size,
        )
        targets = merge_tiles(masks, grid_size=tile_grid_size).to(device)
        context = f"training epoch {epoch + 1}, batch {batch_index + 1}"
        _check_finite(predictions, "full-image predictions", context)

        batch_loss = criterion(predictions, targets)
        _check_finite(batch_loss, "full-image loss", context)

        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Weight the batch mean by its sample count for the epoch mean.
        running_loss += batch_loss.item() * batch_size
        total_samples += batch_size

        # Update the progress bar
        progress_bar.set_postfix(val_loss=f"{running_loss / total_samples:.4f}")

    # Compute training metrics
    train_loss = running_loss / total_samples

    return train_loss


def validate_one_epoch(
    epoch: int,
    num_epochs: int,
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    threshold: float = 0.5,
    tile_grid_size: int = 4,
) -> dict[str, float]:
    """
    Evaluate the segmentation model for one epoch.

    This function performs one complete pass over the validation dataset.
    Model parameters are not updated during validation. Tile logits are
    reconstructed before calculating the full-image Dice loss and binary
    segmentation metrics.

    Parameters
    ----------
    epoch : int
        Current training epoch (zero-based index).
    num_epochs : int
        Total number of training epochs.
    model : torch.nn.Module
        Segmentation model to be evaluated.
    val_loader : DataLoader
        DataLoader providing validation image-mask pairs.
    criterion : torch.nn.Module
        Loss function used to evaluate the model.
    device : str
        Device on which the model and data are stored (e.g., CPU or CUDA).
    threshold : float, default=0.5
        Threshold used to convert predicted probabilities into binary
        segmentation masks.
    tile_grid_size : int, default=4
        Number of tile rows and columns used to reconstruct each image.

    Returns
    -------
    dict[str, float]
        Dictionary containing the following validation metrics:

        - ``loss`` : Average full-image validation Dice loss.
        - ``dice`` : Dice similarity coefficient.
        - ``iou`` : Intersection over Union (Jaccard index).
        - ``precision`` : Positive predictive value.
        - ``sensitivity`` : Recall (true positive rate).
        - ``specificity`` : True negative rate.
    """

    # Enable validation mode
    model.eval()

    # Initialize validation statistics
    running_loss = 0.0
    total_samples = 0

    # Initialize confusion matrix components
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    # Create the validation progress bar
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]",
        unit="batch",
        leave=False,
    )

    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate over all validation mini-batches.
        for batch_index, (images, masks) in enumerate(progress_bar):
            batch_size = images.size(0)
            num_tiles = images.size(1)
            logit_tiles = []

            # Process one tile position at a time to reduce GPU memory usage.
            for tile_index in range(num_tiles):
                image_tiles = images[:, tile_index].to(device)
                predictions = model(image_tiles)

                logit_tiles.append(predictions.float().cpu())

            # Compute loss and metrics after reconstructing the full images.
            logits = merge_tiles(
                torch.stack(logit_tiles, dim=1),
                grid_size=tile_grid_size,
            )
            targets = merge_tiles(masks, grid_size=tile_grid_size)
            context = f"validation epoch {epoch + 1}, batch {batch_index + 1}"
            _check_finite(logits, "full-image predictions", context)

            batch_loss = criterion(logits, targets)
            _check_finite(batch_loss, "full-image loss", context)

            predictions = torch.sigmoid(logits)
            predictions = (predictions > threshold).float()

            tp, fp, tn, fn = update_confusion_matrix(
                predictions=predictions, targets=targets
            )

            true_positive += tp
            false_positive += fp
            true_negative += tn
            false_negative += fn

            # Weight the batch mean by its sample count for the epoch mean.
            running_loss += batch_loss.item() * batch_size
            total_samples += batch_size

            # Update the progress bar
            progress_bar.set_postfix(val_loss=f"{running_loss / total_samples:.4f}")

    # Compute validation metrics
    val_loss = running_loss / total_samples

    # Compute segmentation metrics
    metrics = compute_segmentation_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )

    metrics["loss"] = val_loss

    return metrics
