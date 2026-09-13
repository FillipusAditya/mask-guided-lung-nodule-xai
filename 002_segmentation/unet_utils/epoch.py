"""Training and validation routines for one segmentation epoch."""

import torch
import torch.optim as optim
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader

from tqdm import tqdm

from .metrics import compute_segmentation_metrics
from .tiles import merge_tiles


def _check_finite(tensor: torch.Tensor, name: str, context: str) -> None:
    """Raise an error when a tensor contains NaN or infinity."""

    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"Non-finite {name} detected in {context}.")


def _confusion_matrix_counts(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Return TP, FP, TN, and FN counts without synchronizing the device."""

    return torch.stack(
        (
            ((predictions == 1) & (targets == 1)).sum(),
            ((predictions == 1) & (targets == 0)).sum(),
            ((predictions == 0) & (targets == 0)).sum(),
            ((predictions == 0) & (targets == 1)).sum(),
        )
    )


def _forward_tiles(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str,
    amp_enabled: bool,
    parallel_tile_processing: bool,
    tile_chunk_size: int | None,
    activation_checkpointing: bool,
) -> torch.Tensor:
    """
    Run the model on tiled images and restore the tile batch dimensions.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model used to predict each tile.
    images : torch.Tensor
        Tiled images with shape ``[B, T, C, H, W]`` on the runtime device.
    device : str
        Device type used by automatic mixed precision.
    amp_enabled : bool
        Whether automatic mixed precision is enabled.
    parallel_tile_processing : bool
        Whether to combine the image and tile dimensions before model forward.
    tile_chunk_size : int, optional
        Maximum number of flattened tiles in one model forward. If ``None``,
        all flattened tiles are processed together.
    activation_checkpointing : bool
        Whether to use activation checkpointing during gradient-enabled forward.

    Returns
    -------
    torch.Tensor
        Predicted tiles with shape ``[B, T, C_out, H, W]``.
    """

    if tile_chunk_size is not None and tile_chunk_size <= 0:
        raise ValueError("tile_chunk_size must be a positive integer or None.")

    def forward_tile_batch(tile_batch: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=device, enabled=amp_enabled):
            if activation_checkpointing and torch.is_grad_enabled():
                return activation_checkpoint(model, tile_batch, use_reentrant=False)

            return model(tile_batch)

    batch_size, num_tiles, channels, tile_height, tile_width = images.shape

    if parallel_tile_processing:
        flattened_tiles = images.reshape(
            batch_size * num_tiles,
            channels,
            tile_height,
            tile_width,
        )
        chunk_size = tile_chunk_size or flattened_tiles.size(0)
        prediction_chunks = [
            forward_tile_batch(tile_batch)
            for tile_batch in flattened_tiles.split(chunk_size, dim=0)
        ]
        flattened_predictions = torch.cat(prediction_chunks, dim=0)

        return flattened_predictions.reshape(
            batch_size,
            num_tiles,
            *flattened_predictions.shape[1:],
        )

    prediction_tiles = [
        forward_tile_batch(images[:, tile_index])
        for tile_index in range(num_tiles)
    ]

    return torch.stack(prediction_tiles, dim=1)


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
    parallel_tile_processing: bool = False,
    tile_chunk_size: int | None = None,
    activation_checkpointing: bool = True,
    non_blocking_transfer: bool = False,
) -> float:
    """
    Train the segmentation model for one epoch.

    This function performs one complete pass over the training dataset.
    For each mini-batch, it processes the image tiles, reconstructs the
    full-size logits, computes one segmentation loss per image, performs
    backpropagation, and updates the model parameters.

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
    parallel_tile_processing : bool, default=False
        Whether to combine the image and tile dimensions for parallel forward.
    tile_chunk_size : int, optional
        Maximum number of flattened tiles in one model forward. If ``None``,
        all flattened tiles are processed together.
    activation_checkpointing : bool, default=True
        Whether to trade additional computation for lower activation memory.
    non_blocking_transfer : bool, default=False
        Whether host-to-device tensor transfers may run asynchronously.

    Returns
    -------
    float
        Average full-image segmentation loss for the epoch.
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

        # Clear gradients before processing the current image batch.
        optimizer.zero_grad(set_to_none=True)

        # Transfer the complete tiled batch once so parallel processing can keep
        # the accelerator supplied with a larger model batch.
        images = images.to(device, non_blocking=non_blocking_transfer)
        masks = masks.to(device, non_blocking=non_blocking_transfer)

        prediction_tiles = _forward_tiles(
            model=model,
            images=images,
            device=device,
            amp_enabled=amp_enabled,
            parallel_tile_processing=parallel_tile_processing,
            tile_chunk_size=tile_chunk_size,
            activation_checkpointing=activation_checkpointing,
        )

        # Reconstruct full-size logits and masks before computing the loss.
        predictions = merge_tiles(
            prediction_tiles,
            grid_size=tile_grid_size,
        )
        targets = merge_tiles(masks, grid_size=tile_grid_size)
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
        progress_bar.set_postfix(train_loss=f"{running_loss / total_samples:.4f}")

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
    amp_enabled: bool = False,
    parallel_tile_processing: bool = False,
    tile_chunk_size: int | None = None,
    validation_on_gpu: bool = False,
    non_blocking_transfer: bool = False,
) -> dict[str, float]:
    """
    Evaluate the segmentation model for one epoch.

    This function performs one complete pass over the validation dataset.
    Model parameters are not updated during validation. Tile logits are
    reconstructed before calculating the full-image segmentation loss and binary
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
    amp_enabled : bool, default=False
        Whether automatic mixed precision is enabled for model forward.
    parallel_tile_processing : bool, default=False
        Whether to combine the image and tile dimensions for parallel forward.
    tile_chunk_size : int, optional
        Maximum number of flattened tiles in one model forward. If ``None``,
        all flattened tiles are processed together.
    validation_on_gpu : bool, default=False
        Whether to calculate full-image loss and metrics on the runtime device.
    non_blocking_transfer : bool, default=False
        Whether host-to-device tensor transfers may run asynchronously.

    Returns
    -------
    dict[str, float]
        Dictionary containing the following validation metrics:

        - ``loss`` : Average full-image validation segmentation loss.
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

    # Keep confusion matrix counts on the selected validation device and only
    # synchronize them with the CPU once after the epoch.
    metric_device = device if validation_on_gpu else "cpu"
    confusion_matrix = torch.zeros(4, dtype=torch.int64, device=metric_device)

    # Create the validation progress bar
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]",
        unit="batch",
        leave=False,
    )

    # Disable autograd bookkeeping during validation for lower overhead.
    with torch.inference_mode():
        # Iterate over all validation mini-batches.
        for batch_index, (images, masks) in enumerate(progress_bar):
            batch_size = images.size(0)

            images = images.to(device, non_blocking=non_blocking_transfer)
            prediction_tiles = _forward_tiles(
                model=model,
                images=images,
                device=device,
                amp_enabled=amp_enabled,
                parallel_tile_processing=parallel_tile_processing,
                tile_chunk_size=tile_chunk_size,
                activation_checkpointing=False,
            ).float()

            if validation_on_gpu:
                masks = masks.to(device, non_blocking=non_blocking_transfer)
            else:
                prediction_tiles = prediction_tiles.cpu()

            # Compute loss and metrics after reconstructing the full images.
            logits = merge_tiles(
                prediction_tiles,
                grid_size=tile_grid_size,
            )
            targets = merge_tiles(masks, grid_size=tile_grid_size)
            context = f"validation epoch {epoch + 1}, batch {batch_index + 1}"
            _check_finite(logits, "full-image predictions", context)

            batch_loss = criterion(logits, targets)
            _check_finite(batch_loss, "full-image loss", context)

            predictions = torch.sigmoid(logits)
            predictions = (predictions > threshold).float()

            confusion_matrix += _confusion_matrix_counts(
                predictions=predictions,
                targets=targets,
            )

            # Weight the batch mean by its sample count for the epoch mean.
            running_loss += batch_loss.item() * batch_size
            total_samples += batch_size

            # Update the progress bar
            progress_bar.set_postfix(val_loss=f"{running_loss / total_samples:.4f}")

    # Compute validation metrics
    val_loss = running_loss / total_samples
    true_positive, false_positive, true_negative, false_negative = (
        int(value) for value in confusion_matrix.cpu().tolist()
    )

    # Compute segmentation metrics
    metrics = compute_segmentation_metrics(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )

    metrics["loss"] = val_loss

    return metrics
