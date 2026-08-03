from pathlib import Path

import albumentations as A
import torch
import torch.optim as optim

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from loss import BCEDiceLoss
from model import UNET
from utils import (
    create_dataloader,
    save_history,
    set_seed,
)


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "_segmentation_dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "segmentation"
    / "unet_model"
    / "train_result"
)


# ---------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 1

NUM_WORKERS = 0
PIN_MEMORY = False

PRED_THRESHOLD = 0.5

SEED = 42

POS_WEIGHT = 20.0

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "dice",
    "iou",
    "precision",
    "sensitivity",
    "specificity",
)

def train_one_epoch(
    loader,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> float:
    """
    Train the model for one epoch.

    Parameters
    ----------
    loader : DataLoader
        Training dataloader.
    model : nn.Module
        Segmentation model.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    loss_fn : nn.Module
        Loss function.

    Returns
    -------
    float
        Average training loss.
    """

    # Switch the model to training mode.
    model.train()

    epoch_loss = 0.0

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=True,
    )

    for images, masks in progress_bar:

        # Move the current batch to the selected device.
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clear gradients from the previous iteration.
        optimizer.zero_grad()

        # Forward propagation.
        predictions = model(images)

        # Compute the training loss.
        loss = loss_fn(
            predictions,
            masks,
        )

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Accumulate the batch loss.
        epoch_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
        )

    # Compute the average training loss.
    epoch_loss /= len(loader)

    return epoch_loss

def validate_one_epoch(
    loader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
) -> dict[str, float]:
    """
    Evaluate the model on the validation dataset.

    Parameters
    ----------
    loader : DataLoader
        Validation dataloader.
    model : nn.Module
        Segmentation model.
    loss_fn : nn.Module
        Loss function.

    Returns
    -------
    dict[str, float]
        Dictionary containing the validation loss
        and segmentation metrics.
    """

    eps = 1e-8

    # Switch the model to evaluation mode.
    model.eval()

    epoch_loss = 0.0

    # Initialize confusion matrix components.
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():

        for images, masks in progress_bar:

            # Move the current batch to the selected device.
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward propagation.
            predictions = model(images)

            # Compute validation loss.
            loss = loss_fn(
                predictions,
                masks,
            )

            epoch_loss += loss.item()

            # Convert logits to binary predictions.
            predictions = torch.sigmoid(predictions)
            predictions = (
                predictions > PRED_THRESHOLD
            ).float()

            # Update confusion matrix.
            true_positive += (
                (predictions == 1) &
                (masks == 1)
            ).sum().item()

            false_positive += (
                (predictions == 1) &
                (masks == 0)
            ).sum().item()

            true_negative += (
                (predictions == 0) &
                (masks == 0)
            ).sum().item()

            false_negative += (
                (predictions == 0) &
                (masks == 1)
            ).sum().item()

    # Compute the average validation loss.
    epoch_loss /= len(loader)

    # Compute segmentation metrics.
    dice = (
        2 * true_positive
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + eps
    )

    iou = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + false_negative
        + eps
    )

    precision = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + eps
    )

    sensitivity = (
        true_positive
    ) / (
        true_positive
        + false_negative
        + eps
    )

    specificity = (
        true_negative
    ) / (
        true_negative
        + false_positive
        + eps
    )

    # Restore training mode.
    model.train()

    return {
        "loss": epoch_loss,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def main() -> None:
    """
    Execute the complete training pipeline.
    """

    # -------------------------------------------------------------
    # Initialize the training environment
    # -------------------------------------------------------------

    # Set random seed.
    set_seed(SEED)

    # Create the output directory.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    
    # -------------------------------------------------------------
    # Create data augmentation pipelines
    # -------------------------------------------------------------

    train_transforms = A.Compose([
        A.Resize(
            height=512,
            width=512,
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(
            height=512,
            width=512,
        ),
        ToTensorV2(),
    ])
    
    # -------------------------------------------------------------
    # Create dataloaders
    # -------------------------------------------------------------

    train_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split="train",
        transform=train_transforms,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split="val",
        transform=val_transforms,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    # -------------------------------------------------------------
    # Initialize the model
    # -------------------------------------------------------------

    model = UNET(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)
    
    # -------------------------------------------------------------
    # Configure the loss function
    # -------------------------------------------------------------

    pos_weight = torch.tensor(
        [POS_WEIGHT],
        device=DEVICE,
    )

    loss_fn = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )
    
    # -------------------------------------------------------------
    # Configure the optimizer
    # -------------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    
    # -------------------------------------------------------------
    # Initialize training variables
    # -------------------------------------------------------------

    best_dice = 0.0

    history = {
        key: []
        for key in HISTORY_KEYS
    }
    
    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 60)

        # ---------------------------------------------------------
        # Training
        # ---------------------------------------------------------

        train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        val_metrics = validate_one_epoch(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
        )

        # ---------------------------------------------------------
        # Store history
        # ---------------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["dice"].append(val_metrics["dice"])
        history["iou"].append(val_metrics["iou"])
        history["precision"].append(val_metrics["precision"])
        history["sensitivity"].append(val_metrics["sensitivity"])
        history["specificity"].append(val_metrics["specificity"])

        # ---------------------------------------------------------
        # Print epoch summary
        # ---------------------------------------------------------

        print()

        print("Training")
        print(f"  Loss         : {train_loss:.4f}")

        print()

        print("Validation")
        print(f"  Loss         : {val_metrics['loss']:.4f}")
        print(f"  Dice Score   : {val_metrics['dice']:.4f}")
        print(f"  IoU          : {val_metrics['iou']:.4f}")
        print(f"  Precision    : {val_metrics['precision']:.4f}")
        print(f"  Sensitivity  : {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity  : {val_metrics['specificity']:.4f}")

        # ---------------------------------------------------------
        # Save best model
        # ---------------------------------------------------------

        if val_metrics["dice"] > best_dice:

            best_dice = val_metrics["dice"]

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            checkpoint_path = (
                OUTPUT_DIR
                / "best_checkpoint.pth.tar"
            )

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print()
            print(
                f"New best Dice Score : "
                f"{best_dice:.4f}"
            )

            print(
                f"Checkpoint saved to:"
            )

            print(
                f"  {checkpoint_path}"
            )