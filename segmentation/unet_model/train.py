from datetime import datetime
from pathlib import Path

import albumentations as A
import torch
import torch.optim as optim

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from segmentation.unet_model.model import UNET

from segmentation.unet_utils import (
    BCEDiceLoss,
    create_dataloader,
    create_training_log,
    append_training_log,
    save_training_config,
    save_checkpoint,
    save_best_model,
    set_seed,
)

# ---------------------------------------------------------------------
# PROJECT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#---------------------------------
# OUTPUT
#---------------------------------
RESULT_DIR_NAME = (
    datetime.now()
    .strftime("result_%Y%m%d_%H%M")
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "segmentation"
    / "unet_model"
    / RESULT_DIR_NAME
)

TRAINING_LOG_PATH = (
    OUTPUT_DIR
    / "training_log.csv"
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pth"
)

CHECKPOINT_PATH = (
    OUTPUT_DIR
    / "checkpoint_latest.pth"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

#---------------------------------
# DATASET
#---------------------------------
DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "_segmentation_dataset"
)

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 1

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

PRED_THRESHOLD = 0.5

SEED = 42

POS_WEIGHT = 20.0

BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

#---------------------------------
# DEVICE
#---------------------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#---------------------------------
# TRAIN ONE EPOCH
#---------------------------------
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

#---------------------------------
# VALIDATE ONE EPOCH
#---------------------------------
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


#---------------------------------
# MAIN FUNCTION
#---------------------------------
def main() -> None:
    """
    Execute the complete training pipeline.
    """
    
    # Set random seed
    set_seed(SEED)

    # Create training log header
    create_training_log(TRAINING_LOG_PATH)
    
    # -------------------------------------------------------------
    # PREPARE AUGMENTATIONS PIPELINE
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
    # PREPARE DATASET & DATA LOADER
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
    # PREPARE MODEL
    # -------------------------------------------------------------

    model = UNET(
        in_channels=1,
        out_channels=1,
    ).to(DEVICE)

    pos_weight = torch.tensor(
        [POS_WEIGHT],
        device=DEVICE,
    )

    loss_fn = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=BCE_WEIGHT,
        dice_weight=DICE_WEIGHT,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    
    #---------------------------------
    # TRAINING CONFIG
    #---------------------------------
    training_config = {
        "architecture": "UNet",
        "in_channels": 1,
        "out_channels": 1,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "loss_function": "BCEDiceLoss",
        "bce_weight": BCE_WEIGHT,
        "dice_weight": DICE_WEIGHT,
        "pos_weight": POS_WEIGHT,
        "prediction_threshold": PRED_THRESHOLD,
        "seed": SEED,
        "device": DEVICE,
    }

    save_training_config(
        config=training_config,
        save_path=OUTPUT_DIR / "training_config.json",
    )
    
    # Initialize training variables
    best_dice = 0.0
    
    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    for epoch in range(NUM_EPOCHS):
        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 60)

        # Train One Epoch
        train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        # Validation One Epoch
        val_metrics = validate_one_epoch(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
        )

        # Write Training Log
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            dice_score=val_metrics["dice"],
            iou=val_metrics["iou"],
            precision=val_metrics["precision"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
        )
        
        # Save Best Model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            save_best_model(
                model=model,
                save_path=BEST_MODEL_PATH,
            )
            
            print(
                f"Best model updated "
                f"(Validation Loss: {val_metrics['loss']:.4f})"
            )
        
        # Save complete checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            dice_score=val_metrics["dice"],
            iou=val_metrics["iou"],
            precision=val_metrics["precision"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
            best_dice=best_dice,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            save_path=CHECKPOINT_PATH,
            architecture="UNet",
        )
        
        # Display Epoch Summary
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

if __name__ == "__main__":
    main()