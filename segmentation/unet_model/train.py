from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNET
from loss import BCEDiceLoss
from utils import (
    create_dataloader,
    check_accuracy,
    save_checkpoint,
)


# Project directory
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

# Training configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 1
NUM_WORKERS = 2
PIN_MEMORY = True


def train_one_epoch(loader, model, optimizer, loss_fn,):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, masks in progress_bar:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader)

    return epoch_loss


def main():
    """
    Execute the complete training pipeline.
    """

    train_transforms = A.Compose([
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        ToTensorV2(),
    ])

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

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)

    pos_weight = torch.tensor(
        [20.0],
        device=DEVICE,
    )

    loss_fn = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        print(f"Training Loss : {train_loss:.4f}")

        check_accuracy(val_loader, model, device=DEVICE)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(
            checkpoint,
            filename= OUTPUT_DIR / "checkpoint.pth.tar",
        )


if __name__ == "__main__":
    main()