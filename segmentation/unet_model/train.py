import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

from loss import (
    BCEDiceLoss
)


# Training configuration
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False

# LEARNING_RATE = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 8
# NUM_EPOCHS = 100
# NUM_WORKERS = 8
# IMAGE_HEIGHT = 256
# IMAGE_WIDTH = 256
# PIN_MEMORY = True
# LOAD_MODEL = False

# LEARNING_RATE = 1e-4
# DEVICE = "cuda"
# BATCH_SIZE = 8
# NUM_EPOCHS = 100
# NUM_WORKERS = 6
# IMAGE_HEIGHT = 256
# IMAGE_WIDTH = 256
# PIN_MEMORY = True
# LOAD_MODEL = False

# Dataset directories
TRAIN_IMG_DIR = "../../dataset/_lndb/008_consensus_v1_split/train/ct"
TRAIN_MASK_DIR = "../../dataset/_lndb/008_consensus_v1_split/train/mask"
VAL_IMG_DIR = "../../dataset/_lndb/008_consensus_v1_split/val/ct"
VAL_MASK_DIR = "../../dataset/_lndb/008_consensus_v1_split/val/mask"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Train the model for one epoch.
    """

    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        # Move data to the selected device
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward pass with automatic mixed precision
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backpropagation and parameter update
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Display current training loss
        loop.set_postfix(loss=loss.item())


def main():
    """
    Configure data, initialize the model,
    and execute the training process.
    """

    # Data augmentation for the training set
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=10,
                p=0.5,
                border_mode=0,
            ),
            ToTensorV2(),
        ],
    )

    # Preprocessing for the validation set
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
    )

    # Initialize model, loss function, and optimizer
    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    pos_weight = torch.tensor([20.0], device=DEVICE)

    loss_fn = BCEDiceLoss(
        pos_weight=pos_weight,
        bce_weight=0.5,
        dice_weight=0.5,
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create training and validation data loaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    # Optionally resume training from a saved checkpoint
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # Evaluate the model before training
    check_accuracy(val_loader, model, device=DEVICE)

    # Enable mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save the current model checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # Evaluate model performance on the validation set
        check_accuracy(val_loader, model, device=DEVICE)

        # Save predicted masks for visual inspection
        save_predictions_as_imgs(
            val_loader,
            model,
            folder="saved_images/",
            device=DEVICE,
        )


if __name__ == "__main__":
    main()