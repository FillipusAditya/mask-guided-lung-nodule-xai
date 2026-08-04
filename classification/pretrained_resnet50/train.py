from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from classification.utils import (
    append_training_log,
    create_training_log,
    save_training_config,
    plot_accuracy_curve,
    plot_loss_curve,
    save_best_model,
    save_checkpoint,
    set_seed,
)

#---------------------------------
# PROJECT
#---------------------------------
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
    / "classification"
    / "pretrained_resnet50"
    / RESULT_DIR_NAME
)

FIGURES_DIR = (
    OUTPUT_DIR
    / "figures"
)

TRAINING_LOG_PATH = (
    OUTPUT_DIR
    / "training_log.csv"
)

TRAINING_CONFIG_PATH = (
    OUTPUT_DIR
    / "training_config.json"
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

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

#---------------------------------
# DATASET
#---------------------------------
TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_hymenoptera_dataset"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_hymenoptera_dataset"
    / "val"
)

#---------------------------------
# MODEL
#---------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT

#---------------------------------
# TRAINING
#---------------------------------
SEED = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

#---------------------------------
# DEVICE
#---------------------------------
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#---------------------------------
# TRAIN ONE EPOCH
#---------------------------------
def train_one_epoch(
    epoch,
    num_epochs,
    model,
    train_loader,
    optimizer,
    criterion,
    device,
):
    """
    Train the model for one epoch.
    """

    # Enable training mode
    model.train()

    # Initialize training statistics
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Create the training progress bar
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Train]",
        unit="batch",
        leave=False,
    )

    # Iterate over all training mini-batches
    for images, labels in progress_bar:

        # Move the mini-batch to the selected device
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous iteration
        optimizer.zero_grad(
            set_to_none=True
        )

        # Perform the forward pass
        outputs = model(images)

        # Compute the training loss
        loss = criterion(
            outputs,
            labels,
        )

        # Compute gradients
        loss.backward()

        # Update the trainable parameters
        optimizer.step()

        # Convert logits into predicted class indices
        predictions = outputs.argmax(dim=1)

        # Update training statistics
        running_loss += (
            loss.item()
            * labels.size(0)
        )

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        # Update the progress bar
        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            acc=f"{100 * correct_predictions / total_samples:.2f}%"
        )

    # Compute training metrics
    train_loss = running_loss / total_samples
    train_accuracy = correct_predictions / total_samples

    return (
        train_loss,
        train_accuracy,
    )


#---------------------------------
# VALIDATE ONE EPOCH
#---------------------------------
def validate_one_epoch(
    epoch,
    num_epochs,
    model,
    val_loader,
    criterion,
    device,
):
    """
    Validate the model for one epoch.
    """

    # Enable evaluation mode
    model.eval()

    # Initialize validation statistics
    val_running_loss = 0.0
    val_correct_predictions = 0
    val_total_samples = 0

    # Create the validation progress bar
    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]",
        unit="batch",
        leave=False,
    )

    # Disable gradient computation during validation
    with torch.no_grad():

        # Iterate over all validation mini-batches
        for images, labels in progress_bar:

            # Move the mini-batch to the selected device
            images = images.to(device)
            labels = labels.to(device)

            # Perform the forward pass
            outputs = model(images)

            # Compute the validation loss
            loss = criterion(
                outputs,
                labels,
            )

            # Convert logits into predicted class indices
            predictions = outputs.argmax(dim=1)

            # Update validation statistics
            val_running_loss += (
                loss.item()
                * labels.size(0)
            )

            val_correct_predictions += (
                predictions == labels
            ).sum().item()

            val_total_samples += labels.size(0)

            # Update the progress bar
            progress_bar.set_postfix(
                loss=f"{val_running_loss / val_total_samples:.4f}",
                acc=f"{100 * val_correct_predictions / val_total_samples:.2f}%",
            )

    # Compute validation metrics
    val_loss = val_running_loss / val_total_samples
    val_accuracy = val_correct_predictions / val_total_samples

    return (
        val_loss,
        val_accuracy,
    )
    
    
#---------------------------------
# MAIN FUNCTION
#---------------------------------
def main():
    # Set random seed
    set_seed(seed=SEED, deterministic=True)
    
    # Create training log header
    create_training_log(TRAINING_LOG_PATH)
    
    #---------------------------------
    # PREPARE DATASET & DATA LOADER
    #---------------------------------

    preprocess = WEIGHTS.transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=preprocess
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=preprocess
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    #---------------------------------
    # PREPARE MODEL
    #---------------------------------
    
    model = models.resnet50(weights=WEIGHTS)

    num_classes = len(train_dataset.classes)

    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(DEVICE)

    optimizer = torch.optim.Adam(
        params=model.fc.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    
    #---------------------------------
    # TRAINING CONFIG
    #---------------------------------
    training_config = {
        "model": {
            "architecture": "Transfer Learning ResNet50",
            "pretrained_weights": str(WEIGHTS),
            "num_classes": num_classes,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "seed": SEED,
        },
        "dataloader": {
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        },
        "device": {
            "device": str(DEVICE),
        },
    }

    save_training_config(
        config=training_config,
        save_path=TRAINING_CONFIG_PATH,
    )
    
    #---------------------------------
    # TRAINING
    #---------------------------------
    for epoch in range(NUM_EPOCHS):
        # Training One Epoch
        train_loss, train_accuracy = train_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )
        
        # Validation One Epoch
        val_loss, val_accuracy = validate_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
                    
            save_best_model(
                model=model,
                save_path=BEST_MODEL_PATH,
            )
            
            print(
                f"Best model updated "
                f"(Validation Loss: {val_loss:.4f})"
            )
        
        # Save Checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            best_val_loss=best_val_loss,
            num_classes=num_classes,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            save_path=CHECKPOINT_PATH,
        )
        
        # Write Training Log
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
            
        # Display the training and validation results
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"Training Loss       : {train_loss:.4f}")
        print(f"Training Accuracy   : {train_accuracy:.2%}")
        print(f"Validation Loss     : {val_loss:.4f}")
        print(f"Validation Accuracy : {val_accuracy:.2%}")
        print()
        
    #---------------------------------
    # PLOT TRAINING CURVE
    #---------------------------------
    history = pd.read_csv(
        TRAINING_LOG_PATH
    )

    plot_loss_curve(
        history,
        FIGURES_DIR
    )

    plot_accuracy_curve(
        history,
        FIGURES_DIR
    )

if __name__ == "__main__":
    main()