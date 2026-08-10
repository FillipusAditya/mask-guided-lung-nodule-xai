"""Train a pretrained ResNet-50 model for image classification."""

from collections import Counter
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
    EarlyStopping,
    append_training_log,
    compute_auc,
    compute_classification_metrics,
    create_training_log,
    plot_accuracy_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_roc_curve,
    plot_validation_metrics_curve,
    save_best_model,
    save_checkpoint,
    save_training_config,
    set_seed,
    update_confusion_matrix,
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

TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TRAIN_DROP_LAST = False
VAL_DROP_LAST = False

#---------------------------------
# MODEL
#---------------------------------
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL_ARCHITECTURE = "ResNet50"
TRAINING_STRATEGY = "feature_extraction"

#---------------------------------
# TRAINING
#---------------------------------

NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()

SEED = 42

#---------------------------------

LEARNING_RATE = 1e-3
BATCH_SIZE = 4
NUM_EPOCHS = 5

WEIGHT_DECAY_OPTM = 1e-4

SCHEDULER_MODE = "min"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_THRESHOLD = 1e-3
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_COOLDOWN = 1
SCHEDULER_MIN_LR = 1e-6
SCHEDULER_MONITOR = "val_loss"

STOPPING_PATIENCE = 20
STOPPING_MODE = "min"
STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MONITOR = "val_loss"

BEST_MODEL_MONITOR = "val_loss"
BEST_MODEL_MODE = "min"
SAVE_LATEST_CHECKPOINT = True

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
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Train the classification model for one epoch.

    This function performs one complete pass over the training dataset,
    computes the loss and gradients for each mini-batch, updates the trainable
    model parameters, and accumulates sample-weighted training metrics.

    Parameters
    ----------
    epoch : int
        Current training epoch as a zero-based index.
    num_epochs : int
        Total number of training epochs.
    model : nn.Module
        Classification model to train.
    train_loader : DataLoader
        DataLoader providing training image-label pairs.
    optimizer : torch.optim.Optimizer
        Optimizer used to update trainable model parameters.
    criterion : nn.Module
        Loss function used to optimize the model.
    device : torch.device
        Device on which the model and mini-batches are stored.

    Returns
    -------
    tuple[float, float]
        Average training loss and classification accuracy for the epoch.
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
        optimizer.zero_grad(set_to_none=True)

        # Perform the forward pass
        outputs = model(images)

        # Compute the training loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
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
    epoch: int,
    num_epochs: int,
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the classification model for one epoch.

    Model parameters are not updated during validation. The function performs
    one complete pass over the validation dataset and computes sample-weighted
    loss, confusion-matrix metrics, and ROC AUC.

    Parameters
    ----------
    epoch : int
        Current validation epoch as a zero-based index.
    num_epochs : int
        Total number of training epochs.
    model : nn.Module
        Classification model to evaluate.
    val_loader : DataLoader
        DataLoader providing validation image-label pairs.
    criterion : nn.Module
        Loss function used to evaluate model predictions.
    device : torch.device
        Device on which the model and mini-batches are stored.

    num_classes : int
        Number of classification classes.

    Returns
    -------
    tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]
        Validation metrics, confusion matrix, targets, and probabilities.
    """

    # Enable evaluation mode
    model.eval()

    # Initialize validation statistics
    running_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )
    all_targets = []
    all_probabilities = []

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

            # Convert logits into class probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = probabilities.argmax(dim=1)

            # Update validation statistics
            running_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += labels.size(0)

            confusion_matrix = update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=labels,
                num_classes=num_classes,
            )

            all_targets.append(labels.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())

            running_accuracy = (
                confusion_matrix.diag().sum().item()
                / total_samples
            )

            # Update the progress bar
            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{100 * running_accuracy:.2f}%",
            )

    # Compute validation metrics
    val_loss = running_loss / total_samples
    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)

    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = val_loss
    metrics["auc"] = compute_auc(
        targets=targets.numpy(),
        probabilities=probabilities.numpy(),
    )

    return metrics, confusion_matrix, targets, probabilities

    
#---------------------------------
# MAIN FUNCTION
#---------------------------------
def main() -> None:
    """Execute the complete pretrained ResNet-50 training pipeline."""

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

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class mappings do not match. "
            f"Train: {train_dataset.class_to_idx}; "
            f"Validation: {val_dataset.class_to_idx}"
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        drop_last=TRAIN_DROP_LAST,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        drop_last=VAL_DROP_LAST,
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

    optimizer = torch.optim.AdamW(
        params=model.fc.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY_OPTM,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        threshold=SCHEDULER_THRESHOLD,
        threshold_mode=SCHEDULER_THRESHOLD_MODE,
        cooldown=SCHEDULER_COOLDOWN,
        min_lr=SCHEDULER_MIN_LR,
    )

    early_stopping = EarlyStopping(
        patience=STOPPING_PATIENCE,
        mode=STOPPING_MODE,
        min_delta=STOPPING_MIN_DELTA,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    optimizer_parameters = optimizer.param_groups[0]

    train_class_counts = Counter(train_dataset.targets)
    val_class_counts = Counter(val_dataset.targets)

    #---------------------------------
    # TRAINING CONFIG
    #---------------------------------

    training_config = {
        "experiment": {
            "result_directory": RESULT_DIR_NAME,
            "output_directory": str(OUTPUT_DIR),
            "created_at": datetime.now().isoformat(
                timespec="seconds",
            ),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "training_strategy": TRAINING_STRATEGY,
            "pretrained_weights": str(WEIGHTS),
            "num_classes": num_classes,
            "backbone_frozen": True,
            "trainable_component": "fc",
            "total_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "data": {
            "train_directory": str(TRAIN_DIR),
            "val_directory": str(VAL_DIR),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            "val_class_to_idx": val_dataset.class_to_idx,
            "train_class_distribution": {
                class_name: train_class_counts[class_index]
                for class_name, class_index
                in train_dataset.class_to_idx.items()
            },
            "val_class_distribution": {
                class_name: val_class_counts[class_index]
                for class_name, class_index
                in val_dataset.class_to_idx.items()
            },
            "preprocessing": str(preprocess),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "seed": SEED,
        },
        "loss": {
            "name": criterion.__class__.__name__,
        },
        "metrics": {
            "names": [
                "confusion_matrix",
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1_score",
                "roc_curve",
                "auc",
            ],
            "binary_positive_class_index": (
                1 if num_classes == 2 else None
            ),
            "binary_positive_class_name": (
                train_dataset.classes[1]
                if num_classes == 2
                else None
            ),
            "multiclass_average": (
                "macro" if num_classes > 2 else None
            ),
            "multiclass_auc_strategy": (
                "one-vs-rest" if num_classes > 2 else None
            ),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "initial_learning_rate": optimizer_parameters["lr"],
            "weight_decay": optimizer_parameters["weight_decay"],
            "betas": optimizer_parameters["betas"],
            "eps": optimizer_parameters["eps"],
        },
        "scheduler": {
            "name": scheduler.__class__.__name__,
            "monitor": SCHEDULER_MONITOR,
            "mode": SCHEDULER_MODE,
            "factor": SCHEDULER_FACTOR,
            "patience": SCHEDULER_PATIENCE,
            "threshold": SCHEDULER_THRESHOLD,
            "threshold_mode": SCHEDULER_THRESHOLD_MODE,
            "cooldown": SCHEDULER_COOLDOWN,
            "min_lr": SCHEDULER_MIN_LR,
        },
        "early_stopping": {
            "monitor": EARLY_STOPPING_MONITOR,
            "mode": STOPPING_MODE,
            "patience": STOPPING_PATIENCE,
            "min_delta": STOPPING_MIN_DELTA,
        },
        "checkpoint": {
            "best_model_monitor": BEST_MODEL_MONITOR,
            "best_model_mode": BEST_MODEL_MODE,
            "save_latest_checkpoint": SAVE_LATEST_CHECKPOINT,
            "best_model_path": str(BEST_MODEL_PATH),
            "checkpoint_path": str(CHECKPOINT_PATH),
        },
        "dataloader": {
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "train_shuffle": TRAIN_SHUFFLE,
            "val_shuffle": VAL_SHUFFLE,
            "train_drop_last": TRAIN_DROP_LAST,
            "val_drop_last": VAL_DROP_LAST,
        },
        "device": {
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
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
        (
            val_metrics,
            val_confusion_matrix,
            val_targets,
            val_probabilities,
        ) = validate_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            num_classes=num_classes,
        )

        validation_metrics = {
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }

        # Apply learning rate scheduler
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(validation_metrics[SCHEDULER_MONITOR])
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler_updated = current_lr != previous_lr
        patience_counter = scheduler.num_bad_epochs

        # Update early-stopping state before logging and checkpointing
        stop = early_stopping(
            metric=validation_metrics[EARLY_STOPPING_MONITOR],
            epoch=epoch + 1,
        )
        early_stop_counter = early_stopping.counter
        
        # Save Best Model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
                    
            save_best_model(
                model=model,
                save_path=BEST_MODEL_PATH,
            )
            
            print(
                f"Best model updated "
                f"(Validation Loss: {val_metrics['loss']:.4f})"
            )
        
        # Save Checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
            precision=val_metrics["precision"],
            f1_score=val_metrics["f1_score"],
            auc_score=val_metrics["auc"],
            best_val_loss=best_val_loss,
            num_classes=num_classes,
            learning_rate=current_lr,
            batch_size=BATCH_SIZE,
            save_path=CHECKPOINT_PATH,
            architecture=MODEL_ARCHITECTURE,
        )
        
        # Write Training Log
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=epoch + 1,
            learning_rate=current_lr,
            scheduler_updated=scheduler_updated,
            patience_counter=patience_counter,
            early_stop_counter=early_stop_counter,
            stopped_early=stop,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
            precision=val_metrics["precision"],
            f1_score=val_metrics["f1_score"],
            auc_score=val_metrics["auc"],
        )
            
        # Display the training and validation results
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        if scheduler_updated:
            print(
                f"Learning Rate        : {previous_lr:.2e} -> "
                f"{current_lr:.2e}"
            )
        else:
            print(f"Learning Rate        : {current_lr:.2e}")
        print(f"Training Loss       : {train_loss:.4f}")
        print(f"Training Accuracy   : {train_accuracy:.2%}")
        print(f"Validation Loss     : {val_metrics['loss']:.4f}")
        print(f"Validation Accuracy : {val_metrics['accuracy']:.2%}")
        print(f"Sensitivity         : {val_metrics['sensitivity']:.4f}")
        print(f"Specificity         : {val_metrics['specificity']:.4f}")
        print(f"Precision           : {val_metrics['precision']:.4f}")
        print(f"F1-score            : {val_metrics['f1_score']:.4f}")
        print(f"ROC AUC             : {val_metrics['auc']:.4f}")
        print()

        if stop:
            print("Early stopping triggered.")
            break
        
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

    plot_validation_metrics_curve(
        history,
        FIGURES_DIR,
    )

    plot_confusion_matrix(
        confusion_matrix=val_confusion_matrix,
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )

    plot_roc_curve(
        targets=val_targets.numpy(),
        probabilities=val_probabilities.numpy(),
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )

if __name__ == "__main__":
    main()
