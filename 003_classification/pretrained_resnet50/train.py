"""Train a pretrained ResNet-50 model for image classification."""

from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm

from ..utils import (
    EarlyStopping,
    append_training_log,
    compute_auc,
    compute_classification_metrics,
    create_dataloader,
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
DATASET_ROOT = (
    PROJECT_ROOT
    / "000_dataset"
    / "_segmentation_dataset_v2"
)

CT_PATH_COLUMN = "ct_windowed_path"

INPUT_HEIGHT = 224
INPUT_WIDTH = 224

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

CLASS_TO_IDX = {
    "benign": 0,
    "malignant": 1,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()

SEED = 42

#---------------------------------

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 100

WEIGHT_DECAY_OPTM = 1e-4

MONITOR_TO_METRIC_KEY = {
    "val_loss": "loss",
    "accuracy": "accuracy",
    "precision": "precision",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
    "f1_score": "f1_score",
}

MONITOR_TO_MODE = {
    "val_loss": "min",
    "accuracy": "max",
    "precision": "max",
    "sensitivity": "max",
    "specificity": "max",
    "f1_score": "max",
}

SCHEDULER_MONITOR = "val_loss"
SCHEDULER_MODE = MONITOR_TO_MODE[SCHEDULER_MONITOR]
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_THRESHOLD = 1e-3
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_COOLDOWN = 1
SCHEDULER_MIN_LR = 1e-6

STOPPING_PATIENCE = 15
STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MONITOR = "val_loss"
STOPPING_MODE = MONITOR_TO_MODE[EARLY_STOPPING_MONITOR]

BEST_MODEL_MONITOR = "val_loss"
BEST_MODEL_MODE = MONITOR_TO_MODE[BEST_MODEL_MONITOR]
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
) -> tuple[float, float, int]:
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
    tuple[float, float, int]
        Average training loss, classification accuracy, and number of samples
        processed during the epoch.
    """

    # Keep the frozen pretrained backbone, including its BatchNorm running
    # statistics, in evaluation mode. Only the classifier head is trained.
    model.eval()
    model.fc.train()

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
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

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
        total_samples,
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
            images = images.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)

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

    train_transforms = A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.2),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])

    train_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split=TRAIN_SPLIT,
        transform=train_transforms,
        class_to_idx=CLASS_TO_IDX,
        ct_path_column=CT_PATH_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=TRAIN_DROP_LAST,
    )

    val_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split=VAL_SPLIT,
        transform=val_transforms,
        class_to_idx=CLASS_TO_IDX,
        ct_path_column=CT_PATH_COLUMN,
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=VAL_DROP_LAST,
    )

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Train and validation class mappings do not match. "
            f"Train: {train_dataset.class_to_idx}; "
            f"Validation: {val_dataset.class_to_idx}"
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

    best_metric = (
        float("-inf")
        if BEST_MODEL_MODE == "max"
        else float("inf")
    )

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
            "dataset_root": str(DATASET_ROOT),
            "metadata_path": str(DATASET_ROOT / "split_metadata.csv"),
            "ct_path_column": CT_PATH_COLUMN,
            "image_height": INPUT_HEIGHT,
            "image_width": INPUT_WIDTH,
            "input_channels": 3,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
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
            "train_transforms": str(train_transforms),
            "val_transforms": str(val_transforms),
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
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
            "monitor_to_metric_key": MONITOR_TO_METRIC_KEY,
            "monitor_to_mode": MONITOR_TO_MODE,
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
            "persistent_workers": PERSISTENT_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
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

    training_start_time = time.perf_counter()
    
    #---------------------------------
    # TRAINING
    #---------------------------------
    for epoch in range(NUM_EPOCHS):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        # Training One Epoch
        train_loss, train_accuracy, train_total_samples = train_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_time_sec = time.perf_counter() - epoch_start_time
        
        # Validation One Epoch
        val_start_time = time.perf_counter()
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
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        val_time_sec = time.perf_counter() - val_start_time

        # Apply learning rate scheduler
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(
            val_metrics[
                MONITOR_TO_METRIC_KEY[SCHEDULER_MONITOR]
            ]
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler_updated = current_lr != previous_lr
        patience_counter = scheduler.num_bad_epochs

        # Update early-stopping state before logging and checkpointing
        stop = early_stopping(
            metric=val_metrics[
                MONITOR_TO_METRIC_KEY[EARLY_STOPPING_MONITOR]
            ],
            epoch=epoch + 1,
        )
        early_stop_counter = early_stopping.counter
        
        # Save Best Model
        current_best_metric = val_metrics[
            MONITOR_TO_METRIC_KEY[BEST_MODEL_MONITOR]
        ]
        is_best = (
            current_best_metric > best_metric
            if BEST_MODEL_MODE == "max"
            else current_best_metric < best_metric
        )

        if is_best:
            best_metric = current_best_metric
                    
            save_best_model(
                model=model,
                save_path=BEST_MODEL_PATH,
            )
            
            print(
                f"Best model updated "
                f"({BEST_MODEL_MONITOR}: {current_best_metric:.4f})"
            )
        
        # Save Checkpoint
        checkpoint_saved = False
        if SAVE_LATEST_CHECKPOINT:
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
                best_metric=best_metric,
                best_metric_name=BEST_MODEL_MONITOR,
                best_metric_mode=BEST_MODEL_MODE,
                num_classes=num_classes,
                learning_rate=current_lr,
                batch_size=BATCH_SIZE,
                save_path=CHECKPOINT_PATH,
                architecture=MODEL_ARCHITECTURE,
            )
            checkpoint_saved = True

        epoch_time = time.perf_counter() - epoch_start_time
        elapsed_time_sec = time.perf_counter() - training_start_time
        gpu_memory_allocated_mb = (
            torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
            if DEVICE.type == "cuda" else 0.0
        )
        gpu_memory_reserved_mb = (
            torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
            if DEVICE.type == "cuda" else 0.0
        )
        samples_per_sec = (
            train_total_samples / train_time_sec
            if train_time_sec > 0.0 else 0.0
        )
        
        # Write Training Log
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=epoch + 1,
            epoch_time=epoch_time,
            elapsed_time_sec=elapsed_time_sec,
            is_best=is_best,
            early_stop_counter=early_stop_counter,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            train_time_sec=train_time_sec,
            val_time_sec=val_time_sec,
            scheduler_updated=scheduler_updated,
            patience_counter=patience_counter,
            best_metric=best_metric,
            checkpoint_saved=checkpoint_saved,
            samples_per_sec=samples_per_sec,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            stopped_early=stop,
            learning_rate=current_lr,
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

    # Restore and evaluate the best checkpoint so that the final validation
    # artifacts represent the selected model rather than the last epoch.
    best_model_state = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(best_model_state)

    (
        best_val_metrics,
        best_val_confusion_matrix,
        best_val_targets,
        best_val_probabilities,
    ) = validate_one_epoch(
        epoch=0,
        num_epochs=1,
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=num_classes,
    )

    print(f"Best Model Validation Loss     : {best_val_metrics['loss']:.4f}")
    print(
        "Best Model Validation Accuracy : "
        f"{best_val_metrics['accuracy']:.2%}"
    )
    print(f"Best Model ROC AUC             : {best_val_metrics['auc']:.4f}")

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
        confusion_matrix=best_val_confusion_matrix,
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )

    plot_roc_curve(
        targets=best_val_targets.numpy(),
        probabilities=best_val_probabilities.numpy(),
        class_names=train_dataset.classes,
        output_dir=FIGURES_DIR,
    )

if __name__ == "__main__":
    main()
