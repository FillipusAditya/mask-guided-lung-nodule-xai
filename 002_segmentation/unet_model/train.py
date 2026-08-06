from datetime import datetime
from pathlib import Path

import albumentations as A
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from segmentation.unet_model.model import UNET

from segmentation.unet_utils import (
    BCEDiceLoss,
    EarlyStopping,
    create_dataloader,
    create_training_log,
    append_training_log,
    save_training_config,
    save_checkpoint,
    save_best_model,
    update_confusion_matrix,
    compute_segmentation_metrics,
    set_seed,
    plot_all_curves,
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

INPUT_HEIGHT = 512
INPUT_WIDTH = 512

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"

TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TRAIN_DROP_LAST = False
VAL_DROP_LAST = False

#---------------------------------
# TRAINING
#---------------------------------

NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()

SEED = 42

MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = 1
MODEL_ARCHITECTURE = "UNet"


MONITOR_TO_METRIC_KEY = {
    "val_loss": "loss",
    "dice_score": "dice",
}

SAVE_LATEST_CHECKPOINT = True

#---------------------------------

LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_EPOCHS = 150

GRADIENT_CLIP = 1.0

POS_WEIGHT = 20.0

BCE_WEIGHT = 0.4
DICE_WEIGHT = 0.6

PRED_THRESHOLD = 0.5

WEIGHT_DECAY_OPTM = 1e-4

SCHEDULER_MODE = "max"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_THRESHOLD = 1e-3
SCHEDULER_THRESHOLD_MODE = "rel"
SCHEDULER_COOLDOWN = 1
SCHEDULER_MIN_LR = 1e-6

STOPPING_PATIENCE = 20
STOPPING_MODE = "max"

SCHEDULER_MONITOR = "dice_score"
EARLY_STOPPING_MONITOR = "dice_score"
BEST_MODEL_MONITOR = "dice_score"
BEST_MODEL_MODE = "max"

#---------------------------------
# DEVICE
#---------------------------------
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

AMP_ENABLED = DEVICE == "cuda"

#---------------------------------
# TRAIN ONE EPOCH
#---------------------------------
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
        max_gradient_norm: float | None = None,
) -> tuple[float, int]:
    """
    Train the segmentation model for one epoch.

    This function performs one complete pass over the training dataset.
    For each mini-batch, it computes the forward pass, evaluates the loss,
    performs backpropagation, updates the model parameters, and accumulates
    the average training loss.

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
    amp_enabled : bool
        Whether automatic mixed precision is enabled.
    max_gradient_norm : float | None, default=None
        Maximum L2 norm allowed for the gradients.
        If provided, gradient clipping is applied after
        backpropagation using
        ``torch.nn.utils.clip_grad_norm_``.
        If ``None``, gradient clipping is disabled.

    Returns
    -------
    tuple[float, int]
        Average training loss and the number of samples actually processed
        during the epoch.
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

    # Iterate over all training mini-batches
    for images, masks in progress_bar:

        # Move the mini-batch to the selected device
        images = images.to(device)
        masks = masks.to(device)

        # Clear gradients from the previous iteration
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device,
            enabled=amp_enabled,
        ):
            # Forward propagation
            predictions = model(images)

            # Compute the training loss
            loss = criterion(predictions, masks)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        if max_gradient_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_gradient_norm,
            )

        scaler.step(optimizer)

        scaler.update()

        # Update training statistics
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        # Update the progress bar
        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
        )

    # Compute training metrics
    train_loss = running_loss / total_samples

    return train_loss, total_samples

#---------------------------------
# VALIDATE ONE EPOCH
#---------------------------------
def validate_one_epoch(
        epoch: int,
        num_epochs: int,
        model: torch.nn.Module,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        device: str,
        amp_enabled: bool,
        threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate the segmentation model for one epoch.

    This function performs one complete pass over the validation dataset.
    Model parameters are not updated during validation. The function computes
    the average validation loss and evaluates the segmentation performance
    using Dice score, Intersection over Union (IoU), Precision, Sensitivity,
    and Specificity.

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
    amp_enabled : bool
        Whether automatic mixed precision is enabled.
    threshold : float, default=0.5
        Threshold used to convert predicted probabilities into binary
        segmentation masks.

    Returns
    -------
    dict[str, float]
        Dictionary containing the following validation metrics:

        - ``loss`` : Average validation loss.
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

        # Iterate over all validation mini-batches
        for images, masks in progress_bar:

            # Move the mini-batch to the selected device
            images = images.to(device)
            masks = masks.to(device)

            with torch.amp.autocast(
                device_type=device,
                enabled=amp_enabled,
            ):
                predictions = model(images)
                loss = criterion(predictions, masks)

            # Update validation statistics
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # Update the progress bar
            progress_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
            )

            # Convert logits to binary predictions
            predictions = torch.sigmoid(predictions)
            predictions = (
                predictions > threshold
            ).float()

            # update_confusion_matrix
            tp, fp, tn, fn = update_confusion_matrix(
                predictions=predictions,
                targets=masks,
            )
            
            true_positive += tp
            false_positive += fp
            true_negative += tn
            false_negative += fn

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
    
    #---------------------------------
    # PREPARE AUGMENTATIONS PIPELINE
    #---------------------------------

    train_transforms = A.Compose([
        A.Resize(
            height=INPUT_HEIGHT,
            width=INPUT_WIDTH,
        ),

        A.HorizontalFlip(
            p=0.5,
        ),

        A.VerticalFlip(
            p=0.2,
        ),

        A.Rotate(
            limit=15,
            border_mode=0,
            p=0.5,
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.10,
            p=0.3,
        ),

        A.GaussNoise(
            std_range=(0.01, 0.03),
            p=0.2,
        ),

        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(
            height=INPUT_HEIGHT,
            width=INPUT_WIDTH,
        ),
        ToTensorV2(),
    ])
    
    #---------------------------------
    # PREPARE DATASET & DATA LOADER
    #---------------------------------

    train_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split=TRAIN_SPLIT,
        transform=train_transforms,
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
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=VAL_DROP_LAST,
    )
    
    #---------------------------------
    # PREPARE MODEL
    #---------------------------------

    model = UNET(
        in_channels=MODEL_IN_CHANNELS,
        out_channels=MODEL_OUT_CHANNELS,
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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY_OPTM,
    )
    
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
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
    )

    optimizer_parameters = optimizer.param_groups[0]
    
    #---------------------------------
    # TRAINING CONFIG
    #---------------------------------
    
    training_config = {
        "experiment": {
            "result_directory": RESULT_DIR_NAME,
            "created_at": datetime.now().isoformat(
                timespec="seconds",
            ),
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "in_channels": MODEL_IN_CHANNELS,
            "out_channels": MODEL_OUT_CHANNELS,
        },
        "data": {
            "dataset_root": str(DATASET_ROOT),
            "image_height": INPUT_HEIGHT,
            "image_width": INPUT_WIDTH,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "prediction_threshold": PRED_THRESHOLD,
            "seed": SEED,
            "gradient_clip": GRADIENT_CLIP,
        },
        "loss": {
            "name": loss_fn.__class__.__name__,
            "bce_weight": BCE_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "pos_weight": POS_WEIGHT,
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "initial_learning_rate": optimizer_parameters["lr"],
            "weight_decay": optimizer_parameters["weight_decay"],
            "betas": optimizer_parameters["betas"],
            "eps": optimizer_parameters["eps"],
        },
        "amp": {
            "enabled": AMP_ENABLED,
            "dtype": (
                "float16"
                if AMP_ENABLED
                else None
            ),
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
            "device": DEVICE,
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
        save_path=OUTPUT_DIR / "training_config.json",
    )
    
    # Initialize training variables
    best_metric = (
        float("-inf")
        if BEST_MODEL_MODE == "max"
        else float("inf")
    )
    training_start_time = time.perf_counter()
    
    #---------------------------------
    # Training
    #---------------------------------
    
    for epoch in range(NUM_EPOCHS):
        print()
        print("=" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("=" * 60)

        # Start epoch timer
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        # Train One Epoch
        train_loss, train_total_samples = train_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=DEVICE,
            scaler=scaler,
            amp_enabled=AMP_ENABLED,
            max_gradient_norm=GRADIENT_CLIP,
        )
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        train_time_sec = time.perf_counter() - epoch_start_time

        # Validation One Epoch
        val_start_time = time.perf_counter()
        val_metrics = validate_one_epoch(
            epoch=epoch,
            num_epochs=NUM_EPOCHS,
            model=model,
            val_loader=val_loader,
            criterion=loss_fn,
            device=DEVICE,
            amp_enabled=AMP_ENABLED,
            threshold=PRED_THRESHOLD
        )
        if DEVICE == "cuda":
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
            model=model,
            epoch=epoch + 1,
        )
        early_stop_counter = early_stopping.counter

        # Compute epoch duration
        epoch_time = time.perf_counter() - epoch_start_time
        
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

        # Save complete checkpoint
        checkpoint_saved = False
        if SAVE_LATEST_CHECKPOINT:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopping=early_stopping,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                dice_score=val_metrics["dice"],
                iou=val_metrics["iou"],
                precision=val_metrics["precision"],
                sensitivity=val_metrics["sensitivity"],
                specificity=val_metrics["specificity"],
                best_metric=best_metric,
                best_metric_name=BEST_MODEL_MONITOR,
                best_metric_mode=BEST_MODEL_MODE,
                learning_rate=current_lr,
                batch_size=BATCH_SIZE,
                save_path=CHECKPOINT_PATH,
                architecture=MODEL_ARCHITECTURE,
            )
            checkpoint_saved = True

        # Collect end-of-epoch operational metrics
        elapsed_time_sec = time.perf_counter() - training_start_time
        gpu_memory_allocated_mb = (
            torch.cuda.memory_allocated(DEVICE) / (1024 ** 2)
            if DEVICE == "cuda"
            else 0.0
        )
        gpu_memory_reserved_mb = (
            torch.cuda.memory_reserved(DEVICE) / (1024 ** 2)
            if DEVICE == "cuda"
            else 0.0
        )
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        samples_per_sec = (
            train_total_samples / train_time_sec
            if train_time_sec > 0.0
            else 0.0
        )

        # Write Training Log after checkpoint operations complete
        append_training_log(
            log_path=TRAINING_LOG_PATH,
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
            train_batches=train_batches,
            val_batches=val_batches,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
            stopped_early=stop,
            epoch=epoch + 1,
            learning_rate=current_lr,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            dice_score=val_metrics["dice"],
            iou=val_metrics["iou"],
            precision=val_metrics["precision"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
        )
        
        # Display Epoch Summary
        print()

        print("Training")
        if current_lr < previous_lr:
            print(
                f"  Learning rate reduced "
                f"from {previous_lr:.2e} "
                f"to {current_lr:.2e}"
            )
        print(f"  Current LR   : {current_lr:.2e}")
        print(f"  Loss         : {train_loss:.4f}")

        print()

        print("Validation")
        print(f"  Loss         : {val_metrics['loss']:.4f}")
        print(f"  Dice Score   : {val_metrics['dice']:.4f}")
        print(f"  IoU          : {val_metrics['iou']:.4f}")
        print(f"  Precision    : {val_metrics['precision']:.4f}")
        print(f"  Sensitivity  : {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity  : {val_metrics['specificity']:.4f}")
        
        if stop:
            print("Early stopping triggered.")
            break
    
    # ---------------------------------
    # TRAINING VISUALIZATION
    # ---------------------------------

    training_log = pd.read_csv(
        TRAINING_LOG_PATH,
    )

    plot_all_curves(
        training_log=training_log,
        output_dir=OUTPUT_DIR,
    )

    print("Training visualizations generated successfully.")

if __name__ == "__main__":
    main()
