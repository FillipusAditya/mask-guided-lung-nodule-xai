import json
from pathlib import Path
import sys
from uuid import uuid4

import albumentations as A
import pandas as pd

import torch
import torch.optim as optim

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEGMENTATION_ROOT))

from unet_arch import UNET
from unet_utils import (
    DiceLoss,
    append_training_log,
    create_dataloader,
    create_training_log,
    load_checkpoint,
    plot_all_curves,
    save_best_model,
    save_checkpoint,
    set_seed,
    synchronize_training_log,
    train_one_epoch,
    validate_one_epoch,
)

# load experiment settings from the json configuration file
CONFIG_PATH = SEGMENTATION_ROOT / "configs" / "unet_holdout.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

OUTPUT_CONFIG = CONFIG["output"]
DATA_CONFIG = CONFIG["data"]
TRAINING_CONFIG = CONFIG["training"]
OPTIMIZER_CONFIG = CONFIG["optimizer"]
DATALOADER_CONFIG = CONFIG["dataloader"]
AMP_CONFIG = CONFIG["amp"]
CHECKPOINT_CONFIG = CONFIG["checkpoint"]

# resolve the project root from the training script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# resolve the checkpoint selected for continuing an existing experiment
resume_checkpoint_path = CHECKPOINT_CONFIG["resume_checkpoint_path"]
RESUME_CHECKPOINT_PATH = (
    PROJECT_ROOT / resume_checkpoint_path
    if resume_checkpoint_path is not None
    else None
)

# reuse the checkpoint directory when resuming or create one for a new experiment
if RESUME_CHECKPOINT_PATH is not None:
    OUTPUT_DIR = RESUME_CHECKPOINT_PATH.parent
else:
    RUN_ID = uuid4()
    RESULT_DIR_NAME = str(RUN_ID)
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_CONFIG["root_directory"] / RESULT_DIR_NAME

TRAINING_LOG_PATH = OUTPUT_DIR / "training_log.csv"
LAST_CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pth"

# prepare dataset, split, and tiling settings
DATASET_ROOT = PROJECT_ROOT / DATA_CONFIG["dataset_root"]
SPLIT_METHOD = DATA_CONFIG["split_method"]
METADATA_FILENAME = DATA_CONFIG["metadata_filename"]
FOLD = DATA_CONFIG["fold"]
IMAGE_PATH_COLUMN = DATA_CONFIG["image_path_column"]

INPUT_HEIGHT = DATA_CONFIG["input_height"]
INPUT_WIDTH = DATA_CONFIG["input_width"]
TILE_GRID_SIZE = DATA_CONFIG["tile_grid_size"]

TRAIN_SHUFFLE = DATALOADER_CONFIG["train_shuffle"]

# prepare dataloader, model, loss, optimizer, and checkpoint settings
NUM_WORKERS = DATALOADER_CONFIG["num_workers"]
PERSISTENT_WORKERS = DATALOADER_CONFIG["persistent_workers"]
PREFETCH_FACTOR = DATALOADER_CONFIG["prefetch_factor"]
PIN_MEMORY = torch.cuda.is_available()

SEED = TRAINING_CONFIG["seed"]

LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
TOTAL_EPOCHS = TRAINING_CONFIG["total_epochs"]
EARLY_STOPPING_PATIENCE = TRAINING_CONFIG["early_stopping_patience"]

PRED_THRESHOLD = TRAINING_CONFIG["prediction_threshold"]

WEIGHT_DECAY_OPTM = OPTIMIZER_CONFIG["weight_decay"]

# select the runtime device and configure automatic mixed precision
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_AMP_ENABLED = AMP_CONFIG["training_enabled"] and DEVICE == "cuda"


def main() -> None:
    """
    Execute the complete training pipeline.
    """

    # create output files only when starting a new experiment
    if RESUME_CHECKPOINT_PATH is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        create_training_log(TRAINING_LOG_PATH)
    elif not TRAINING_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Training log not found for resumed experiment: {TRAINING_LOG_PATH}"
        )

    # initialize random number generators for reproducible training
    set_seed(SEED)

    # resize each image-mask pair and convert both arrays to tensors
    transforms = A.Compose([A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH)])

    # create data loaders for the configured training and validation splits
    train_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split="train",
        batch_size=BATCH_SIZE,
        transform=transforms,
        split_method=SPLIT_METHOD,
        metadata_filename=METADATA_FILENAME,
        fold=FOLD,
        image_path_column=IMAGE_PATH_COLUMN,
        tile_grid_size=TILE_GRID_SIZE,
        shuffle=TRAIN_SHUFFLE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_loader = create_dataloader(
        root_dir=DATASET_ROOT,
        split="val",
        batch_size=BATCH_SIZE,
        transform=transforms,
        split_method=SPLIT_METHOD,
        metadata_filename=METADATA_FILENAME,
        fold=FOLD,
        image_path_column=IMAGE_PATH_COLUMN,
        tile_grid_size=TILE_GRID_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # initialize the model, loss function, optimizer, and gradient scaler
    model = UNET(
        features=[16, 32, 64, 128],
    ).to(DEVICE)

    loss_fn = DiceLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY_OPTM
    )

    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN_AMP_ENABLED)

    # restore all training states when continuing from an earlier run
    completed_epochs = 0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    if RESUME_CHECKPOINT_PATH is not None:
        (
            completed_epochs,
            best_val_loss,
            best_epoch,
            epochs_without_improvement,
        ) = load_checkpoint(
            checkpoint_path=RESUME_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )

        # discard log entries that were written after the latest checkpoint
        synchronize_training_log(TRAINING_LOG_PATH, completed_epochs)

        print(f"Training resumed from epoch {completed_epochs}.")

    if completed_epochs >= TOTAL_EPOCHS:
        print(f"Training has already completed {TOTAL_EPOCHS} epochs.")
        return

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            "Early stopping has already been reached after "
            f"{epochs_without_improvement} epochs without improvement."
        )
        return

    # continue training until the configured total number of epochs
    for epoch in range(completed_epochs, TOTAL_EPOCHS):
        # update model parameters using every training mini-batch
        train_loss = train_one_epoch(
            epoch=epoch,
            num_epochs=TOTAL_EPOCHS,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=DEVICE,
            scaler=scaler,
            amp_enabled=TRAIN_AMP_ENABLED,
            tile_grid_size=TILE_GRID_SIZE,
        )

        # evaluate the updated model on the validation split
        val_metrics = validate_one_epoch(
            epoch=epoch,
            num_epochs=TOTAL_EPOCHS,
            model=model,
            val_loader=val_loader,
            criterion=loss_fn,
            device=DEVICE,
            threshold=PRED_THRESHOLD,
            tile_grid_size=TILE_GRID_SIZE,
        )

        current_epoch = epoch + 1
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            best_epoch = current_epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # replace the best model when validation loss reaches a new minimum
        if is_best:
            best_model_path = OUTPUT_DIR / "best_model.pth"

            save_best_model(model=model, save_path=best_model_path)

            print(
                f"New best validation loss: {best_val_loss:.4f} "
                f"at epoch {best_epoch}."
            )

        # write the completed epoch metrics immediately before its checkpoint
        append_training_log(
            log_path=TRAINING_LOG_PATH,
            epoch=current_epoch,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            dice_score=val_metrics["dice"],
            iou=val_metrics["iou"],
            precision=val_metrics["precision"],
            sensitivity=val_metrics["sensitivity"],
            specificity=val_metrics["specificity"],
        )

        # keep the latest state in the current experiment result directory
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=current_epoch,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
            save_path=LAST_CHECKPOINT_PATH,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping at epoch {current_epoch}: validation loss "
                f"did not improve for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    # generate metric curves from the completed training log
    training_log = pd.read_csv(TRAINING_LOG_PATH)

    plot_all_curves(training_log=training_log, output_dir=OUTPUT_DIR)

    print("Training visualizations generated successfully.")
    print(f"Last checkpoint saved to: {LAST_CHECKPOINT_PATH}")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
