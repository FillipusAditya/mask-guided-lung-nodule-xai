"""Train a five-fold ResNet-50 baseline from an explicit JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights

from ..fulltuning_cv_resnet50 import train as engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "003_classification/configs/cv_resnet50.json"

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    CONFIG = json.load(file)

EXPERIMENT_CONFIG = CONFIG["experiment"]
OUTPUT_CONFIG = CONFIG["output"]
DATA_CONFIG = CONFIG["data"]
CV_CONFIG = CONFIG["cross_validation"]
MODEL_CONFIG = CONFIG["model"]
TRAINING_CONFIG = CONFIG["training"]
OPTIMIZER_CONFIG = CONFIG["optimizer"]
DATALOADER_CONFIG = CONFIG["dataloader"]
EARLY_STOPPING_CONFIG = CONFIG["early_stopping"]
CHECKPOINT_CONFIG = CONFIG["checkpoint"]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


EXPERIMENT_ID = str(EXPERIMENT_CONFIG["id"])
EXPERIMENT_COMPONENT = str(EXPERIMENT_CONFIG["component"])
OUTPUT_DIR = (
    resolve_project_path(OUTPUT_CONFIG["root_directory"])
    / EXPERIMENT_ID
    / EXPERIMENT_COMPONENT
)
CONFIG_SNAPSHOT_PATH = OUTPUT_DIR / str(
    OUTPUT_CONFIG["config_snapshot_filename"]
)
DATASET_ROOT = resolve_project_path(DATA_CONFIG["dataset_root"])
METADATA_PATH = resolve_project_path(DATA_CONFIG["metadata_path"])


def _configured_weights() -> ResNet50_Weights | None:
    name = str(MODEL_CONFIG["pretrained_weights"]).upper()
    if name == "DEFAULT":
        return ResNet50_Weights.DEFAULT
    if name == "IMAGENET1K_V2":
        return ResNet50_Weights.IMAGENET1K_V2
    if name == "NONE":
        return None
    raise ValueError(f"Unsupported pretrained_weights: {name}")


def configure_engine() -> None:
    """Apply the JSON settings to the established CV training engine."""

    if str(MODEL_CONFIG["architecture"]) != "ResNet50":
        raise ValueError("cv_resnet50 only supports architecture='ResNet50'.")
    if str(OPTIMIZER_CONFIG["name"]).upper() != "SGD":
        raise ValueError("cv_resnet50 only supports the SGD optimizer.")
    if not bool(EARLY_STOPPING_CONFIG["enabled"]):
        raise ValueError("Early stopping must remain enabled for this baseline.")
    if not bool(EARLY_STOPPING_CONFIG["restore_best_weights"]):
        raise ValueError("restore_best_weights must be true.")
    if (
        str(CV_CONFIG["role_column"]) != "cv_role"
        or str(CV_CONFIG["fold_column"]) != "cv_fold"
    ):
        raise ValueError("The dataset requires cv_role and cv_fold columns.")

    device_name = str(TRAINING_CONFIG["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but unavailable.")
    device = torch.device(device_name)
    weights = _configured_weights()

    # Output and experiment identity.
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.EXPERIMENT_COMPONENT = EXPERIMENT_COMPONENT
    engine.RESULT_DIR_NAME = Path(EXPERIMENT_COMPONENT).name
    engine.OUTPUT_DIR = OUTPUT_DIR
    engine.CONFIG_SNAPSHOT_SOURCE = CONFIG_PATH
    engine.CONFIG_SNAPSHOT_PATH = CONFIG_SNAPSHOT_PATH
    engine.CV_CONFIG_PATH = OUTPUT_DIR / "cv_config.json"
    engine.CV_SUMMARY_PATH = OUTPUT_DIR / "cv_summary.csv"
    engine.CV_SUMMARY_JSON_PATH = OUTPUT_DIR / "cv_summary.json"
    engine.OOF_PREDICTIONS_PATH = OUTPUT_DIR / "out_of_fold_predictions.csv"
    engine.CV_FIGURES_DIR = OUTPUT_DIR / "figures"

    # Dataset and CV.
    engine.DATASET_ROOT = DATASET_ROOT
    engine.METADATA_PATH = METADATA_PATH
    engine.CT_PATH_COLUMN = str(DATA_CONFIG["ct_path_column"])
    engine.INPUT_HEIGHT = int(DATA_CONFIG["input_height"])
    engine.INPUT_WIDTH = int(DATA_CONFIG["input_width"])
    engine.CLASS_TO_IDX = {
        str(name): int(index)
        for name, index in DATA_CONFIG["class_to_idx"].items()
    }
    engine.IMAGENET_MEAN = tuple(DATA_CONFIG["normalization_mean"])
    engine.IMAGENET_STD = tuple(DATA_CONFIG["normalization_std"])
    engine.N_SPLITS = int(CV_CONFIG["num_folds"])
    engine.CV_FOLDS = tuple(range(engine.N_SPLITS))
    engine.DEVELOPMENT_ROLE = str(CV_CONFIG["development_role"])
    engine.HOLDOUT_ROLE = str(CV_CONFIG["holdout_role"])
    engine.HOLDOUT_FOLD = int(CV_CONFIG["holdout_fold"])

    # Model and optimization.
    engine.WEIGHTS = weights
    engine.MODEL_ARCHITECTURE = "ResNet50"
    engine.TRAINING_STRATEGY = str(MODEL_CONFIG["training_strategy"])
    engine.TRAINABLE_COMPONENT = str(MODEL_CONFIG["trainable_component"])
    engine.CLASSIFIER_DROPOUT = float(MODEL_CONFIG["classifier_dropout"])
    engine.CLASSIFICATION_THRESHOLD = float(
        TRAINING_CONFIG["classification_threshold"]
    )
    engine.SEED = int(TRAINING_CONFIG["seed"])
    engine.TRANSFORM_SEED = int(TRAINING_CONFIG["transform_seed"])
    engine.LEARNING_RATE = float(TRAINING_CONFIG["learning_rate"])
    engine.BATCH_SIZE = int(TRAINING_CONFIG["batch_size"])
    engine.NUM_EPOCHS = int(TRAINING_CONFIG["num_epochs"])
    engine.WEIGHT_DECAY_OPTM = float(OPTIMIZER_CONFIG["weight_decay"])
    engine.MOMENTUM_OPTM = float(OPTIMIZER_CONFIG["momentum"])
    engine.NESTEROV_OPTM = bool(OPTIMIZER_CONFIG["nesterov"])
    engine.NUM_WORKERS = int(DATALOADER_CONFIG["num_workers"])
    engine.PERSISTENT_WORKERS = bool(DATALOADER_CONFIG["persistent_workers"])
    engine.PREFETCH_FACTOR = int(DATALOADER_CONFIG["prefetch_factor"])
    engine.PIN_MEMORY = (
        bool(DATALOADER_CONFIG["pin_memory"]) and torch.cuda.is_available()
    )
    engine.TRAIN_SHUFFLE = bool(DATALOADER_CONFIG["train_shuffle"])
    engine.VAL_SHUFFLE = bool(DATALOADER_CONFIG["val_shuffle"])
    engine.TRAIN_DROP_LAST = bool(DATALOADER_CONFIG["train_drop_last"])
    engine.VAL_DROP_LAST = bool(DATALOADER_CONFIG["val_drop_last"])
    engine.BEST_MODEL_MONITOR = str(EARLY_STOPPING_CONFIG["monitor"])
    engine.BEST_MODEL_MODE = str(EARLY_STOPPING_CONFIG["mode"])
    engine.EARLY_STOPPING_ENABLED = True
    engine.EARLY_STOPPING_PATIENCE = int(EARLY_STOPPING_CONFIG["patience"])
    engine.EARLY_STOPPING_MIN_DELTA = float(EARLY_STOPPING_CONFIG["min_delta"])
    engine.EARLY_STOPPING_VERBOSE = bool(EARLY_STOPPING_CONFIG["verbose"])
    engine.SAVE_LATEST_CHECKPOINT = bool(CHECKPOINT_CONFIG["save_latest"])
    engine.DEVICE = device

    # The reused transform/model builders read their own module globals.
    base = engine.base_train
    base.DATASET_ROOT = DATASET_ROOT
    base.METADATA_PATH = METADATA_PATH
    base.CT_PATH_COLUMN = engine.CT_PATH_COLUMN
    base.INPUT_HEIGHT = engine.INPUT_HEIGHT
    base.INPUT_WIDTH = engine.INPUT_WIDTH
    base.CLASS_TO_IDX = engine.CLASS_TO_IDX
    base.IMAGENET_MEAN = engine.IMAGENET_MEAN
    base.IMAGENET_STD = engine.IMAGENET_STD
    base.TRANSFORM_SEED = engine.TRANSFORM_SEED
    base.WEIGHTS = weights
    base.MODEL_ARCHITECTURE = engine.MODEL_ARCHITECTURE
    base.TRAINING_STRATEGY = engine.TRAINING_STRATEGY
    base.TRAINABLE_COMPONENT = engine.TRAINABLE_COMPONENT
    base.CLASSIFIER_DROPOUT = engine.CLASSIFIER_DROPOUT
    base.DEVICE = device


configure_engine()


def main() -> None:
    """Run the baseline with the same five-fold lifecycle as guided training."""

    engine.main()


if __name__ == "__main__":
    main()
