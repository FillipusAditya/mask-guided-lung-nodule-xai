"""Evaluate layer-wise XAI fidelity with a Gaussian-blur deletion test.

The program uses the Grad-CAM and LRP maps already saved by classification
inference. It does not retrain a model and does not generate XAI maps again.
For every explanation, the most important pixels are cumulatively replaced by
pixels from a Gaussian-blurred copy of the original CT image. The probability
of the original predicted class is measured after every deletion step.

The supported experiment types are:

* baseline five-fold ResNet-50; and
* five-fold multi-stage direct-guided ResNet-50.

For the direct-guided model, only the CT is perturbed. The segmentation
probability map remains unchanged throughout the deletion curve.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models as torchvision_models
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ID = "fa028196-fd4c-441f-a5d3-3efb9707e7f9"
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiment_results"
    / DEFAULT_EXPERIMENT_ID
    / "classification"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "000_dataset_v2/_segmentation_dataset"
DEFAULT_MODEL_NAMES = (
    "cv_resnet50",
    "multistage_direct_guided_cv_resnet50",
)

LAYERS = ("layer1", "layer2", "layer3", "layer4")
METHOD_DIRECTORIES = {
    "gradcam": "gradcam_npy",
    "lrp": "lrp_npy",
}
DEFAULT_DELETION_PERCENTAGES = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
PRIMARY_BLUR_NAME = "medium"
EPSILON = 1e-12


@dataclass(frozen=True)
class BlurConfiguration:
    """One Gaussian blur setting."""

    name: str
    sigma: float
    kernel_size: int


BLUR_CONFIGURATIONS = {
    "weak": BlurConfiguration("weak", sigma=3.0, kernel_size=19),
    "medium": BlurConfiguration("medium", sigma=5.0, kernel_size=31),
    "strong": BlurConfiguration("strong", sigma=10.0, kernel_size=61),
}


@dataclass
class ModelContext:
    """Loaded model ensemble and the preprocessing settings it requires."""

    name: str
    architecture: str
    result_directory: Path
    models: list[nn.Module]
    guided: bool
    image_height: int
    image_width: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    class_to_index: dict[str, int]
    classes: tuple[str, ...]
    ct_path_column: str
    probability_root: Path | None


CURVE_COLUMNS = (
    "model",
    "architecture",
    "xai_method",
    "layer",
    "blur_name",
    "blur_sigma",
    "blur_kernel_size",
    "deletion_percent",
    "deleted_pixels",
    "total_pixels",
    "dataset",
    "patient_id",
    "nodule_id",
    "filename",
    "true_class",
    "target_class",
    "target_index",
    "correct_prediction",
    "original_probability_cached",
    "original_probability",
    "original_probability_absolute_error",
    "perturbed_probability",
    "absolute_probability_drop",
    "relative_probability",
    "positive_attribution_pixels",
    "map_has_positive_attribution",
    "map_is_constant_after_normalization",
)

CONSISTENCY_COLUMNS = (
    "model",
    "filename",
    "cached_predicted_index",
    "cached_predicted_class",
    "recomputed_predicted_index",
    "recomputed_predicted_class",
    "cached_target_probability",
    "recomputed_target_probability",
    "absolute_probability_error",
    "prediction_matches_cached",
    "probability_within_tolerance",
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Grad-CAM and LRP fidelity using Gaussian-blur deletion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Classification directory containing model result folders.",
    )
    parser.add_argument(
        "--model-dir",
        "--model-dirs",
        dest="model_dirs",
        type=Path,
        nargs="+",
        default=[Path(name) for name in DEFAULT_MODEL_NAMES],
        help=(
            "One or more model result folders. Relative paths are resolved "
            "inside --experiment-dir."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset directory used as the base for relative CT paths.",
    )
    parser.add_argument(
        "--probability-root",
        type=Path,
        default=None,
        help=(
            "Segmentation probability-map directory for direct-guided models. "
            "The local experiment path is detected automatically by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root. Defaults to EXPERIMENT/evaluation/xai_fidelity.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_DIRECTORIES),
        default=list(METHOD_DIRECTORIES),
        help="XAI methods to evaluate.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=LAYERS,
        default=list(LAYERS),
        help="ResNet layers to evaluate.",
    )
    parser.add_argument(
        "--blur-config",
        nargs="+",
        choices=tuple(BLUR_CONFIGURATIONS),
        default=list(BLUR_CONFIGURATIONS),
        help="Gaussian blur configurations to evaluate.",
    )
    parser.add_argument(
        "--deletion-percentages",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELETION_PERCENTAGES),
        metavar="PERCENT",
        help="Cumulative percentages of highest-attribution pixels to delete.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="Number of perturbed images per inference batch.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to accelerate inference.",
    )
    parser.add_argument(
        "--probability-tolerance",
        type=float,
        default=5e-3,
        help=(
            "Maximum absolute difference allowed between recomputed and "
            "cached original target-class probability."
        ),
    )
    parser.add_argument(
        "--strict-original-check",
        action="store_true",
        help=(
            "Stop when recomputed original inference differs from cached "
            "inference. By default, record the issue and continue using the "
            "cached XAI target class."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a compatible interrupted run from fidelity_curve.csv.tmp."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N test samples for a smoke test.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""

    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_model_directory(experiment_dir: Path, path: Path) -> Path:
    """Resolve a model folder relative to the classification experiment."""

    path = path.expanduser()
    if not path.is_absolute():
        path = experiment_dir / path
    return path.resolve()


def select_device(requested: str) -> torch.device:
    """Choose the requested PyTorch device."""

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def validate_percentages(values: Iterable[float]) -> tuple[float, ...]:
    """Validate and sort the cumulative deletion percentages."""

    percentages = tuple(sorted(set(float(value) for value in values)))
    if not percentages:
        raise ValueError("At least one deletion percentage is required.")
    if any(
        not math.isfinite(value) or value <= 0.0 or value > 100.0
        for value in percentages
    ):
        raise ValueError(
            "Deletion percentages must be finite, greater than 0, "
            "and at most 100."
        )
    return percentages


def percentage_label(value: float) -> str:
    """Convert a percentage into a safe column-name suffix."""

    return f"{value:g}".replace(".", "_")


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate one JSON object."""

    if not path.is_file():
        raise FileNotFoundError(f"JSON file was not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return values


def ordered_class_names(class_to_index: dict[str, int]) -> tuple[str, ...]:
    """Return class names ordered by their numeric class index."""

    expected_indices = set(range(len(class_to_index)))
    if set(class_to_index.values()) != expected_indices:
        raise ValueError("Class indices must be contiguous and start at zero.")
    return tuple(
        name
        for name, _ in sorted(class_to_index.items(), key=lambda item: item[1])
    )


def build_baseline_model(config: dict[str, Any]) -> nn.Module:
    """Create the baseline architecture without downloading weights."""

    model = torchvision_models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(
            float(config["model"]["classifier"]["dropout_probability"])
        ),
        nn.Linear(
            model.fc.in_features,
            int(config["model"]["num_classes"]),
        ),
    )
    return model


def load_baseline_context(
    result_directory: Path,
    device: torch.device,
) -> ModelContext:
    """Load the baseline five-fold ensemble and preprocessing settings."""

    config = load_json(result_directory / "fold_0/training_config.json")
    if config.get("model", {}).get("architecture") != "ResNet50":
        raise ValueError(f"Unsupported baseline architecture: {result_directory}")

    data = config["data"]
    class_to_index = {
        str(name): int(index)
        for name, index in data["class_to_idx"].items()
    }
    fold_ids = tuple(
        int(value) for value in config["cross_validation"]["all_folds"]
    )

    loaded_models: list[nn.Module] = []
    for fold in tqdm(fold_ids, desc="Loading baseline folds", unit="fold"):
        checkpoint_path = result_directory / f"fold_{fold}/best_model.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = build_baseline_model(config)
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        loaded_models.append(model.to(device).eval())

    return ModelContext(
        name=result_directory.name,
        architecture="ResNet50",
        result_directory=result_directory,
        models=loaded_models,
        guided=False,
        image_height=int(data.get("image_height", data.get("input_height", 224))),
        image_width=int(data.get("image_width", data.get("input_width", 224))),
        normalization_mean=tuple(float(value) for value in data["normalization_mean"]),
        normalization_std=tuple(float(value) for value in data["normalization_std"]),
        class_to_index=class_to_index,
        classes=ordered_class_names(class_to_index),
        ct_path_column=str(data["ct_path_column"]),
        probability_root=None,
    )


def import_direct_guided_model_class():
    """Import the project direct-guided architecture."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    module = importlib.import_module(
        "003_classification.multistage_direct_guided_cv_resnet50."
        "direct_guided_resnet50"
    )
    return module.MultiStageDirectGuidedResNet50


def find_probability_root(
    experiment_dir: Path,
    configured_value: object,
    override: Path | None,
) -> Path:
    """Find the local U-Net probability-map directory."""

    candidates: list[Path] = []
    if override is not None:
        candidates.append(resolve_project_path(override))

    configured_path = Path(str(configured_value)).expanduser()
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        candidates.append(resolve_project_path(configured_path))

    experiment_root = experiment_dir.parent
    candidates.append(
        experiment_root / "segmentation/unet/inference/probability_npy"
    )

    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_dir():
            return candidate

    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Segmentation probability-map directory was not found. Checked:\n"
        f"{formatted}"
    )


def load_guided_context(
    result_directory: Path,
    experiment_dir: Path,
    probability_root_override: Path | None,
    device: torch.device,
) -> ModelContext:
    """Load the direct-guided five-fold ensemble and its settings."""

    run_config = load_json(result_directory / "run_config.json")
    if run_config.get("architecture") != "MultiStageDirectGuidedResNet50":
        raise ValueError(f"Unsupported guided architecture: {result_directory}")

    config = run_config["comparison_settings"]
    snapshot_name = str(
        config.get("output", {}).get(
            "config_snapshot_filename",
            "multistage_direct_guided_cv_resnet50.json",
        )
    )
    snapshot_path = result_directory / snapshot_name
    if snapshot_path.is_file():
        config = load_json(snapshot_path)

    data = config["data"]
    model_options = run_config["model_options"]
    class_to_index = {
        str(name): int(index)
        for name, index in data["class_to_idx"].items()
    }
    probability_root = find_probability_root(
        experiment_dir=experiment_dir,
        configured_value=data["probability_root"],
        override=probability_root_override,
    )

    model_class = import_direct_guided_model_class()
    number_of_folds = int(config["cross_validation"]["num_folds"])
    loaded_models: list[nn.Module] = []
    for fold in tqdm(
        range(number_of_folds),
        desc="Loading direct-guided folds",
        unit="fold",
    ):
        checkpoint_path = result_directory / f"fold_{fold}/best_model.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = model_class(
            num_classes=len(class_to_index),
            dropout=float(config["model"]["classifier_dropout"]),
            weights=None,
            alpha_layer1=float(model_options["alpha_layer1"]),
            alpha_layer2=float(model_options["alpha_layer2"]),
            alpha_layer3=float(model_options["alpha_layer3"]),
            learnable_alpha=bool(model_options["learnable_alpha"]),
            guidance_resize_mode=str(model_options["guidance_resize_mode"]),
        )
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        loaded_models.append(model.to(device).eval())

    return ModelContext(
        name=result_directory.name,
        architecture="MultiStageDirectGuidedResNet50",
        result_directory=result_directory,
        models=loaded_models,
        guided=True,
        image_height=int(data["input_height"]),
        image_width=int(data["input_width"]),
        normalization_mean=tuple(float(value) for value in data["normalization_mean"]),
        normalization_std=tuple(float(value) for value in data["normalization_std"]),
        class_to_index=class_to_index,
        classes=ordered_class_names(class_to_index),
        ct_path_column=str(data["ct_path_column"]),
        probability_root=probability_root,
    )


def detect_and_load_model(
    result_directory: Path,
    experiment_dir: Path,
    probability_root_override: Path | None,
    device: torch.device,
) -> ModelContext:
    """Detect one supported result type and load its ensemble."""

    if (result_directory / "run_config.json").is_file():
        return load_guided_context(
            result_directory=result_directory,
            experiment_dir=experiment_dir,
            probability_root_override=probability_root_override,
            device=device,
        )
    if (result_directory / "fold_0/training_config.json").is_file():
        return load_baseline_context(result_directory, device)
    raise ValueError(
        "Could not identify a supported model result directory: "
        f"{result_directory}"
    )


def load_prediction_rows(
    context: ModelContext,
    methods: tuple[str, ...],
    layers: tuple[str, ...],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Load test metadata and verify all requested explanation maps."""

    predictions_path = context.result_directory / "test/test_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Test predictions not found: {predictions_path}")
    predictions = pd.read_csv(predictions_path)

    required_columns = {
        "filename",
        context.ct_path_column,
        "label",
        "predicted_index",
        "predicted_class",
        "cv_nodule_id",
    }
    required_columns.update(
        f"probability_{class_name}" for class_name in context.classes
    )
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing_columns)}"
        )

    if max_samples is not None:
        predictions = predictions.iloc[:max_samples].copy()
    if predictions.empty:
        raise ValueError(f"No prediction rows available: {predictions_path}")
    if predictions["filename"].duplicated().any():
        raise ValueError("Prediction filenames must be unique.")

    test_directory = context.result_directory / "test"
    missing_maps: list[Path] = []
    for filename in predictions["filename"]:
        filename = Path(str(filename)).name
        for method in methods:
            for layer in layers:
                map_path = (
                    test_directory
                    / METHOD_DIRECTORIES[method]
                    / layer
                    / filename
                )
                if not map_path.is_file():
                    missing_maps.append(map_path)
    if missing_maps:
        examples = "\n".join(f"  - {path}" for path in missing_maps[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_maps)} explanation maps. Examples:\n"
            f"{examples}"
        )

    if context.guided:
        assert context.probability_root is not None
        missing_probability_maps = [
            context.probability_root / Path(str(filename)).name
            for filename in predictions["filename"]
            if not (
                context.probability_root / Path(str(filename)).name
            ).is_file()
        ]
        if missing_probability_maps:
            examples = "\n".join(
                f"  - {path}" for path in missing_probability_maps[:5]
            )
            raise FileNotFoundError(
                f"Missing {len(missing_probability_maps)} probability maps. "
                f"Examples:\n{examples}"
            )

    return predictions.to_dict(orient="records")


def resolve_data_path(dataset_root: Path, value: object) -> Path:
    """Resolve one absolute or dataset-relative data path."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def load_two_dimensional_array(path: Path, description: str) -> np.ndarray:
    """Load and validate a finite two-dimensional NumPy array."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    values = np.squeeze(np.load(path, allow_pickle=False))
    if values.ndim != 2:
        raise ValueError(
            f"{description} must be 2D after squeeze, received {values.shape}: "
            f"{path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{description} contains NaN or infinity: {path}")
    return values.astype(np.float32, copy=False)


def resize_image(
    image: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Resize one image exactly like the classification input pipeline."""

    if image.shape == (height, width):
        return image.astype(np.float32, copy=True)
    resized = cv2.resize(
        image,
        dsize=(width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized.astype(np.float32, copy=False)


def normalize_ct_batch(images: np.ndarray, context: ModelContext) -> Tensor:
    """Convert unnormalized grayscale CT images to normalized RGB tensors."""

    if images.ndim != 3:
        raise ValueError(
            "CT batch must have shape [batch, height, width], "
            f"received {images.shape}."
        )
    tensor = torch.from_numpy(images.astype(np.float32, copy=False))
    tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor(context.normalization_mean).view(1, 3, 1, 1)
    std = torch.tensor(context.normalization_std).view(1, 3, 1, 1)
    # Albumentations uses multiplication by reciprocal standard deviation.
    # Keeping the same float32 operation reproduces cached inference exactly.
    return (tensor - mean) * torch.reciprocal(std)


def prepare_probability_map(
    path: Path,
    context: ModelContext,
) -> Tensor:
    """Load and resize one direct-guidance probability map."""

    values = load_two_dimensional_array(path, "segmentation probability map")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError(f"Probability map is outside [0, 1]: {path}")
    values = resize_image(
        values,
        height=context.image_height,
        width=context.image_width,
    )
    values = np.clip(values, 0.0, 1.0)
    return torch.from_numpy(values).unsqueeze(0).unsqueeze(0).float()


def ensemble_probabilities(
    context: ModelContext,
    normalized_ct: Tensor,
    probability_map: Tensor | None,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> Tensor:
    """Return mean softmax probabilities from all fold models."""

    probability_batches: list[Tensor] = []
    amp_enabled = use_amp and device.type == "cuda"

    with torch.inference_mode():
        for start in range(0, normalized_ct.shape[0], batch_size):
            end = min(start + batch_size, normalized_ct.shape[0])
            ct_batch = normalized_ct[start:end].to(
                device,
                non_blocking=device.type == "cuda",
            )
            guided_batch = None
            if context.guided:
                if probability_map is None:
                    raise ValueError("Direct-guided inference needs a probability map.")
                guided_batch = probability_map.expand(
                    end - start,
                    -1,
                    -1,
                    -1,
                ).to(device, non_blocking=device.type == "cuda")

            ensemble_sum: Tensor | None = None
            for model in context.models:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    if context.guided:
                        logits = model(ct_batch, guided_batch)
                    else:
                        logits = model(ct_batch)
                    probabilities = torch.softmax(logits.float(), dim=1)
                if ensemble_sum is None:
                    ensemble_sum = probabilities
                else:
                    ensemble_sum = ensemble_sum + probabilities

            assert ensemble_sum is not None
            probability_batches.append(
                (ensemble_sum / len(context.models)).cpu()
            )

    return torch.cat(probability_batches, dim=0)


def prepare_attribution_ranking(
    path: Path,
    height: int,
    width: int,
) -> tuple[np.ndarray, int, bool, bool]:
    """Return a stable descending pixel ranking from positive attribution."""

    attribution = load_two_dimensional_array(path, "explanation map")
    attribution = np.maximum(attribution, 0.0)
    attribution = resize_image(attribution, height=height, width=width)
    positive_pixel_count = int(np.count_nonzero(attribution > 0.0))
    has_positive_attribution = positive_pixel_count > 0

    minimum = float(attribution.min())
    maximum = float(attribution.max())
    value_range = maximum - minimum
    is_constant = value_range <= np.finfo(np.float32).eps
    if is_constant:
        normalized = np.zeros_like(attribution, dtype=np.float32)
    else:
        normalized = (attribution - minimum) / value_range
        normalized = np.clip(normalized, 0.0, 1.0)

    # Stable sorting gives deterministic behavior when attribution values tie.
    ranking = np.argsort(-normalized.reshape(-1), kind="stable")
    return ranking, positive_pixel_count, has_positive_attribution, is_constant


def gaussian_blur(
    original_image: np.ndarray,
    configuration: BlurConfiguration,
) -> np.ndarray:
    """Create one Gaussian-blurred copy of a resized original CT."""

    radius = configuration.kernel_size // 2
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=torch.float32,
    )
    kernel_1d = torch.exp(
        -(coordinates**2) / (2.0 * configuration.sigma**2)
    )
    kernel_1d = kernel_1d / kernel_1d.sum()
    image_tensor = torch.from_numpy(original_image).unsqueeze(0).unsqueeze(0)
    horizontal_kernel = kernel_1d.view(1, 1, 1, -1)
    vertical_kernel = kernel_1d.view(1, 1, -1, 1)

    horizontally_padded = F.pad(
        image_tensor,
        (radius, radius, 0, 0),
        mode="reflect",
    )
    horizontally_blurred = F.conv2d(
        horizontally_padded,
        horizontal_kernel,
    )
    vertically_padded = F.pad(
        horizontally_blurred,
        (0, 0, radius, radius),
        mode="reflect",
    )
    blurred_tensor = F.conv2d(vertically_padded, vertical_kernel)
    blurred = blurred_tensor[0, 0].numpy()

    if blurred.shape != original_image.shape or not np.isfinite(blurred).all():
        raise ValueError("Gaussian blur produced an invalid CT image.")
    return blurred.astype(np.float32, copy=False)


def create_deleted_image(
    original_image: np.ndarray,
    blurred_image: np.ndarray,
    ranking: np.ndarray,
    percentage: float,
) -> tuple[np.ndarray, int]:
    """Replace the exact top percentage of pixels with blurred values."""

    total_pixels = original_image.size
    deleted_pixels = min(
        total_pixels,
        max(1, math.ceil(total_pixels * percentage / 100.0)),
    )
    deleted = original_image.copy().reshape(-1)
    blurred_flat = blurred_image.reshape(-1)
    selected_indices = ranking[:deleted_pixels]
    deleted[selected_indices] = blurred_flat[selected_indices]
    return deleted.reshape(original_image.shape), deleted_pixels


def trapezoid_auc(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Calculate trapezoidal area without relying on a NumPy version alias."""

    widths = np.diff(x_values)
    heights = (y_values[:-1] + y_values[1:]) / 2.0
    return float(np.sum(widths * heights))


def calculate_curve_metrics(
    deletion_percentages: tuple[float, ...],
    original_probability: float,
    perturbed_probabilities: list[float],
) -> dict[str, float]:
    """Calculate raw, range-normalized, and relative deletion AUC."""

    x_values = np.asarray(
        (0.0, *deletion_percentages),
        dtype=np.float64,
    ) / 100.0
    y_values = np.asarray(
        (original_probability, *perturbed_probabilities),
        dtype=np.float64,
    )
    if len(x_values) != len(y_values):
        raise ValueError("Deletion percentages and probabilities do not match.")

    x_range = float(x_values[-1] - x_values[0])
    if x_range <= 0.0:
        raise ValueError("Deletion AUC requires a positive x-axis range.")

    raw_auc = trapezoid_auc(x_values, y_values)
    normalized_auc = raw_auc / x_range
    relative_values = y_values / max(original_probability, EPSILON)
    relative_auc = trapezoid_auc(x_values, relative_values) / x_range

    metrics = {
        "deletion_auc_raw": raw_auc,
        "deletion_auc_normalized": normalized_auc,
        "relative_deletion_auc": relative_auc,
        "maximum_deletion_percent": deletion_percentages[-1],
    }
    for percentage, probability in zip(
        deletion_percentages,
        perturbed_probabilities,
        strict=True,
    ):
        label = percentage_label(percentage)
        metrics[f"probability_at_{label}_percent"] = probability
        metrics[f"probability_drop_at_{label}_percent"] = (
            original_probability - probability
        )
        metrics[f"relative_probability_at_{label}_percent"] = (
            probability / max(original_probability, EPSILON)
        )
    return metrics


def prepare_resume_curve(
    curve_temporary_path: Path,
    context: ModelContext,
    prediction_rows: list[dict[str, Any]],
    methods: tuple[str, ...],
    layers: tuple[str, ...],
    blur_configurations: tuple[BlurConfiguration, ...],
    deletion_percentages: tuple[float, ...],
) -> set[str]:
    """Validate a partial curve file and return completed filenames."""

    curve_frame = pd.read_csv(curve_temporary_path)
    if curve_frame.empty:
        return set()

    missing_columns = set(CURVE_COLUMNS) - set(curve_frame.columns)
    if missing_columns:
        raise ValueError(
            "Cannot resume because the temporary curve is missing columns: "
            f"{sorted(missing_columns)}"
        )

    expected_values = {
        "model": {context.name},
        "xai_method": set(methods),
        "layer": set(layers),
        "blur_name": {
            configuration.name for configuration in blur_configurations
        },
        "deletion_percent": {0.0, *deletion_percentages},
    }
    for column, expected in expected_values.items():
        actual = set(curve_frame[column].unique())
        if actual != expected:
            raise ValueError(
                "Cannot resume because the temporary curve uses a different "
                f"configuration for {column}: expected={sorted(expected)}, "
                f"actual={sorted(actual)}."
            )

    expected_filenames = {
        Path(str(row["filename"])).name for row in prediction_rows
    }
    unknown_filenames = set(curve_frame["filename"]) - expected_filenames
    if unknown_filenames:
        examples = sorted(unknown_filenames)[:5]
        raise ValueError(
            "Cannot resume because the temporary curve contains filenames "
            f"outside the selected test rows: {examples}"
        )

    rows_per_sample = (
        len(methods)
        * len(layers)
        * len(blur_configurations)
        * (len(deletion_percentages) + 1)
    )
    row_counts = curve_frame.groupby("filename").size()
    incomplete_filenames = set(
        row_counts[row_counts.ne(rows_per_sample)].index
    )
    if incomplete_filenames:
        curve_frame = curve_frame[
            ~curve_frame["filename"].isin(incomplete_filenames)
        ].copy()
        write_dataframe_atomic(curve_frame, curve_temporary_path)
        print(
            "Removed incomplete temporary rows for "
            f"{len(incomplete_filenames)} sample(s)."
        )

    completed_filenames = set(curve_frame["filename"].unique())
    print(
        f"Resume found {len(completed_filenames):,} completed "
        f"sample(s) for {context.name}."
    )
    return completed_filenames


def create_slice_frame_from_curve(
    curve_path: Path,
    deletion_percentages: tuple[float, ...],
) -> pd.DataFrame:
    """Calculate one slice-level AUC record from every saved curve."""

    curve_frame = pd.read_csv(curve_path)
    grouping_columns = [
        "model",
        "xai_method",
        "layer",
        "blur_name",
        "filename",
    ]
    base_columns = [
        "model",
        "architecture",
        "dataset",
        "patient_id",
        "nodule_id",
        "filename",
        "true_class",
        "target_class",
        "target_index",
        "correct_prediction",
        "original_probability_cached",
        "original_probability",
        "original_probability_absolute_error",
        "xai_method",
        "layer",
        "blur_name",
        "blur_sigma",
        "blur_kernel_size",
        "positive_attribution_pixels",
        "map_has_positive_attribution",
        "map_is_constant_after_normalization",
    ]
    expected_points = np.asarray(
        (0.0, *deletion_percentages),
        dtype=np.float64,
    )
    slice_records: list[dict[str, Any]] = []

    groups = curve_frame.groupby(grouping_columns, sort=False, dropna=False)
    for group_name, group in groups:
        group = group.sort_values("deletion_percent")
        actual_points = group["deletion_percent"].to_numpy(dtype=np.float64)
        if not np.array_equal(actual_points, expected_points):
            raise ValueError(
                "Incomplete or duplicated deletion curve for "
                f"{group_name}: {actual_points.tolist()}"
            )

        first = group.iloc[0]
        original_probability = float(first["original_probability"])
        perturbed_probabilities = [
            float(value)
            for value in group.iloc[1:]["perturbed_probability"]
        ]
        metrics = calculate_curve_metrics(
            deletion_percentages=deletion_percentages,
            original_probability=original_probability,
            perturbed_probabilities=perturbed_probabilities,
        )
        base_record = {column: first[column] for column in base_columns}
        slice_records.append({**base_record, **metrics})

    slice_frame = pd.DataFrame(slice_records)
    return slice_frame.sort_values(
        ["model", "xai_method", "layer", "blur_name", "nodule_id", "filename"]
    ).reset_index(drop=True)


def mixed_value(values: pd.Series) -> str:
    """Return one shared string value, or 'mixed' when values differ."""

    unique_values = sorted(set(str(value) for value in values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def aggregate_fidelity_results(
    slice_frame: pd.DataFrame,
    deletion_percentages: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate slice AUC first within nodules and then across nodules."""

    nodule_group_columns = [
        "model",
        "architecture",
        "xai_method",
        "layer",
        "blur_name",
        "blur_sigma",
        "blur_kernel_size",
        "dataset",
        "patient_id",
        "nodule_id",
        "true_class",
    ]
    nodule_aggregations: dict[str, tuple[str, Any]] = {
        "target_class": ("target_class", mixed_value),
        "number_of_slices": ("filename", "nunique"),
        "mean_slice_deletion_auc": (
            "deletion_auc_normalized",
            "mean",
        ),
        "std_slice_deletion_auc": (
            "deletion_auc_normalized",
            "std",
        ),
        "median_slice_deletion_auc": (
            "deletion_auc_normalized",
            "median",
        ),
        "mean_slice_relative_deletion_auc": (
            "relative_deletion_auc",
            "mean",
        ),
        "mean_original_probability": (
            "original_probability",
            "mean",
        ),
    }
    for percentage in deletion_percentages:
        label = percentage_label(percentage)
        drop_column = f"probability_drop_at_{label}_percent"
        nodule_aggregations[f"mean_{drop_column}"] = (drop_column, "mean")

    nodule_frame = (
        slice_frame.groupby(nodule_group_columns, as_index=False)
        .agg(**nodule_aggregations)
        .sort_values(
            [
                "model",
                "xai_method",
                "layer",
                "blur_name",
                "nodule_id",
            ]
        )
        .reset_index(drop=True)
    )
    nodule_frame["std_slice_deletion_auc"] = nodule_frame[
        "std_slice_deletion_auc"
    ].fillna(0.0)

    summary_group_columns = [
        "model",
        "architecture",
        "xai_method",
        "layer",
        "blur_name",
        "blur_sigma",
        "blur_kernel_size",
    ]
    summary_aggregations: dict[str, tuple[str, Any]] = {
        "number_of_nodules": ("nodule_id", "count"),
        "number_of_slices": ("number_of_slices", "sum"),
        "mean_deletion_auc_across_nodules": (
            "mean_slice_deletion_auc",
            "mean",
        ),
        "std_deletion_auc_across_nodules": (
            "mean_slice_deletion_auc",
            "std",
        ),
        "median_deletion_auc_across_nodules": (
            "mean_slice_deletion_auc",
            "median",
        ),
        "min_deletion_auc_across_nodules": (
            "mean_slice_deletion_auc",
            "min",
        ),
        "max_deletion_auc_across_nodules": (
            "mean_slice_deletion_auc",
            "max",
        ),
        "mean_relative_deletion_auc_across_nodules": (
            "mean_slice_relative_deletion_auc",
            "mean",
        ),
        "std_relative_deletion_auc_across_nodules": (
            "mean_slice_relative_deletion_auc",
            "std",
        ),
        "mean_original_probability": (
            "mean_original_probability",
            "mean",
        ),
    }
    for percentage in deletion_percentages:
        label = percentage_label(percentage)
        nodule_drop_column = f"mean_probability_drop_at_{label}_percent"
        summary_aggregations[
            f"mean_probability_drop_at_{label}_percent"
        ] = (nodule_drop_column, "mean")

    summary_frame = (
        nodule_frame.groupby(summary_group_columns, as_index=False)
        .agg(**summary_aggregations)
        .sort_values(["model", "xai_method", "layer", "blur_name"])
        .reset_index(drop=True)
    )
    summary_frame["std_deletion_auc_across_nodules"] = summary_frame[
        "std_deletion_auc_across_nodules"
    ].fillna(0.0)
    summary_frame["std_relative_deletion_auc_across_nodules"] = summary_frame[
        "std_relative_deletion_auc_across_nodules"
    ].fillna(0.0)
    summary_frame["primary_metric"] = summary_frame["blur_name"].eq(
        PRIMARY_BLUR_NAME
    )
    return nodule_frame, summary_frame


def create_curve_summary(curve_path: Path) -> pd.DataFrame:
    """Create nodule-macro mean deletion curves from detailed curve rows."""

    curve_frame = pd.read_csv(curve_path)
    nodule_group_columns = [
        "model",
        "xai_method",
        "layer",
        "blur_name",
        "blur_sigma",
        "blur_kernel_size",
        "nodule_id",
        "deletion_percent",
    ]
    nodule_curves = (
        curve_frame.groupby(nodule_group_columns, as_index=False)
        .agg(
            target_probability=("perturbed_probability", "mean"),
            relative_probability=("relative_probability", "mean"),
        )
    )

    summary_group_columns = [
        "model",
        "xai_method",
        "layer",
        "blur_name",
        "blur_sigma",
        "blur_kernel_size",
        "deletion_percent",
    ]
    curve_summary = (
        nodule_curves.groupby(summary_group_columns, as_index=False)
        .agg(
            number_of_nodules=("nodule_id", "count"),
            mean_target_probability=("target_probability", "mean"),
            std_target_probability=("target_probability", "std"),
            mean_relative_probability=("relative_probability", "mean"),
            std_relative_probability=("relative_probability", "std"),
        )
        .sort_values(
            ["model", "xai_method", "blur_name", "layer", "deletion_percent"]
        )
        .reset_index(drop=True)
    )
    for column in ("std_target_probability", "std_relative_probability"):
        curve_summary[column] = curve_summary[column].fillna(0.0)
    return curve_summary


def save_curve_figures(
    curve_summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save one four-layer deletion-curve figure per method and blur."""

    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    model_name = str(curve_summary["model"].iloc[0])

    groups = curve_summary.groupby(["xai_method", "blur_name"], sort=False)
    for (method, blur_name), group in groups:
        figure, axis = plt.subplots(figsize=(8, 5))
        for layer in LAYERS:
            layer_frame = group[group["layer"].eq(layer)].sort_values(
                "deletion_percent"
            )
            if layer_frame.empty:
                continue
            axis.plot(
                layer_frame["deletion_percent"],
                layer_frame["mean_target_probability"],
                marker="o",
                linewidth=2,
                label=layer,
            )

        axis.set_title(
            f"Deletion curve: {model_name}\n{method.upper()} - {blur_name} blur"
        )
        axis.set_xlabel("Deleted highest-attribution pixels (%)")
        axis.set_ylabel("Mean target-class probability across nodules")
        axis.set_xlim(left=0.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
        axis.legend(title="Layer")
        figure.tight_layout()
        figure.savefig(
            figure_directory / f"{method}_{blur_name}_deletion_curve.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)


def curve_row(
    base: dict[str, Any],
    deletion_percent: float,
    deleted_pixels: int,
    total_pixels: int,
    original_probability: float,
    perturbed_probability: float,
) -> dict[str, Any]:
    """Build one tidy deletion-curve record."""

    return {
        **base,
        "deletion_percent": deletion_percent,
        "deleted_pixels": deleted_pixels,
        "total_pixels": total_pixels,
        "perturbed_probability": perturbed_probability,
        "absolute_probability_drop": (
            original_probability - perturbed_probability
        ),
        "relative_probability": (
            perturbed_probability / max(original_probability, EPSILON)
        ),
    }


def evaluate_model(
    context: ModelContext,
    prediction_rows: list[dict[str, Any]],
    dataset_root: Path,
    methods: tuple[str, ...],
    layers: tuple[str, ...],
    blur_configurations: tuple[BlurConfiguration, ...],
    deletion_percentages: tuple[float, ...],
    device: torch.device,
    inference_batch_size: int,
    use_amp: bool,
    probability_tolerance: float,
    strict_original_check: bool,
    resume: bool,
    output_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run deletion fidelity for one model ensemble."""

    output_directory.mkdir(parents=True, exist_ok=True)
    curve_path = output_directory / "fidelity_curve.csv"
    curve_temporary_path = curve_path.with_suffix(".csv.tmp")
    consistency_path = output_directory / "inference_consistency_issues.csv"
    consistency_temporary_path = consistency_path.with_suffix(".csv.tmp")
    started = time.perf_counter()

    if resume and not curve_temporary_path.is_file() and curve_path.is_file():
        curve_path.replace(curve_temporary_path)
        if (
            consistency_path.is_file()
            and not consistency_temporary_path.is_file()
        ):
            consistency_path.replace(consistency_temporary_path)

    resume_existing_curve = (
        resume
        and curve_temporary_path.is_file()
        and curve_temporary_path.stat().st_size > 0
    )
    completed_filenames: set[str] = set()
    if resume_existing_curve:
        completed_filenames = prepare_resume_curve(
            curve_temporary_path=curve_temporary_path,
            context=context,
            prediction_rows=prediction_rows,
            methods=methods,
            layers=layers,
            blur_configurations=blur_configurations,
            deletion_percentages=deletion_percentages,
        )
    elif resume:
        print(
            f"No temporary curve found for {context.name}; starting a new run."
        )

    remaining_rows = [
        row
        for row in prediction_rows
        if Path(str(row["filename"])).name not in completed_filenames
    ]
    curve_mode = "a" if resume_existing_curve else "w"
    resume_existing_issues = (
        resume_existing_curve
        and consistency_temporary_path.is_file()
        and consistency_temporary_path.stat().st_size > 0
    )
    consistency_mode = "a" if resume_existing_issues else "w"

    with (
        curve_temporary_path.open(
            curve_mode,
            encoding="utf-8",
            newline="",
        ) as file,
        consistency_temporary_path.open(
            consistency_mode,
            encoding="utf-8",
            newline="",
        ) as consistency_file,
    ):
        writer = csv.DictWriter(file, fieldnames=CURVE_COLUMNS)
        consistency_writer = csv.DictWriter(
            consistency_file,
            fieldnames=CONSISTENCY_COLUMNS,
        )
        if not resume_existing_curve:
            writer.writeheader()
        if not resume_existing_issues:
            consistency_writer.writeheader()

        progress = tqdm(
            remaining_rows,
            desc=f"Fidelity {context.name}",
            unit="slice",
        )
        for row in progress:
            filename = Path(str(row["filename"])).name
            ct_path = resolve_data_path(
                dataset_root,
                row[context.ct_path_column],
            )
            original_ct = load_two_dimensional_array(ct_path, "CT image")
            original_ct = resize_image(
                original_ct,
                height=context.image_height,
                width=context.image_width,
            )
            total_pixels = original_ct.size

            probability_map = None
            if context.guided:
                assert context.probability_root is not None
                probability_map = prepare_probability_map(
                    context.probability_root / filename,
                    context,
                )

            normalized_original = normalize_ct_batch(
                original_ct[np.newaxis, ...],
                context,
            )
            original_probabilities = ensemble_probabilities(
                context=context,
                normalized_ct=normalized_original,
                probability_map=probability_map,
                device=device,
                batch_size=1,
                use_amp=use_amp,
            )[0]

            target_index = int(row["predicted_index"])
            if target_index < 0 or target_index >= len(context.classes):
                raise ValueError(
                    f"Invalid predicted_index for {filename}: {target_index}"
                )
            target_class = context.classes[target_index]
            if str(row["predicted_class"]) != target_class:
                raise ValueError(
                    f"predicted_class and predicted_index disagree for {filename}."
                )
            recomputed_index = int(original_probabilities.argmax().item())
            recomputed_class = context.classes[recomputed_index]
            original_probability = float(original_probabilities[target_index])
            cached_probability = float(row[f"probability_{target_class}"])
            probability_error = abs(original_probability - cached_probability)
            prediction_matches_cached = recomputed_index == target_index
            probability_within_tolerance = (
                probability_error <= probability_tolerance
            )
            if not prediction_matches_cached or not probability_within_tolerance:
                consistency_writer.writerow(
                    {
                        "model": context.name,
                        "filename": filename,
                        "cached_predicted_index": target_index,
                        "cached_predicted_class": target_class,
                        "recomputed_predicted_index": recomputed_index,
                        "recomputed_predicted_class": recomputed_class,
                        "cached_target_probability": cached_probability,
                        "recomputed_target_probability": original_probability,
                        "absolute_probability_error": probability_error,
                        "prediction_matches_cached": prediction_matches_cached,
                        "probability_within_tolerance": (
                            probability_within_tolerance
                        ),
                    }
                )
                consistency_file.flush()
                message = (
                    "Original-inference consistency issue for "
                    f"{filename}: cached={target_class} "
                    f"({cached_probability:.8f}), "
                    f"recomputed={recomputed_class} "
                    f"(cached-target probability={original_probability:.8f}), "
                    f"error={probability_error:.8f}. The cached class remains "
                    "the XAI target."
                )
                if strict_original_check:
                    raise RuntimeError(message)
                tqdm.write(f"WARNING: {message}")

            attribution_details: dict[
                tuple[str, str],
                tuple[np.ndarray, int, bool, bool],
            ] = {}
            test_directory = context.result_directory / "test"
            for method in methods:
                for layer in layers:
                    explanation_path = (
                        test_directory
                        / METHOD_DIRECTORIES[method]
                        / layer
                        / filename
                    )
                    attribution_details[(method, layer)] = (
                        prepare_attribution_ranking(
                            explanation_path,
                            height=context.image_height,
                            width=context.image_width,
                        )
                    )

            nodule_id = str(row.get("cv_nodule_id") or row.get("nodule_id"))
            if not nodule_id or nodule_id == "nan":
                raise ValueError(f"Missing nodule identifier for {filename}.")
            true_class = str(row["label"])
            metadata = {
                "model": context.name,
                "architecture": context.architecture,
                "dataset": str(row.get("dataset", "")),
                "patient_id": str(row.get("patient_id", "")),
                "nodule_id": nodule_id,
                "filename": filename,
                "true_class": true_class,
                "target_class": target_class,
                "target_index": target_index,
                "correct_prediction": true_class == target_class,
                "original_probability_cached": cached_probability,
                "original_probability": original_probability,
                "original_probability_absolute_error": probability_error,
            }

            for blur_configuration in blur_configurations:
                blurred_ct = gaussian_blur(original_ct, blur_configuration)
                deleted_images: list[np.ndarray] = []
                task_details: list[tuple[str, str, float, int]] = []

                for method in methods:
                    for layer in layers:
                        ranking = attribution_details[(method, layer)][0]
                        for percentage in deletion_percentages:
                            deleted_image, deleted_pixel_count = (
                                create_deleted_image(
                                    original_image=original_ct,
                                    blurred_image=blurred_ct,
                                    ranking=ranking,
                                    percentage=percentage,
                                )
                            )
                            deleted_images.append(deleted_image)
                            task_details.append(
                                (
                                    method,
                                    layer,
                                    percentage,
                                    deleted_pixel_count,
                                )
                            )

                deleted_batch = normalize_ct_batch(
                    np.stack(deleted_images),
                    context,
                )
                deleted_probabilities = ensemble_probabilities(
                    context=context,
                    normalized_ct=deleted_batch,
                    probability_map=probability_map,
                    device=device,
                    batch_size=inference_batch_size,
                    use_amp=use_amp,
                )[:, target_index].numpy()

                probability_by_curve: dict[
                    tuple[str, str],
                    list[float],
                ] = {
                    (method, layer): []
                    for method in methods
                    for layer in layers
                }
                for task, perturbed_probability in zip(
                    task_details,
                    deleted_probabilities,
                    strict=True,
                ):
                    method, layer, percentage, deleted_pixel_count = task
                    probability_by_curve[(method, layer)].append(
                        float(perturbed_probability)
                    )

                for method in methods:
                    for layer in layers:
                        (
                            _,
                            positive_pixel_count,
                            has_positive_attribution,
                            is_constant,
                        ) = attribution_details[(method, layer)]
                        base_record = {
                            **metadata,
                            "xai_method": method,
                            "layer": layer,
                            "blur_name": blur_configuration.name,
                            "blur_sigma": blur_configuration.sigma,
                            "blur_kernel_size": blur_configuration.kernel_size,
                            "positive_attribution_pixels": positive_pixel_count,
                            "map_has_positive_attribution": (
                                has_positive_attribution
                            ),
                            "map_is_constant_after_normalization": is_constant,
                        }
                        writer.writerow(
                            curve_row(
                                base=base_record,
                                deletion_percent=0.0,
                                deleted_pixels=0,
                                total_pixels=total_pixels,
                                original_probability=original_probability,
                                perturbed_probability=original_probability,
                            )
                        )

                        curve_probabilities = probability_by_curve[
                            (method, layer)
                        ]
                        for percentage, deleted_pixel_count, probability in zip(
                            deletion_percentages,
                            (
                                math.ceil(total_pixels * value / 100.0)
                                for value in deletion_percentages
                            ),
                            curve_probabilities,
                            strict=True,
                        ):
                            writer.writerow(
                                curve_row(
                                    base=base_record,
                                    deletion_percent=percentage,
                                    deleted_pixels=min(
                                        total_pixels,
                                        deleted_pixel_count,
                                    ),
                                    total_pixels=total_pixels,
                                    original_probability=original_probability,
                                    perturbed_probability=probability,
                                )
                            )

            file.flush()

    curve_temporary_path.replace(curve_path)
    consistency_temporary_path.replace(consistency_path)

    slice_frame = create_slice_frame_from_curve(
        curve_path=curve_path,
        deletion_percentages=deletion_percentages,
    )
    consistency_frame = pd.read_csv(consistency_path)
    nodule_frame, summary_frame = aggregate_fidelity_results(
        slice_frame=slice_frame,
        deletion_percentages=deletion_percentages,
    )
    curve_summary = create_curve_summary(curve_path)
    primary_frame = summary_frame[
        summary_frame["blur_name"].eq(PRIMARY_BLUR_NAME)
    ].reset_index(drop=True)

    write_dataframe_atomic(slice_frame, output_directory / "fidelity_slice.csv")
    write_dataframe_atomic(nodule_frame, output_directory / "fidelity_nodule.csv")
    write_dataframe_atomic(summary_frame, output_directory / "fidelity_summary.csv")
    write_dataframe_atomic(
        primary_frame,
        output_directory / "fidelity_primary_medium_blur.csv",
    )
    write_dataframe_atomic(
        curve_summary,
        output_directory / "fidelity_curve_summary.csv",
    )
    save_curve_figures(curve_summary, output_directory)

    elapsed_seconds = time.perf_counter() - started
    status = {
        "model": context.name,
        "architecture": context.architecture,
        "status": "evaluated",
        "number_of_folds": len(context.models),
        "number_of_samples": len(prediction_rows),
        "number_of_resumed_samples": len(completed_filenames),
        "number_of_nodules": int(slice_frame["nodule_id"].nunique()),
        "number_of_inference_consistency_issues": len(consistency_frame),
        "methods": ",".join(methods),
        "layers": ",".join(layers),
        "blur_configurations": ",".join(
            configuration.name for configuration in blur_configurations
        ),
        "deletion_percentages": ",".join(
            f"{value:g}" for value in deletion_percentages
        ),
        "device": str(device),
        "amp_enabled": use_amp and device.type == "cuda",
        "runtime_seconds": elapsed_seconds,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_directory": str(output_directory),
    }
    write_dataframe_atomic(
        pd.DataFrame([status]),
        output_directory / "model_status.csv",
    )
    return summary_frame, primary_frame, status


def determine_output_root(
    experiment_dir: Path,
    requested_output: Path | None,
    max_samples: int | None,
) -> Path:
    """Choose a full-run or smoke-test output root."""

    if requested_output is None:
        output_root = experiment_dir.parent / "evaluation/xai_fidelity"
    else:
        output_root = resolve_project_path(requested_output)
    if max_samples is not None:
        output_root = output_root.with_name(
            f"{output_root.name}_smoke_{max_samples}"
        )
    return output_root


def release_models(context: ModelContext) -> None:
    """Release one ensemble before loading the next model type."""

    context.models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run fidelity evaluation for all requested model result folders."""

    arguments = parse_arguments()
    if arguments.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive.")
    if arguments.max_samples is not None and arguments.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if arguments.probability_tolerance < 0.0:
        raise ValueError("--probability-tolerance cannot be negative.")

    deletion_percentages = validate_percentages(
        arguments.deletion_percentages
    )
    experiment_dir = resolve_project_path(arguments.experiment_dir)
    dataset_root = resolve_project_path(arguments.dataset_root)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    model_directories = [
        resolve_model_directory(experiment_dir, value)
        for value in arguments.model_dirs
    ]
    missing_model_directories = [
        path for path in model_directories if not path.is_dir()
    ]
    if missing_model_directories:
        formatted = "\n".join(
            f"  - {path}" for path in missing_model_directories
        )
        raise FileNotFoundError(
            f"Model result directories were not found:\n{formatted}"
        )

    methods = tuple(dict.fromkeys(arguments.methods))
    layers = tuple(dict.fromkeys(arguments.layers))
    blur_configurations = tuple(
        BLUR_CONFIGURATIONS[name]
        for name in dict.fromkeys(arguments.blur_config)
    )
    device = select_device(arguments.device)
    output_root = determine_output_root(
        experiment_dir=experiment_dir,
        requested_output=arguments.output_dir,
        max_samples=arguments.max_samples,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print("XAI fidelity deletion evaluation")
    print(f"Experiment directory : {experiment_dir}")
    print(f"Dataset root         : {dataset_root}")
    print(f"Models               : {[path.name for path in model_directories]}")
    print(f"Methods              : {methods}")
    print(f"Layers               : {layers}")
    print(
        "Blur configurations  : "
        f"{[configuration.name for configuration in blur_configurations]}"
    )
    print(f"Deletion percentages : {deletion_percentages}")
    print(f"Device               : {device}")
    print(f"AMP                  : {arguments.amp and device.type == 'cuda'}")
    print(f"Resume               : {arguments.resume}")
    print(f"Strict original check: {arguments.strict_original_check}")
    print(f"Output root          : {output_root}")

    summary_frames: list[pd.DataFrame] = []
    primary_frames: list[pd.DataFrame] = []
    status_records: list[dict[str, Any]] = []

    for result_directory in model_directories:
        print(f"\nLoading {result_directory.name}")
        context = detect_and_load_model(
            result_directory=result_directory,
            experiment_dir=experiment_dir,
            probability_root_override=arguments.probability_root,
            device=device,
        )
        try:
            prediction_rows = load_prediction_rows(
                context=context,
                methods=methods,
                layers=layers,
                max_samples=arguments.max_samples,
            )
            model_output_directory = output_root / context.name
            summary, primary, status = evaluate_model(
                context=context,
                prediction_rows=prediction_rows,
                dataset_root=dataset_root,
                methods=methods,
                layers=layers,
                blur_configurations=blur_configurations,
                deletion_percentages=deletion_percentages,
                device=device,
                inference_batch_size=arguments.inference_batch_size,
                use_amp=arguments.amp,
                probability_tolerance=arguments.probability_tolerance,
                strict_original_check=arguments.strict_original_check,
                resume=arguments.resume,
                output_directory=model_output_directory,
            )
            summary_frames.append(summary)
            primary_frames.append(primary)
            status_records.append(status)
        finally:
            release_models(context)

    combined_summary = pd.concat(summary_frames, ignore_index=True)
    combined_primary = pd.concat(primary_frames, ignore_index=True)
    write_dataframe_atomic(
        combined_summary,
        output_root / "fidelity_summary_combined.csv",
    )
    write_dataframe_atomic(
        combined_primary,
        output_root / "fidelity_primary_medium_blur_combined.csv",
    )
    write_dataframe_atomic(
        pd.DataFrame(status_records),
        output_root / "model_status.csv",
    )

    print("\nFidelity evaluation complete.")
    print(f"Evaluated models : {len(status_records)}")
    print(f"Summary rows     : {len(combined_summary):,}")
    print(f"Outputs          : {output_root}")


if __name__ == "__main__":
    main()
