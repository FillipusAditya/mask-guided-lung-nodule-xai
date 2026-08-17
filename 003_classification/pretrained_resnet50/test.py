"""Evaluate a trained ResNet-50 on the held-out classification test split."""

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from classification.utils import (
    compute_auc,
    compute_classification_metrics,
    create_dataloader,
    plot_confusion_matrix,
    plot_roc_curve,
    update_confusion_matrix,
)


# Project and trained model
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "classification"
    / "pretrained_resnet50"
    / "result_xxxxx"
)

TRAINING_CONFIG_PATH = RESULT_DIR / "training_config.json"
BEST_MODEL_PATH = RESULT_DIR / "best_model.pth"

TEST_OUTPUT_DIR = RESULT_DIR / "test"
TEST_RESULTS_PATH = TEST_OUTPUT_DIR / "test_results.json"
TEST_PREDICTIONS_PATH = TEST_OUTPUT_DIR / "test_predictions.csv"
GRADCAM_OUTPUT_DIR = TEST_OUTPUT_DIR / "gradcam"
GRADCAM_HEATMAP_DIR = TEST_OUTPUT_DIR / "gradcam_npy"


# Test DataLoader
TEST_SPLIT = "test"
TEST_BATCH_SIZE = 4
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()


# Device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


class SimpleGradCAM:
    """Generate Grad-CAM maps from one convolutional model layer."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hook = target_layer.register_forward_hook(
            self._save_activations
        )

    def _save_activations(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Save activations and register a hook for their gradients."""

        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients: torch.Tensor) -> None:
        """Save gradients propagated to the target activations."""

        self.gradients = gradients

    def generate(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized heatmaps, predictions, and probabilities."""

        self.model.zero_grad(set_to_none=True)
        images = images.requires_grad_(True)
        logits = self.model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        if class_indices is None:
            class_indices = predictions

        selected_scores = logits.gather(
            dim=1,
            index=class_indices.unsqueeze(1),
        ).sum()
        selected_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture model tensors.")

        channel_weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )
        heatmaps = torch.relu(
            (channel_weights * self.activations).sum(
                dim=1,
                keepdim=True,
            )
        )
        heatmaps = F.interpolate(
            heatmaps,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        flat_heatmaps = heatmaps.flatten(start_dim=1)
        minimum = flat_heatmaps.min(dim=1).values[:, None, None]
        maximum = flat_heatmaps.max(dim=1).values[:, None, None]
        heatmaps = (heatmaps - minimum) / (maximum - minimum).clamp_min(1e-8)

        return (
            heatmaps.detach().cpu(),
            predictions.detach().cpu(),
            probabilities.detach().cpu(),
        )

    def close(self) -> None:
        """Remove the registered forward hook."""

        self.hook.remove()


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Normalize one CT image to [0, 1] for visualization only."""

    image = image.astype(np.float32, copy=False)
    minimum = float(image.min())
    maximum = float(image.max())

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)

    return (image - minimum) / (maximum - minimum)


def save_gradcam_figure(
    ct_image: np.ndarray,
    heatmap: np.ndarray,
    true_class: str,
    predicted_class: str,
    predicted_probability: float,
    save_path: Path,
) -> None:
    """Save CT, Grad-CAM heatmap, and their overlay in a three-panel layout."""

    display_ct = _normalize_for_display(ct_image)
    if display_ct.shape != heatmap.shape:
        display_ct = (
            F.interpolate(
                torch.from_numpy(display_ct)[None, None],
                size=heatmap.shape,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(display_ct, cmap="gray")
    axes[0].set_title(f"CT\nTrue: {true_class}")

    axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(display_ct, cmap="gray")
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title(
        f"Overlay\nPred: {predicted_class} "
        f"({predicted_probability:.3f})"
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_gradcam_visualizations(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    figure_output_dir: Path,
    heatmap_output_dir: Path,
) -> int:
    """Generate Grad-CAM figures and NumPy heatmaps for all test samples."""

    dataset = test_loader.dataset
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_output_dir.mkdir(parents=True, exist_ok=True)
    gradcam = SimpleGradCAM(
        model=model,
        target_layer=model.layer4[-1],
    )
    sample_index = 0

    try:
        for images, _ in tqdm(
            test_loader,
            desc="Generating Grad-CAM",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=PIN_MEMORY,
            )
            heatmaps, predictions, probabilities = gradcam.generate(images)

            for batch_index in range(images.size(0)):
                row = dataset.metadata.iloc[sample_index]
                filename = str(row["filename"])
                true_class = str(row["label"])
                predicted_index = int(predictions[batch_index])
                predicted_class = dataset.classes[predicted_index]
                predicted_probability = float(
                    probabilities[batch_index, predicted_index]
                )

                ct_path = dataset.get_ct_path(sample_index)
                ct_image = np.load(
                    ct_path,
                    allow_pickle=False,
                )
                heatmap = heatmaps[batch_index].numpy().astype(
                    np.float32,
                    copy=False,
                )
                figure_filename = Path(filename).with_suffix(".png").name
                heatmap_filename = Path(filename).with_suffix(".npy").name

                np.save(
                    heatmap_output_dir / heatmap_filename,
                    heatmap,
                    allow_pickle=False,
                )

                save_gradcam_figure(
                    ct_image=ct_image,
                    heatmap=heatmap,
                    true_class=true_class,
                    predicted_class=predicted_class,
                    predicted_probability=predicted_probability,
                    save_path=figure_output_dir / figure_filename,
                )
                sample_index += 1
    finally:
        gradcam.close()

    if sample_index != len(dataset):
        raise RuntimeError(
            "Grad-CAM output count does not match test dataset size: "
            f"{sample_index} versus {len(dataset)}."
        )

    return sample_index


def load_training_config(config_path: Path) -> dict[str, object]:
    """Load and minimally validate the configuration of a training run."""

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    with open(config_path) as file:
        config = json.load(file)

    required_sections = {"model", "data", "loss", "checkpoint"}
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ValueError(
            "Training configuration is missing required sections: "
            f"{sorted(missing_sections)}"
        )

    return config


def create_test_transform(
    training_config: dict[str, object],
) -> A.Compose:
    """Create the deterministic preprocessing used for test inference."""

    data_config = training_config["data"]

    return A.Compose([
        A.Resize(
            height=int(data_config["image_height"]),
            width=int(data_config["image_width"]),
        ),
        A.Normalize(
            mean=tuple(data_config["normalization_mean"]),
            std=tuple(data_config["normalization_std"]),
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])


def build_model(
    training_config: dict[str, object],
    weights_path: Path,
    device: torch.device,
) -> nn.Module:
    """Reconstruct ResNet-50 and restore the saved best-model weights."""

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Best model weights not found: {weights_path}"
        )

    model_config = training_config["model"]
    architecture = model_config["architecture"]
    if architecture != "ResNet50":
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Expected ResNet50."
        )

    num_classes = int(model_config["num_classes"])
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
    )

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[
    dict[str, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """Evaluate all test batches and return metrics and predictions."""

    running_loss = 0.0
    total_samples = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )
    all_targets = []
    all_predictions = []
    all_probabilities = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    test_start_time = time.perf_counter()

    with torch.inference_mode():
        for images, labels in tqdm(
            test_loader,
            desc="Testing",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=PIN_MEMORY,
            )
            labels = labels.to(
                device,
                non_blocking=PIN_MEMORY,
            )

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            confusion_matrix = update_confusion_matrix(
                confusion_matrix=confusion_matrix,
                predictions=predictions,
                targets=labels,
                num_classes=num_classes,
            )

            all_targets.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_probabilities.append(probabilities.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    test_time_sec = time.perf_counter() - test_start_time

    if total_samples == 0:
        raise RuntimeError("The test DataLoader did not produce any samples.")

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)

    metrics = compute_classification_metrics(confusion_matrix)
    metrics["loss"] = running_loss / total_samples
    metrics["auc"] = compute_auc(
        targets=targets.numpy(),
        probabilities=probabilities.numpy(),
    )

    return (
        metrics,
        confusion_matrix,
        targets,
        predictions,
        probabilities,
        test_time_sec,
    )


def build_predictions_dataframe(
    test_loader: DataLoader,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: torch.Tensor,
) -> pd.DataFrame:
    """Combine sample metadata with targets and model predictions."""

    dataset = test_loader.dataset
    metadata = dataset.metadata.copy().reset_index(drop=True)

    if len(metadata) != len(targets):
        raise RuntimeError(
            "Test metadata and prediction counts do not match: "
            f"{len(metadata)} metadata rows versus {len(targets)} predictions."
        )

    target_indices = targets.numpy()
    prediction_indices = predictions.numpy()
    probability_values = probabilities.numpy()

    metadata["true_index"] = target_indices
    metadata["true_class"] = [
        dataset.classes[index]
        for index in target_indices
    ]
    metadata["predicted_index"] = prediction_indices
    metadata["predicted_class"] = [
        dataset.classes[index]
        for index in prediction_indices
    ]

    for class_index, class_name in enumerate(dataset.classes):
        metadata[f"probability_{class_name}"] = (
            probability_values[:, class_index]
        )

    metadata["is_correct"] = target_indices == prediction_indices

    return metadata


def build_test_results(
    training_config: dict[str, object],
    test_loader: DataLoader,
    metrics: dict[str, float],
    confusion_matrix: torch.Tensor,
    test_time_sec: float,
) -> dict[str, object]:
    """Build the serializable test-results document."""

    dataset = test_loader.dataset
    class_counts = Counter(dataset.targets)
    class_distribution = {
        class_name: class_counts[class_index]
        for class_name, class_index in dataset.class_to_idx.items()
    }

    matrix = confusion_matrix.tolist()
    confusion_result: dict[str, object] = {
        "matrix": matrix,
        "row_definition": "true_class",
        "column_definition": "predicted_class",
    }

    if len(dataset.classes) == 2:
        confusion_result.update({
            "true_negative": matrix[0][0],
            "false_positive": matrix[0][1],
            "false_negative": matrix[1][0],
            "true_positive": matrix[1][1],
        })

    num_samples = len(dataset)
    samples_per_sec = (
        num_samples / test_time_sec
        if test_time_sec > 0.0
        else 0.0
    )

    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "architecture": training_config["model"]["architecture"],
            "weights_path": str(BEST_MODEL_PATH),
            "best_model_monitor": training_config["checkpoint"][
                "best_model_monitor"
            ],
            "best_model_mode": training_config["checkpoint"][
                "best_model_mode"
            ],
        },
        "data": {
            "dataset_root": str(dataset.root_dir),
            "metadata_path": str(dataset.metadata_path),
            "split": TEST_SPLIT,
            "num_samples": num_samples,
            "num_batches": len(test_loader),
            "classes": dataset.classes,
            "class_to_idx": dataset.class_to_idx,
            "class_distribution": class_distribution,
        },
        "metrics": metrics,
        "confusion_matrix": confusion_result,
        "runtime": {
            "test_time_sec": test_time_sec,
            "samples_per_sec": samples_per_sec,
            "batch_size": TEST_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "device": str(DEVICE),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def print_test_results(
    results: dict[str, object],
) -> None:
    """Print the principal test metrics and binary confusion counts."""

    metrics = results["metrics"]
    confusion = results["confusion_matrix"]

    print()
    print("=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Model               : {results['model']['architecture']}")
    print(
        f"Best Model Monitor  : "
        f"{results['model']['best_model_monitor']}"
    )
    print(f"Test Samples        : {results['data']['num_samples']}")
    print(f"Test Loss           : {metrics['loss']:.4f}")
    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"Sensitivity         : {metrics['sensitivity']:.4f}")
    print(f"Specificity         : {metrics['specificity']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"F1-score            : {metrics['f1_score']:.4f}")
    print(f"ROC AUC             : {metrics['auc']:.4f}")

    if "true_positive" in confusion:
        print()
        print(f"True Negative       : {confusion['true_negative']}")
        print(f"False Positive      : {confusion['false_positive']}")
        print(f"False Negative      : {confusion['false_negative']}")
        print(f"True Positive       : {confusion['true_positive']}")

    print()
    print(
        f"Test Time           : "
        f"{results['runtime']['test_time_sec']:.2f} seconds"
    )
    print(
        f"Samples/Second      : "
        f"{results['runtime']['samples_per_sec']:.2f}"
    )


def main() -> None:
    """Run the complete held-out test evaluation pipeline."""

    training_config = load_training_config(TRAINING_CONFIG_PATH)
    data_config = training_config["data"]
    model_config = training_config["model"]

    class_to_idx = {
        class_name: int(class_index)
        for class_name, class_index
        in data_config["class_to_idx"].items()
    }
    num_classes = int(model_config["num_classes"])

    if len(class_to_idx) != num_classes:
        raise ValueError(
            "Number of classes in class_to_idx does not match model output: "
            f"{len(class_to_idx)} versus {num_classes}."
        )

    test_transform = create_test_transform(training_config)
    test_loader = create_dataloader(
        root_dir=Path(data_config["dataset_root"]),
        split=TEST_SPLIT,
        transform=test_transform,
        class_to_idx=class_to_idx,
        ct_path_column=str(
            data_config.get("ct_path_column", "ct_windowed_path")
        ),
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last=False,
    )

    test_dataset = test_loader.dataset
    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError(
            "Test class mapping does not match the training configuration."
        )

    missing_class_indices = (
        set(range(num_classes)) - set(test_dataset.targets)
    )
    if missing_class_indices:
        missing_class_names = [
            test_dataset.classes[index]
            for index in sorted(missing_class_indices)
        ]
        raise ValueError(
            "The test split must contain every class to compute all metrics. "
            f"Missing classes: {missing_class_names}"
        )

    loss_name = training_config["loss"]["name"]
    if loss_name != "CrossEntropyLoss":
        raise ValueError(
            f"Unsupported test loss '{loss_name}'. Expected CrossEntropyLoss."
        )

    model = build_model(
        training_config=training_config,
        weights_path=BEST_MODEL_PATH,
        device=DEVICE,
    )
    criterion = nn.CrossEntropyLoss()

    (
        metrics,
        confusion_matrix,
        targets,
        predictions,
        probabilities,
        test_time_sec,
    ) = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=num_classes,
    )

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gradcam_count = generate_gradcam_visualizations(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        figure_output_dir=GRADCAM_OUTPUT_DIR,
        heatmap_output_dir=GRADCAM_HEATMAP_DIR,
    )

    predictions_df = build_predictions_dataframe(
        test_loader=test_loader,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
    )
    predictions_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    results = build_test_results(
        training_config=training_config,
        test_loader=test_loader,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        test_time_sec=test_time_sec,
    )
    results["gradcam"] = {
        "target_layer": "layer4[-1]",
        "target_class": "predicted_class",
        "num_visualizations": gradcam_count,
        "figure_output_directory": str(GRADCAM_OUTPUT_DIR),
        "heatmap_output_directory": str(GRADCAM_HEATMAP_DIR),
        "heatmap_format": "NumPy float32 [H, W], normalized to [0, 1]",
        "layout": ["ct", "heatmap", "overlay"],
    }
    with open(TEST_RESULTS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=test_dataset.classes,
        output_dir=TEST_OUTPUT_DIR,
    )
    plot_roc_curve(
        targets=targets.numpy(),
        probabilities=probabilities.numpy(),
        class_names=test_dataset.classes,
        output_dir=TEST_OUTPUT_DIR,
    )

    print_test_results(results)
    print()
    print(f"Test outputs saved to: {TEST_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
