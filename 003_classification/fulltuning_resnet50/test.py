"""Evaluate a fully fine-tuned ResNet-50 on the held-out test split."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from ..pretrained_resnet50 import test as shared_test
from ..utils import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    PROJECT_ROOT
    / "classification"
    / "fulltuning_resnet50"
    / "result_xxxxx"
)
TRAINING_CONFIG_PATH = RESULT_DIR / "training_config.json"
BEST_MODEL_PATH = RESULT_DIR / "best_model.pth"
TEST_OUTPUT_DIR = RESULT_DIR / "test"
TEST_RESULTS_PATH = TEST_OUTPUT_DIR / "test_results.json"
TEST_PREDICTIONS_PATH = TEST_OUTPUT_DIR / "test_predictions.csv"
GRADCAM_OUTPUT_DIR = TEST_OUTPUT_DIR / "gradcam"
GRADCAM_HEATMAP_DIR = TEST_OUTPUT_DIR / "gradcam_npy"

TEST_SPLIT = "test"
TEST_BATCH_SIZE = 4
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
PIN_MEMORY = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_shared_test_module() -> None:
    """Point reused evaluation helpers to this full-fine-tuning run."""

    shared_test.BEST_MODEL_PATH = BEST_MODEL_PATH
    shared_test.TEST_SPLIT = TEST_SPLIT
    shared_test.TEST_BATCH_SIZE = TEST_BATCH_SIZE
    shared_test.NUM_WORKERS = NUM_WORKERS
    shared_test.PERSISTENT_WORKERS = PERSISTENT_WORKERS
    shared_test.PREFETCH_FACTOR = PREFETCH_FACTOR
    shared_test.PIN_MEMORY = PIN_MEMORY
    shared_test.DEVICE = DEVICE


def main() -> None:
    """Run test evaluation using artifacts produced by the paired train.py."""

    _configure_shared_test_module()
    training_config = shared_test.load_training_config(TRAINING_CONFIG_PATH)
    data_config = training_config["data"]
    model_config = training_config["model"]
    metrics_config = training_config.get("metrics", {})

    if model_config.get("training_strategy") != "full_fine_tuning":
        raise ValueError(
            "Expected a full_fine_tuning training artifact, received "
            f"{model_config.get('training_strategy')!r}."
        )
    class_to_idx = {
        class_name: int(class_index)
        for class_name, class_index in data_config["class_to_idx"].items()
    }
    num_classes = int(model_config["num_classes"])
    classification_threshold = float(
        metrics_config.get("classification_threshold", 0.5)
    )
    positive_class_index = int(
        metrics_config.get("binary_positive_class_index", 1)
    )
    if len(class_to_idx) != num_classes:
        raise ValueError(
            "class_to_idx count does not match the model output classes."
        )

    test_transform = shared_test.create_test_transform(training_config)
    test_loader = create_dataloader(
        root_dir=Path(data_config["dataset_root"]),
        metadata_path=Path(data_config["metadata_path"]),
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
    missing_class_indices = set(range(num_classes)) - set(
        test_dataset.targets
    )
    if missing_class_indices:
        missing_class_names = [
            test_dataset.classes[index]
            for index in sorted(missing_class_indices)
        ]
        raise ValueError(
            "The test split must contain every class. Missing classes: "
            f"{missing_class_names}"
        )
    if training_config["loss"]["name"] != "CrossEntropyLoss":
        raise ValueError("Only CrossEntropyLoss test artifacts are supported.")

    model = shared_test.build_model(
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
    ) = shared_test.evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=DEVICE,
        num_classes=num_classes,
        classification_threshold=classification_threshold,
        positive_class_index=positive_class_index,
    )

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gradcam_count = shared_test.generate_gradcam_visualizations(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        figure_output_dir=GRADCAM_OUTPUT_DIR,
        heatmap_output_dir=GRADCAM_HEATMAP_DIR,
        classification_threshold=classification_threshold,
        positive_class_index=positive_class_index,
    )
    predictions_frame = shared_test.build_predictions_dataframe(
        test_loader=test_loader,
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
    )
    predictions_frame.to_csv(TEST_PREDICTIONS_PATH, index=False)

    results = shared_test.build_test_results(
        training_config=training_config,
        test_loader=test_loader,
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        test_time_sec=test_time_sec,
        classification_threshold=classification_threshold,
        positive_class_index=positive_class_index,
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
    with TEST_RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    shared_test.plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=test_dataset.classes,
        output_dir=TEST_OUTPUT_DIR,
    )
    shared_test.plot_roc_curve(
        targets=targets.numpy(),
        probabilities=probabilities.numpy(),
        class_names=test_dataset.classes,
        output_dir=TEST_OUTPUT_DIR,
    )
    shared_test.print_test_results(results)
    print()
    print(f"Test outputs saved to: {TEST_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
