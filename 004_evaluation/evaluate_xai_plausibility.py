"""Evaluate layer-wise XAI plausibility with top-k IoU.

This program evaluates Grad-CAM and LRP maps that were saved by each model's
test/inference program.  It does not run the neural networks again.

For every test slice, the program:

1. loads one explanation map and its radiologist nodule mask;
2. keeps only positive attribution (values greater than zero);
3. resizes the explanation map to the original mask resolution;
4. normalizes the resized map to the range 0--1;
5. selects exactly the top-k percent of pixels; and
6. calculates intersection-over-union (IoU).

Slice IoUs are first averaged within each nodule.  The final mean and standard
deviation are then calculated across nodules, so nodules with more CT slices do
not receive more weight.

The inference programs in ``003_classification`` create explanations for the
predicted class.  Therefore, ``predicted_class`` in ``test_predictions.csv`` is
reported as the explanation target in the output files.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ID = "fa028196-fd4c-441f-a5d3-3efb9707e7f9"
DEFAULT_TOP_K_PERCENTAGES = (1.0, 5.0, 10.0, 20.0)
DEFAULT_PRIMARY_TOP_K = 10.0
LAYERS = ("layer1", "layer2", "layer3", "layer4")

# One entry is a model family.  The first existing directory in each family is
# used.  The aliases make the program ready for the guided run even when its
# final directory name differs slightly.
DEFAULT_MODEL_FAMILIES = (
    ("cv_resnet50",),
    ("multistage_direct_guided_cv_resnet50",),
    (
        "guided_cv_resnet_50",
        "guided_cv_resnet50",
        "segmentation_guided_cv_resnet50",
        "guided_resnet50",
    ),
)

METHOD_DIRECTORIES = {
    "gradcam": "gradcam_npy",
    "lrp": "lrp_npy",
}

SLICE_COLUMNS = (
    "model",
    "xai_method",
    "layer",
    "top_k_percent",
    "primary_metric",
    "dataset",
    "patient_id",
    "nodule_id",
    "filename",
    "true_class",
    "target_class",
    "correct_prediction",
    "mask_height",
    "mask_width",
    "ground_truth_pixels",
    "selected_explanation_pixels",
    "intersection_pixels",
    "union_pixels",
    "attribution_threshold",
    "positive_attribution_pixels",
    "map_has_positive_attribution",
    "map_is_constant_after_normalization",
    "iou",
)

NODULE_COLUMNS = (
    "model",
    "xai_method",
    "layer",
    "top_k_percent",
    "primary_metric",
    "dataset",
    "patient_id",
    "nodule_id",
    "true_class",
    "target_class",
    "number_of_slices",
    "mean_slice_iou",
    "std_slice_iou",
    "min_slice_iou",
    "max_slice_iou",
)

SUMMARY_COLUMNS = (
    "model",
    "xai_method",
    "layer",
    "top_k_percent",
    "primary_metric",
    "number_of_nodules",
    "number_of_slices",
    "mean_iou_across_nodules",
    "std_iou_across_nodules",
    "median_iou_across_nodules",
    "min_iou_across_nodules",
    "max_iou_across_nodules",
)

STATUS_COLUMNS = (
    "requested_model",
    "resolved_model",
    "result_directory",
    "status",
    "number_of_samples",
    "message",
)


def parse_arguments() -> argparse.Namespace:
    """Read command-line options."""

    default_experiment_dir = (
        PROJECT_ROOT
        / "experiment_results"
        / DEFAULT_EXPERIMENT_ID
        / "classification"
    )

    parser = argparse.ArgumentParser(
        description="Evaluate Grad-CAM and LRP plausibility using top-k IoU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=default_experiment_dir,
        help="Classification directory containing the model result folders.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "000_dataset_v2",
        help=(
            "Dataset root. It may be 000_dataset_v2 or its "
            "_segmentation_dataset subdirectory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "CSV output directory. By default, files are saved below the "
            "experiment in evaluation/xai_plausibility."
        ),
    )
    parser.add_argument(
        "--model-dir",
        "--model-dirs",
        dest="model_dirs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more model folders to evaluate. A folder name is resolved "
            "inside --experiment-dir; an absolute folder path is also accepted."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=float,
        nargs="+",
        default=list(DEFAULT_TOP_K_PERCENTAGES),
        metavar="PERCENT",
        help="Top attribution percentages used to create binary masks.",
    )
    parser.add_argument(
        "--primary-top-k",
        type=float,
        default=DEFAULT_PRIMARY_TOP_K,
        help="Top-k percentage copied to the primary-metric CSV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples for a quick smoke test.",
    )
    return parser.parse_args()


def resolve_from_project(path: Path) -> Path:
    """Resolve a user path relative to the project root."""

    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_top_k_values(
    top_k_values: Iterable[float],
    primary_top_k: float,
) -> tuple[float, ...]:
    """Validate and sort the requested top-k percentages."""

    values = tuple(sorted(set(float(value) for value in top_k_values)))
    if not values:
        raise ValueError("At least one --top-k value is required.")
    invalid_value_exists = any(
        not math.isfinite(value) or value <= 0.0 or value > 100.0
        for value in values
    )
    if invalid_value_exists:
        raise ValueError("Every --top-k value must be greater than 0 and at most 100.")
    if not any(math.isclose(value, primary_top_k) for value in values):
        raise ValueError("--primary-top-k must also be present in --top-k.")
    return values


def find_mask_base_directory(dataset_root: Path) -> Path:
    """Return the directory against which CSV mask paths are resolved."""

    dataset_root = resolve_from_project(dataset_root)
    metadata_directory = dataset_root / "_segmentation_dataset"
    if metadata_directory.is_dir():
        return metadata_directory
    if dataset_root.is_dir():
        return dataset_root
    raise FileNotFoundError(f"Dataset root was not found: {dataset_root}")


def resolve_model_directory(experiment_dir: Path, value: Path) -> Path:
    """Resolve an explicit model folder."""

    value = value.expanduser()
    if not value.is_absolute():
        value = experiment_dir / value
    return value.resolve()


def discover_default_models(
    experiment_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Find one result directory for each default model family."""

    found_models: list[Path] = []
    status_rows: list[dict[str, Any]] = []

    for family in DEFAULT_MODEL_FAMILIES:
        found = next(
            (
                experiment_dir / name
                for name in family
                if (experiment_dir / name).is_dir()
            ),
            None,
        )
        if found is None:
            status_rows.append(
                {
                    "requested_model": family[0],
                    "resolved_model": "",
                    "result_directory": str(experiment_dir / family[0]),
                    "status": "skipped_missing_result",
                    "number_of_samples": 0,
                    "message": "Result directory is not available yet.",
                }
            )
        else:
            found_models.append(found.resolve())

    return found_models, status_rows


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def load_two_dimensional_array(path: Path, description: str) -> np.ndarray:
    """Load and validate one two-dimensional NumPy array."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")

    values = np.load(path, allow_pickle=False)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(
            f"{description} must be two-dimensional after squeeze, "
            f"but {path} has shape {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{description} contains NaN or infinity: {path}")
    return values


def resize_bilinear(values: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a map to ``(height, width)`` with bilinear interpolation."""

    if values.shape == target_shape:
        return values.astype(np.float32, copy=True)

    target_height, target_width = target_shape
    image = Image.fromarray(values.astype(np.float32, copy=False))
    resized = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def prepare_explanation_map(
    explanation: np.ndarray,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, int, bool, bool]:
    """Keep positive evidence, resize it, and normalize it to 0--1."""

    positive = np.maximum(explanation.astype(np.float32, copy=False), 0.0)
    resized = resize_bilinear(positive, target_shape)
    positive_pixel_count = int(np.count_nonzero(resized > 0.0))
    has_positive_attribution = positive_pixel_count > 0

    minimum = float(resized.min())
    maximum = float(resized.max())
    value_range = maximum - minimum
    is_constant = value_range <= np.finfo(np.float32).eps

    if is_constant:
        normalized = np.zeros_like(resized, dtype=np.float32)
    else:
        normalized = (resized - minimum) / value_range
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    return normalized, positive_pixel_count, has_positive_attribution, is_constant


def compact_percentage(value: float) -> int | float:
    """Write integer percentages as 1 rather than 1.0 in CSV files."""

    rounded = round(value)
    if math.isclose(value, rounded):
        return int(rounded)
    return value


def select_top_k_statistics(
    explanation: np.ndarray,
    ground_truth: np.ndarray,
    top_k_percentages: tuple[float, ...],
) -> dict[float, dict[str, int | float]]:
    """Calculate exact, deterministic top-k selection statistics.

    Pixels above the boundary value are always selected.  If several pixels
    equal the boundary value, the pixels with the smallest flattened indices
    are used first.  This rule keeps the selected area exact and reproducible.
    """

    flat_explanation = explanation.reshape(-1)
    flat_ground_truth = ground_truth.reshape(-1)
    number_of_pixels = flat_explanation.size

    pixels_per_percentage = {
        percentage: min(
            number_of_pixels,
            max(1, math.ceil(number_of_pixels * percentage / 100.0)),
        )
        for percentage in top_k_percentages
    }
    partition_positions = sorted(
        {number_of_pixels - count for count in pixels_per_percentage.values()}
    )
    partitioned = np.partition(flat_explanation, partition_positions)

    ground_truth_pixels = int(np.count_nonzero(flat_ground_truth))
    statistics_by_percentage: dict[float, dict[str, int | float]] = {}

    for percentage in top_k_percentages:
        selected_count = pixels_per_percentage[percentage]
        boundary_index = number_of_pixels - selected_count
        threshold = float(partitioned[boundary_index])

        indices_above = np.flatnonzero(flat_explanation > threshold)
        remaining_count = selected_count - indices_above.size
        indices_equal = np.flatnonzero(flat_explanation == threshold)
        selected_indices = np.concatenate(
            (indices_above, indices_equal[:remaining_count])
        )

        intersection = int(np.count_nonzero(flat_ground_truth[selected_indices]))
        union = selected_count + ground_truth_pixels - intersection
        iou = float(intersection / union) if union > 0 else 1.0

        statistics_by_percentage[percentage] = {
            "selected_explanation_pixels": selected_count,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "attribution_threshold": threshold,
            "iou": iou,
        }

    return statistics_by_percentage


def choose_nodule_id(row: dict[str, str]) -> str:
    """Choose a nodule identifier that is unique across the combined dataset."""

    for column in ("cv_nodule_id", "nodule_id"):
        value = row.get(column, "").strip()
        if value:
            return value
    raise ValueError(
        "test_predictions.csv must contain a non-empty cv_nodule_id or nodule_id."
    )


def resolve_mask_path(mask_base_directory: Path, value: str) -> Path:
    """Resolve one absolute or metadata-relative mask path."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = mask_base_directory / path
    return path.resolve()


def same_value_or_mixed(values: Iterable[str]) -> str:
    """Return one common value, or 'mixed' when values differ."""

    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    return "mixed"


def sample_standard_deviation(values: list[float]) -> float:
    """Return sample standard deviation, or zero for one observation."""

    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def rounded(value: float) -> float:
    """Round metric values while retaining enough precision for analysis."""

    return round(float(value), 10)


def validate_prediction_rows(rows: list[dict[str, str]], path: Path) -> None:
    """Validate columns required by the evaluation."""

    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")

    required_columns = {"filename", "mask_path", "label", "predicted_class"}
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise ValueError(
            f"Prediction CSV is missing columns {sorted(missing_columns)}: {path}"
        )

    filenames = [Path(row["filename"]).name for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Prediction filenames must be unique: {path}")


def evaluate_model(
    model_directory: Path,
    mask_base_directory: Path,
    top_k_percentages: tuple[float, ...],
    primary_top_k: float,
    max_samples: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all methods and layers for one model result directory."""

    model_name = model_directory.name
    test_directory = model_directory / "test"
    predictions_path = test_directory / "test_predictions.csv"
    status_row = {
        "requested_model": model_name,
        "resolved_model": model_name,
        "result_directory": str(model_directory),
        "status": "evaluated",
        "number_of_samples": 0,
        "message": "",
    }

    if not predictions_path.is_file():
        status_row["status"] = "skipped_no_inference"
        status_row["message"] = "test/test_predictions.csv is not available yet."
        return [], status_row

    rows = load_csv_rows(predictions_path)
    validate_prediction_rows(rows, predictions_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    status_row["number_of_samples"] = len(rows)

    slice_records: list[dict[str, Any]] = []
    total_samples = len(rows)

    for sample_number, row in enumerate(rows, start=1):
        filename = Path(row["filename"]).name
        mask_path = resolve_mask_path(mask_base_directory, row["mask_path"])
        ground_truth_values = load_two_dimensional_array(
            mask_path,
            description="ground-truth nodule mask",
        )
        ground_truth = ground_truth_values > 0
        ground_truth_pixels = int(np.count_nonzero(ground_truth))
        if ground_truth_pixels == 0:
            raise ValueError(f"Ground-truth nodule mask is empty: {mask_path}")

        nodule_id = choose_nodule_id(row)
        true_class = row["label"]
        target_class = row["predicted_class"]
        correct_prediction = true_class == target_class

        for method, method_directory in METHOD_DIRECTORIES.items():
            for layer in LAYERS:
                explanation_path = test_directory / method_directory / layer / filename
                explanation_values = load_two_dimensional_array(
                    explanation_path,
                    description=f"{method} {layer} explanation map",
                )
                (
                    explanation,
                    positive_pixel_count,
                    has_positive_attribution,
                    is_constant,
                ) = prepare_explanation_map(
                    explanation_values,
                    target_shape=ground_truth.shape,
                )
                top_k_statistics = select_top_k_statistics(
                    explanation,
                    ground_truth,
                    top_k_percentages,
                )

                for percentage, values in top_k_statistics.items():
                    record = {
                        "model": model_name,
                        "xai_method": method,
                        "layer": layer,
                        "top_k_percent": compact_percentage(percentage),
                        "primary_metric": math.isclose(percentage, primary_top_k),
                        "dataset": row.get("dataset", ""),
                        "patient_id": row.get("patient_id", ""),
                        "nodule_id": nodule_id,
                        "filename": filename,
                        "true_class": true_class,
                        "target_class": target_class,
                        "correct_prediction": correct_prediction,
                        "mask_height": ground_truth.shape[0],
                        "mask_width": ground_truth.shape[1],
                        "ground_truth_pixels": ground_truth_pixels,
                        "positive_attribution_pixels": positive_pixel_count,
                        "map_has_positive_attribution": has_positive_attribution,
                        "map_is_constant_after_normalization": is_constant,
                        **values,
                    }
                    record["attribution_threshold"] = rounded(
                        float(record["attribution_threshold"])
                    )
                    record["iou"] = rounded(float(record["iou"]))
                    slice_records.append(record)

        should_report_progress = (
            sample_number == 1
            or sample_number % 100 == 0
            or sample_number == total_samples
        )
        if should_report_progress:
            print(f"  {model_name}: {sample_number:,}/{total_samples:,} slices")

    return slice_records, status_row


def aggregate_by_nodule(
    slice_records: list[dict[str, Any]],
    primary_top_k: float,
) -> list[dict[str, Any]]:
    """Average slice IoUs within each nodule."""

    grouped_records: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in slice_records:
        key = (
            record["model"],
            record["xai_method"],
            record["layer"],
            record["top_k_percent"],
            record["nodule_id"],
        )
        grouped_records[key].append(record)

    nodule_records: list[dict[str, Any]] = []
    for key, records in grouped_records.items():
        model, method, layer, percentage, nodule_id = key
        iou_values = [float(record["iou"]) for record in records]
        nodule_records.append(
            {
                "model": model,
                "xai_method": method,
                "layer": layer,
                "top_k_percent": percentage,
                "primary_metric": math.isclose(float(percentage), primary_top_k),
                "dataset": same_value_or_mixed(record["dataset"] for record in records),
                "patient_id": same_value_or_mixed(
                    record["patient_id"] for record in records
                ),
                "nodule_id": nodule_id,
                "true_class": same_value_or_mixed(
                    record["true_class"] for record in records
                ),
                "target_class": same_value_or_mixed(
                    record["target_class"] for record in records
                ),
                "number_of_slices": len(records),
                "mean_slice_iou": rounded(statistics.fmean(iou_values)),
                "std_slice_iou": rounded(sample_standard_deviation(iou_values)),
                "min_slice_iou": rounded(min(iou_values)),
                "max_slice_iou": rounded(max(iou_values)),
            }
        )

    nodule_records.sort(
        key=lambda row: (
            row["model"],
            row["xai_method"],
            row["layer"],
            float(row["top_k_percent"]),
            row["nodule_id"],
        )
    )
    return nodule_records


def summarize_across_nodules(
    nodule_records: list[dict[str, Any]],
    primary_top_k: float,
) -> list[dict[str, Any]]:
    """Calculate macro mean and standard deviation across nodules."""

    grouped_records: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in nodule_records:
        key = (
            record["model"],
            record["xai_method"],
            record["layer"],
            record["top_k_percent"],
        )
        grouped_records[key].append(record)

    summary_records: list[dict[str, Any]] = []
    for key, records in grouped_records.items():
        model, method, layer, percentage = key
        nodule_iou_values = [float(record["mean_slice_iou"]) for record in records]
        summary_records.append(
            {
                "model": model,
                "xai_method": method,
                "layer": layer,
                "top_k_percent": percentage,
                "primary_metric": math.isclose(float(percentage), primary_top_k),
                "number_of_nodules": len(records),
                "number_of_slices": sum(
                    int(record["number_of_slices"]) for record in records
                ),
                "mean_iou_across_nodules": rounded(
                    statistics.fmean(nodule_iou_values)
                ),
                "std_iou_across_nodules": rounded(
                    sample_standard_deviation(nodule_iou_values)
                ),
                "median_iou_across_nodules": rounded(
                    statistics.median(nodule_iou_values)
                ),
                "min_iou_across_nodules": rounded(min(nodule_iou_values)),
                "max_iou_across_nodules": rounded(max(nodule_iou_values)),
            }
        )

    summary_records.sort(
        key=lambda row: (
            row["model"],
            row["xai_method"],
            row["layer"],
            float(row["top_k_percent"]),
        )
    )
    return summary_records


def write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    """Write one CSV atomically, so interrupted runs do not leave partial files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def primary_csv_filename(primary_top_k: float) -> str:
    """Create a filename that reflects the selected primary threshold."""

    percentage = f"{primary_top_k:g}".replace(".", "_")
    return f"plausibility_primary_iou_at_{percentage}.csv"


def main() -> None:
    """Run the complete plausibility evaluation."""

    arguments = parse_arguments()
    if arguments.max_samples is not None and arguments.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")

    top_k_percentages = validate_top_k_values(
        arguments.top_k,
        arguments.primary_top_k,
    )
    experiment_dir = resolve_from_project(arguments.experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory was not found: {experiment_dir}")
    mask_base_directory = find_mask_base_directory(arguments.dataset_root)

    default_status_rows: list[dict[str, Any]] = []
    if arguments.model_dirs is None:
        model_directories, default_status_rows = discover_default_models(experiment_dir)
    else:
        model_directories = [
            resolve_model_directory(experiment_dir, value)
            for value in arguments.model_dirs
        ]
        missing_models = [path for path in model_directories if not path.is_dir()]
        if missing_models:
            formatted = "\n".join(f"  - {path}" for path in missing_models)
            raise FileNotFoundError(
                f"Explicit model directories were not found:\n{formatted}"
            )

    if arguments.output_dir is None:
        output_dir = experiment_dir.parent / "evaluation" / "xai_plausibility"

        # Keep an explicitly selected run separate from the combined default
        # run. This prevents a one-model run from replacing the combined CSVs.
        if arguments.model_dirs is not None:
            if len(model_directories) == 1:
                output_dir = output_dir / model_directories[0].name
            else:
                output_dir = output_dir / "selected_models"

        if arguments.max_samples is not None:
            output_dir = output_dir.with_name(
                f"{output_dir.name}_smoke_{arguments.max_samples}"
            )
    else:
        output_dir = resolve_from_project(arguments.output_dir)

    print("XAI plausibility evaluation")
    print(f"Experiment directory : {experiment_dir}")
    print(f"Mask base directory  : {mask_base_directory}")
    print(f"Top-k percentages    : {top_k_percentages}")
    print(f"Primary metric       : IoU@{arguments.primary_top_k:g}%")
    print(f"Output directory     : {output_dir}")

    all_slice_records: list[dict[str, Any]] = []
    status_rows = list(default_status_rows)
    for model_directory in model_directories:
        print(f"\nEvaluating {model_directory.name}")
        records, status = evaluate_model(
            model_directory=model_directory,
            mask_base_directory=mask_base_directory,
            top_k_percentages=top_k_percentages,
            primary_top_k=arguments.primary_top_k,
            max_samples=arguments.max_samples,
        )
        all_slice_records.extend(records)
        status_rows.append(status)

    write_csv(output_dir / "model_status.csv", STATUS_COLUMNS, status_rows)
    if not all_slice_records:
        raise RuntimeError(
            "No model could be evaluated. See model_status.csv for details."
        )

    all_slice_records.sort(
        key=lambda row: (
            row["model"],
            row["xai_method"],
            row["layer"],
            float(row["top_k_percent"]),
            row["nodule_id"],
            row["filename"],
        )
    )
    nodule_records = aggregate_by_nodule(
        all_slice_records,
        primary_top_k=arguments.primary_top_k,
    )
    summary_records = summarize_across_nodules(
        nodule_records,
        primary_top_k=arguments.primary_top_k,
    )
    primary_records = [
        record for record in summary_records if record["primary_metric"]
    ]

    write_csv(
        output_dir / "plausibility_slice.csv",
        SLICE_COLUMNS,
        all_slice_records,
    )
    write_csv(
        output_dir / "plausibility_nodule.csv",
        NODULE_COLUMNS,
        nodule_records,
    )
    write_csv(
        output_dir / "plausibility_summary.csv",
        SUMMARY_COLUMNS,
        summary_records,
    )
    write_csv(
        output_dir / primary_csv_filename(arguments.primary_top_k),
        SUMMARY_COLUMNS,
        primary_records,
    )

    print("\nEvaluation complete.")
    print(f"Slice rows  : {len(all_slice_records):,}")
    print(f"Nodule rows : {len(nodule_records):,}")
    print(f"Summary rows: {len(summary_records):,}")
    print(f"CSV files   : {output_dir}")


if __name__ == "__main__":
    main()
