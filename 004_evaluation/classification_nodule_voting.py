"""Aggregate slice-level classification predictions at nodule level.

The script supports the paired experiments used in this repository:

* conventional ResNet50, whose validation OOF predictions are already saved;
* multi-stage direct-guided ResNet50, whose validation predictions are rebuilt
  from the five saved best-model checkpoints before aggregation.

Discrete metrics use majority voting across slices. Mean malignant
probability breaks exact vote ties and is also used for log loss and ROC-AUC.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ID = "fa028196-fd4c-441f-a5d3-3efb9707e7f9"
BASELINE_COMPONENT = "classification/cv_resnet50"
GUIDED_COMPONENT = "classification/multistage_direct_guided_cv_resnet50"

METRIC_COLUMNS = [
    "loss",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
    "auc",
]


def parse_args() -> argparse.Namespace:
    """Parse paths and guided-validation inference settings."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "000_dataset_v2" / "_segmentation_dataset",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "000_dataset_v2"
            / "_segmentation_dataset"
            / "004_classification_cv_5fold_seed42.csv"
        ),
    )
    parser.add_argument(
        "--probability-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiment_results"
            / DEFAULT_EXPERIMENT_ID
            / "segmentation"
            / "unet"
            / "inference"
            / "probability_npy"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--reuse-guided-validation",
        action="store_true",
        help="Reuse an existing guided validation_slice_predictions.csv.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_input_frame(frame: pd.DataFrame, source: Path) -> None:
    """Validate columns and per-nodule ground-truth consistency."""

    required = {
        "cv_nodule_id",
        "probability_benign",
        "probability_malignant",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    target_column = "target" if "target" in frame else "true_index"
    prediction_column = "prediction" if "prediction" in frame else "predicted_index"
    for column in (target_column, prediction_column):
        if column not in frame:
            raise ValueError(f"{source} is missing {column!r}.")

    if frame["cv_nodule_id"].isna().any():
        raise ValueError(f"{source} contains missing cv_nodule_id values.")
    if frame[target_column].isna().any() or frame[prediction_column].isna().any():
        raise ValueError(f"{source} contains missing targets or predictions.")

    label_counts = frame.groupby("cv_nodule_id", sort=False)[target_column].nunique()
    inconsistent = label_counts[label_counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{source} has {len(inconsistent)} nodules with inconsistent targets."
        )


def first_if_present(group: pd.DataFrame, column: str) -> Any:
    """Return the first value of an optional metadata column."""

    return group[column].iloc[0] if column in group else None


def aggregate_predictions(source: Path, validation: bool) -> pd.DataFrame:
    """Aggregate one slice-prediction CSV to one row per nodule."""

    frame = pd.read_csv(source)
    validate_input_frame(frame, source)

    target_column = "target" if "target" in frame else "true_index"
    prediction_column = "prediction" if "prediction" in frame else "predicted_index"
    fold_column = None
    if validation:
        if "validation_fold" in frame:
            fold_column = "validation_fold"
        elif "cv_fold" in frame:
            fold_column = "cv_fold"
        else:
            raise ValueError(f"Validation predictions lack a fold column: {source}")

    group_columns = ["cv_nodule_id"]
    if fold_column is not None:
        group_columns.insert(0, fold_column)

    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fold = int(keys[0]) if fold_column is not None else None
        nodule_id = str(keys[-1])

        target = int(group[target_column].iloc[0])
        slice_predictions = group[prediction_column].astype(int).to_numpy()
        benign_votes = int(np.sum(slice_predictions == 0))
        malignant_votes = int(np.sum(slice_predictions == 1))
        mean_benign = float(group["probability_benign"].mean())
        mean_malignant = float(group["probability_malignant"].mean())
        tied = benign_votes == malignant_votes
        if tied:
            prediction = int(mean_malignant >= 0.5)
        else:
            prediction = int(malignant_votes > benign_votes)

        record = {
            "dataset": first_if_present(group, "dataset"),
            "patient_id": first_if_present(group, "patient_id"),
            "cv_group_id": first_if_present(group, "cv_group_id"),
            "nodule_id": first_if_present(group, "nodule_id"),
            "cv_nodule_id": nodule_id,
            "label": first_if_present(group, "label"),
            "target": target,
            "num_slices": int(len(group)),
            "benign_votes": benign_votes,
            "malignant_votes": malignant_votes,
            "malignant_vote_fraction": malignant_votes / len(group),
            "mean_probability_benign": mean_benign,
            "mean_probability_malignant": mean_malignant,
            "tie_break_used": tied,
            "prediction": prediction,
            "predicted_class": "malignant" if prediction == 1 else "benign",
            "correct": prediction == target,
        }
        if fold is not None:
            record = {"validation_fold": fold, **record}
        records.append(record)

    result = pd.DataFrame.from_records(records)
    sort_columns = ["validation_fold", "cv_nodule_id"] if validation else ["cv_nodule_id"]
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Calculate nodule-level binary metrics from voting predictions."""

    targets = frame["target"].astype(int).to_numpy()
    predictions = frame["prediction"].astype(int).to_numpy()
    probabilities = frame["mean_probability_malignant"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

    def safe_ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    sensitivity = safe_ratio(int(tp), int(tp + fn))
    specificity = safe_ratio(int(tn), int(tn + fp))
    precision = safe_ratio(int(tp), int(tp + fp))
    f1_denominator = 2 * int(tp) + int(fp) + int(fn)

    return {
        "n_nodules": int(len(frame)),
        "benign_nodules": int(np.sum(targets == 0)),
        "malignant_nodules": int(np.sum(targets == 1)),
        "tie_breaks": int(frame["tie_break_used"].sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "accuracy": safe_ratio(int(tp + tn), int(len(frame))),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": safe_ratio(2 * int(tp), f1_denominator),
        "auc": float(roc_auc_score(targets, probabilities)),
    }


def validation_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build per-fold, mean/SD, and pooled OOF validation metrics."""

    fold_records: list[dict[str, Any]] = []
    for fold, group in predictions.groupby("validation_fold", sort=True):
        fold_records.append(
            {
                "scope": "fold",
                "fold": int(fold),
                **calculate_metrics(group),
            }
        )
    folds = pd.DataFrame.from_records(fold_records)
    numeric_columns = [column for column in folds.columns if column not in {"scope", "fold"}]
    mean_record = {"scope": "cv_mean", "fold": np.nan}
    std_record = {"scope": "cv_std", "fold": np.nan}
    mean_record.update(folds[numeric_columns].mean().to_dict())
    std_record.update(folds[numeric_columns].std(ddof=1).to_dict())
    pooled_record = {
        "scope": "oof_pooled",
        "fold": np.nan,
        **calculate_metrics(predictions),
    }
    return pd.concat(
        [folds, pd.DataFrame([mean_record, std_record, pooled_record])],
        ignore_index=True,
    )


def test_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build the single independent-test metric row."""

    return pd.DataFrame([{"scope": "test_ensemble", **calculate_metrics(predictions)}])


def infer_guided_validation(
    guided_dir: Path,
    output_path: Path,
    dataset_root: Path,
    metadata_path: Path,
    probability_root: Path,
    requested_device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    """Rebuild missing guided-model OOF slice predictions from checkpoints."""

    import torch
    from torch import nn

    module_root = PROJECT_ROOT / "003_classification" / "multistage_direct_guided_cv_resnet50"
    sys.path.insert(0, str(module_root))
    import train as guided_train  # noqa: PLC0415

    run_config = load_json(guided_dir / "run_config.json")
    config = dict(run_config["comparison_settings"])
    config["data"] = dict(config["data"])
    config["dataloader"] = dict(config["dataloader"])
    config["training"] = dict(config["training"])
    config["data"]["dataset_root"] = str(dataset_root.resolve())
    config["data"]["metadata_path"] = str(metadata_path.resolve())
    config["data"]["probability_root"] = str(probability_root.resolve())
    config["training"]["batch_size"] = int(batch_size)
    config["dataloader"]["num_workers"] = int(num_workers)
    config["dataloader"]["persistent_workers"] = num_workers > 0

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(requested_device)
    threshold = float(config["training"]["classification_threshold"])
    expected_metrics = pd.read_csv(guided_dir / "cross_validation_summary.csv")
    all_predictions: list[pd.DataFrame] = []

    for fold in range(int(config["cross_validation"]["num_folds"])):
        _, validation_loader = guided_train.create_fold_dataloaders(config, fold)
        model = guided_train.create_model(
            config,
            run_config["model_options"],
            use_pretrained_weights=False,
        ).to(device)
        state = torch.load(
            guided_dir / f"fold_{fold}" / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        metrics, targets, probabilities = guided_train.evaluate(
            model,
            validation_loader,
            nn.CrossEntropyLoss(),
            device,
            threshold,
            description=f"Guided fold {fold} validation reconstruction",
        )

        expected = expected_metrics.loc[expected_metrics["fold"].astype(int) == fold].iloc[0]
        for metric in METRIC_COLUMNS:
            # ROC-AUC is rank-based and can move slightly when near-tied logits
            # differ across PyTorch/CUDA versions. Discrete metrics and loss
            # should remain substantially tighter.
            tolerance = 1e-4 if metric == "auc" else (1e-5 if metric == "loss" else 2e-6)
            if not np.isclose(
                metrics[metric],
                float(expected[metric]),
                atol=tolerance,
                rtol=0,
            ):
                raise RuntimeError(
                    f"Fold {fold} reconstructed {metric}={metrics[metric]} does not "
                    f"match saved value {expected[metric]}."
                )

        metadata = validation_loader.dataset.metadata.copy().reset_index(drop=True)
        if len(metadata) != len(targets):
            raise RuntimeError(f"Fold {fold} metadata and prediction lengths differ.")
        metadata.insert(0, "validation_fold", fold)
        metadata["target"] = targets.numpy().astype(int)
        metadata["prediction"] = (
            probabilities[:, 1].numpy() >= threshold
        ).astype(int)
        metadata["probability_benign"] = probabilities[:, 0].numpy()
        metadata["probability_malignant"] = probabilities[:, 1].numpy()
        all_predictions.append(metadata)

        del model, state, validation_loader, probabilities, targets
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(output_path, index=False)


def write_model_outputs(
    model_dir: Path,
    validation_source: Path,
    test_source: Path,
) -> None:
    """Write per-nodule predictions and metrics for validation and test."""

    output_dir = model_dir / "nodule_voting"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_predictions = aggregate_predictions(validation_source, validation=True)
    validation_metrics = validation_metric_table(validation_predictions)
    test_predictions = aggregate_predictions(test_source, validation=False)
    test_metrics = test_metric_table(test_predictions)

    validation_predictions.to_csv(
        output_dir / "validation_nodule_predictions.csv", index=False
    )
    validation_metrics.to_csv(output_dir / "validation_nodule_metrics.csv", index=False)
    test_predictions.to_csv(output_dir / "test_nodule_predictions.csv", index=False)
    test_metrics.to_csv(output_dir / "test_nodule_metrics.csv", index=False)

    print(f"Wrote nodule-voting outputs: {output_dir}")


def main() -> None:
    """Reconstruct missing predictions and write both models' outputs."""

    args = parse_args()
    experiment_root = PROJECT_ROOT / "experiment_results" / args.experiment_id
    baseline_dir = experiment_root / BASELINE_COMPONENT
    guided_dir = experiment_root / GUIDED_COMPONENT
    guided_slice_path = guided_dir / "nodule_voting" / "validation_slice_predictions.csv"

    if not args.reuse_guided_validation or not guided_slice_path.is_file():
        infer_guided_validation(
            guided_dir=guided_dir,
            output_path=guided_slice_path,
            dataset_root=args.dataset_root,
            metadata_path=args.metadata_path,
            probability_root=args.probability_root,
            requested_device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    write_model_outputs(
        model_dir=baseline_dir,
        validation_source=baseline_dir / "out_of_fold_predictions.csv",
        test_source=baseline_dir / "test" / "test_predictions.csv",
    )
    write_model_outputs(
        model_dir=guided_dir,
        validation_source=guided_slice_path,
        test_source=guided_dir / "test" / "test_predictions.csv",
    )


if __name__ == "__main__":
    main()
