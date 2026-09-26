"""Train a patient-grouped five-fold 2.5D ResNet-50 classifier."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils import (
    EarlyStopping,
    binary_probabilities_to_predictions,
    compute_auc,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    save_best_model,
    set_seed,
    update_confusion_matrix,
)
from .aggregation import aggregate_nodule_predictions
from .dataset import TwoPointFiveDLungDataset
from .model import build_model
from .transforms import build_eval_transform, build_train_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "003_classification/configs/cv_2_5d_resnet50.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path, output_override: Path | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config = deepcopy(config)
    config["data"]["dataset_root"] = str(resolve_project_path(config["data"]["dataset_root"]))
    config["data"]["metadata_path"] = str(resolve_project_path(config["data"]["metadata_path"]))
    if output_override is None:
        output = (
            resolve_project_path(config["output"]["root_directory"])
            / str(config["experiment"]["id"])
            / str(config["experiment"]["component"])
        )
    else:
        output = resolve_project_path(output_override)
    config["resolved_output_directory"] = str(output)
    return config


def select_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def validate_configuration(config: dict[str, Any]) -> None:
    """Fail early when the configuration is incompatible with this program."""

    if config["model"]["architecture"] != "2.5D ResNet50":
        raise ValueError("model.architecture must be '2.5D ResNet50'.")
    if config["model"]["training_strategy"] != "full_fine_tuning":
        raise ValueError("Only full_fine_tuning is supported.")
    if int(config["model"]["num_classes"]) != 2:
        raise ValueError("This program supports exactly two classes.")
    if list(config["data"]["slice_offsets"]) != [-1, 0, 1]:
        raise ValueError("data.slice_offsets must be [-1, 0, 1].")
    if str(config["optimizer"]["name"]).upper() != "SGD":
        raise ValueError("Only the SGD optimizer is supported.")
    if config["aggregation"]["nodule_method"] != "mean_probability":
        raise ValueError("Only mean_probability nodule aggregation is supported.")
    if config["xai"]["gradcam_layer"] != "layer4[-1]":
        raise ValueError("This implementation requires Grad-CAM layer4[-1].")
    if config["xai"]["lrp_rule"] != "EpsilonPlusFlat":
        raise ValueError("This implementation requires EpsilonPlusFlat LRP.")
    if config["xai"]["target"] not in {"predicted_class", "true_class"}:
        raise ValueError("xai.target must be predicted_class or true_class.")
    if config["early_stopping"]["monitor"] != "val_loss":
        raise ValueError("Early stopping must monitor val_loss.")
    if config["early_stopping"]["mode"] != "min":
        raise ValueError("Early-stopping mode must be min.")
    threshold = float(config["training"]["classification_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("classification_threshold must be between zero and one.")
    if int(config["training"]["num_epochs"]) <= 0:
        raise ValueError("num_epochs must be positive.")
    if int(config["training"]["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive.")


def validate_metadata(config: dict[str, Any]) -> pd.DataFrame:
    data, cv = config["data"], config["cross_validation"]
    path = Path(data["metadata_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Metadata not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "dataset", "patient_id", "filename", "label", data["ct_path_column"],
        cv["group_column"], cv["nodule_column"], cv["fold_column"], cv["role_column"],
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata columns are missing: {sorted(missing)}")
    frame = frame.copy()
    frame[cv["role_column"]] = frame[cv["role_column"]].astype(str).str.lower()
    frame[cv["fold_column"]] = pd.to_numeric(frame[cv["fold_column"]], errors="raise").astype(int)
    development = frame[frame[cv["role_column"]].eq(cv["development_role"])]
    holdout = frame[frame[cv["role_column"]].eq(cv["holdout_role"])]
    expected_folds = set(range(int(cv["num_folds"])))
    if set(development[cv["fold_column"]].unique()) != expected_folds:
        raise ValueError("Development metadata does not contain every configured fold.")
    if set(development[cv["group_column"]]) & set(holdout[cv["group_column"]]):
        raise RuntimeError("Patient leakage exists between development and holdout.")
    if development.groupby(cv["group_column"])[cv["fold_column"]].nunique().gt(1).any():
        raise RuntimeError("A development patient appears in multiple folds.")
    if development.groupby(cv["nodule_column"])[cv["fold_column"]].nunique().gt(1).any():
        raise RuntimeError("A development nodule appears in multiple folds.")
    return frame


def dataset_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    data, cv = config["data"], config["cross_validation"]
    return {
        "root_dir": data["dataset_root"],
        "metadata_path": data["metadata_path"],
        "class_to_idx": {name: int(index) for name, index in data["class_to_idx"].items()},
        "ct_path_column": data["ct_path_column"],
        "nodule_column": cv["nodule_column"],
        "group_column": cv["group_column"],
        "role_column": cv["role_column"],
        "fold_column": cv["fold_column"],
        "development_role": cv["development_role"],
        "holdout_role": cv["holdout_role"],
        "slice_offsets": tuple(data["slice_offsets"]),
        "boundary_mode": data["boundary_mode"],
    }


def make_loader(dataset: TwoPointFiveDLungDataset, config: dict, training: bool) -> DataLoader:
    options = config["dataloader"]
    workers = int(options["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=bool(options["train_shuffle"] if training else options["val_shuffle"]),
        num_workers=workers,
        pin_memory=bool(options["pin_memory"]) and torch.cuda.is_available(),
        drop_last=bool(options["train_drop_last"] if training else options["val_drop_last"]),
        persistent_workers=bool(options["persistent_workers"]) if workers > 0 else False,
        prefetch_factor=int(options["prefetch_factor"]) if workers > 0 else None,
    )


def make_fold_loaders(config: dict, fold: int) -> tuple[DataLoader, DataLoader]:
    common = dataset_kwargs(config)
    train_dataset = TwoPointFiveDLungDataset(
        **common, split="train", cv_fold=fold, transform=build_train_transform(config)
    )
    val_dataset = TwoPointFiveDLungDataset(
        **common, split="val", cv_fold=fold, transform=build_eval_transform(config)
    )
    train_groups = set(train_dataset.metadata[config["cross_validation"]["group_column"]])
    val_groups = set(val_dataset.metadata[config["cross_validation"]["group_column"]])
    if train_groups & val_groups:
        raise RuntimeError(f"Patient leakage detected in fold {fold}.")
    return make_loader(train_dataset, config, True), make_loader(val_dataset, config, False)


def metrics_from_probabilities(
    targets: torch.Tensor, probabilities: torch.Tensor, threshold: float
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    predictions = binary_probabilities_to_predictions(probabilities, threshold, 1)
    confusion = update_confusion_matrix(
        torch.zeros((2, 2), dtype=torch.int64), predictions, targets, 2
    )
    metrics = compute_classification_metrics(confusion)
    metrics["auc"] = compute_auc(targets.numpy(), probabilities.numpy())
    return metrics, confusion, predictions


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all, probabilities_all, indices_all = [], [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for inputs, targets, indices in tqdm(loader, leave=False, unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            targets_all.append(targets.detach().cpu())
            probabilities_all.append(torch.softmax(logits.detach(), dim=1).cpu())
            indices_all.append(indices.cpu())
    targets_tensor = torch.cat(targets_all)
    probabilities_tensor = torch.cat(probabilities_all)
    metrics, _, _ = metrics_from_probabilities(targets_tensor, probabilities_tensor, threshold)
    return (
        total_loss / len(targets_tensor), metrics, targets_tensor,
        probabilities_tensor, torch.cat(indices_all),
    )


def build_prediction_frame(
    dataset: TwoPointFiveDLungDataset,
    indices: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
    fold: int,
) -> pd.DataFrame:
    order = indices.numpy().astype(int)
    frame = dataset.metadata.iloc[order].reset_index(drop=True).copy()
    predictions = binary_probabilities_to_predictions(probabilities, threshold, 1).numpy()
    frame.insert(0, "validation_fold", fold)
    frame["true_index"] = targets.numpy()
    frame["predicted_index"] = predictions
    frame["predicted_class"] = np.where(predictions == 1, "malignant", "benign")
    frame["probability_benign"] = probabilities[:, 0].numpy()
    frame["probability_malignant"] = probabilities[:, 1].numpy()
    frame["input_slice_indices"] = [
        "|".join(map(str, dataset.windows[index]["input_slice_indices"])) for index in order
    ]
    frame["input_filenames"] = [
        "|".join(dataset.windows[index]["input_filenames"]) for index in order
    ]
    frame["boundary_replicated"] = [
        dataset.windows[index]["boundary_replicated"] for index in order
    ]
    return frame


def nodule_metrics(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict, torch.Tensor]:
    threshold = float(config["training"]["classification_threshold"])
    nodules = aggregate_nodule_predictions(
        frame, config["cross_validation"]["nodule_column"], threshold
    )
    targets = torch.tensor(nodules["true_index"].to_numpy(), dtype=torch.long)
    probabilities = torch.tensor(
        nodules[["probability_benign", "probability_malignant"]].to_numpy(),
        dtype=torch.float32,
    )
    metrics, confusion, _ = metrics_from_probabilities(targets, probabilities, threshold)
    return nodules, metrics, confusion


def save_training_curves(log: pd.DataFrame, directory: Path) -> None:
    for columns, title, filename in (
        (("train_loss", "val_loss"), "Loss", "loss_curve.png"),
        (("train_accuracy", "val_accuracy"), "Accuracy", "accuracy_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for column in columns:
            axis.plot(log["epoch"], log[column], label=column.replace("_", " ").title())
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=200)
        plt.close(figure)


def save_cv_summary_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for column, label in (
        ("window_accuracy", "Window accuracy"),
        ("window_auc", "Window AUC"),
        ("nodule_accuracy", "Nodule accuracy"),
        ("nodule_auc", "Nodule AUC"),
    ):
        axis.plot(summary["fold"], summary[column], marker="o", label=label)
    axis.set_xlabel("Validation fold")
    axis.set_ylabel("Metric")
    axis.set_xticks(summary["fold"])
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "cv_fold_metrics.png", dpi=250)
    plt.close(figure)


def run_fold(config: dict, fold: int, output_dir: Path, device: torch.device):
    set_seed(int(config["training"]["seed"]), deterministic=True)
    fold_dir = output_dir / f"fold_{fold}"
    figures = fold_dir / "figures"
    window_figures, nodule_figures = figures / "window", figures / "nodule"
    for path in (fold_dir, figures, window_figures, nodule_figures):
        path.mkdir(parents=True, exist_ok=False if path == fold_dir else True)
    train_loader, val_loader = make_fold_loaders(config, fold)
    model = build_model(config, load_pretrained=True).to(device)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        nesterov=bool(optimizer_config["nesterov"]),
    )
    criterion = nn.CrossEntropyLoss()
    stop_config = config["early_stopping"]
    stopper = EarlyStopping(
        patience=int(stop_config["patience"]), mode="min",
        min_delta=float(stop_config["min_delta"]), verbose=True,
    )
    threshold = float(config["training"]["classification_threshold"])
    best_loss = float("inf")
    best_frame = best_nodules = best_window_metrics = best_nodule_metrics = None
    best_window_confusion = best_nodule_confusion = None
    log_rows = []
    started = time.perf_counter()

    print(f"\nFold {fold + 1}/{config['cross_validation']['num_folds']}")
    print(f"Train windows: {len(train_loader.dataset):,}; validation windows: {len(val_loader.dataset):,}")
    for epoch in range(1, int(config["training"]["num_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss, train_metrics, *_ = run_epoch(
            model, train_loader, criterion, device, threshold, optimizer
        )
        val_loss, val_metrics, targets, probabilities, indices = run_epoch(
            model, val_loader, criterion, device, threshold
        )
        frame = build_prediction_frame(
            val_loader.dataset, indices, targets, probabilities, threshold, fold
        )
        nodules, node_metrics, node_confusion = nodule_metrics(frame, config)
        _, window_confusion, _ = metrics_from_probabilities(targets, probabilities, threshold)
        is_best = val_loss < best_loss - float(stop_config["min_delta"])
        if is_best:
            best_loss = val_loss
            save_best_model(model, fold_dir / "best_model.pth")
            best_frame, best_nodules = frame, nodules
            best_window_metrics, best_nodule_metrics = val_metrics, node_metrics
            best_window_confusion, best_nodule_confusion = window_confusion, node_confusion
        should_stop = stopper(val_loss, epoch)
        torch.save(
            {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "early_stopping_state_dict": stopper.state_dict(), "val_loss": val_loss,
            },
            fold_dir / "checkpoint_latest.pth",
        )
        row = {
            "epoch": epoch, "epoch_time_seconds": time.perf_counter() - epoch_started,
            "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"], "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"], "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"], "val_precision": val_metrics["precision"],
            "val_f1_score": val_metrics["f1_score"], "val_auc": val_metrics["auc"],
            "nodule_val_accuracy": node_metrics["accuracy"], "nodule_val_auc": node_metrics["auc"],
            "is_best": is_best, "early_stop_counter": stopper.counter,
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(fold_dir / "training_log.csv", index=False)
        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"window_auc={val_metrics['auc']:.4f} nodule_auc={node_metrics['auc']:.4f}"
        )
        if should_stop:
            break

    assert best_frame is not None and best_nodules is not None
    best_frame.to_csv(fold_dir / "validation_predictions.csv", index=False)
    best_nodules.to_csv(fold_dir / "validation_nodule_predictions.csv", index=False)
    plot_confusion_matrix(best_window_confusion, ["benign", "malignant"], window_figures)
    plot_roc_curve(
        best_frame["true_index"].to_numpy(),
        best_frame[["probability_benign", "probability_malignant"]].to_numpy(),
        ["benign", "malignant"], window_figures,
    )
    plot_confusion_matrix(best_nodule_confusion, ["benign", "malignant"], nodule_figures)
    plot_roc_curve(
        best_nodules["true_index"].to_numpy(),
        best_nodules[["probability_benign", "probability_malignant"]].to_numpy(),
        ["benign", "malignant"], nodule_figures,
    )
    save_training_curves(pd.DataFrame(log_rows), figures)
    fold_config = deepcopy(config)
    fold_config["cross_validation"]["active_fold"] = fold
    fold_config["data_summary"] = {
        "train_windows": len(train_loader.dataset), "validation_windows": len(val_loader.dataset),
        "train_boundary_replicated": train_loader.dataset.boundary_replication_count,
        "validation_boundary_replicated": val_loader.dataset.boundary_replication_count,
    }
    with (fold_dir / "training_config.json").open("w", encoding="utf-8") as file:
        json.dump(fold_config, file, indent=2)
        file.write("\n")
    summary = {
        "fold": fold, "best_epoch": stopper.best_epoch, "epochs_completed": len(log_rows),
        "best_val_loss": best_loss, "training_seconds": time.perf_counter() - started,
        **{f"window_{key}": value for key, value in best_window_metrics.items()},
        **{f"nodule_{key}": value for key, value in best_nodule_metrics.items()},
    }
    return summary, best_frame, best_nodules


def main() -> None:
    args = parse_args()
    config = load_config(resolve_project_path(args.config), args.output_dir)
    validate_configuration(config)
    metadata = validate_metadata(config)
    output_dir = Path(config["resolved_output_directory"])
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copy2(resolve_project_path(args.config), output_dir / "cv_2_5d_resnet50.json")
    device = select_device(config["training"]["device"])
    with (output_dir / "cv_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "device": str(device), "configuration": config,
                "metadata_rows": len(metadata),
            }, file, indent=2,
        )
        file.write("\n")
    print("2.5D ResNet-50 patient-grouped cross-validation")
    print(f"Output: {output_dir}\nDevice: {device}")
    summaries, oof, oof_nodules = [], [], []
    for fold in range(int(config["cross_validation"]["num_folds"])):
        summary, predictions, nodules = run_fold(config, fold, output_dir, device)
        summaries.append(summary)
        oof.append(predictions)
        oof_nodules.append(nodules)
        pd.DataFrame(summaries).to_csv(output_dir / "cv_summary.csv", index=False)
    oof_frame = pd.concat(oof, ignore_index=True)
    if oof_frame.duplicated(["dataset", "filename"]).any():
        raise RuntimeError("Duplicate out-of-fold window predictions were produced.")
    oof_frame.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    pd.concat(oof_nodules, ignore_index=True).to_csv(
        output_dir / "out_of_fold_nodule_predictions.csv", index=False
    )
    summary_frame = pd.DataFrame(summaries)
    save_cv_summary_figure(summary_frame, output_dir)
    numeric = summary_frame.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
    summary_json = {
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "folds": len(summaries),
        "holdout_used_during_training": False,
        "aggregate_metrics": {
            column: {"mean": float(numeric[column].mean()), "std": float(numeric[column].std(ddof=1))}
            for column in numeric
        },
    }
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_json, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"\nCross-validation complete: {output_dir}")


if __name__ == "__main__":
    main()
