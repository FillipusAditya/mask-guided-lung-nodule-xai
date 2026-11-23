"""Reproducible classification evaluation for the AAIML retraining runs.

The module consumes saved held-out predictions. It never selects models or
changes thresholds using the held-out test set. Nodule probabilities are the
mean malignant probability across slices, while primary nodule predictions
remain the original majority vote with mean-probability tie-breaking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


METRIC_NAMES = (
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
)
SOURCE_SCOPES = (
    ("Combined", None),
    ("LIDC-IDRI", "LIDC-IDRI"),
    ("LNDb", "LNDb"),
)
MODEL_STYLES = {
    "Baseline": {
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "Proposed": {
        "color": "#D55E00",
        "linestyle": "--",
        "marker": "s",
    },
}


def configure_plot_style() -> str:
    """Apply the manuscript plotting style and return the selected font."""

    tinos_path = Path("/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf")
    if tinos_path.is_file():
        font_manager.fontManager.addfont(tinos_path)
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ("Times New Roman", "Tinos", "Liberation Serif")
    family = next((name for name in candidates if name in available), "serif")
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family


def style_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    """Apply thin grids, spines, and ticks to one axis."""

    axis.grid(True, axis=grid_axis, color="#D0D0D0", linewidth=0.35)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(width=0.6, length=2.5)


def save_figure(figure: plt.Figure, path: Path, dpi: int = 500) -> None:
    """Save a publication figure as a tightly cropped PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
        facecolor="white",
    )
    plt.close(figure)


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio or NaN when the denominator is zero."""

    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Calculate binary metrics without replacing undefined values by zero."""

    targets = np.asarray(targets, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tn = int(np.sum((targets == 0) & (predictions == 0)))
    fp = int(np.sum((targets == 0) & (predictions == 1)))
    fn = int(np.sum((targets == 1) & (predictions == 0)))
    tp = int(np.sum((targets == 1) & (predictions == 1)))
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": safe_ratio(tp + tn, len(targets)),
        "sensitivity": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "precision": safe_ratio(tp, tp + fp),
        "f1_score": safe_ratio(2 * tp, 2 * tp + fp + fn),
    }


def select_scope(frame: pd.DataFrame, dataset: str | None) -> pd.DataFrame:
    """Select the combined cohort or one dataset source."""

    return frame if dataset is None else frame.loc[frame["dataset"].eq(dataset)]


def prepare_predictions(path: Path, level: str) -> pd.DataFrame:
    """Standardize one slice- or nodule-level prediction table."""

    frame = pd.read_csv(path)
    if level == "slice":
        aliases = {
            "target": "true_index" if "true_index" in frame else "target",
            "prediction": (
                "predicted_index" if "predicted_index" in frame else "prediction"
            ),
            "probability": "probability_malignant",
        }
    elif level == "nodule":
        aliases = {
            "target": "target",
            "prediction": "prediction",
            "probability": "mean_probability_malignant",
        }
    else:
        raise ValueError(f"Unsupported evaluation level: {level}")

    required = {
        "dataset",
        "patient_id",
        "cv_nodule_id",
        aliases["target"],
        aliases["prediction"],
        aliases["probability"],
    }
    if level == "slice":
        required.add("filename")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")

    result = frame.copy()
    result["target"] = result[aliases["target"]].astype(int)
    result["prediction"] = result[aliases["prediction"]].astype(int)
    result["probability"] = result[aliases["probability"]].astype(float)
    result["unique_patient_id"] = (
        result["dataset"].astype(str)
        + "::"
        + result["patient_id"].astype(str)
    )
    if not result["target"].isin([0, 1]).all():
        raise ValueError(f"Invalid targets in {path}.")
    if not result["prediction"].isin([0, 1]).all():
        raise ValueError(f"Invalid predictions in {path}.")
    if not result["probability"].between(0.0, 1.0).all():
        raise ValueError(f"Probabilities outside [0, 1] in {path}.")
    return result


def count_entities(frame: pd.DataFrame) -> dict[str, int]:
    """Count patients, nodules, slices, and labels in one metadata subset."""

    nodules = frame.drop_duplicates("cv_nodule_id")
    return {
        "patients": int(frame["unique_patient_id"].nunique()),
        "nodules": int(nodules["cv_nodule_id"].nunique()),
        "slices": int(len(frame)),
        "benign_patients": int(
            frame.loc[frame["label"].eq("benign"), "unique_patient_id"].nunique()
        ),
        "malignant_patients": int(
            frame.loc[
                frame["label"].eq("malignant"), "unique_patient_id"
            ].nunique()
        ),
        "benign_nodules": int(nodules["label"].eq("benign").sum()),
        "malignant_nodules": int(nodules["label"].eq("malignant").sum()),
        "benign_slices": int(frame["label"].eq("benign").sum()),
        "malignant_slices": int(frame["label"].eq("malignant").sum()),
    }


def build_split_distribution(metadata_path: Path, output_path: Path) -> pd.DataFrame:
    """Save source/class counts for every CV partition and held-out test set."""

    metadata = pd.read_csv(metadata_path)
    required = {
        "dataset",
        "patient_id",
        "cv_nodule_id",
        "cv_role",
        "cv_fold",
        "label",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    metadata = metadata.copy()
    metadata["label"] = metadata["label"].astype(str).str.lower()
    metadata["unique_patient_id"] = (
        metadata["dataset"].astype(str)
        + "::"
        + metadata["patient_id"].astype(str)
    )
    development = metadata.loc[metadata["cv_role"].eq("development")]
    holdout = metadata.loc[metadata["cv_role"].eq("holdout_test")]
    records: list[dict[str, Any]] = []

    for fold in range(5):
        partitions = (
            ("Training", development.loc[development["cv_fold"].ne(fold)]),
            ("Validation", development.loc[development["cv_fold"].eq(fold)]),
        )
        for split_name, split_frame in partitions:
            for scope, dataset in SOURCE_SCOPES:
                subset = select_scope(split_frame, dataset)
                records.append(
                    {
                        "validation_fold": str(fold),
                        "dataset_scope": scope,
                        "split": split_name,
                        **count_entities(subset),
                    }
                )

    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(holdout, dataset)
        records.append(
            {
                "validation_fold": "Independent",
                "dataset_scope": scope,
                "split": "Held-Out Test",
                **count_entities(subset),
            }
        )

    result = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def source_metric_table(
    frame: pd.DataFrame,
    model_name: str,
    level: str,
) -> pd.DataFrame:
    """Calculate classification metrics for combined and source subsets."""

    records = []
    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(frame, dataset)
        metrics = binary_metrics(subset["target"], subset["prediction"])
        records.append(
            {
                "model": model_name,
                "dataset_scope": scope,
                "evaluation_level": level,
                "samples": int(len(subset)),
                "patients": int(subset["unique_patient_id"].nunique()),
                "nodules": int(subset["cv_nodule_id"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records)


def export_prediction_table(
    frame: pd.DataFrame,
    level: str,
    output_path: Path,
) -> pd.DataFrame:
    """Export reviewer-ready predictions with explicit identifier columns."""

    identifier_columns = ["dataset", "patient_id", "cv_nodule_id"]
    if level == "slice":
        identifier_columns.append("filename")
    if frame.duplicated(identifier_columns).any():
        raise ValueError(
            f"Duplicate {level}-level prediction identifiers were found."
        )

    exported = pd.DataFrame(
        {
            "dataset_source": frame["dataset"].astype(str),
            "patient_id": frame["patient_id"].astype(str),
            "nodule_id": frame["cv_nodule_id"].astype(str),
            "ground_truth_label": np.where(
                frame["target"].astype(int).eq(1), "malignant", "benign"
            ),
            "ensemble_malignant_probability": frame["probability"].astype(float),
            "predicted_label": np.where(
                frame["prediction"].astype(int).eq(1), "malignant", "benign"
            ),
        }
    )
    if level == "slice":
        exported.insert(3, "slice_id", frame["filename"].astype(str))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(output_path, index=False)
    return exported


def plot_source_specific_metrics(
    metrics: pd.DataFrame,
    model_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Summarize source-specific slice- and nodule-level performance."""

    metric_labels = {
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1_score": "F1-score",
    }
    scope_styles = {
        "Combined": {"color": "#000000", "marker": "o", "linestyle": "-"},
        "LIDC-IDRI": {"color": "#0072B2", "marker": "s", "linestyle": "--"},
        "LNDb": {"color": "#D55E00", "marker": "^", "linestyle": ":"},
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.20), sharey=True)
    positions = np.arange(len(METRIC_NAMES))
    for axis, level in zip(axes, ("slice", "nodule"), strict=True):
        level_metrics = metrics.loc[
            metrics["evaluation_level"].eq(level)
        ].set_index("dataset_scope")
        for scope, style in scope_styles.items():
            values = level_metrics.loc[scope, list(METRIC_NAMES)].to_numpy(float)
            axis.plot(
                positions,
                values,
                label=scope,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                markersize=3.2,
                markerfacecolor="white",
            )
        axis.set_xticks(
            positions,
            [metric_labels[name] for name in METRIC_NAMES],
            rotation=30,
            ha="right",
        )
        axis.set_ylim(0, 1.02)
        axis.set_title(f"{level.capitalize()} level")
        axis.set_ylabel("Metric value" if level == "slice" else "")
        style_axis(axis)
    axes[1].legend(loc="lower right", frameon=False)
    figure.suptitle(model_name, fontsize=8)
    figure.subplots_adjust(
        left=0.09, right=0.995, bottom=0.24, top=0.88, wspace=0.16
    )
    save_figure(figure, output_path, dpi)


def plot_confusion(
    row: pd.Series,
    model_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Save one two-class confusion matrix."""

    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    figure, axis = plt.subplots(figsize=(3.15, 2.85))
    image = axis.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )
    axis.set_xticks([0, 1], ["Benign", "Malignant"])
    axis.set_yticks([0, 1], ["Benign", "Malignant"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(model_name)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.subplots_adjust(left=0.20, right=0.88, bottom=0.18, top=0.90)
    save_figure(figure, output_path, dpi)


def roc_pr_outputs(
    frame: pd.DataFrame,
    model_name: str,
    level: str,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    """Save ROC/PR points, figures, AUC, and average precision."""

    records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    style = MODEL_STYLES[model_name]
    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(frame, dataset)
        targets = subset["target"].to_numpy(dtype=int)
        probabilities = subset["probability"].to_numpy(dtype=float)
        slug = scope.lower().replace("-", "_").replace(" ", "_")
        if np.unique(targets).size < 2:
            records.append(
                {
                    "model": model_name,
                    "dataset_scope": scope,
                    "evaluation_level": level,
                    "roc_auc": np.nan,
                    "average_precision": np.nan,
                    "status": "undefined_single_class",
                }
            )
            continue

        false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
            targets, probabilities
        )
        precision, recall, pr_thresholds = precision_recall_curve(
            targets, probabilities
        )
        roc_auc = float(roc_auc_score(targets, probabilities))
        average_precision = float(average_precision_score(targets, probabilities))
        pd.DataFrame(
            {
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": roc_thresholds,
            }
        ).to_csv(output_dir / f"{level}_{slug}_roc_points.csv", index=False)
        pd.DataFrame(
            {
                "recall": recall,
                "precision": precision,
                "threshold": np.append(pr_thresholds, np.nan),
            }
        ).to_csv(output_dir / f"{level}_{slug}_pr_points.csv", index=False)

        figure, axis = plt.subplots(figsize=(3.55, 3.20))
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.25,
            label=f"{model_name} (AUC={roc_auc:.3f})",
        )
        axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        axis.legend(loc="lower right", frameon=False)
        save_figure(figure, output_dir / f"{level}_{slug}_roc_curve.png", dpi)

        figure, axis = plt.subplots(figsize=(3.55, 3.20))
        axis.plot(
            recall,
            precision,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.25,
            label=f"{model_name} (AP={average_precision:.3f})",
        )
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        axis.legend(loc="lower left", frameon=False)
        save_figure(figure, output_dir / f"{level}_{slug}_pr_curve.png", dpi)
        records.append(
            {
                "model": model_name,
                "dataset_scope": scope,
                "evaluation_level": level,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "status": "ok",
            }
        )
    return pd.DataFrame.from_records(records)


def calibration_outputs(
    frame: pd.DataFrame,
    model_name: str,
    level: str,
    output_dir: Path,
    bins: int,
    dpi: int,
) -> pd.DataFrame:
    """Save equal-width reliability diagrams, Brier score, and ECE."""

    if bins < 2:
        raise ValueError("Calibration bins must be at least two.")
    output_dir.mkdir(parents=True, exist_ok=True)
    style = MODEL_STYLES[model_name]
    summaries: list[dict[str, Any]] = []
    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(frame, dataset)
        targets = subset["target"].to_numpy(dtype=int)
        probabilities = subset["probability"].to_numpy(dtype=float)
        bin_indices = np.minimum((probabilities * bins).astype(int), bins - 1)
        rows = []
        for bin_index in range(bins):
            selected = bin_indices == bin_index
            count = int(selected.sum())
            mean_probability = float(probabilities[selected].mean()) if count else np.nan
            observed_rate = float(targets[selected].mean()) if count else np.nan
            rows.append(
                {
                    "bin": bin_index + 1,
                    "lower_bound": bin_index / bins,
                    "upper_bound": (bin_index + 1) / bins,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_frequency": observed_rate,
                    "absolute_gap": (
                        abs(observed_rate - mean_probability) if count else np.nan
                    ),
                }
            )
        statistics = pd.DataFrame.from_records(rows)
        ece = float(
            np.nansum(
                statistics["count"]
                / max(len(subset), 1)
                * statistics["absolute_gap"]
            )
        )
        brier = float(brier_score_loss(targets, probabilities))
        slug = scope.lower().replace("-", "_").replace(" ", "_")
        statistics.to_csv(
            output_dir / f"{level}_{slug}_calibration_bins.csv", index=False
        )

        nonempty = statistics.loc[statistics["count"].gt(0)]
        figure, axis = plt.subplots(figsize=(3.55, 3.20))
        axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8)
        axis.plot(
            nonempty["mean_probability"],
            nonempty["observed_frequency"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2,
            markerfacecolor="white",
            linewidth=1.25,
            label=f"{model_name} (ECE={ece:.3f})",
        )
        axis.set_xlabel("Mean Predicted Probability")
        axis.set_ylabel("Observed Malignant Fraction")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        style_axis(axis)
        axis.legend(loc="upper left", frameon=False)
        save_figure(
            figure,
            output_dir / f"{level}_{slug}_reliability_diagram.png",
            dpi,
        )
        summaries.append(
            {
                "model": model_name,
                "dataset_scope": scope,
                "evaluation_level": level,
                "samples": int(len(subset)),
                "calibration_bins": bins,
                "binning_method": "equal_width_[0,1]",
                "brier_score": brier,
                "expected_calibration_error": ece,
            }
        )
    return pd.DataFrame.from_records(summaries)


def training_curve_outputs(
    model_dir: Path,
    model_name: str,
    output_dir: Path,
    dpi: int,
) -> None:
    """Regenerate five fold training curves with the shared visual style."""

    output_dir.mkdir(parents=True, exist_ok=True)
    style = MODEL_STYLES[model_name]
    for fold in range(5):
        baseline_path = model_dir / f"fold_{fold}/training_log.csv"
        guided_path = model_dir / f"fold_{fold}/history.csv"
        path = baseline_path if baseline_path.is_file() else guided_path
        if not path.is_file():
            raise FileNotFoundError(f"Training history not found for fold {fold}.")
        history = pd.read_csv(path)
        figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.15))
        axes[0].plot(
            history["epoch"], history["train_loss"],
            color=style["color"], linewidth=1.25, label="Training",
        )
        axes[0].plot(
            history["epoch"], history["val_loss"],
            color=style["color"], linestyle="--", linewidth=1.25,
            label="Validation",
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-Entropy Loss")
        style_axis(axes[0])
        axes[0].legend(frameon=False)

        axes[1].plot(
            history["epoch"], history["train_accuracy"],
            color=style["color"], linewidth=1.25, label="Training",
        )
        axes[1].plot(
            history["epoch"], history["val_accuracy"],
            color=style["color"], linestyle="--", linewidth=1.25,
            label="Validation",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1.02)
        style_axis(axes[1])
        axes[1].legend(frameon=False)
        figure.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.98, wspace=0.25)
        save_figure(figure, output_dir / f"fold_{fold}_training_curves.png", dpi)


def refresh_legacy_figures(
    model_dir: Path,
    model_name: str,
    slice_predictions: pd.DataFrame,
    dpi: int,
) -> None:
    """Overwrite legacy summary figures with the shared 500-DPI style."""

    combined_metrics = source_metric_table(slice_predictions, model_name, "slice").iloc[0]
    plot_confusion(
        combined_metrics,
        model_name,
        model_dir / "test/confusion_matrix.png",
        dpi,
    )
    targets = slice_predictions["target"].to_numpy(dtype=int)
    probabilities = slice_predictions["probability"].to_numpy(dtype=float)
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, probabilities)
    auc_value = roc_auc_score(targets, probabilities)
    style = MODEL_STYLES[model_name]
    figure, axis = plt.subplots(figsize=(3.55, 3.20))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=1.25,
        label=f"{model_name} (AUC={auc_value:.3f})",
    )
    axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    style_axis(axis)
    axis.legend(loc="lower right", frameon=False)
    save_figure(figure, model_dir / "test/roc_curve.png", dpi)

    summary_path = model_dir / "cv_summary.csv"
    if not summary_path.is_file():
        summary_path = model_dir / "cross_validation_summary.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        figure, axis = plt.subplots(figsize=(4.80, 3.20))
        fold_values = summary["fold"].astype(int)
        for column, label, marker in (
            ("best_val_accuracy", "Accuracy", "o"),
            ("best_sensitivity", "Sensitivity", "s"),
            ("best_specificity", "Specificity", "^"),
        ):
            candidates = (
                column,
                column.replace("best_val_", ""),
                column.replace("best_", ""),
            )
            source_column = next(
                candidate for candidate in candidates if candidate in summary
            )
            axis.plot(
                fold_values,
                summary[source_column],
                marker=marker,
                markersize=3.2,
                linewidth=1.1,
                label=label,
            )
        axis.set_xlabel("Validation Fold")
        axis.set_ylabel("Metric Value")
        axis.set_xticks(range(5))
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        axis.legend(loc="lower right", frameon=False)
        save_figure(figure, model_dir / "figures/cv_fold_metrics.png", dpi)

    for fold in range(5):
        history_path = model_dir / f"fold_{fold}/training_log.csv"
        prediction_path = model_dir / f"fold_{fold}/validation_predictions.csv"
        if not history_path.is_file() or not prediction_path.is_file():
            continue
        figure_dir = model_dir / f"fold_{fold}/figures"
        history = pd.read_csv(history_path)
        for output_name, train_column, validation_column, y_label in (
            ("loss_curve.png", "train_loss", "val_loss", "Cross-Entropy Loss"),
            ("accuracy_curve.png", "train_accuracy", "val_accuracy", "Accuracy"),
        ):
            figure, axis = plt.subplots(figsize=(3.55, 3.20))
            axis.plot(history["epoch"], history[train_column], label="Training")
            axis.plot(
                history["epoch"],
                history[validation_column],
                linestyle="--",
                label="Validation",
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(y_label)
            style_axis(axis)
            axis.legend(frameon=False)
            save_figure(figure, figure_dir / output_name, dpi)

        figure, axis = plt.subplots(figsize=(4.80, 3.20))
        for column, label in (
            ("sensitivity", "Sensitivity"),
            ("specificity", "Specificity"),
            ("precision", "Precision"),
            ("f1_score", "F1-score"),
            ("auc", "ROC-AUC"),
        ):
            axis.plot(history["epoch"], history[column], linewidth=1.0, label=label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation Metric")
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        axis.legend(ncol=2, frameon=False)
        save_figure(figure, figure_dir / "metrics_curve.png", dpi)

        validation = prepare_predictions(prediction_path, "slice")
        fold_metrics = source_metric_table(validation, model_name, "slice").iloc[0]
        plot_confusion(
            fold_metrics,
            model_name,
            figure_dir / "confusion_matrix.png",
            dpi,
        )
        fold_targets = validation["target"].to_numpy(dtype=int)
        fold_probabilities = validation["probability"].to_numpy(dtype=float)
        false_positive_rate, true_positive_rate, _ = roc_curve(
            fold_targets, fold_probabilities
        )
        figure, axis = plt.subplots(figsize=(3.55, 3.20))
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            color=style["color"],
            linewidth=1.25,
        )
        axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8)
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        style_axis(axis)
        save_figure(figure, figure_dir / "roc_curve.png", dpi)


def run_single_model_evaluation(
    model_dir: str | Path,
    model_name: str,
    metadata_path: str | Path,
    calibration_bins: int = 10,
    dpi: int = 500,
) -> dict[str, Path]:
    """Generate all single-model held-out and source-specific outputs."""

    if model_name not in MODEL_STYLES:
        raise ValueError(f"Model name must be one of {tuple(MODEL_STYLES)}.")
    configure_plot_style()
    model_dir = Path(model_dir)
    metadata_path = Path(metadata_path)
    slice_path = model_dir / "test/test_predictions.csv"
    nodule_path = model_dir / "nodule_voting/test_nodule_predictions.csv"
    slice_predictions = prepare_predictions(slice_path, "slice")
    nodule_predictions = prepare_predictions(nodule_path, "nodule")

    source_dir = model_dir / "source_specific_evaluation"
    confusion_dir = model_dir / "confusion_matrices"
    roc_pr_dir = model_dir / "roc_pr_evaluation"
    calibration_dir = model_dir / "calibration"
    training_dir = model_dir / "training_curves"
    for directory in (
        source_dir,
        confusion_dir,
        roc_pr_dir,
        calibration_dir,
        training_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    build_split_distribution(
        metadata_path,
        source_dir / "source_class_split_distribution.csv",
    )
    export_prediction_table(
        slice_predictions,
        "slice",
        source_dir / "held_out_slice_predictions.csv",
    )
    export_prediction_table(
        nodule_predictions,
        "nodule",
        source_dir / "held_out_nodule_predictions.csv",
    )
    metric_frames = []
    roc_frames = []
    calibration_frames = []
    for level, predictions in (
        ("slice", slice_predictions),
        ("nodule", nodule_predictions),
    ):
        metrics = source_metric_table(predictions, model_name, level)
        metric_frames.append(metrics)
        for _, row in metrics.iterrows():
            slug = row["dataset_scope"].lower().replace("-", "_").replace(" ", "_")
            values = row[
                ["tn", "fp", "fn", "tp", *METRIC_NAMES]
            ].to_frame().T
            values.to_csv(
                confusion_dir / f"{level}_{slug}_confusion_matrix.csv",
                index=False,
            )
            plot_confusion(
                row,
                model_name,
                confusion_dir / f"{level}_{slug}_confusion_matrix.png",
                dpi,
            )
        roc_frames.append(
            roc_pr_outputs(predictions, model_name, level, roc_pr_dir, dpi)
        )
        calibration_frames.append(
            calibration_outputs(
                predictions,
                model_name,
                level,
                calibration_dir,
                calibration_bins,
                dpi,
            )
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(source_dir / "source_specific_metrics.csv", index=False)
    plot_source_specific_metrics(
        metrics,
        model_name,
        source_dir / "source_specific_performance.png",
        dpi,
    )
    metrics.to_csv(confusion_dir / "classification_metrics_summary.csv", index=False)
    pd.concat(roc_frames, ignore_index=True).to_csv(
        roc_pr_dir / "roc_auc_average_precision_summary.csv", index=False
    )
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        calibration_dir / "calibration_summary.csv", index=False
    )
    training_curve_outputs(model_dir, model_name, training_dir, dpi)
    return {
        "source_specific": source_dir,
        "confusion_matrices": confusion_dir,
        "roc_pr": roc_pr_dir,
        "calibration": calibration_dir,
        "training_curves": training_dir,
    }


def align_predictions(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    level: str,
) -> pd.DataFrame:
    """Align paired predictions and verify identical samples and targets."""

    keys = ["dataset", "patient_id", "cv_nodule_id"]
    if level == "slice":
        keys.append("filename")
    columns = keys + ["unique_patient_id", "target", "prediction", "probability"]
    paired = baseline[columns].merge(
        proposed[columns],
        on=keys,
        how="outer",
        suffixes=("_baseline", "_proposed"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise RuntimeError(f"Baseline and proposed {level} samples do not match.")
    if not paired["target_baseline"].eq(paired["target_proposed"]).all():
        raise RuntimeError(f"Baseline and proposed {level} targets do not match.")
    paired["target"] = paired["target_baseline"].astype(int)
    paired["unique_patient_id"] = paired["unique_patient_id_baseline"]
    return paired.drop(columns="_merge")


def paired_metric_summary(paired: pd.DataFrame, level: str) -> pd.DataFrame:
    """Build a baseline-versus-proposed source-specific metric table."""

    records = []
    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(paired, dataset)
        for model in ("Baseline", "Proposed"):
            suffix = model.lower()
            metrics = binary_metrics(
                subset["target"], subset[f"prediction_{suffix}"]
            )
            records.append(
                {
                    "model": model,
                    "dataset_scope": scope,
                    "evaluation_level": level,
                    "samples": int(len(subset)),
                    "patients": int(subset["unique_patient_id"].nunique()),
                    "nodules": int(subset["cv_nodule_id"].nunique()),
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def clustered_bootstrap(
    paired_nodules: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired patient-clustered bootstrap on nodule-level predictions."""

    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    random = np.random.default_rng(seed)
    all_iterations: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for scope, dataset in SOURCE_SCOPES:
        subset = select_scope(paired_nodules, dataset).reset_index(drop=True)
        positive_patients = subset.loc[
            subset["target"].eq(1), "unique_patient_id"
        ].nunique()
        negative_patients = subset.loc[
            subset["target"].eq(0), "unique_patient_id"
        ].nunique()
        patients = subset["unique_patient_id"].drop_duplicates().to_numpy()
        if positive_patients < 2 or negative_patients < 2:
            for metric in METRIC_NAMES:
                summaries.append(
                    {
                        "dataset_scope": scope,
                        "metric": metric,
                        "iterations_requested": iterations,
                        "confidence_level": 0.95,
                        "random_seed": seed,
                        "resampling_unit": "patient",
                        "paired": True,
                        "valid_iterations": 0,
                        "mean_difference": np.nan,
                        "ci_lower": np.nan,
                        "ci_upper": np.nan,
                        "status": "insufficient_class_specific_patients",
                    }
                )
            continue

        patient_rows = {
            patient: np.flatnonzero(subset["unique_patient_id"].eq(patient))
            for patient in patients
        }
        for iteration in range(iterations):
            sampled_patients = random.choice(
                patients,
                size=len(patients),
                replace=True,
            )
            sampled_rows = np.concatenate(
                [patient_rows[patient] for patient in sampled_patients]
            )
            sample = subset.iloc[sampled_rows]
            baseline_metrics = binary_metrics(
                sample["target"], sample["prediction_baseline"]
            )
            proposed_metrics = binary_metrics(
                sample["target"], sample["prediction_proposed"]
            )
            record: dict[str, Any] = {
                "dataset_scope": scope,
                "iteration": iteration + 1,
                "sampled_patients": len(sampled_patients),
                "sampled_nodules": len(sample),
            }
            for metric in METRIC_NAMES:
                baseline_value = baseline_metrics[metric]
                proposed_value = proposed_metrics[metric]
                record[f"baseline_{metric}"] = baseline_value
                record[f"proposed_{metric}"] = proposed_value
                record[f"difference_{metric}"] = (
                    proposed_value - baseline_value
                    if np.isfinite(baseline_value) and np.isfinite(proposed_value)
                    else np.nan
                )
            all_iterations.append(record)

        scope_iterations = pd.DataFrame(
            record for record in all_iterations if record["dataset_scope"] == scope
        )
        for metric in METRIC_NAMES:
            differences = scope_iterations[f"difference_{metric}"].dropna()
            has_valid_iterations = not differences.empty
            summaries.append(
                {
                    "dataset_scope": scope,
                    "metric": metric,
                    "iterations_requested": iterations,
                    "confidence_level": 0.95,
                    "random_seed": seed,
                    "resampling_unit": "patient",
                    "paired": True,
                    "valid_iterations": int(len(differences)),
                    "mean_difference": (
                        float(differences.mean())
                        if has_valid_iterations
                        else np.nan
                    ),
                    "ci_lower": (
                        float(differences.quantile(0.025))
                        if has_valid_iterations
                        else np.nan
                    ),
                    "ci_upper": (
                        float(differences.quantile(0.975))
                        if has_valid_iterations
                        else np.nan
                    ),
                    "status": (
                        "ok" if has_valid_iterations else "no_valid_iterations"
                    ),
                }
            )
    return pd.DataFrame.from_records(all_iterations), pd.DataFrame.from_records(summaries)


def plot_bootstrap_intervals(summary: pd.DataFrame, path: Path, dpi: int) -> None:
    """Plot paired performance differences and 95% confidence intervals."""

    valid = summary.loc[summary["status"].eq("ok")].copy()
    if valid.empty:
        return
    metric_labels = {
        "accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1_score": "F1-score",
    }
    scopes = [scope for scope, _ in SOURCE_SCOPES if scope in set(valid["dataset_scope"])]
    figure, axes = plt.subplots(
        1,
        len(scopes),
        figsize=(3.55 * len(scopes), 3.25),
        sharey=True,
        squeeze=False,
    )
    for axis, scope in zip(axes[0], scopes, strict=True):
        panel = valid.loc[valid["dataset_scope"].eq(scope)].set_index("metric")
        panel = panel.reindex(METRIC_NAMES).dropna(subset=["mean_difference"])
        positions = np.arange(len(panel))
        means = panel["mean_difference"].to_numpy()
        lower = means - panel["ci_lower"].to_numpy()
        upper = panel["ci_upper"].to_numpy() - means
        axis.errorbar(
            means,
            positions,
            xerr=np.vstack([lower, upper]),
            fmt="s",
            color=MODEL_STYLES["Proposed"]["color"],
            markersize=4,
            capsize=2.5,
            linewidth=1.0,
        )
        axis.axvline(0, color="#606060", linestyle=":", linewidth=0.8)
        axis.set_yticks(positions, [metric_labels[name] for name in panel.index])
        axis.invert_yaxis()
        axis.set_title(scope)
        axis.set_xlabel("Proposed - Baseline")
        style_axis(axis, grid_axis="x")
    figure.subplots_adjust(left=0.13, right=0.995, bottom=0.17, top=0.90, wspace=0.20)
    save_figure(figure, path, dpi)


def paired_roc_pr_outputs(
    pairs: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    """Save overlaid ROC and precision-recall comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for level, paired in pairs.items():
        for scope, dataset in SOURCE_SCOPES:
            subset = select_scope(paired, dataset)
            targets = subset["target"].to_numpy(dtype=int)
            slug = scope.lower().replace("-", "_").replace(" ", "_")
            if np.unique(targets).size < 2:
                for model in ("Baseline", "Proposed"):
                    summaries.append(
                        {
                            "model": model,
                            "dataset_scope": scope,
                            "evaluation_level": level,
                            "roc_auc": np.nan,
                            "average_precision": np.nan,
                            "status": "undefined_single_class",
                        }
                    )
                continue

            roc_figure, roc_axis = plt.subplots(figsize=(3.55, 3.20))
            pr_figure, pr_axis = plt.subplots(figsize=(3.55, 3.20))
            roc_point_frames = []
            pr_point_frames = []
            for model in ("Baseline", "Proposed"):
                suffix = model.lower()
                probabilities = subset[f"probability_{suffix}"].to_numpy(dtype=float)
                false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(
                    targets, probabilities
                )
                precision, recall, pr_thresholds = precision_recall_curve(
                    targets, probabilities
                )
                roc_auc = float(roc_auc_score(targets, probabilities))
                average_precision = float(
                    average_precision_score(targets, probabilities)
                )
                style = MODEL_STYLES[model]
                roc_axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.25,
                    label=f"{model} (AUC={roc_auc:.3f})",
                )
                pr_axis.plot(
                    recall,
                    precision,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.25,
                    label=f"{model} (AP={average_precision:.3f})",
                )
                roc_point_frames.append(
                    pd.DataFrame(
                        {
                            "model": model,
                            "false_positive_rate": false_positive_rate,
                            "true_positive_rate": true_positive_rate,
                            "threshold": roc_thresholds,
                        }
                    )
                )
                pr_point_frames.append(
                    pd.DataFrame(
                        {
                            "model": model,
                            "recall": recall,
                            "precision": precision,
                            "threshold": np.append(pr_thresholds, np.nan),
                        }
                    )
                )
                summaries.append(
                    {
                        "model": model,
                        "dataset_scope": scope,
                        "evaluation_level": level,
                        "roc_auc": roc_auc,
                        "average_precision": average_precision,
                        "status": "ok",
                    }
                )

            pd.concat(roc_point_frames, ignore_index=True).to_csv(
                output_dir / f"{level}_{slug}_roc_points.csv", index=False
            )
            pd.concat(pr_point_frames, ignore_index=True).to_csv(
                output_dir / f"{level}_{slug}_pr_points.csv", index=False
            )
            roc_axis.plot(
                [0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8
            )
            roc_axis.set_xlabel("False Positive Rate")
            roc_axis.set_ylabel("True Positive Rate")
            roc_axis.set_xlim(0, 1)
            roc_axis.set_ylim(0, 1.02)
            style_axis(roc_axis)
            roc_axis.legend(loc="lower right", frameon=False)
            save_figure(
                roc_figure,
                output_dir / f"{level}_{slug}_roc_comparison.png",
                dpi,
            )
            pr_axis.set_xlabel("Recall")
            pr_axis.set_ylabel("Precision")
            pr_axis.set_xlim(0, 1)
            pr_axis.set_ylim(0, 1.02)
            style_axis(pr_axis)
            pr_axis.legend(loc="lower left", frameon=False)
            save_figure(
                pr_figure,
                output_dir / f"{level}_{slug}_pr_comparison.png",
                dpi,
            )
    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(output_dir / "roc_auc_average_precision_comparison.csv", index=False)
    return summary


def paired_calibration_outputs(
    pairs: dict[str, pd.DataFrame],
    output_dir: Path,
    bins: int,
    dpi: int,
) -> pd.DataFrame:
    """Save overlaid reliability diagrams and calibration summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for level, paired in pairs.items():
        for scope, dataset in SOURCE_SCOPES:
            subset = select_scope(paired, dataset)
            targets = subset["target"].to_numpy(dtype=int)
            slug = scope.lower().replace("-", "_").replace(" ", "_")
            figure, axis = plt.subplots(figsize=(3.55, 3.20))
            axis.plot([0, 1], [0, 1], color="#707070", linestyle=":", linewidth=0.8)
            bin_frames = []
            for model in ("Baseline", "Proposed"):
                suffix = model.lower()
                probabilities = subset[f"probability_{suffix}"].to_numpy(dtype=float)
                bin_indices = np.minimum((probabilities * bins).astype(int), bins - 1)
                rows = []
                for bin_index in range(bins):
                    selected = bin_indices == bin_index
                    count = int(selected.sum())
                    mean_probability = (
                        float(probabilities[selected].mean()) if count else np.nan
                    )
                    observed_rate = float(targets[selected].mean()) if count else np.nan
                    rows.append(
                        {
                            "model": model,
                            "bin": bin_index + 1,
                            "lower_bound": bin_index / bins,
                            "upper_bound": (bin_index + 1) / bins,
                            "count": count,
                            "mean_probability": mean_probability,
                            "observed_frequency": observed_rate,
                            "absolute_gap": (
                                abs(observed_rate - mean_probability)
                                if count
                                else np.nan
                            ),
                        }
                    )
                statistics = pd.DataFrame.from_records(rows)
                bin_frames.append(statistics)
                ece = float(
                    np.nansum(
                        statistics["count"]
                        / max(len(subset), 1)
                        * statistics["absolute_gap"]
                    )
                )
                brier = float(brier_score_loss(targets, probabilities))
                nonempty = statistics.loc[statistics["count"].gt(0)]
                style = MODEL_STYLES[model]
                axis.plot(
                    nonempty["mean_probability"],
                    nonempty["observed_frequency"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=3.2,
                    markerfacecolor="white",
                    linewidth=1.25,
                    label=f"{model} (ECE={ece:.3f})",
                )
                summaries.append(
                    {
                        "model": model,
                        "dataset_scope": scope,
                        "evaluation_level": level,
                        "samples": int(len(subset)),
                        "calibration_bins": bins,
                        "binning_method": "equal_width_[0,1]",
                        "brier_score": brier,
                        "expected_calibration_error": ece,
                    }
                )
            pd.concat(bin_frames, ignore_index=True).to_csv(
                output_dir / f"{level}_{slug}_calibration_bins.csv", index=False
            )
            axis.set_xlabel("Mean Predicted Probability")
            axis.set_ylabel("Observed Malignant Fraction")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            style_axis(axis)
            axis.legend(loc="upper left", frameon=False)
            save_figure(
                figure,
                output_dir / f"{level}_{slug}_reliability_comparison.png",
                dpi,
            )
    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(output_dir / "calibration_comparison.csv", index=False)
    return summary


def run_paired_comparison(
    baseline_dir: str | Path,
    proposed_dir: str | Path,
    output_dir: str | Path,
    bootstrap_iterations: int = 10_000,
    seed: int = 42,
    dpi: int = 500,
) -> dict[str, Path]:
    """Compare retrained models using identical held-out samples."""

    configure_plot_style()
    baseline_dir = Path(baseline_dir)
    proposed_dir = Path(proposed_dir)
    output_dir = Path(output_dir)
    uncertainty_dir = output_dir / "statistical_uncertainty"
    roc_pr_dir = output_dir / "roc_pr_evaluation"
    calibration_dir = output_dir / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)

    pairs = {}
    metric_frames = []
    for level, relative_path in (
        ("slice", Path("test/test_predictions.csv")),
        ("nodule", Path("nodule_voting/test_nodule_predictions.csv")),
    ):
        baseline = prepare_predictions(baseline_dir / relative_path, level)
        proposed = prepare_predictions(proposed_dir / relative_path, level)
        paired = align_predictions(baseline, proposed, level)
        pairs[level] = paired
        metric_frames.append(paired_metric_summary(paired, level))

    comparison = pd.concat(metric_frames, ignore_index=True)
    comparison.to_csv(output_dir / "baseline_proposed_metrics_summary.csv", index=False)
    paired_roc_pr_outputs(pairs, roc_pr_dir, dpi)
    paired_calibration_outputs(pairs, calibration_dir, bins=10, dpi=dpi)

    iterations, confidence_intervals = clustered_bootstrap(
        pairs["nodule"],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    iterations.to_csv(
        uncertainty_dir / "paired_patient_clustered_bootstrap_iterations.csv",
        index=False,
    )
    confidence_intervals.to_csv(
        uncertainty_dir / "paired_patient_clustered_bootstrap_summary.csv",
        index=False,
    )
    plot_bootstrap_intervals(
        confidence_intervals,
        uncertainty_dir / "paired_patient_clustered_confidence_intervals.png",
        dpi,
    )
    return {
        "comparison": output_dir / "baseline_proposed_metrics_summary.csv",
        "roc_pr": roc_pr_dir / "roc_auc_average_precision_comparison.csv",
        "calibration": calibration_dir / "calibration_comparison.csv",
        "bootstrap_iterations": (
            uncertainty_dir / "paired_patient_clustered_bootstrap_iterations.csv"
        ),
        "bootstrap_summary": (
            uncertainty_dir / "paired_patient_clustered_bootstrap_summary.csv"
        ),
    }
