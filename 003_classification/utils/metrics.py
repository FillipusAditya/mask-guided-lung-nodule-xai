"""Compute and plot classification evaluation metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


#---------------------------------
# CONFUSION MATRIX
#---------------------------------
def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Accumulate a batch into a multiclass confusion matrix."""

    indices = targets.to(torch.int64) * num_classes
    indices += predictions.to(torch.int64)

    batch_matrix = torch.bincount(
        indices.cpu(),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

    return confusion_matrix + batch_matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two values, returning zero when the denominator is zero."""

    return numerator / denominator if denominator > 0.0 else 0.0


def compute_classification_metrics(
    confusion_matrix: torch.Tensor,
) -> dict[str, float]:
    """Compute accuracy, sensitivity, specificity, precision, and F1-score."""

    matrix = confusion_matrix.to(torch.float64)
    total = matrix.sum().item()
    accuracy = _safe_divide(matrix.diag().sum().item(), total)

    true_positive = matrix.diag()
    false_positive = matrix.sum(dim=0) - true_positive
    false_negative = matrix.sum(dim=1) - true_positive
    true_negative = total - true_positive - false_positive - false_negative

    precision_per_class = true_positive / (
        true_positive + false_positive
    ).clamp_min(1.0)
    sensitivity_per_class = true_positive / (
        true_positive + false_negative
    ).clamp_min(1.0)
    specificity_per_class = true_negative / (
        true_negative + false_positive
    ).clamp_min(1.0)
    f1_per_class = (
        2.0 * precision_per_class * sensitivity_per_class
        / (precision_per_class + sensitivity_per_class).clamp_min(1e-12)
    )

    # For binary classification, report metrics for class index 1 as the
    # positive class. For multiclass classification, report macro averages.
    if matrix.shape[0] == 2:
        metric_index = 1
        precision = precision_per_class[metric_index].item()
        sensitivity = sensitivity_per_class[metric_index].item()
        specificity = specificity_per_class[metric_index].item()
        f1_score = f1_per_class[metric_index].item()
    else:
        precision = precision_per_class.mean().item()
        sensitivity = sensitivity_per_class.mean().item()
        specificity = specificity_per_class.mean().item()
        f1_score = f1_per_class.mean().item()

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
    }


def compute_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Compute binary or macro one-vs-rest multiclass ROC AUC."""

    try:
        if probabilities.shape[1] == 2:
            return float(
                roc_auc_score(targets, probabilities[:, 1])
            )

        return float(
            roc_auc_score(
                targets,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        # AUC is undefined when the validation set does not contain all
        # required target classes.
        return float("nan")


def plot_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Plot and save the validation confusion matrix."""

    matrix = confusion_matrix.cpu().numpy()
    figure_size = max(6, len(class_names))

    pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    ).to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="true_label",
    )

    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.colorbar()

    tick_positions = np.arange(len(class_names))
    plt.xticks(tick_positions, class_names, rotation=45, ha="right")
    plt.yticks(tick_positions, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    """Plot binary or one-vs-rest multiclass ROC curves."""

    plt.figure(figsize=(8, 6))

    if probabilities.shape[1] == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            targets,
            probabilities[:, 1],
        )
        curve_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{class_names[1]} (AUC = {curve_auc:.4f})",
        )
    else:
        binary_targets = label_binarize(
            targets,
            classes=np.arange(probabilities.shape[1]),
        )

        for class_index, class_name in enumerate(class_names):
            if np.unique(binary_targets[:, class_index]).size < 2:
                continue

            false_positive_rate, true_positive_rate, _ = roc_curve(
                binary_targets[:, class_index],
                probabilities[:, class_index],
            )
            curve_auc = auc(false_positive_rate, true_positive_rate)
            plt.plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=2,
                label=f"{class_name} (AUC = {curve_auc:.4f})",
            )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()


#---------------------------------
# PLOT TRAINING CURVES
#---------------------------------
def plot_curve(
    df: pd.DataFrame,
    train_column: str,
    val_column: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    """
    Plot the training and validation metric curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training history.

    train_column : str
        Column name for the training metric.

    val_column : str
        Column name for the validation metric.

    ylabel : str
        Label for the y-axis.

    title : str
        Figure title.

    save_path : Path
        Output path for the generated figure.
    """

    # Create a new figure
    plt.figure(
        figsize=(8, 5)
    )

    # Plot the training metric
    plt.plot(
        df["epoch"],
        df[train_column],
        label=f"Training {ylabel}",
        linewidth=2,
    )

    # Plot the validation metric
    plt.plot(
        df["epoch"],
        df[val_column],
        label=f"Validation {ylabel}",
        linewidth=2,
    )

    # Configure the figure
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.xticks(df["epoch"])

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(
        save_path,
        dpi=300,
    )

    # Close the figure to release memory
    plt.close()


def plot_loss_curve(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the training and validation loss curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training history.

    output_dir : Path
        Directory where the figure will be saved.
    """

    plot_curve(
        df=df,
        train_column="train_loss",
        val_column="val_loss",
        ylabel="Loss",
        title="Training and Validation Loss",
        save_path=output_dir / "loss_curve.png",
    )


def plot_accuracy_curve(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Plot the training and validation accuracy curves.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the training history.

    output_dir : Path
        Directory where the figure will be saved.
    """

    plot_curve(
        df=df,
        train_column="train_accuracy",
        val_column="val_accuracy",
        ylabel="Accuracy",
        title="Training and Validation Accuracy",
        save_path=output_dir / "accuracy_curve.png",
    )


def plot_validation_metrics_curve(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot validation classification metrics across epochs."""

    metric_columns = {
        "val_accuracy": "Accuracy",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1_score": "F1-score",
        "auc": "ROC AUC",
    }

    plt.figure(figsize=(10, 6))

    for column, label in metric_columns.items():
        plt.plot(
            df["epoch"],
            df[column],
            label=label,
            linewidth=2,
        )

    plt.title("Validation Classification Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.xticks(df["epoch"])
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_curve.png", dpi=300)
    plt.close()
