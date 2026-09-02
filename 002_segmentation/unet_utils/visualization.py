"""Plot and save visual summaries of U-Net training progress."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_curve(
    training_log: pd.DataFrame, save_path: str | Path, dpi: int = 300
) -> None:
    """
    Plot the training and validation loss curves.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.
    save_path : str or Path
        Output path of the figure.
    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        training_log["epoch"],
        training_log["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        training_log["epoch"],
        training_log["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_dice_curve(
    training_log: pd.DataFrame, save_path: str | Path, dpi: int = 300
) -> None:
    """
    Plot the Dice score curve.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.
    save_path : str or Path
        Output path of the figure.
    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        training_log["epoch"],
        training_log["dice_score"],
        linewidth=2,
        label="Dice Score",
    )

    plt.title("Validation Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_iou_curve(
    training_log: pd.DataFrame, save_path: str | Path, dpi: int = 300
) -> None:
    """
    Plot the Intersection-over-Union (IoU) curve.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.
    save_path : str or Path
        Output path of the figure.
    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(training_log["epoch"], training_log["iou"], linewidth=2, label="IoU")

    plt.title("Validation Intersection-over-Union")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metrics_curve(
    training_log: pd.DataFrame, save_path: str | Path, dpi: int = 300
) -> None:
    """
    Plot the segmentation metrics curves.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.
    save_path : str or Path
        Output path of the figure.
    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(figsize=(8, 6))

    plt.plot(
        training_log["epoch"], training_log["precision"], linewidth=2, label="Precision"
    )

    plt.plot(
        training_log["epoch"],
        training_log["sensitivity"],
        linewidth=2,
        label="Sensitivity",
    )

    plt.plot(
        training_log["epoch"],
        training_log["specificity"],
        linewidth=2,
        label="Specificity",
    )

    plt.title("Segmentation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_all_curves(
    training_log: pd.DataFrame, output_dir: str | Path, dpi: int = 300
) -> None:
    """
    Generate all training visualization figures.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.
    output_dir : str or Path
        Directory where all figures will be saved.
    dpi : int, default=300
        Figure resolution.
    """

    output_dir = Path(output_dir)

    visualization_dir = output_dir / "visualizations"

    visualization_dir.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(
        training_log=training_log,
        save_path=visualization_dir / "loss_curve.png",
        dpi=dpi,
    )

    plot_dice_curve(
        training_log=training_log,
        save_path=visualization_dir / "dice_curve.png",
        dpi=dpi,
    )

    plot_iou_curve(
        training_log=training_log,
        save_path=visualization_dir / "iou_curve.png",
        dpi=dpi,
    )

    plot_metrics_curve(
        training_log=training_log,
        save_path=visualization_dir / "metrics_curve.png",
        dpi=dpi,
    )
