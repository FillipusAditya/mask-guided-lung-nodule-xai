"""Plot and save visual summaries of U-Net training progress."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "segmentation"
    / "unet_model"
    / "result_20260804_1043"
)

TRAINING_LOG_PATH = (
    OUTPUT_DIR
    / "training_log.csv"
)

# ---------------------------------------------------------------------
# Loss Curve
# ---------------------------------------------------------------------
def plot_loss_curve(
    training_log: pd.DataFrame,
    save_path: str | Path,
    dpi: int = 300,
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

    plt.figure(
        figsize=(8, 6),
    )

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

    plt.title(
        "Training and Validation Loss",
    )

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Loss",
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()

# ---------------------------------------------------------------------
# Dice Score Curve
# ---------------------------------------------------------------------
def plot_dice_curve(
    training_log: pd.DataFrame,
    save_path: str | Path,
    dpi: int = 300,
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

    # Find the best Dice score.
    best_index = training_log["dice_score"].idxmax()

    best_epoch = training_log.loc[
        best_index,
        "epoch",
    ]

    best_dice = training_log.loc[
        best_index,
        "dice_score",
    ]

    plt.figure(
        figsize=(8, 6),
    )

    plt.plot(
        training_log["epoch"],
        training_log["dice_score"],
        linewidth=2,
        label="Dice Score",
    )

    # Highlight the best Dice score.
    plt.scatter(
        best_epoch,
        best_dice,
        marker="o",
        s=80,
        label="Best Dice",
    )

    plt.annotate(
        f"{best_dice:.4f}",
        xy=(
            best_epoch,
            best_dice,
        ),
        xytext=(
            best_epoch,
            best_dice + 0.02,
        ),
        arrowprops={
            "arrowstyle": "->",
        },
    )

    plt.title(
        "Validation Dice Score",
    )

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Dice Score",
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()

# ---------------------------------------------------------------------
# IoU Curve
# ---------------------------------------------------------------------
def plot_iou_curve(
    training_log: pd.DataFrame,
    save_path: str | Path,
    dpi: int = 300,
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

    # Find the best IoU score.
    best_index = training_log["iou"].idxmax()

    best_epoch = training_log.loc[
        best_index,
        "epoch",
    ]

    best_iou = training_log.loc[
        best_index,
        "iou",
    ]

    plt.figure(
        figsize=(8, 6),
    )

    plt.plot(
        training_log["epoch"],
        training_log["iou"],
        linewidth=2,
        label="IoU",
    )

    # Highlight the best IoU score.
    plt.scatter(
        best_epoch,
        best_iou,
        marker="o",
        s=80,
        label="Best IoU",
    )

    plt.annotate(
        f"{best_iou:.4f}",
        xy=(
            best_epoch,
            best_iou,
        ),
        xytext=(
            best_epoch,
            best_iou + 0.02,
        ),
        arrowprops={
            "arrowstyle": "->",
        },
    )

    plt.title(
        "Validation Intersection-over-Union",
    )

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "IoU",
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Segmentation Metrics Curve
# ---------------------------------------------------------------------
def plot_metrics_curve(
    training_log: pd.DataFrame,
    save_path: str | Path,
    dpi: int = 300,
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

    plt.figure(
        figsize=(8, 6),
    )

    plt.plot(
        training_log["epoch"],
        training_log["precision"],
        linewidth=2,
        label="Precision",
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

    plt.title(
        "Segmentation Metrics",
    )

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Score",
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Learning Rate Curve
# ---------------------------------------------------------------------
def plot_learning_rate_curve(
    training_log: pd.DataFrame,
    save_path: str | Path,
    dpi: int = 300,
) -> None:
    """
    Plot the learning rate schedule.

    Parameters
    ----------
    training_log : pd.DataFrame
        DataFrame containing the training log.

    save_path : str or Path
        Output path of the figure.

    dpi : int, default=300
        Figure resolution.
    """

    plt.figure(
        figsize=(8, 6),
    )

    plt.plot(
        training_log["epoch"],
        training_log["learning_rate"],
        linewidth=2,
        label="Learning Rate",
    )

    # Find epochs where the learning rate changes.
    learning_rates = training_log["learning_rate"]

    changed = learning_rates.diff().fillna(0) != 0

    for _, row in training_log.loc[changed].iterrows():

        plt.scatter(
            row["epoch"],
            row["learning_rate"],
            s=60,
            marker="o",
        )

        plt.annotate(
            f"{row['learning_rate']:.1e}",
            xy=(
                row["epoch"],
                row["learning_rate"],
            ),
            xytext=(
                row["epoch"],
                row["learning_rate"] * 1.2,
            ),
            arrowprops={
                "arrowstyle": "->",
            },
            fontsize=8,
        )

    plt.title(
        "Learning Rate Schedule",
    )

    plt.xlabel(
        "Epoch",
    )

    plt.ylabel(
        "Learning Rate",
    )

    plt.yscale(
        "log",
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close()

# ---------------------------------------------------------------------
# Plot All Curves
# ---------------------------------------------------------------------
def plot_all_curves(
    training_log: pd.DataFrame,
    output_dir: str | Path,
    dpi: int = 300,
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

    visualization_dir = (
        output_dir
        / "visualizations"
    )

    visualization_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # Only create the learning rate curve if the column exists.
    if "learning_rate" in training_log.columns:

        plot_learning_rate_curve(
            training_log=training_log,
            save_path=visualization_dir / "learning_rate_curve.png",
            dpi=dpi,
        )


# ---------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------
def main() -> None:
    """
    Generate all training visualization figures.
    """

    # Verify that the training log exists.
    if not TRAINING_LOG_PATH.exists():

        raise FileNotFoundError(
            f"Training log not found: {TRAINING_LOG_PATH}"
        )

    # Load the training log.
    training_log = pd.read_csv(
        TRAINING_LOG_PATH,
    )

    # Generate all visualization figures.
    plot_all_curves(
        training_log=training_log,
        output_dir=OUTPUT_DIR,
    )

    print()
    print("=" * 60)
    print("Training visualizations generated successfully.")
    print(f"Output directory : {OUTPUT_DIR / 'visualizations'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
