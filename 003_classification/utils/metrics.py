"""Plot classification training and validation metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
