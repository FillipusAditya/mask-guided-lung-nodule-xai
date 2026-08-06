"""Metrics for evaluating binary segmentation predictions."""

import torch


# ---------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------
def update_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    """
    Compute confusion matrix components for binary segmentation.

    Parameters
    ----------
    predictions : torch.Tensor
        Binary predicted segmentation masks.

    targets : torch.Tensor
        Ground-truth binary masks.

    Returns
    -------
    tuple[int, int, int, int]
        Tuple containing:

        - True Positive (TP)
        - False Positive (FP)
        - True Negative (TN)
        - False Negative (FN)
    """

    true_positive = (
        (predictions == 1) &
        (targets == 1)
    ).sum().item()

    false_positive = (
        (predictions == 1) &
        (targets == 0)
    ).sum().item()

    true_negative = (
        (predictions == 0) &
        (targets == 0)
    ).sum().item()

    false_negative = (
        (predictions == 0) &
        (targets == 1)
    ).sum().item()

    return (
        true_positive,
        false_positive,
        true_negative,
        false_negative,
    )


# ---------------------------------------------------------------------
# Segmentation Metrics
# ---------------------------------------------------------------------
def compute_segmentation_metrics(
    true_positive: int,
    false_positive: int,
    true_negative: int,
    false_negative: int,
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Compute binary segmentation metrics from the confusion matrix.

    Parameters
    ----------
    true_positive : int
        Number of true positive pixels.

    false_positive : int
        Number of false positive pixels.

    true_negative : int
        Number of true negative pixels.

    false_negative : int
        Number of false negative pixels.

    eps : float, default=1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        - ``dice``
        - ``iou``
        - ``precision``
        - ``sensitivity``
        - ``specificity``
    """

    dice = (
        2 * true_positive
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + eps
    )

    iou = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + false_negative
        + eps
    )

    precision = (
        true_positive
    ) / (
        true_positive
        + false_positive
        + eps
    )

    sensitivity = (
        true_positive
    ) / (
        true_positive
        + false_negative
        + eps
    )

    specificity = (
        true_negative
    ) / (
        true_negative
        + false_positive
        + eps
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
