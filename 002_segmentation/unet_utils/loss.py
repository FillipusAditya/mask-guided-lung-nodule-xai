"""Loss functions for binary lung nodule segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Compute Dice loss for binary image segmentation.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize the Dice loss.

        Parameters
        ----------
        smooth : float, default=1e-6
            Small constant added to avoid division by zero.
        """

        super().__init__()

        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model predictions.
        targets : torch.Tensor
            Ground-truth segmentation masks.

        Returns
        -------
        torch.Tensor
            Dice loss.
        """

        # Compute Dice entirely in FP32 to prevent overflow in large reductions
        # when model logits are produced under mixed precision.
        logits = logits.float()
        targets = targets.float()

        # Convert logits to probabilities.
        probabilities = torch.sigmoid(logits)

        # Flatten spatial dimensions while preserving the batch dimension.
        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        # Compute the intersection for each sample.
        intersection = (probabilities * targets).sum(dim=1)

        # Compute the Dice coefficient for each sample.
        dice_score = (2.0 * intersection + self.smooth) / (
            probabilities.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        # Return the mean Dice loss across the mini-batch.
        return 1.0 - dice_score.mean()


class IoULoss(nn.Module):
    """
    Compute Intersection over Union loss for binary image segmentation.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize the IoU loss.

        Parameters
        ----------
        smooth : float, default=1e-6
            Small constant added to avoid division by zero.
        """

        super().__init__()

        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the IoU loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model predictions.
        targets : torch.Tensor
            Ground-truth segmentation masks.

        Returns
        -------
        torch.Tensor
            Mean IoU loss across the mini-batch.
        """

        # Compute IoU in FP32 for stable reductions with mixed precision.
        logits = logits.float()
        targets = targets.float()

        # Convert raw logits into foreground probabilities.
        probabilities = torch.sigmoid(logits)

        # Flatten spatial dimensions while preserving the batch dimension.
        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        # Calculate intersection and union separately for each sample.
        intersection = (probabilities * targets).sum(dim=1)
        union = probabilities.sum(dim=1) + targets.sum(dim=1) - intersection

        # IoU loss is one minus the mean IoU score across the mini-batch.
        iou_score = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou_score.mean()
