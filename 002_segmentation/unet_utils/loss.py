"""Loss functions for binary lung nodule segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Compute Dice loss for binary image segmentation.
    """

    def __init__(
        self,
        smooth: float = 1e-6,
    ) -> None:
        """
        Initialize the Dice loss.

        Parameters
        ----------
        smooth : float, default=1e-6
            Small constant added to avoid division by zero.
        """

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
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

        # Flatten both tensors.
        probabilities = probabilities.view(-1)
        targets = targets.view(-1)

        # Compute the intersection.
        intersection = (probabilities * targets).sum()

        # Compute the Dice coefficient.
        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum()
            + targets.sum()
            + self.smooth
        )

        # Return Dice loss.
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combine BCE-with-logits and Dice losses using configurable weights.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        """
        Initialize the combined BCE and Dice loss.

        Parameters
        ----------
        pos_weight : torch.Tensor, optional
            Positive class weight used by BCEWithLogitsLoss.
        bce_weight : float, default=0.5
            Weight assigned to BCE loss.
        dice_weight : float, default=0.5
            Weight assigned to Dice loss.
        """

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
        )

        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted BCE-Dice loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model predictions.
        targets : torch.Tensor
            Ground-truth segmentation masks.

        Returns
        -------
        torch.Tensor
            Combined BCE-Dice loss.
        """

        bce_loss = self.bce(
            logits,
            targets,
        )

        dice_loss = self.dice(
            logits,
            targets,
        )

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss
