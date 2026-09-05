"""Voxel-wise consensus computation for LNDb nodule annotations."""

import math
from typing import Any

import numpy as np


def stack_masks(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Stack all cropped binary masks into a single 4D array.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing a cropped mask for each radiologist.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the stacked radiologist masks.
    """

    # Stack cropped binary masks from all radiologists
    mask_stack = np.stack(
        [rad["mask_crop"] for rad in scan["radiologists"]],
        axis=0
    )

    return {
        **scan,
        "mask_stack": mask_stack
    }


def compute_agreement_map(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute the voxel-wise agreement map from the stacked binary masks.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing the stacked radiologist masks.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the voxel-wise annotation counts.
    """

    # Count how many radiologists annotated each voxel
    agreement_map = np.sum(scan["mask_stack"], axis=0)

    return {
        **scan,
        "agreement_map": agreement_map
    }



def create_consensus_mask(
    scan: dict[str, Any],
    clevel: float = 0.5,
) -> dict[str, Any]:
    """
    Create a binary consensus mask at the requested agreement level.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing the agreement map and radiologist annotations.
    clevel : float, default=0.5
        Fraction of radiologists required to include a voxel in the consensus
        mask. The value must be in the interval ``(0, 1]``.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the consensus level, integer agreement
        threshold, and binary consensus mask.

    Raises
    ------
    ValueError
        If ``clevel`` is outside the interval ``(0, 1]``.
    """

    if not 0 < clevel <= 1:
        raise ValueError("clevel must be between 0 and 1.")

    agreement_map = scan["agreement_map"]
    num_radiologists = len(scan["radiologists"])

    threshold = math.ceil(clevel * num_radiologists)

    consensus_mask = agreement_map >= threshold

    return {
        **scan,
        "clevel": clevel,
        "consensus_threshold": threshold,
        "consensus_mask": consensus_mask,
    }


def restore_consensus_mask(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Restore the cropped consensus mask to the original CT volume size.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing the cropped consensus mask, consensus bounding
        box, and original CT volume.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the full-size consensus mask and indices
        of slices that contain consensus voxels.
    """

    # Create an empty mask with the same shape as the CT volume
    consensus_mask_full = np.zeros_like(
        scan["ct_volume"],
        dtype=scan["consensus_mask"].dtype
    )

    # Retrieve the consensus bounding box
    bbox = scan["consensus_bbox"]

    zmin, zmax = bbox["zmin"], bbox["zmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    xmin, xmax = bbox["xmin"], bbox["xmax"]

    # Insert the cropped consensus mask into the original volume
    consensus_mask_full[
        zmin:zmax + 1,
        ymin:ymax + 1,
        xmin:xmax + 1
    ] = scan["consensus_mask"]

    consensus_slices = np.where(
        np.any(
            consensus_mask_full,
            axis=(1, 2),
        )
    )[0]

    return {
        **scan,
        "consensus_mask_full": consensus_mask_full,
        "consensus_slices": consensus_slices,
    }
