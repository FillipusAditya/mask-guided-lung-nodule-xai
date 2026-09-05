"""Consensus-mask calculation and restoration for LIDC-IDRI clusters."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from pylidc.utils import consensus

from .compat import enable_pylidc_numpy_compatibility


def validate_consensus_level(consensus_level: float) -> None:
    """Require a consensus fraction in the interval ``(0, 1]``."""
    if not 0 < consensus_level <= 1:
        raise ValueError("consensus_level must be between 0 and 1.")


def restore_consensus_slices(
    cropped_mask: np.ndarray,
    bounding_box: Sequence[slice],
    image_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Restore nonempty cropped-mask slices to full CT slice dimensions.

    Parameters
    ----------
    cropped_mask
        Boolean consensus mask in ``(y, x, z)`` order as returned by pylidc.
    bounding_box
        Three slices in ``(y, x, z)`` order locating the crop in the CT volume.
    image_shape
        Original two-dimensional CT slice shape ``(height, width)``.
    """
    if cropped_mask.ndim != 3:
        raise ValueError(
            f"Expected a 3D cropped mask, received shape {cropped_mask.shape}."
        )
    if len(bounding_box) != 3:
        raise ValueError("Expected a bounding box containing y, x, and z slices.")
    if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
        raise ValueError(f"Invalid CT slice shape: {image_shape}")

    y_box, x_box, z_box = bounding_box
    expected_shape = (
        y_box.stop - y_box.start,
        x_box.stop - x_box.start,
        z_box.stop - z_box.start,
    )
    if cropped_mask.shape != expected_shape:
        raise ValueError(
            "Consensus mask and bounding-box shapes differ: "
            f"mask={cropped_mask.shape}, bbox={expected_shape}."
        )
    if not (
        0 <= y_box.start < y_box.stop <= image_shape[0]
        and 0 <= x_box.start < x_box.stop <= image_shape[1]
        and 0 <= z_box.start < z_box.stop
    ):
        raise ValueError("Consensus bounding box is outside the CT dimensions.")

    restored: dict[int, np.ndarray] = {}
    for local_index, slice_index in enumerate(range(z_box.start, z_box.stop)):
        local_mask = cropped_mask[:, :, local_index]
        if not np.any(local_mask):
            continue

        full_mask = np.zeros(image_shape, dtype=bool)
        full_mask[y_box, x_box] = local_mask.astype(bool, copy=False)
        restored[slice_index] = full_mask

    return restored


def compute_consensus_slices(
    annotation_cluster: Sequence[Any],
    image_shape: tuple[int, int],
    consensus_level: float = 0.5,
) -> dict[int, np.ndarray]:
    """Compute pylidc consensus and return full-size masks by slice index."""
    validate_consensus_level(consensus_level)
    if not annotation_cluster:
        raise ValueError("annotation_cluster must contain at least one annotation.")

    enable_pylidc_numpy_compatibility()
    cropped_mask, bounding_box, _ = consensus(
        annotation_cluster,
        clevel=consensus_level,
    )
    return restore_consensus_slices(
        cropped_mask=cropped_mask,
        bounding_box=bounding_box,
        image_shape=image_shape,
    )
