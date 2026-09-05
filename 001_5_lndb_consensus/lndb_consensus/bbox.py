"""Bounding-box computation for LNDb radiologist annotations."""

from typing import Any

import numpy as np


def compute_bounding_boxes(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute the axis-aligned bounding box for each radiologist's binary mask.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing radiologist-specific binary nodule masks.

    Returns
    -------
    dict[str, Any]
        Updated scan state with a bounding box for each radiologist.

    Raises
    ------
    ValueError
        If a radiologist annotation contains no foreground voxel.
    """

    radiologists = []

    for rad in scan["radiologists"]:
        # Retrieve the binary mask
        binary_mask = rad["mask_nodule"]

        # Find the coordinates of all foreground voxels
        coords = np.argwhere(binary_mask)

        # Ensure the mask contains at least one foreground voxel
        if coords.size == 0:
            raise ValueError(
                f"Empty binary mask found for radiologist {rad['radid']}."
            )

        # Compute the minimum and maximum coordinates
        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0)

        # Store the bounding box together with the radiologist information
        radiologists.append({
            **rad,
            "bbox": {
                "xmin": int(xmin),
                "xmax": int(xmax),

                "ymin": int(ymin),
                "ymax": int(ymax),

                "zmin": int(zmin),
                "zmax": int(zmax),
            }

        })

    return {
        **scan,
        "radiologists": radiologists
    }


def compute_consensus_bounding_box(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute a bounding box that encloses all radiologist annotations.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan state containing radiologist-specific bounding boxes.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the enclosing consensus bounding box.
    """

    radiologists = scan["radiologists"]

    consensus_bbox = {
        "xmin": min(rad["bbox"]["xmin"] for rad in radiologists),
        "xmax": max(rad["bbox"]["xmax"] for rad in radiologists),

        "ymin": min(rad["bbox"]["ymin"] for rad in radiologists),
        "ymax": max(rad["bbox"]["ymax"] for rad in radiologists),

        "zmin": min(rad["bbox"]["zmin"] for rad in radiologists),
        "zmax": max(rad["bbox"]["zmax"] for rad in radiologists),
    }

    return {
        **scan,
        "consensus_bbox": consensus_bbox
    }
