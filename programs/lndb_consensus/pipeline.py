"""Orchestration for the complete LNDb annotation consensus pipeline."""

from typing import Any

from .bbox import (
    compute_bounding_boxes,
    compute_consensus_bounding_box,
)

from .consensus import (
    create_consensus_mask,
    compute_agreement_map,
    restore_consensus_mask,
    stack_masks,
)

from .crop import crop_scan

from .loader import (
    load_scan,
    verify_scan_metadata,
)


def process_scan(
    scan: dict[str, Any],
    clevel: float = 0.5,
) -> dict[str, Any]:
    """
    Execute the complete LNDb consensus pipeline.

    The pipeline loads the CT and masks, computes individual and enclosing
    bounding boxes, crops all volumes, calculates voxel-wise agreement, creates
    the consensus mask, and restores it to the original CT dimensions.

    Parameters
    ----------
    scan : dict[str, Any]
        Prepared scan metadata for one LNDb finding.
    clevel : float, default=0.5
        Fraction of radiologists required to include a voxel in the consensus
        mask.

    Returns
    -------
    dict[str, Any]
        Fully processed scan state containing cropped and restored consensus
        data.
    """

    scan = load_scan(scan)
    scan = compute_bounding_boxes(scan)
    scan = compute_consensus_bounding_box(scan)
    scan = crop_scan(scan)
    scan = stack_masks(scan)
    scan = compute_agreement_map(scan)
    scan = create_consensus_mask(
        scan,
        clevel=clevel,
    )
    scan = restore_consensus_mask(scan)

    return scan
