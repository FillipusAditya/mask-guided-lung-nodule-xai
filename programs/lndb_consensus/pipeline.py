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


def process_scan(scan, clevel=0.5):
    """
    Execute the complete LNDb consensus pipeline.
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