import math

import numpy as np

def stack_masks(scan):
    """
    Stack all cropped binary masks into a single 4D array.
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
    

def compute_agreement_map(scan):
    """
    Compute the voxel-wise agreement map from the stacked binary masks.

    Returns
    -------
    dict
        Updated scan dictionary containing the agreement map.
    """

    # Count how many radiologists annotated each voxel
    agreement_map = np.sum(scan["mask_stack"], axis=0)

    return {
        **scan,
        "agreement_map": agreement_map
    }



def create_consensus_mask(scan, clevel=0.5):
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


def restore_consensus_mask(scan):
    """
    Restore the cropped consensus mask to the original CT volume size.
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