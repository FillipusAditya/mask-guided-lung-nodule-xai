from pathlib import Path

import numpy as np

from .visualize import (
    show_agreement_map,
    show_consensus_bbox,
    show_consensus_mask,
    show_ct_mask_overlay,
    show_restored_consensus_mask,
)


def save_ct_slices(scan, output_dir):
    """
    Save CT slices containing the consensus mask.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for z in scan["consensus_slices"]:

        filename = (
            output_dir
            / (
                f"LNDb-{scan['lndb_id']:04d}"
                f"_finding{scan['finding_id']}"
                f"_slice{z}.npy"
            )
        )

        np.save(
            filename,
            scan["ct_volume"][z],
        )


def save_consensus_slices(scan, output_dir):
    """
    Save consensus mask slices.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for z in scan["consensus_slices"]:

        filename = (
            output_dir
            / (
                f"LNDb-{scan['lndb_id']:04d}"
                f"_finding{scan['finding_id']}"
                f"_slice{z}.npy"
            )
        )

        np.save(
            filename,
            scan["consensus_mask_full"][z],
        )


def save_visualizations(scan, output_dir):
    """
    Save all visualization figures.
    """

    output_dir = Path(output_dir)

    overlay_dir = output_dir / "overlay"
    agreement_dir = output_dir / "agreement"
    consensus_dir = output_dir / "consensus"
    restored_dir = output_dir / "restored"
    bbox_dir = output_dir / "bbox"

    for folder in [
        overlay_dir,
        agreement_dir,
        consensus_dir,
        restored_dir,
        bbox_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    filename = (
        f"LNDb-{scan['lndb_id']:04d}"
        f"_finding{scan['finding_id']}.png"
    )

    # Restored Consensus
    show_restored_consensus_mask(
        scan,
        save_path=restored_dir / filename,
    )

    # Consensus Mask
    show_consensus_mask(
        scan,
        save_path=consensus_dir / filename,
    )

    # Agreement Map
    show_agreement_map(
        scan,
        save_path=agreement_dir / filename,
    )

    # Consensus Bounding Box
    show_consensus_bbox(
        scan,
        save_path=bbox_dir / filename,
    )

    # CT + Consensus Overlay
    show_ct_mask_overlay(
        ct_volume=scan["ct_volume"],
        mask_volume=scan["consensus_mask_full"],
        slices=scan["consensus_slices"],
        bbox=scan["consensus_bbox"],
        save_path=overlay_dir / filename,
    )