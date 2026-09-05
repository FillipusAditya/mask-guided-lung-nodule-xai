"""Export CT slices, consensus masks, and visualizations for LNDb findings."""

from pathlib import Path
from typing import Any

import numpy as np

from .visualize import (
    show_agreement_map,
    show_consensus_bbox,
    show_consensus_mask,
    show_ct_mask_overlay,
    show_restored_consensus_mask,
)


def save_ct_slices(
    scan: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Save CT slices containing the consensus mask.

    Parameters
    ----------
    scan : dict[str, Any]
        Processed scan state containing the CT volume and consensus slice
        indices.
    output_dir : str or Path
        Directory where CT slices will be saved as NumPy arrays.
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


def save_consensus_slices(
    scan: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Save consensus mask slices.

    Parameters
    ----------
    scan : dict[str, Any]
        Processed scan state containing the restored consensus mask and slice
        indices.
    output_dir : str or Path
        Directory where consensus slices will be saved as NumPy arrays.
    overwrite : bool, default=False
        Replace existing slice files when true.

    Raises
    ------
    FileExistsError
        If one or more target files exist and ``overwrite`` is false.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [
        output_dir / f"slice_{int(z)}.npy"
        for z in scan["consensus_slices"]
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Consensus mask output already exists: {existing[0]}. "
            "Use overwrite=True to replace existing slices."
        )

    for z, output_path in zip(scan["consensus_slices"], output_paths):
        np.save(output_path, scan["consensus_mask_full"][z])


def save_visualizations(
    scan: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """
    Save all visualization figures.

    Parameters
    ----------
    scan : dict[str, Any]
        Fully processed scan state containing all arrays and metadata required
        by the visualization functions.
    output_dir : str or Path
        Root directory where visualization subdirectories will be created.
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
