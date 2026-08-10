"""Load LNDb CT volumes, annotations, and image geometry metadata."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import SimpleITK as sitk


def prepare_scan_data(
    row: Mapping[str, Any],
    data_dir: Path,
    mask_dir: Path,
) -> dict[str, Any]:
    """
    Prepare CT scan and annotation metadata for a single finding.

    Parameters
    ----------
    row : Mapping[str, Any]
        Metadata record containing LNDb scan, finding, label, and radiologist
        identifiers.
    data_dir : Path
        Directory containing LNDb CT volumes in MetaImage format.
    mask_dir : Path
        Directory containing radiologist annotation volumes.

    Returns
    -------
    dict[str, Any]
        Scan metadata and radiologist-specific annotation metadata required by
        the consensus pipeline.
    """

    # Retrieve scan and finding identifiers
    lndb_id = row["lndbid"]
    finding_id = row["findingid"]
    label = row["label"]

    # Construct CT image path and load image metadata
    ct_path = data_dir / f"LNDb-{lndb_id:04d}.mhd"
    ct_image = sitk.ReadImage(ct_path)

    # Extract CT image metadata
    ct_size = ct_image.GetSize()
    ct_spacing = ct_image.GetSpacing()
    ct_origin = ct_image.GetOrigin()
    ct_direction = ct_image.GetDirection()

    # Parse radiologist IDs
    rad_ids = [
        int(idx.strip())
        for idx in str(row["radid"]).split(",")
    ]

    # Parse finding IDs assigned by each radiologist
    rad_finding_ids = [
        int(idx.strip())
        for idx in str(row["radfindingid"]).split(",")
    ]

    # Store metadata for each radiologist annotation
    radiologists = []

    for rad_id, rad_finding_id in zip(rad_ids, rad_finding_ids):

        # Construct mask path and load mask metadata
        mask_path = mask_dir / f"LNDb-{lndb_id:04d}_rad{rad_id}.mhd"
        mask_image = sitk.ReadImage(mask_path)

        # Save radiologist-specific information
        radiologists.append({
            "radid": rad_id,
            "radfindingid": rad_finding_id,
            "mask_path": mask_path,
            "size": mask_image.GetSize(),
            "spacing": mask_image.GetSpacing(),
            "origin": mask_image.GetOrigin(),
            "direction": mask_image.GetDirection()
        })

    # Return all metadata for this finding
    return {
        "lndb_id": lndb_id,
        "finding_id": finding_id,
        "label": label, 
        "ct_path": ct_path,
        "ct_size": ct_size,
        "ct_spacing": ct_spacing,
        "ct_origin": ct_origin,
        "ct_direction": ct_direction,
        "radiologists": radiologists
    }


def verify_scan_metadata(
    scan: dict[str, Any],
) -> bool:
    """
    Verify that all radiologist masks have the same image geometry
    as the corresponding CT scan.

    Parameters
    ----------
    scan : dict[str, Any]
        Scan metadata containing CT and radiologist mask geometry.

    Returns
    -------
    bool
        True when every radiologist mask matches the CT image geometry.

    Raises
    ------
    ValueError
        If any mask differs from the CT in size, spacing, origin, or direction.
    """

    for radiologist in scan["radiologists"]:

        # Verify image size
        if scan["ct_size"] != radiologist["size"]:
            raise ValueError(
                f"Size mismatch "
                f"(LNDb-{scan['lndb_id']:04d}, "
                f"Radiologist {radiologist['radid']}): "
                f"CT={scan['ct_size']}, "
                f"Mask={radiologist['size']}"
            )

        # Verify voxel spacing
        if scan["ct_spacing"] != radiologist["spacing"]:
            raise ValueError(
                f"Spacing mismatch "
                f"(LNDb-{scan['lndb_id']:04d}, "
                f"Radiologist {radiologist['radid']}): "
                f"CT={scan['ct_spacing']}, "
                f"Mask={radiologist['spacing']}"
            )

        # Verify image origin
        if scan["ct_origin"] != radiologist["origin"]:
            raise ValueError(
                f"Origin mismatch "
                f"(LNDb-{scan['lndb_id']:04d}, "
                f"Radiologist {radiologist['radid']}): "
                f"CT={scan['ct_origin']}, "
                f"Mask={radiologist['origin']}"
            )

        # Verify image direction
        if scan["ct_direction"] != radiologist["direction"]:
            raise ValueError(
                f"Direction mismatch "
                f"(LNDb-{scan['lndb_id']:04d}, "
                f"Radiologist {radiologist['radid']}): "
                f"CT={scan['ct_direction']}, "
                f"Mask={radiologist['direction']}"
            )

    return True


def load_scan(
    scan: dict[str, Any],
) -> dict[str, Any]:
    """
    Load CT volume and extract binary masks for each radiologist annotation.

    Parameters
    ----------
    scan : dict[str, Any]
        Prepared scan metadata containing CT and radiologist mask paths.

    Returns
    -------
    dict[str, Any]
        Updated scan state containing the CT array, binary nodule masks, and
        annotated slice indices for every radiologist.
    """

    # Load CT volume
    ct_image = sitk.ReadImage(scan["ct_path"])
    ct_volume = sitk.GetArrayFromImage(ct_image)

    radiologists = []

    for rad in scan["radiologists"]:

        # Load mask volume
        mask_image = sitk.ReadImage(rad["mask_path"])
        mask_volume = sitk.GetArrayFromImage(mask_image)

        # Extract binary mask for the target nodule
        mask_nodule = (mask_volume == rad["radfindingid"])

        # Find slices containing the selected rad finding id
        nodule_slices = np.where(
            np.any(mask_volume == rad["radfindingid"], axis=(1, 2))
            )[0]
        
        radiologists.append({
            **rad,
            "nodule_slices": nodule_slices,
            "mask_nodule": mask_nodule
        })

    return {
        **scan,
        "ct_volume": ct_volume,
        "radiologists": radiologists
    }
