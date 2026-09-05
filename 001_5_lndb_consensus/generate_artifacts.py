"""Generate optional segmented PNG and QC artifacts for LNDb scans."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    LNDB_DATA_DIR,
    LNDB_MASK_DIR,
    MEDIAN_FILTER_SIZE,
    QUALITY_CONTROL_DIR,
    SEGMENTED_NODULE_PNG_DIR,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)
from generate_consensus_metadata import require_input_path, validate_input_metadata
from lndb_consensus import prepare_scan_data, process_scan
from lndb_consensus.artifacts import (
    preprocess_ct_for_display,
    save_consensus_quality_control,
    save_segmented_nodule_png,
)


def generate_consensus_artifacts(
    input_csv: str | Path = INPUT_METADATA_CSV,
    data_dir: str | Path = LNDB_DATA_DIR,
    mask_dir: str | Path = LNDB_MASK_DIR,
    segmented_png_dir: str | Path = SEGMENTED_NODULE_PNG_DIR,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    scan_id: int | None = None,
    segmented_png: bool = False,
    quality_control: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """Generate selected artifacts and group every finding by scan directory."""
    if not segmented_png and not quality_control:
        return {"findings": 0, "segmented_png": 0, "quality_control": 0}

    input_csv = require_input_path(input_csv, "Input metadata CSV")
    data_dir = require_input_path(data_dir, "LNDb CT directory")
    mask_dir = require_input_path(mask_dir, "LNDb mask directory")
    metadata = pd.read_csv(input_csv)
    validate_input_metadata(metadata)

    if scan_id is not None:
        ids = pd.to_numeric(metadata["lndbid"], errors="raise").astype(int)
        metadata = metadata.loc[ids == int(scan_id)].copy()
        if metadata.empty:
            raise ValueError(f"No LNDb findings found for scan {scan_id}.")

    metadata = metadata.sort_values(["lndbid", "findingid"])
    normalized_by_scan = {}
    counts = {"findings": 0, "segmented_png": 0, "quality_control": 0}

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Generating LNDb consensus artifacts",
        unit="finding",
        dynamic_ncols=True,
    ):
        scan = prepare_scan_data(row=row, data_dir=data_dir, mask_dir=mask_dir)
        scan = process_scan(scan, clevel=consensus_level)
        lndb_id = int(scan["lndb_id"])
        finding_id = int(scan["finding_id"])
        scan_name = f"LNDb-{lndb_id:04d}"

        if lndb_id not in normalized_by_scan:
            normalized_by_scan[lndb_id] = preprocess_ct_for_display(
                scan["ct_volume"],
                window_level=WINDOW_LEVEL,
                window_width=WINDOW_WIDTH,
                median_filter_size=MEDIAN_FILTER_SIZE,
            )
        normalized_ct = normalized_by_scan[lndb_id]

        if segmented_png:
            result = save_segmented_nodule_png(
                scan,
                normalized_ct,
                Path(segmented_png_dir) / scan_name / f"finding_{finding_id}",
                overwrite=overwrite,
            )
            counts["segmented_png"] += result.status == "written"

        if quality_control:
            result = save_consensus_quality_control(
                scan,
                normalized_ct,
                Path(quality_control_dir) / scan_name,
                overwrite=overwrite,
            )
            counts["quality_control"] += result.status == "written"

        counts["findings"] += 1

    return counts
