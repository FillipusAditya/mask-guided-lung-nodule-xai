"""Generate optional segmented PNG and QC artifacts for LIDC-IDRI scans."""

from pathlib import Path

import numpy as np
import pandas as pd
import pylidc as pl
from tqdm import tqdm

from config import (
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    MEDIAN_FILTER_SIZE,
    QUALITY_CONTROL_DIR,
    SEGMENTED_NODULE_PNG_DIR,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)
from generate_consensus_masks import scan_key, validate_input_metadata
from lidc_consensus.artifacts import (
    prepare_cluster_artifact,
    preprocess_ct_for_display,
    save_consensus_quality_control,
    save_segmented_nodule_png,
)
from lidc_consensus.identifiers import scan_directory_name


def generate_consensus_artifacts(
    metadata_csv: str | Path = INPUT_METADATA_CSV,
    segmented_png_dir: str | Path = SEGMENTED_NODULE_PNG_DIR,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    patient_id: str | None = None,
    segmented_png: bool = False,
    quality_control: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    """Generate selected artifacts and group every cluster by scan directory."""
    if not segmented_png and not quality_control:
        return {"clusters": 0, "segmented_png": 0, "quality_control": 0}

    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Input metadata was not found: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    validate_input_metadata(metadata)
    if patient_id is not None:
        metadata = metadata.loc[metadata["patient_id"] == patient_id].copy()
        if metadata.empty:
            raise ValueError(f"No LIDC clusters found for {patient_id}.")

    query = pl.query(pl.Scan)
    if patient_id is not None:
        query = query.filter(pl.Scan.patient_id == patient_id)
    scans = query.all()
    scans_by_key = {scan_key(scan): scan for scan in scans}
    groups = list(
        metadata.groupby(
            ["patient_id", "study_instance_uid", "series_instance_uid"],
            sort=True,
        )
    )
    counts = {"clusters": 0, "segmented_png": 0, "quality_control": 0}

    for raw_key, rows in tqdm(
        groups,
        desc="Generating LIDC consensus artifacts",
        unit="scan",
        dynamic_ncols=True,
    ):
        key = tuple(str(value) for value in raw_key)
        if key not in scans_by_key:
            raise RuntimeError(f"Could not match LIDC metadata scan: {key}")
        scan = scans_by_key[key]
        scan_name = scan_directory_name(scan)
        ct_volume = np.transpose(scan.to_volume(), (2, 0, 1)).astype(np.int16)
        normalized_ct = preprocess_ct_for_display(
            ct_volume,
            window_level=WINDOW_LEVEL,
            window_width=WINDOW_WIDTH,
            median_filter_size=MEDIAN_FILTER_SIZE,
        )
        annotation_clusters = scan.cluster_annotations()

        for cluster_id in sorted(rows["cluster_id"].astype(int).unique()):
            if not 0 <= cluster_id < len(annotation_clusters):
                raise IndexError(
                    f"{scan.patient_id} cluster {cluster_id} is unavailable."
                )
            artifact = prepare_cluster_artifact(
                annotation_clusters[cluster_id],
                normalized_ct,
                consensus_level=consensus_level,
            )
            title = f"{scan.patient_id} | Cluster {cluster_id}"

            if segmented_png:
                result = save_segmented_nodule_png(
                    artifact,
                    Path(segmented_png_dir) / scan_name / f"cluster_{cluster_id}",
                    overwrite=overwrite,
                )
                counts["segmented_png"] += result.status == "written"

            if quality_control:
                result = save_consensus_quality_control(
                    artifact,
                    title,
                    Path(quality_control_dir) / scan_name,
                    cluster_id,
                    overwrite=overwrite,
                )
                counts["quality_control"] += result.status == "written"

            counts["clusters"] += 1

    return counts
