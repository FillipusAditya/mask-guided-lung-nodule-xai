"""Generate per-slice LIDC-IDRI consensus masks and path metadata."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pylidc as pl
from tqdm import tqdm

from config import (
    CONSENSUS_MASK_DIR,
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    OUTPUT_METADATA_CSV,
    PROJECT_ROOT,
)
from lidc_consensus import (
    cluster_uid,
    compute_consensus_slices,
    enable_pylidc_numpy_compatibility,
    save_consensus_slices,
    scan_directory_name,
)


REQUIRED_COLUMNS = {
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "cluster_id",
    "cluster_uid",
}


def validate_input_metadata(metadata: pd.DataFrame) -> None:
    """Validate the fields used to select scans and annotation clusters."""
    missing = sorted(REQUIRED_COLUMNS.difference(metadata.columns))
    if missing:
        raise ValueError(
            "Input metadata is missing required columns: " + ", ".join(missing)
        )
    if metadata.empty:
        raise ValueError("Input metadata contains no annotation clusters.")
    if metadata["cluster_uid"].duplicated().any():
        duplicates = metadata.loc[
            metadata["cluster_uid"].duplicated(), "cluster_uid"
        ].unique()
        raise ValueError(f"Duplicate cluster_uid values: {duplicates.tolist()}")

    expected_uids = metadata.apply(
        lambda row: cluster_uid(str(row["patient_id"]), int(row["cluster_id"])),
        axis=1,
    )
    mismatched = metadata.loc[
        metadata["cluster_uid"].astype(str) != expected_uids,
        "cluster_uid",
    ]
    if not mismatched.empty:
        raise ValueError(
            "cluster_uid does not match patient_id and cluster_id: "
            f"{mismatched.iloc[0]}"
        )


def metadata_path_value(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def scan_key(scan: Any) -> tuple[str, str, str]:
    """Return the patient, study, and series identifiers for a scan."""
    return (
        str(scan.patient_id),
        str(scan.study_instance_uid),
        str(scan.series_instance_uid),
    )


def generate_consensus_masks(
    metadata_csv: str | Path = INPUT_METADATA_CSV,
    output_dir: str | Path = CONSENSUS_MASK_DIR,
    output_metadata_csv: str | Path = OUTPUT_METADATA_CSV,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    overwrite: bool = False,
    scans: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Generate selected LIDC-IDRI masks and write portable path metadata.

    Parameters
    ----------
    metadata_csv
        Cleaned cluster metadata used to select annotation clusters.
    output_dir
        Root directory for full-size two-dimensional mask slices.
    output_metadata_csv
        Copy of the input metadata with ``consensus_mask_path`` added.
    consensus_level
        Fraction of annotations required for a consensus voxel.
    overwrite
        Replace existing metadata and mask files when true.
    scans
        Optional iterable of pylidc-compatible scan objects, primarily useful
        for testing. By default all scans in the configured pylidc database
        are queried.
    """
    metadata_csv = Path(metadata_csv)
    output_dir = Path(output_dir)
    output_metadata_csv = Path(output_metadata_csv)

    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Input metadata was not found: {metadata_csv}")
    if output_metadata_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output metadata already exists: {output_metadata_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    metadata = pd.read_csv(metadata_csv)
    validate_input_metadata(metadata)
    output_metadata = metadata.copy()
    output_metadata["consensus_mask_path"] = None

    enable_pylidc_numpy_compatibility()
    available_scans = list(scans) if scans is not None else pl.query(pl.Scan).all()
    if not available_scans:
        raise RuntimeError("No scans were found in the configured pylidc database.")

    scans_by_key: dict[tuple[str, str, str], Any] = {}
    for scan in available_scans:
        key = scan_key(scan)
        if key in scans_by_key:
            raise ValueError(f"Duplicate pylidc scan identifiers: {key}")
        scans_by_key[key] = scan

    metadata_groups = list(
        metadata.groupby(
            ["patient_id", "study_instance_uid", "series_instance_uid"],
            sort=True,
        )
    )
    metadata_keys = {
        tuple(str(value) for value in key)
        for key, _ in metadata_groups
    }
    missing_scans = sorted(metadata_keys.difference(scans_by_key))
    if missing_scans:
        preview = ", ".join(key[0] for key in missing_scans[:10])
        suffix = " ..." if len(missing_scans) > 10 else ""
        raise RuntimeError(
            f"Could not match {len(missing_scans)} metadata scans in pylidc: "
            f"{preview}{suffix}"
        )

    missing_dicom: list[str] = []
    for key in tqdm(
        sorted(metadata_keys),
        desc="Checking LIDC DICOM",
        unit="scan",
        dynamic_ncols=True,
    ):
        scan = scans_by_key[key]
        try:
            scan.get_path_to_dicom_files()
        except RuntimeError:
            missing_dicom.append(str(scan.patient_id))
    if missing_dicom:
        preview = ", ".join(missing_dicom[:10])
        suffix = " ..." if len(missing_dicom) > 10 else ""
        raise RuntimeError(
            f"DICOM files are unavailable for {len(missing_dicom)} selected "
            f"scans: {preview}{suffix}"
        )

    for raw_key, selected_rows in tqdm(
        metadata_groups,
        desc="Generating LIDC consensus masks",
        unit="scan",
        dynamic_ncols=True,
    ):
        key = tuple(str(value) for value in raw_key)
        scan = scans_by_key[key]

        annotation_clusters = scan.cluster_annotations()
        selected_indices = set(selected_rows["cluster_id"].astype(int))
        unavailable = sorted(
            index for index in selected_indices if index >= len(annotation_clusters)
        )
        if unavailable:
            raise IndexError(
                f"{scan.patient_id} metadata references unavailable clusters: "
                f"{unavailable}."
            )

        image_shape = tuple(int(value) for value in scan.to_volume().shape[:2])
        scan_output_dir = output_dir / scan_directory_name(scan)

        for cluster_index in sorted(selected_indices):
            uid = cluster_uid(str(scan.patient_id), cluster_index)
            masks = compute_consensus_slices(
                annotation_cluster=annotation_clusters[cluster_index],
                image_shape=image_shape,
                consensus_level=consensus_level,
            )
            cluster_output_dir = scan_output_dir / f"cluster_{cluster_index}"
            save_consensus_slices(
                masks=masks,
                output_dir=cluster_output_dir,
                overwrite=overwrite,
            )
            output_metadata.loc[
                output_metadata["cluster_uid"].astype(str) == uid,
                "consensus_mask_path",
            ] = metadata_path_value(cluster_output_dir)

    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.to_csv(output_metadata_csv, index=False)
    return output_metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=INPUT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=CONSENSUS_MASK_DIR)
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=OUTPUT_METADATA_CSV,
    )
    parser.add_argument(
        "--consensus-level",
        type=float,
        default=DEFAULT_CONSENSUS_LEVEL,
        help="Required annotation agreement fraction in the interval (0, 1].",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Generate masks using default or user-supplied paths."""
    args = build_parser().parse_args()
    output = generate_consensus_masks(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_metadata_csv,
        consensus_level=args.consensus_level,
        overwrite=args.overwrite,
    )
    print(
        f"LIDC consensus | Clusters: {len(output):,} | "
        f"Masks: {args.output_dir} | Metadata: {args.output_metadata_csv}"
    )


if __name__ == "__main__":
    main()
