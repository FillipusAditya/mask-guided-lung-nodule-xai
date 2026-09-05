"""Create final 3D lung-parenchyma volumes for LIDC-IDRI and LNDb."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import (
    LIDC_OUTPUT_DIR,
    LNDB_INPUT_DIR,
    LNDB_OUTPUT_DIR,
    QUALITY_CONTROL_DIR,
)
from step_1_ct_to_numpy import (
    get_lidc_scans,
    lidc_filename,
    load_lidc_scan,
    load_lndb_scan,
)
from step_2_segmentation import segment_volume
from step_3_erosion import erode_volume
from step_4_median_filter import filter_volume
from step_5_normalize_and_mask import normalize_and_mask


def create_lung_parenchyma(volume: np.ndarray) -> np.ndarray:
    """Run all processing stages on one CT volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")

    # The mask branch uses the original Hounsfield Unit values.
    lung_mask = segment_volume(volume)
    lung_mask = erode_volume(lung_mask)

    # The image branch is filtered before windowing and normalization.
    filtered_volume = filter_volume(volume)
    return normalize_and_mask(filtered_volume, lung_mask)


def save_lung_parenchyma(
    volume: np.ndarray,
    output_path: str | Path,
) -> None:
    """Process and save one final float32 lung-parenchyma volume."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = create_lung_parenchyma(volume)
    np.save(output_path, result.astype(np.float32, copy=False))


def save_lidc_quality_control(
    scan,
    parenchyma_path: str | Path,
    output_path: str | Path,
) -> None:
    """Create one LIDC-IDRI nodule-overlay quality-control canvas."""
    from quality_control import (  # Imported only when QC is requested.
        prepare_lidc_quality_control,
        show_nodule_overlays,
    )

    prepared = prepare_lidc_quality_control(
        patient_id=str(scan.patient_id),
        parenchyma_file=parenchyma_path,
        study_instance_uid=str(scan.study_instance_uid),
        series_instance_uid=str(scan.series_instance_uid),
    )
    ct_volume, parenchyma, findings, title = prepared
    show_nodule_overlays(
        ct_volume=ct_volume,
        lung_parenchyma=parenchyma,
        findings=findings,
        title=title,
        output_path=output_path,
    )


def save_lndb_quality_control(
    mhd_path: str | Path,
    parenchyma_path: str | Path,
    output_path: str | Path,
) -> None:
    """Create one LNDb nodule-overlay quality-control canvas."""
    from quality_control import (  # Imported only when QC is requested.
        prepare_lndb_quality_control,
        show_nodule_overlays,
    )

    mhd_path = Path(mhd_path)
    try:
        scan_id = int(mhd_path.stem.split("-")[-1])
    except ValueError as error:
        raise ValueError(
            f"Expected an LNDb filename such as LNDb-0001.mhd: {mhd_path.name}"
        ) from error

    prepared = prepare_lndb_quality_control(
        scan_id=scan_id,
        data_dir=mhd_path.parent,
        parenchyma_file=parenchyma_path,
    )
    ct_volume, parenchyma, findings, title = prepared
    show_nodule_overlays(
        ct_volume=ct_volume,
        lung_parenchyma=parenchyma,
        findings=findings,
        title=title,
        output_path=output_path,
    )


def process_lidc_dataset(
    output_dir: str | Path = LIDC_OUTPUT_DIR,
    patient_id: str | None = None,
    overwrite: bool = False,
    quality_control: bool = False,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
) -> tuple[int, int]:
    """Process all configured LIDC scans or one selected patient."""
    output_dir = Path(output_dir)
    quality_control_dir = Path(quality_control_dir) / "lidc"
    scans = get_lidc_scans(patient_id)
    completed = 0
    qc_completed = 0
    qc_skipped = 0

    for scan in tqdm(
        scans,
        desc="LIDC-IDRI parenchyma",
        unit="scan",
        dynamic_ncols=True,
    ):
        output_path = output_dir / lidc_filename(scan)

        should_process = overwrite or not output_path.exists()
        if should_process:
            save_lung_parenchyma(load_lidc_scan(scan), output_path)
            completed += 1

        if quality_control:
            quality_control_path = quality_control_dir / f"{output_path.stem}.png"
            if quality_control_path.exists() and not overwrite:
                qc_skipped += 1
            else:
                save_lidc_quality_control(
                    scan=scan,
                    parenchyma_path=output_path,
                    output_path=quality_control_path,
                )
                qc_completed += 1

    if quality_control:
        print(
            "LIDC-IDRI QC | "
            f"Written: {qc_completed} | Skipped: {qc_skipped}"
        )

    return completed, len(scans)


def process_lndb_dataset(
    input_path: str | Path = LNDB_INPUT_DIR,
    output_dir: str | Path = LNDB_OUTPUT_DIR,
    overwrite: bool = False,
    quality_control: bool = False,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
) -> tuple[int, int]:
    """Process one LNDb MHD file or every MHD file in a directory."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    quality_control_dir = Path(quality_control_dir) / "lndb"

    if input_path.is_file() and input_path.suffix.lower() == ".mhd":
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("*.mhd"))
    else:
        raise FileNotFoundError(f"LNDb input was not found: {input_path}")

    if not files:
        raise FileNotFoundError(f"No .mhd files found in {input_path}.")

    completed = 0
    qc_completed = 0
    qc_skipped = 0

    for mhd_path in tqdm(
        files,
        desc="LNDb parenchyma",
        unit="scan",
        dynamic_ncols=True,
    ):
        output_path = output_dir / f"{mhd_path.stem}.npy"

        should_process = overwrite or not output_path.exists()
        if should_process:
            save_lung_parenchyma(load_lndb_scan(mhd_path), output_path)
            completed += 1

        if quality_control:
            quality_control_path = quality_control_dir / f"{mhd_path.stem}.png"
            if quality_control_path.exists() and not overwrite:
                qc_skipped += 1
            else:
                save_lndb_quality_control(
                    mhd_path=mhd_path,
                    parenchyma_path=output_path,
                    output_path=quality_control_path,
                )
                qc_completed += 1

    if quality_control:
        print(f"LNDb QC | Written: {qc_completed} | Skipped: {qc_skipped}")

    return completed, len(files)


def main() -> None:
    """Run the complete pipeline for LIDC-IDRI, LNDb, or both datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("lidc", "lndb", "all"))
    parser.add_argument(
        "--patient-id",
        help="Process only this LIDC patient, for example LIDC-IDRI-0001.",
    )
    parser.add_argument(
        "--lndb-input",
        type=Path,
        default=LNDB_INPUT_DIR,
        help="One LNDb .mhd file or a directory containing .mhd files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quality-control",
        action="store_true",
        help="Create a consensus-nodule overlay PNG for every selected scan.",
    )
    parser.add_argument(
        "--quality-control-dir",
        type=Path,
        default=QUALITY_CONTROL_DIR,
        help="Root directory for LIDC-IDRI and LNDb quality-control images.",
    )
    args = parser.parse_args()

    if args.dataset in ("lidc", "all"):
        completed, total = process_lidc_dataset(
            patient_id=args.patient_id,
            overwrite=args.overwrite,
            quality_control=args.quality_control,
            quality_control_dir=args.quality_control_dir,
        )
        print(
            "LIDC-IDRI | "
            f"Total: {total} | Written: {completed} | "
            f"Skipped: {total - completed}"
        )

    if args.dataset in ("lndb", "all"):
        completed, total = process_lndb_dataset(
            input_path=args.lndb_input,
            overwrite=args.overwrite,
            quality_control=args.quality_control,
            quality_control_dir=args.quality_control_dir,
        )
        print(
            f"LNDb | Total: {total} | Written: {completed} | "
            f"Skipped: {total - completed}"
        )


if __name__ == "__main__":
    main()
