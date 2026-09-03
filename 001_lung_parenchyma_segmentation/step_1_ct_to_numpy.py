"""Step 1: convert LIDC-IDRI DICOM or LNDb MHD scans to NumPy."""

import argparse
from pathlib import Path

import numpy as np
import pylidc as pl
import SimpleITK as sitk


def load_lidc_scan(scan: pl.Scan) -> np.ndarray:
    """Load one pylidc scan as an int16 volume with shape (N, H, W)."""
    volume = scan.to_volume()

    # pylidc returns (H, W, N); the project convention is (N, H, W).
    return np.transpose(volume, (2, 0, 1)).astype(np.int16)


def load_lndb_scan(input_path: str | Path) -> np.ndarray:
    """Load one LNDb MHD scan as an int16 volume with shape (N, H, W)."""
    image = sitk.ReadImage(str(input_path))
    return sitk.GetArrayFromImage(image).astype(np.int16)


def save_volume(volume: np.ndarray, output_path: str | Path) -> None:
    """Save one 3D NumPy volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, volume)


def get_lidc_scans(patient_id: str | None = None) -> list[pl.Scan]:
    """Return all configured LIDC scans or only scans for one patient."""
    query = pl.query(pl.Scan)

    if patient_id is not None:
        query = query.filter(pl.Scan.patient_id == patient_id)

    scans = query.all()
    if not scans:
        raise FileNotFoundError("No matching LIDC-IDRI scans were found.")

    return scans


def lidc_filename(scan: pl.Scan) -> str:
    """Create a unique and stable output filename for one LIDC scan."""
    study_uid = scan.study_instance_uid[-5:]
    series_uid = scan.series_instance_uid[-5:]
    return f"{scan.patient_id}_{study_uid}_{series_uid}.npy"


def convert_lidc_scan(scan: pl.Scan, output_path: str | Path) -> None:
    """Convert and save one LIDC scan."""
    save_volume(load_lidc_scan(scan), output_path)


def convert_lidc_batch(
    output_dir: str | Path,
    patient_id: str | None = None,
) -> None:
    """Convert all matching LIDC scans."""
    output_dir = Path(output_dir)
    scans = get_lidc_scans(patient_id)

    for index, scan in enumerate(scans, start=1):
        print(f"[{index}/{len(scans)}] {scan.patient_id}")
        convert_lidc_scan(scan, output_dir / lidc_filename(scan))


def convert_lndb_file(input_path: str | Path, output_path: str | Path) -> None:
    """Convert and save one LNDb MHD scan."""
    save_volume(load_lndb_scan(input_path), output_path)


def convert_lndb_batch(input_dir: str | Path, output_dir: str | Path) -> None:
    """Convert every LNDb MHD scan in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.rglob("*.mhd"))

    if not files:
        raise FileNotFoundError(f"No .mhd files found in {input_dir}.")

    for index, input_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {input_path.name}")
        convert_lndb_file(input_path, output_dir / f"{input_path.stem}.npy")


def main() -> None:
    """Run Step 1 for one scan or a directory of scans."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lidc_one = subparsers.add_parser("lidc-one")
    lidc_one.add_argument("patient_id")
    lidc_one.add_argument("output_file", type=Path)

    lidc_batch = subparsers.add_parser("lidc-batch")
    lidc_batch.add_argument("output_dir", type=Path)
    lidc_batch.add_argument("--patient-id")

    lndb_one = subparsers.add_parser("lndb-one")
    lndb_one.add_argument("input_file", type=Path)
    lndb_one.add_argument("output_file", type=Path)

    lndb_batch = subparsers.add_parser("lndb-batch")
    lndb_batch.add_argument("input_dir", type=Path)
    lndb_batch.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    if args.command == "lidc-one":
        scan = get_lidc_scans(args.patient_id)[0]
        convert_lidc_scan(scan, args.output_file)
    elif args.command == "lidc-batch":
        convert_lidc_batch(args.output_dir, args.patient_id)
    elif args.command == "lndb-one":
        convert_lndb_file(args.input_file, args.output_file)
    else:
        convert_lndb_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
