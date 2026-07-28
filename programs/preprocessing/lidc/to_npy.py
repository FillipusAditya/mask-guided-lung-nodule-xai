from pathlib import Path

import numpy as np
import pylidc as pl
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "_lidc" / "001_volume_npy"


def generate_filename(scan: pl.Scan) -> str:
    """
    Generate a unique filename for a LIDC-IDRI scan.

    Format
    ------
    lidc_<patient_id>_<study_uid>_<series_uid>.npy
    """

    patient_id = scan.patient_id
    study_uid = scan.study_instance_uid[-5:]
    series_uid = scan.series_instance_uid[-5:]

    return f"lidc_{patient_id}_{study_uid}_{series_uid}.npy"


def load_hu_volume(scan: pl.Scan) -> str:
    """
    Load a CT scan as a 3D Hounsfield Unit (HU) volume.

    Returns
    -------
    numpy.ndarray
        CT volume with shape (z, y, x)
        in Hounsfield Unit (HU).
    """

    return scan.to_volume().astype(np.int16)


def save_volume(
        volume: np.ndarray, 
        output_dir: Path, 
        filename: str,
    ) -> None:
    """
    Save a CT volume as a NumPy (.npy) file.
    """

    output_path = output_dir / filename
    np.save(output_path, volume)


def convert_scan(
        scan: pl.Scan,
        output_dir: Path,
) -> None:
    """
    Convert a single LIDC-IDRI scan into a NumPy file.
    """

    filename = generate_filename(scan)

    volume = load_hu_volume(scan)

    save_volume(
        volume=volume,
        output_dir=output_dir,
        filename=filename,
    )


def convert_dataset(
        output_dir: str | Path
) -> None:
    """
    Convert all LIDC-IDRI scans into NumPy files.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save converted .npy files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scans = pl.query(pl.Scan).all()

    print(f"Total scans : {len(scans)}")

    failed = []

    for scan in tqdm(scans, desc="Converting LIDC-IDRI", unit="scan"):
        try:
            convert_scan(scan=scan, output_dir=output_dir)
        except Exception as e:
            failed.append(scan.patient_id)
            print(
                f"\nFailed processing "
                f"{scan.patient_id}: {e}"
            )

    print("\nConversion finished")
    print(f"Successful : {len(scans) - len(failed)}")
    print(f"Failed     : {len(failed)}")

    if failed:
        print("\nFailed scans:")
        for patient in failed:
            print(f" - {patient}")


if __name__ == "__main__":
    convert_dataset(output_dir=OUTPUT_DIR)