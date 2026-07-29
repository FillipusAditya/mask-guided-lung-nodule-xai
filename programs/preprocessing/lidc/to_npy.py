from pathlib import Path

import numpy as np
import pylidc as pl
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directory where converted CT volumes will be stored.
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "_lidc" / "001_volume_npy"


def convert_dataset(output_dir: str | Path) -> None:
    """
    Convert all LIDC-IDRI CT scans into NumPy (.npy) volumes.
    """

    # Create the output directory if it does not already exist.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve all CT scans from the LIDC database.
    scans = pl.query(pl.Scan).all()

    print(f"Total scans : {len(scans)}")

    failed_patients = []

    # Convert each scan into a NumPy volume.
    for scan in tqdm(
        scans,
        desc="Converting LIDC-IDRI",
        unit="scan",
    ):
        try:
            # Generate a unique filename for the current scan.
            patient_id = scan.patient_id
            study_uid = scan.study_instance_uid[-5:]
            series_uid = scan.series_instance_uid[-5:]

            # LIDC-IDRI-0001_30178_03192.npy
            filename = (
                f"{patient_id}_{study_uid}_{series_uid}.npy"
            )

            # Load the CT volume in Hounsfield Units.
            volume = scan.to_volume().astype(np.int16)

            # Save the volume as a NumPy file.
            np.save(output_dir / filename, volume)

        except Exception as error:
            failed_patients.append(scan.patient_id)

            print(
                f"\nFailed processing "
                f"{scan.patient_id}: {error}"
            )

    successful = len(scans) - len(failed_patients)

    print("\nConversion finished")
    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed_patients)}")

    if failed_patients:
        print("\nFailed scans:")

        for patient_id in failed_patients:
            print(f" - {patient_id}")


if __name__ == "__main__":
    convert_dataset(output_dir=OUTPUT_DIR)