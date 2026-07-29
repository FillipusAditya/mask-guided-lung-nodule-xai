from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Directory containing the original LNDb .mhd files.
DATA_DIR = PROJECT_ROOT / "dataset" / "lndb" / "data"

# Directory where converted CT volumes will be stored.
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "_lndb" / "001_volume_npy"


def convert_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Convert all LNDb CT scans into NumPy (.npy) volumes.
    """

    # Convert input paths into Path objects.
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all MetaImage files.
    mhd_files = sorted(data_dir.glob("*.mhd"))

    print(f"Total scans : {len(mhd_files)}")

    failed_scans = []

    # Convert each CT scan into a NumPy volume.
    for mhd_path in tqdm(
        mhd_files,
        desc="Converting LNDb",
        unit="scan",
    ):
        try:
            # Read the CT scan.
            image = sitk.ReadImage(str(mhd_path))

            # Convert the image into a NumPy array with int16 data type.
            volume = sitk.GetArrayFromImage(image).astype(np.int16)

            # Generate the output filename.
            filename = f"{mhd_path.stem}.npy"

            # Save the CT volume.
            np.save(output_dir / filename, volume)

        except Exception as error:
            failed_scans.append(mhd_path.stem)

            print(
                f"\nFailed processing "
                f"{mhd_path.name}: {error}"
            )

    successful = len(mhd_files) - len(failed_scans)

    print("\nConversion finished")
    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed_scans)}")

    if failed_scans:
        print("\nFailed scans:")

        for scan_id in failed_scans:
            print(f" - {scan_id}")


if __name__ == "__main__":
    convert_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
    )