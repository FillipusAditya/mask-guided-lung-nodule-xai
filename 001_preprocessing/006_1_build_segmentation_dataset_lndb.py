from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing CT volumes (.npy).
CT_VOLUME_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_lndb"
    / "001_volume_npy"
)

# Directory containing consensus masks.
CONSENSUS_MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_lndb"
    / "007_consensus_nodules_npy"
)

# Output directory.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_lndb"
    / "009_segmentation_dataset"
)

# Output directories for CT slices and masks.
CT_OUTPUT_DIR = OUTPUT_DIR / "ct"
MASK_OUTPUT_DIR = OUTPUT_DIR / "mask"

# Metadata file.
METADATA_CSV = OUTPUT_DIR / "metadata.csv"


def load_volume(volume_path):
    """
    Load a CT volume stored as a NumPy array.

    Parameters
    ----------
    volume_path : Path
        Path to the CT volume (.npy).

    Returns
    -------
    numpy.ndarray
        CT volume with shape (num_slices, height, width).
    """
    return np.load(volume_path)


def extract_slice_index(mask_path):
    """
    Extract the slice index from a mask filename.

    Expected filename format
    ------------------------
    slice_257.npy

    Parameters
    ----------
    mask_path : Path
        Path to the mask file.

    Returns
    -------
    int
        Slice index.
    """
    return int(mask_path.stem.split("_")[1])


def save_ct_slice(ct_slice, output_path):
    """
    Save a CT slice.

    Parameters
    ----------
    ct_slice : numpy.ndarray
        CT slice.

    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        ct_slice,
    )


def save_mask(mask, output_path):
    """
    Save a consensus mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Consensus mask.

    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        mask,
    )


def process_scan(scan_dir):
    """
    Generate segmentation samples for a single CT scan.

    Parameters
    ----------
    scan_dir : Path
        Directory containing all findings for one CT scan.

    Returns
    -------
    list
        Metadata records generated for every saved sample.
    """

    metadata_rows = []

    patient_name = scan_dir.name

    volume_path = (
        CT_VOLUME_DIR
        / f"{patient_name}.npy"
    )

    if not volume_path.exists():
        raise FileNotFoundError(
            f"CT volume not found: {volume_path}"
        )

    volume = load_volume(volume_path)

    finding_dirs = sorted(
        [
            path
            for path in scan_dir.iterdir()
            if path.is_dir()
        ]
    )

    for finding_dir in finding_dirs:

        finding_name = finding_dir.name

        mask_paths = sorted(
            finding_dir.glob("slice_*.npy")
        )

        for mask_path in mask_paths:

            slice_index = extract_slice_index(mask_path)

            if slice_index >= volume.shape[0]:
                raise IndexError(
                    f"{patient_name} slice "
                    f"{slice_index} exceeds "
                    f"volume depth "
                    f"{volume.shape[0]}"
                )

            ct_slice = volume[slice_index]

            mask = np.load(mask_path)

            filename = (
                f"{patient_name}_"
                f"{finding_name}_"
                f"slice_{slice_index}.npy"
            )

            ct_output_path = (
                CT_OUTPUT_DIR
                / filename
            )

            mask_output_path = (
                MASK_OUTPUT_DIR
                / filename
            )

            save_ct_slice(
                ct_slice=ct_slice,
                output_path=ct_output_path,
            )

            save_mask(
                mask=mask,
                output_path=mask_output_path,
            )

            metadata_rows.append(
                {
                    "filename": filename,
                    "patient_id": patient_name,
                    "finding_id": int(
                        finding_name.split("_")[1]
                    ),
                    "slice_index": slice_index,
                    "ct_path": str(
                        ct_output_path.relative_to(
                            OUTPUT_DIR
                        )
                    ),
                    "mask_path": str(
                        mask_output_path.relative_to(
                            OUTPUT_DIR
                        )
                    ),
                    "image_height": ct_slice.shape[0],
                    "image_width": ct_slice.shape[1],
                    "mask_height": mask.shape[0],
                    "mask_width": mask.shape[1],
                    "mask_pixels": int(mask.sum()),
                }
            )

    return metadata_rows

def main():
    """
    Generate a segmentation dataset from LNDb consensus masks.

    For every consensus mask slice, the corresponding CT slice is extracted
    from the CT volume and saved using the same filename. A metadata CSV
    containing information about every generated sample is also created.
    """

    CT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MASK_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan_dirs = sorted(
        [
            path
            for path in CONSENSUS_MASK_DIR.iterdir()
            if path.is_dir()
        ]
    )

    print(f"Total scans : {len(scan_dirs)}")

    metadata_rows = []

    success_count = 0
    failed_count = 0

    for scan_dir in tqdm(
        scan_dirs,
        desc="Processing scans",
    ):
        try:
            rows = process_scan(scan_dir)

            metadata_rows.extend(rows)

            success_count += 1

        except Exception as e:
            failed_count += 1

            print()
            print(f"Failed: {scan_dir.name}")
            print(e)

    metadata_df = pd.DataFrame(metadata_rows)

    metadata_df = metadata_df.sort_values(
        by=[
            "patient_id",
            "finding_id",
            "slice_index",
        ]
    ).reset_index(
        drop=True,
    )

    metadata_df.to_csv(
        METADATA_CSV,
        index=False,
    )

    print()
    print("Segmentation dataset generation completed")
    print()

    print(f"Processed scans : {len(scan_dirs)}")
    print(f"Successful      : {success_count}")
    print(f"Failed          : {failed_count}")
    print(f"Generated pairs : {len(metadata_df)}")
    print()

    print(f"CT directory    : {CT_OUTPUT_DIR}")
    print(f"Mask directory  : {MASK_OUTPUT_DIR}")
    print(f"Metadata CSV    : {METADATA_CSV}")


if __name__ == "__main__":
    main()
