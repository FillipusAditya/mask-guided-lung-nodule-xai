from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mapping from CT output directory names to source volume directories.
# Add another entry here to generate an additional CT representation.
CT_VOLUME_DIRS = {
    "ct_windowed": (
        PROJECT_ROOT
        / "000_dataset"
        / "_lndb"
        / "002_windowed_npy"
    ),
    "ct_parenchyma": (
        PROJECT_ROOT
        / "000_dataset"
        / "_lndb"
        / "004_volume_parencyma_npy"
    ),
}

# Directory containing consensus masks.
CONSENSUS_MASK_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "005_mask_consensus_npy"
)

# Output directory.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "007_segmentation_dataset_npy"
)

# Output directories for CT slices and masks.
CT_OUTPUT_DIRS = {
    output_name: OUTPUT_DIR / output_name
    for output_name in CT_VOLUME_DIRS
}
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

    volumes = {}

    for output_name, volume_dir in CT_VOLUME_DIRS.items():
        volume_path = volume_dir / f"{patient_name}.npy"

        if not volume_path.exists():
            raise FileNotFoundError(
                f"CT volume for {output_name} not found: {volume_path}"
            )

        volumes[output_name] = load_volume(volume_path)

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

            ct_slices = {}

            for output_name, volume in volumes.items():
                if not 0 <= slice_index < volume.shape[0]:
                    raise IndexError(
                        f"{patient_name} slice {slice_index} is outside "
                        f"the {output_name} volume depth {volume.shape[0]}"
                    )

                ct_slices[output_name] = volume[slice_index]

            mask = np.load(mask_path)

            filename = (
                f"{patient_name}_"
                f"{finding_name}_"
                f"slice_{slice_index}.npy"
            )

            ct_output_paths = {
                output_name: CT_OUTPUT_DIRS[output_name] / filename
                for output_name in CT_VOLUME_DIRS
            }

            mask_output_path = (
                MASK_OUTPUT_DIR
                / filename
            )

            for output_name, ct_slice in ct_slices.items():
                save_ct_slice(
                    ct_slice=ct_slice,
                    output_path=ct_output_paths[output_name],
                )

            save_mask(
                mask=mask,
                output_path=mask_output_path,
            )

            metadata_row = {
                "filename": filename,
                "patient_id": patient_name,
                "finding_id": int(
                    finding_name.split("_")[1]
                ),
                "slice_index": slice_index,
            }

            for output_name, ct_slice in ct_slices.items():
                metadata_row[f"{output_name}_path"] = str(
                    ct_output_paths[output_name].relative_to(
                        OUTPUT_DIR
                    )
                )
                metadata_row[f"{output_name}_height"] = ct_slice.shape[0]
                metadata_row[f"{output_name}_width"] = ct_slice.shape[1]

            metadata_row.update(
                {
                    "mask_path": str(
                        mask_output_path.relative_to(
                            OUTPUT_DIR
                        )
                    ),
                    "mask_height": mask.shape[0],
                    "mask_width": mask.shape[1],
                    "mask_pixels": int(mask.sum()),
                }
            )

            metadata_rows.append(metadata_row)

    return metadata_rows

def main():
    """
    Generate a segmentation dataset from LNDb consensus masks.

    For every consensus mask slice, the corresponding CT slice is extracted
    from the CT volume and saved using the same filename. A metadata CSV
    containing information about every generated sample is also created.
    """

    for ct_output_dir in CT_OUTPUT_DIRS.values():
        ct_output_dir.mkdir(
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

    for output_name, ct_output_dir in CT_OUTPUT_DIRS.items():
        print(f"{output_name:<15} : {ct_output_dir}")
    print(f"Mask directory  : {MASK_OUTPUT_DIR}")
    print(f"Metadata CSV    : {METADATA_CSV}")


if __name__ == "__main__":
    main()
