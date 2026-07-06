import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def load_scan(mhd_path: Path):
    # Load CT scan
    image = sitk.ReadImage(str(mhd_path))

    # Convert to numpy array
    volume = sitk.GetArrayFromImage(image).astype(np.int16)

    # Extract voxel spacing
    spacing = np.array(image.GetSpacing(), dtype=np.float32)

    return volume, spacing


def save_volume(volume, volume_dir, filename):
    # Save CT volume
    np.save(volume_dir / filename, volume)


def save_spacing(spacing: np.ndarray, spacing_dir, filename,):
    # Save voxel spacing
    np.save(spacing_dir / filename, spacing)


def process_scan(mhd_path, volume_dir, spacing_dir):
    # Load scan data
    volume, spacing = load_scan(mhd_path)

    # Generate output filename
    filename = mhd_path.stem + ".npy"

    # Save volume and spacing
    save_volume(volume, volume_dir, filename)
    save_spacing(spacing, spacing_dir, filename)


def process_all_scans(lndb_dir, output_dir,):
    lndb_dir = Path(lndb_dir)
    output_dir = Path(output_dir)

    volume_dir = output_dir / "volume"
    spacing_dir = output_dir / "spacing"

    # Create output directories
    volume_dir.mkdir(parents=True, exist_ok=True)
    spacing_dir.mkdir(parents=True, exist_ok=True)

    # Find all LNDb scans
    mhd_files = sorted(lndb_dir.glob("*.mhd"))

    # Print total number of scans
    print(f"Total scans: {len(mhd_files)}")

    # Process each scan
    for mhd_path in tqdm(mhd_files, desc="Converting LNDb", unit="scan"):
        try:
            process_scan(
                mhd_path,
                volume_dir,
                spacing_dir,
            )
        except Exception as e:
            print(
                f"\nFailed processing "
                f"{mhd_path.name}: {e}"
            )


if __name__ == "__main__":
    process_all_scans(
        lndb_dir="../../dataset/lndb/data",
        output_dir="../../dataset/npy_files/lndb_npy",
    )