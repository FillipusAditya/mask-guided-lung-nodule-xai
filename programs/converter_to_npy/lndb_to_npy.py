import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

class LNDbConverter:

    def __init__(self, lndb_dir: str, output_dir: str):
        self.lndb_dir = Path(lndb_dir)
        self.output_dir = Path(output_dir)
        self.volume_dir = (
            self.output_dir / "volume"
        )
        self.spacing_dir = (
            self.output_dir / "spacing"
        )

        # Create output directories
        self.volume_dir.mkdir(parents=True, exist_ok=True)
        self.spacing_dir.mkdir(parents=True,exist_ok=True)

    def load_scan(self, mhd_path: Path):
        # Load CT scan
        image = sitk.ReadImage(str(mhd_path))

        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image).astype(np.int16)

        # Extract voxel spacing
        spacing = np.array(image.GetSpacing(), dtype=np.float32)

        return volume, spacing

    def save_volume(self, volume: np.ndarray, filename: str):
        # Save CT volume
        np.save(self.volume_dir / filename, volume)

    def save_spacing(self, spacing: np.ndarray, filename: str):
        # Save voxel spacing
        np.save(self.spacing_dir / filename, spacing)

    def process_scan(self, mhd_path: Path):
        # Load scan data
        volume, spacing = self.load_scan(mhd_path)

        # Generate output filename
        filename = (mhd_path.stem + ".npy")

        # Save volume and spacing
        self.save_volume(volume, filename)

        self.save_spacing(spacing, filename)

    def process_all_scans(self):
        # Find all LNDb scans
        mhd_files = sorted(self.lndb_dir.glob("*.mhd"))

        # Print total number of scans
        print(f"Total scans: {len(mhd_files)}")

        # Process each scan
        for mhd_path in tqdm(mhd_files, desc="Converting LNDb", unit="scan"):
            try:
                self.process_scan(mhd_path)
            except Exception as e:
                print(
                    f"\nFailed processing "
                    f"{mhd_path.name}: {e}"
                )


if __name__ == "__main__":

    converter = LNDbConverter(
        lndb_dir="../../dataset/lndb/data",
        output_dir="../../dataset/npy_files/lndb_npy"
    )

    converter.process_all_scans()