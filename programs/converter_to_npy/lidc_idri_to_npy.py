import os
import numpy as np
import pylidc as pl
from pathlib import Path
from tqdm import tqdm


class LIDCConverter:

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_filename(self, scan) -> str:
        # Get identifiers
        patient_id = scan.patient_id
        study_uid = scan.study_instance_uid[-5:]
        series_uid = scan.series_instance_uid[-5:]

        # Create filename
        filename = (
            f'lidc_{patient_id}_{study_uid}_{series_uid}.npy'
        )

        return filename

    def load_hu_volume(self, scan) -> np.ndarray:
        # Load DICOM images
        images = scan.load_all_dicom_images()
        
        # Stack slices
        volumes = np.stack(
            [image.pixel_array for image in images]
        ).astype(np.int16)

        # HU conversion
        slope = float(images[0].RescaleSlope)
        intercept = float(images[0].RescaleIntercept)
        hu_volumes = volumes * slope + intercept
        
        return hu_volumes.astype(np.int16)

    def save_volume(self, volume: np.ndarray, filename: str):
        output_path = self.output_dir / filename
        np.save(output_path, volume)

    def process_scan(self, scan):
        # Generate filename
        filename = self.generate_filename(scan)

        # Load and convert to HU
        hu_volume = self.load_hu_volume(scan)

        # Save as .npy
        self.save_volume(hu_volume, filename)

    def process_all_scans(self):
        # Query all scans from the LIDC-IDRI dataset
        scans = pl.query(pl.Scan).all()

        # Print total number of scans
        print(f'Total scans: {len(scans)}')

        # Process each scan
        for scan in tqdm(scans, desc='Converting LIDC-IDRI', unit='scan'):
            try:
                self.process_scan(scan)
            except Exception as e:
                print(
                    f'\nFailed processing '
                    f'{scan.patient_id}: {e}'
                )


if __name__ == '__main__':
    converter = LIDCConverter(
        output_dir='../dataset/npy_files/'
    )

    converter.process_all_scans()