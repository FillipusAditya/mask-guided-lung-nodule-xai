import os
from pathlib import Path

import matplotlib.pyplot as plt
import pylidc as pl

from tqdm import tqdm


class LIDCIDRISliceExporter:
    """
    Export CT scan slices as PNG images.
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.scans = pl.query(pl.Scan).all()

        Path(self.output_path).mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Num of Patient: {len(self.scans)}")

    def export(self):
        for scan in tqdm(
            self.scans,
            desc="Exporting Scans"
        ):

            try:

                patient_output_path = os.path.join(
                    self.output_path,
                    scan.patient_id
                )

                Path(patient_output_path).mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Load volume
                volume = scan.to_volume()

                total_slices = volume.shape[2]

                # Check existing PNG files
                existing_pngs = [
                    f for f in os.listdir(patient_output_path)
                    if f.endswith(".png")
                ]

                # If complete, skip patient
                if len(existing_pngs) == total_slices:

                    print(
                        f"Skipping {scan.patient_id} "
                        f"(already complete)"
                    )

                    continue

                print(
                    f"Processing {scan.patient_id} "
                    f"({len(existing_pngs)}/{total_slices})"
                )

                for slice_id in range(total_slices):

                    save_path = os.path.join(
                        patient_output_path,
                        f"{slice_id}.png"
                    )

                    # Skip existing slice
                    if os.path.exists(save_path):
                        continue

                    array_slice = volume[:, :, slice_id]

                    plt.imshow(
                        array_slice,
                        cmap='gray'
                    )

                    plt.axis('off')

                    plt.savefig(
                        save_path,
                        bbox_inches='tight',
                        pad_inches=0
                    )

                    plt.close()

            except Exception as e:

                print(
                    f"Error processing "
                    f"{scan.patient_id}: {e}"
                )

                continue


if __name__ == "__main__":

    OUTPUT_PATH = "../datasets/images/lidc_idri_png"

    exporter = LIDCIDRISliceExporter(
        output_path=OUTPUT_PATH
    )

    exporter.export()