from pathlib import Path

import numpy as np
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm


""" 
    cbbox : bounding box 3D dari consesus mask, memberi tahu cmask ada di mana 
        (slice(294, 328, None), slice(306, 350, None), slice(23, 29, None)) 
        (                    y,                     x,                   z) 
    cmask : mask hasil consesus 
        34, 44, 6) 
        ( y, x, z) 
        -> y = 328 - 294 
        -> x = 350 - 306 
        -> z = 29 - 3 
    mask : mask milik setiap annotation 
"""


# Configuration

CT_DIR = Path("../dataset/npy_files")
IMAGE_OUTPUT_DIR = Path("../dataset/segmentation/images")
MASK_OUTPUT_DIR = Path("../dataset/segmentation/masks")
CLEVEL = 0.5

IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Helper Functions
def create_empty_masks(height, width, n_slices):
    """
    Create an empty binary mask for every CT slice.

    Returns
    -------
    dict[int, np.ndarray]
        key   : slice index
        value : (H, W) binary mask
    """
    return {
        z: np.zeros((height, width), dtype=np.uint8)
        for z in range(n_slices)
    }


def merge_cluster_into_masks(mask_per_slice, cmask, cbbox):
    """
    Merge one consensus mask into the full-size mask dictionary.
    """
    y0, y1 = cbbox[0].start, cbbox[0].stop
    x0, x1 = cbbox[1].start, cbbox[1].stop
    z0 = cbbox[2].start

    for local_z in range(cmask.shape[2]):
        global_z = z0 + local_z

        mask_per_slice[global_z][y0:y1, x0:x1] |= (
            cmask[:, :, local_z].astype(np.uint8)
        )


def save_slice_dataset(ct_volume, mask_per_slice, filename, image_dir, mask_dir):
    """
    Save image-mask pair for every slice containing nodules.
    """
    # If CT shape is (512,512,N)
    for z, mask in mask_per_slice.items():
        if not mask.any():
            continue

        np.save(
            image_dir / f"{filename}_slice{z:03d}.npy",
            ct_volume[:, :, z],
        )

        np.save(
            mask_dir / f"{filename}_slice{z:03d}.npy",
            mask,
        )


# Main Processing

def process_scan(scan, clevel):
    patient_id = scan.patient_id
    study_uid = scan.study_instance_uid[-5:]
    series_uid = scan.series_instance_uid[-5:]

    filename = f"lidc_{patient_id}_{study_uid}_{series_uid}"

    ct_volume = np.load(CT_DIR / f"{filename}.npy")

    height, width, n_slices = ct_volume.shape

    mask_per_slice = create_empty_masks(height, width, n_slices)

    clusters = scan.cluster_annotations()

    for cluster in clusters:
        cmask, cbbox, _ = consensus(cluster, clevel=clevel)

        merge_cluster_into_masks(mask_per_slice, cmask, cbbox)

    save_slice_dataset(ct_volume, mask_per_slice, filename, IMAGE_OUTPUT_DIR, MASK_OUTPUT_DIR)


def main():
    scans = pl.query(pl.Scan).all()

    for scan in tqdm(scans, desc="Extracting segmentation dataset"):
        process_scan(scan, CLEVEL)


if __name__ == "__main__":
    main()