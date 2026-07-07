from pathlib import Path

import numpy as np
from tqdm import tqdm

from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


# Configuration

# Input CT volume (.npy) after Hounsfield Unit (HU) conversion
INPUT_PATH = Path("CT_scan.npy")

# Output paths
MASK_OUTPUT_PATH = Path("output/mask/lung_mask.npy")
LUNG_CT_OUTPUT_PATH = Path("output/lung_ct/lung_ct.npy")

# Lung segmentation parameters
HU_THRESHOLD = -320
NUM_LARGEST_COMPONENTS = 3
TRACHEA_AREA_THRESHOLD = 0.0069
DILATION_ITERATIONS = 5


# Segmentation Functions
def threshold_lung(slice_image: np.ndarray) -> np.ndarray:
    """
    Generate an initial binary mask using HU thresholding.
    Lung tissues contain large amounts of air and therefore have
    Hounsfield Unit values lower than the selected threshold.
    """
    return slice_image < HU_THRESHOLD


def remove_border_objects(mask: np.ndarray) -> np.ndarray:
    """
    Remove connected components that touch the image border.
    These regions are typically outside the patient's body.
    """
    return clear_border(mask)


def keep_largest_components(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected components.
    This removes small noisy regions while preserving the lungs.
    """
    labeled = label(mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        return np.zeros_like(mask, dtype=bool)

    areas = [region.area for region in regions]
    sorted_indices = np.argsort(areas)[::-1]

    output = np.zeros_like(mask, dtype=bool)

    for idx in sorted_indices[:NUM_LARGEST_COMPONENTS]:
        output[tuple(regions[idx].coords.T)] = True

    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes inside the lung regions to obtain complete lung masks.
    """
    return ndi.binary_fill_holes(mask)


def remove_trachea(mask: np.ndarray) -> np.ndarray:
    """
    Remove very small connected components that are likely
    to represent the trachea or airway structures.
    """
    output = mask.copy()

    labeled = label(mask)
    regions = regionprops(labeled)

    image_area = mask.shape[0] * mask.shape[1]

    for region in regions:

        area_ratio = region.area / image_area

        if area_ratio < TRACHEA_AREA_THRESHOLD:
            output[tuple(region.coords.T)] = False

    return output


def remove_table(mask: np.ndarray) -> np.ndarray:
    """
    Remove components located near the top or bottom of the image.
    These structures are usually caused by the CT table or imaging artifacts.
    """
    output = mask.copy()

    labeled = label(mask)

    component_ids = np.unique(labeled)[1:]

    for component_id in component_ids:

        center_y = center_of_mass(labeled == component_id)[0]

        if center_y < 0.30 * mask.shape[0]:
            output[labeled == component_id] = False

        elif center_y > 0.60 * mask.shape[0]:
            output[labeled == component_id] = False

    return output


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    """
    Expand the lung mask slightly to recover lung boundary voxels.
    """
    return binary_dilation(mask, iterations=DILATION_ITERATIONS)


# Slice Processing
def process_slice(slice_image: np.ndarray) -> np.ndarray:
    """
    Perform lung parenchyma segmentation for a single CT slice.
    """

    mask = threshold_lung(slice_image)

    mask = remove_border_objects(mask)

    mask = keep_largest_components(mask)

    mask = fill_holes(mask)

    mask = remove_trachea(mask)

    mask = remove_table(mask)

    mask = dilate_mask(mask)

    return mask.astype(np.uint8)


# Volume Processing
def segment_lung_parenchyma(ct_volume: np.ndarray) -> np.ndarray:
    """
    Perform lung parenchyma segmentation for every slice
    in a 3D CT volume.
    """

    lung_masks = []

    for slice_image in tqdm(
        ct_volume,
        desc="Segmenting Lung Parenchyma",
        unit="slice"
    ):
        lung_masks.append(process_slice(slice_image))

    return np.stack(lung_masks, axis=0)


# Main
def main():

    print("Loading CT volume...")
    ct_volume = np.load(INPUT_PATH)

    print("Running lung parenchyma segmentation...")
    lung_mask = segment_lung_parenchyma(ct_volume)

    # Apply binary mask to preserve only lung regions
    lung_ct = ct_volume * lung_mask

    # Create output directories if they do not exist
    MASK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LUNG_CT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Saving binary lung mask...")
    np.save(MASK_OUTPUT_PATH, lung_mask.astype(np.uint8))

    print("Saving lung-only CT volume...")
    np.save(LUNG_CT_OUTPUT_PATH, lung_ct)

    print("\nProcessing completed successfully.")
    print(f"Input Volume Shape : {ct_volume.shape}")
    print(f"Mask Shape         : {lung_mask.shape}")
    print(f"Lung CT Shape      : {lung_ct.shape}")
    print(f"Binary Mask Saved  : {MASK_OUTPUT_PATH}")
    print(f"Lung CT Saved      : {LUNG_CT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()