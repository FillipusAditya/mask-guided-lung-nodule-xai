from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing CT volumes (.npy).
INPUT_DIR = (
    PROJECT_ROOT 
    / "dataset" 
    / "_lidc" 
    / "001_volume_npy"
)

# Directory to save binary lung masks.
MASK_OUTPUT_DIR = (
    PROJECT_ROOT 
    / "dataset" 
    / "_lidc" 
    / "003_volume_parenchyma_npy"
)

# Segmentation parameters

# HU threshold used to separate air-filled regions from surrounding tissue.
HU_THRESHOLD = -320

# Maximum number of connected components retained after thresholding.
NUM_LARGEST_COMPONENTS = 3

# Minimum component area ratio relative to the image size.
# Smaller components are assumed to belong to the trachea or noise.
TRACHEA_AREA_THRESHOLD = 0.0069

# Number of binary dilation iterations applied to recover lung boundaries.
DILATION_ITERATIONS = 5


def threshold_lung(slice_image: np.ndarray) -> np.ndarray:
    """
    Generate an initial binary lung mask using Hounsfield Unit thresholding.

    Parameters
    ----------
    slice_image : np.ndarray
        One axial CT slice in Hounsfield Units.

    Returns
    -------
    np.ndarray
        Binary mask where air-filled regions are assigned True.
    """
    return slice_image < HU_THRESHOLD


def remove_border_objects(mask: np.ndarray) -> np.ndarray:
    """
    Remove connected components that touch the image border.

    Components connected to the image boundary are usually outside air
    and should not be included as lung regions.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Binary mask after border-connected components are removed.
    """
    return clear_border(mask)


def keep_largest_components(mask: np.ndarray) -> np.ndarray:
    """
    Retain only the largest connected components.

    Lung regions normally form one or two dominant connected components.
    This step removes small isolated regions caused by noise.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Binary mask containing only the largest connected components.
    """
    labeled = label(mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        return np.zeros_like(mask, dtype=bool)

    output = np.zeros_like(mask, dtype=bool)

    areas = [region.area for region in regions]
    sorted_indices = np.argsort(areas)[::-1]

    for idx in sorted_indices[:NUM_LARGEST_COMPONENTS]:
        output[tuple(regions[idx].coords.T)] = True

    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill enclosed holes inside the lung regions.

    This operation restores cavities caused by vessels or segmentation
    discontinuities while preserving the overall lung shape.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Binary mask with internal holes filled.
    """
    return ndi.binary_fill_holes(mask)


def remove_trachea(mask: np.ndarray) -> np.ndarray:
    """
    Remove small connected components that likely correspond to the trachea.

    Components whose area is smaller than the predefined threshold are
    discarded.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Binary mask after trachea removal.
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
    Remove CT table and imaging artifacts based on component position.

    Components located near the top or bottom of the image are assumed to
    belong to scanning artifacts instead of the lungs.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Binary mask after artifact removal.
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
    Expand the lung mask slightly using binary dilation.

    Dilation helps recover lung boundaries that may have been removed during
    previous processing steps.

    Parameters
    ----------
    mask : np.ndarray
        Binary lung mask.

    Returns
    -------
    np.ndarray
        Dilated binary lung mask.
    """
    return binary_dilation(mask, iterations=DILATION_ITERATIONS)


def process_slice(slice_image: np.ndarray) -> np.ndarray:
    """
    Segment the lung parenchyma from a single CT slice.

    The segmentation pipeline consists of:
        1. HU thresholding
        2. Border object removal
        3. Largest component selection
        4. Hole filling
        5. Trachea removal
        6. CT table removal
        7. Binary dilation

    Parameters
    ----------
    slice_image : np.ndarray
        One axial CT slice.

    Returns
    -------
    np.ndarray
        Binary lung mask as uint8.
    """
    mask = threshold_lung(slice_image)
    mask = remove_border_objects(mask)
    mask = keep_largest_components(mask)
    mask = fill_holes(mask)
    mask = remove_trachea(mask)
    mask = remove_table(mask)
    mask = dilate_mask(mask)

    return mask.astype(np.uint8)


def segment_lung_parenchyma(
    ct_volume: np.ndarray,
    volume_name: str,
) -> np.ndarray:
    """
    Segment lung parenchyma for every slice in a CT volume.

    Parameters
    ----------
    ct_volume : np.ndarray
        Three-dimensional CT volume.
    volume_name : str
        Volume name displayed in the progress bar.

    Returns
    -------
    np.ndarray
        Three-dimensional binary lung mask.
    """
    masks = []

    for slice_image in tqdm(
        ct_volume,
        desc=f"Segmenting {volume_name}",
        leave=False,
        unit="slice",
    ):
        masks.append(process_slice(slice_image))

    return np.stack(masks, axis=0)


def process_volume(ct_file: Path) -> None:
    """
    Process one CT volume and save the lung parenchyma mask.

    Parameters
    ----------
    ct_file : Path
        Path to the input CT volume.
    """
    ct_volume = np.load(ct_file)

    lung_mask = segment_lung_parenchyma(
        ct_volume=ct_volume,
        volume_name=ct_file.stem,
    )

    np.save(
        MASK_OUTPUT_DIR / ct_file.name,
        lung_mask.astype(np.uint8),
    )


def main() -> None:
    """
    Perform lung parenchyma segmentation for every CT volume in the dataset.
    """
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(INPUT_DIR.glob("*.npy"))

    if len(ct_files) == 0:
        raise FileNotFoundError(
            f"No .npy files were found in:\n{INPUT_DIR.resolve()}"
        )

    print(f"Found {len(ct_files)} CT volumes.\n")

    for ct_file in tqdm(
        ct_files,
        desc="Overall Progress",
        unit="volume",
    ):
        process_volume(ct_file)

    print("\nProcessing completed successfully.")
    print(f"Processed Volumes : {len(ct_files)}")
    print(f"Mask Directory    : {MASK_OUTPUT_DIR}")


if __name__ == "__main__":
    main()