from collections import OrderedDict
from collections.abc import Mapping
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
INPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "001_volume_npy"

# Directory to save binary lung masks.
MASK_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "003_mask_parenchyma_npy"

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

# Enable or disable individual stages in the segmentation pipeline.
# A disabled post-processing stage passes its input mask through unchanged.
STAGE_ENABLED = OrderedDict(
    [
        ("1. Threshold HU", True),
        ("2. Remove border objects", True),
        ("3. Keep largest components", True),
        ("4. Fill holes", True),
        ("5. Remove trachea", True),
        ("6. Remove table/artifacts", True),
        ("7. Binary dilation", True),
    ]
)

# Optional directory containing an alternative binary initial mask volume for
# every input CT volume. This is required only when HU thresholding is disabled.
# Each mask file must have the same filename and shape as its CT volume.
INITIAL_MASK_DIR: Path | None = None


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


STAGE_FUNCTIONS = OrderedDict(
    [
        ("1. Threshold HU", threshold_lung),
        ("2. Remove border objects", remove_border_objects),
        ("3. Keep largest components", keep_largest_components),
        ("4. Fill holes", fill_holes),
        ("5. Remove trachea", remove_trachea),
        ("6. Remove table/artifacts", remove_table),
        ("7. Binary dilation", dilate_mask),
    ]
)


def validate_stage_configuration(stage_config: Mapping[str, bool]) -> None:
    """Validate the names, order, and values of a stage configuration."""
    expected_stages = list(STAGE_FUNCTIONS)
    configured_stages = list(stage_config)

    if configured_stages != expected_stages:
        raise ValueError(
            "stage_config must contain exactly these stages in order:\n"
            + "\n".join(f"  - {stage}" for stage in expected_stages)
        )

    invalid_values = {
        stage: enabled
        for stage, enabled in stage_config.items()
        if not isinstance(enabled, (bool, np.bool_))
    }
    if invalid_values:
        raise TypeError(
            "Every STAGE_ENABLED value must be True or False. "
            f"Invalid values: {invalid_values}"
        )


def process_slice(
    slice_image: np.ndarray,
    stage_config: Mapping[str, bool] | None = None,
    initial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Segment the lung parenchyma from a single CT slice.

    The configurable segmentation pipeline consists of:
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
    stage_config : mapping or None
        Mapping that enables or disables each pipeline stage.
    initial_mask : np.ndarray or None
        Alternative binary mask used when HU thresholding is disabled.

    Returns
    -------
    np.ndarray
        Binary lung mask as uint8.
    """
    active_stage_config = STAGE_ENABLED if stage_config is None else stage_config
    validate_stage_configuration(active_stage_config)

    mask = None

    for stage_name, stage_function in STAGE_FUNCTIONS.items():
        is_enabled = bool(active_stage_config[stage_name])

        if stage_name == "1. Threshold HU":
            if is_enabled:
                mask = stage_function(slice_image)
            elif initial_mask is not None:
                if initial_mask.shape != slice_image.shape:
                    raise ValueError(
                        "initial_mask and slice_image must have the same shape."
                    )

                unique_values = np.unique(initial_mask)
                if not np.all(np.isin(unique_values, [0, 1])):
                    raise ValueError("initial_mask must be binary with values 0 and 1.")

                mask = np.asarray(initial_mask, dtype=bool).copy()
            else:
                raise ValueError(
                    "HU thresholding cannot be disabled for raw HU input "
                    "unless an alternative binary initial_mask is provided."
                )
        elif is_enabled:
            mask = stage_function(mask)

    return mask.astype(np.uint8)


def segment_lung_parenchyma(
    ct_volume: np.ndarray,
    volume_name: str,
    stage_config: Mapping[str, bool] | None = None,
    initial_mask_volume: np.ndarray | None = None,
) -> np.ndarray:
    """
    Segment lung parenchyma for every slice in a CT volume.

    Parameters
    ----------
    ct_volume : np.ndarray
        Three-dimensional CT volume.
    volume_name : str
        Volume name displayed in the progress bar.
    stage_config : mapping or None
        Mapping that enables or disables each pipeline stage.
    initial_mask_volume : np.ndarray or None
        Alternative 3D binary mask used when HU thresholding is disabled.

    Returns
    -------
    np.ndarray
        Three-dimensional binary lung mask.
    """
    active_stage_config = STAGE_ENABLED if stage_config is None else stage_config
    validate_stage_configuration(active_stage_config)

    if ct_volume.ndim != 3:
        raise ValueError(
            f"ct_volume must be three-dimensional, got shape {ct_volume.shape}."
        )

    if initial_mask_volume is not None:
        if initial_mask_volume.shape != ct_volume.shape:
            raise ValueError(
                "initial_mask_volume and ct_volume must have the same shape."
            )

        unique_values = np.unique(initial_mask_volume)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError("initial_mask_volume must be binary with values 0 and 1.")

    masks = []

    for slice_index, slice_image in enumerate(
        tqdm(
            ct_volume,
            desc=f"Segmenting {volume_name}",
            leave=False,
            unit="slice",
        )
    ):
        initial_mask = (
            None if initial_mask_volume is None else initial_mask_volume[slice_index]
        )
        masks.append(
            process_slice(
                slice_image=slice_image,
                stage_config=active_stage_config,
                initial_mask=initial_mask,
            )
        )

    return np.stack(masks, axis=0)


def load_initial_mask_volume(
    ct_file: Path,
    stage_config: Mapping[str, bool],
) -> np.ndarray | None:
    """Load an alternative initial mask when HU thresholding is disabled."""
    if stage_config["1. Threshold HU"]:
        return None

    if INITIAL_MASK_DIR is None:
        raise ValueError('Set INITIAL_MASK_DIR when "1. Threshold HU" is disabled.')

    initial_mask_file = Path(INITIAL_MASK_DIR) / ct_file.name
    if not initial_mask_file.is_file():
        raise FileNotFoundError(
            f"Initial mask volume was not found: {initial_mask_file}"
        )

    return np.load(initial_mask_file)


def process_volume(
    ct_file: Path,
    stage_config: Mapping[str, bool] | None = None,
) -> None:
    """
    Process one CT volume and save the lung parenchyma mask.

    Parameters
    ----------
    ct_file : Path
        Path to the input CT volume.
    stage_config : mapping or None
        Mapping that enables or disables each pipeline stage.
    """
    active_stage_config = STAGE_ENABLED if stage_config is None else stage_config
    validate_stage_configuration(active_stage_config)

    ct_volume = np.load(ct_file)
    initial_mask_volume = load_initial_mask_volume(
        ct_file=ct_file,
        stage_config=active_stage_config,
    )

    lung_mask = segment_lung_parenchyma(
        ct_volume=ct_volume,
        volume_name=ct_file.stem,
        stage_config=active_stage_config,
        initial_mask_volume=initial_mask_volume,
    )

    np.save(
        MASK_OUTPUT_DIR / ct_file.name,
        lung_mask.astype(np.uint8),
    )


def main() -> None:
    """
    Perform lung parenchyma segmentation for every CT volume in the dataset.
    """
    validate_stage_configuration(STAGE_ENABLED)

    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(INPUT_DIR.glob("*.npy"))

    if len(ct_files) == 0:
        raise FileNotFoundError(f"No .npy files were found in:\n{INPUT_DIR.resolve()}")

    print("Stage Configuration:")
    for stage_name, is_enabled in STAGE_ENABLED.items():
        status = "ON" if is_enabled else "OFF"
        print(f"  [{status:<3}] {stage_name}")

    print(f"\nFound {len(ct_files)} CT volumes.\n")

    for ct_file in tqdm(
        ct_files,
        desc="Overall Progress",
        unit="volume",
    ):
        process_volume(
            ct_file=ct_file,
            stage_config=STAGE_ENABLED,
        )

    print("\nProcessing completed successfully.")
    print(f"Processed Volumes : {len(ct_files)}")
    print(f"Mask Directory    : {MASK_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
