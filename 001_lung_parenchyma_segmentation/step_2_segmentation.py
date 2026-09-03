"""Step 2: generate a 3D lung-parenchyma mask from a CT volume."""

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

from config import (
    DILATION_ITERATIONS,
    HU_THRESHOLD,
    NUM_LARGEST_COMPONENTS,
    STAGE_ENABLED,
    TRACHEA_AREA_THRESHOLD,
)


def threshold_lung(image: np.ndarray) -> np.ndarray:
    """Select air-filled regions below the configured HU threshold."""
    return image < HU_THRESHOLD


def remove_border_objects(mask: np.ndarray) -> np.ndarray:
    """Remove outside air connected to an image border."""
    return clear_border(mask)


def keep_largest_components(mask: np.ndarray) -> np.ndarray:
    """Keep only the configured number of largest components."""
    labeled = label(mask)
    regions = regionprops(labeled)
    output = np.zeros_like(mask, dtype=bool)

    regions.sort(key=lambda region: region.area, reverse=True)
    for region in regions[:NUM_LARGEST_COMPONENTS]:
        output[tuple(region.coords.T)] = True

    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes inside lung components."""
    return ndi.binary_fill_holes(mask)


def remove_trachea(mask: np.ndarray) -> np.ndarray:
    """Remove small components that may represent the trachea or noise."""
    output = mask.copy()
    image_area = mask.shape[0] * mask.shape[1]

    for region in regionprops(label(mask)):
        if region.area / image_area < TRACHEA_AREA_THRESHOLD:
            output[tuple(region.coords.T)] = False

    return output


def remove_table(mask: np.ndarray) -> np.ndarray:
    """Remove components outside the expected vertical lung region."""
    output = mask.copy()
    labeled = label(mask)

    for component_id in np.unique(labeled)[1:]:
        center_y = center_of_mass(labeled == component_id)[0]

        if center_y < 0.30 * mask.shape[0] or center_y > 0.60 * mask.shape[0]:
            output[labeled == component_id] = False

    return output


def dilate_mask(mask: np.ndarray) -> np.ndarray:
    """Expand the lung mask to recover its boundary."""
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


def segment_image(
    image: np.ndarray,
    stage_config: Mapping[str, bool] = STAGE_ENABLED,
    initial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Segment one 2D CT image and return a boolean lung mask."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received {image.shape}.")

    if list(stage_config) != list(STAGE_FUNCTIONS):
        raise ValueError("stage_config must contain the seven stages in order.")

    mask = None

    for stage_name, stage_function in STAGE_FUNCTIONS.items():
        is_enabled = bool(stage_config[stage_name])

        if stage_name == "1. Threshold HU":
            if is_enabled:
                mask = stage_function(image)
            elif initial_mask is not None:
                if initial_mask.shape != image.shape:
                    raise ValueError("initial_mask and image shapes must match.")
                mask = initial_mask.astype(bool, copy=True)
            else:
                raise ValueError(
                    "An initial_mask is required when HU thresholding is disabled."
                )
        elif is_enabled:
            mask = stage_function(mask)

    return mask.astype(bool, copy=False)


def segment_volume(
    volume: np.ndarray,
    stage_config: Mapping[str, bool] = STAGE_ENABLED,
    initial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Segment every axial slice and return a uint8 mask with shape (N, H, W)."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")

    if initial_mask is not None and initial_mask.shape != volume.shape:
        raise ValueError("initial_mask and volume shapes must match.")

    masks = np.empty(volume.shape, dtype=np.uint8)

    for slice_index, image in enumerate(volume):
        initial_slice = None if initial_mask is None else initial_mask[slice_index]
        masks[slice_index] = segment_image(image, stage_config, initial_slice)

    return masks


def process_file(
    input_path: str | Path,
    output_path: str | Path,
    initial_mask_path: str | Path | None = None,
) -> None:
    """Segment one NumPy image or volume and save its binary mask."""
    data = np.load(input_path, allow_pickle=False)
    initial_mask = (
        None
        if initial_mask_path is None
        else np.load(initial_mask_path, allow_pickle=False)
    )

    if data.ndim == 2:
        mask = segment_image(data, initial_mask=initial_mask).astype(np.uint8)
    elif data.ndim == 3:
        mask = segment_volume(data, initial_mask=initial_mask)
    else:
        raise ValueError(f"Expected a 2D or 3D array, received {data.shape}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mask)


def process_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    initial_mask_dir: str | Path | None = None,
) -> None:
    """Segment every NumPy file in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")

    for index, input_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {input_path.name}")
        initial_path = (
            None
            if initial_mask_dir is None
            else Path(initial_mask_dir) / input_path.name
        )
        process_file(input_path, output_dir / input_path.name, initial_path)


def main() -> None:
    """Run segmentation for one NumPy file or a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("input_file", type=Path)
    single.add_argument("output_file", type=Path)
    single.add_argument("--initial-mask", type=Path)

    batch = subparsers.add_parser("batch")
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("output_dir", type=Path)
    batch.add_argument("--initial-mask-dir", type=Path)

    args = parser.parse_args()

    if args.command == "single":
        process_file(args.input_file, args.output_file, args.initial_mask)
    else:
        process_batch(args.input_dir, args.output_dir, args.initial_mask_dir)


if __name__ == "__main__":
    main()
