"""Protected clean-seed bidirectional lung-mask segmentation."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import disk
from skimage.segmentation import clear_border
from tqdm import tqdm

from config import (
    BOUNDARY_REPAIR_RADIUS,
    HU_THRESHOLD,
    MIN_COMPONENT_OVERLAP_PIXELS,
    NUM_LARGEST_COMPONENTS,
    REFERENCE_DILATION_SCHEDULE,
    TABLE_COMPONENT_AREA_THRESHOLD,
    TABLE_LOWER_CENTER_Y_RATIO,
    TABLE_MAX_HEIGHT_WIDTH_RATIO,
    TABLE_MAX_Y_RATIO,
    TABLE_MIN_WIDTH_RATIO,
    TABLE_MIN_Y_RATIO,
    TRACHEA_AREA_THRESHOLD,
    TRACHEA_CENTER_HALF_WIDTH_RATIO,
    TRACHEA_MAX_Y_RATIO,
)


@dataclass
class SegmentationMetrics:
    """Compact diagnostics for one segmented volume."""

    reference_index: int
    failed_overlap_count: int
    failed_overlap_slices: list[int]
    non_empty_start: int | None
    non_empty_end: int | None
    foreground_fraction: float
    mean_adjacent_dice: float

    def to_dict(self) -> dict:
        return asdict(self)


def threshold_lung(image: np.ndarray) -> np.ndarray:
    return image < HU_THRESHOLD


def keep_largest_components(mask: np.ndarray, number: int) -> np.ndarray:
    labeled = label(mask)
    regions = sorted(regionprops(labeled), key=lambda item: item.area, reverse=True)
    output = np.zeros_like(mask, dtype=bool)
    for region in regions[:number]:
        output[labeled == region.label] = True
    return output


def remove_wide_flat_table(mask: np.ndarray) -> np.ndarray:
    """Remove a wide, flat component in the lower part of an axial image."""
    labeled = label(mask)
    output = mask.copy()
    height, width = mask.shape

    for region in regionprops(labeled):
        min_y, min_x, max_y, max_x = region.bbox
        component_height = max_y - min_y
        component_width = max_x - min_x
        height_width_ratio = component_height / max(component_width, 1)
        is_lower = region.centroid[0] > TABLE_LOWER_CENTER_Y_RATIO * height
        is_wide = component_width > TABLE_MIN_WIDTH_RATIO * width
        is_flat = height_width_ratio < TABLE_MAX_HEIGHT_WIDTH_RATIO
        if is_lower and is_wide and is_flat:
            output[labeled == region.label] = False
    return output


def candidate_mask(image: np.ndarray) -> np.ndarray:
    """Build a sensitive candidate with high-specificity table removal."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, received {image.shape}.")
    mask = threshold_lung(image)
    mask = clear_border(mask)
    mask = remove_wide_flat_table(mask)
    mask = keep_largest_components(mask, NUM_LARGEST_COMPONENTS)
    return ndi.binary_fill_holes(mask).astype(bool)


def build_candidate_volume(volume: np.ndarray, show_progress: bool = False) -> np.ndarray:
    """Build uint8 candidate masks for all axial slices."""
    masks = np.zeros(volume.shape, dtype=np.uint8)
    indices = range(len(volume))
    if show_progress:
        indices = tqdm(indices, desc="Candidate masks", unit="slice", leave=False)
    for index in indices:
        masks[index] = candidate_mask(volume[index])
    return masks


def remove_table(mask: np.ndarray) -> np.ndarray:
    """Remove table geometry plus small peripheral reference components."""
    output = remove_wide_flat_table(mask)
    labeled = label(output)
    height, width = mask.shape
    image_area = height * width

    for region in regionprops(labeled):
        center_y = region.centroid[0]
        is_small = region.area / image_area < TABLE_COMPONENT_AREA_THRESHOLD
        is_peripheral = not (
            TABLE_MIN_Y_RATIO * height <= center_y <= TABLE_MAX_Y_RATIO * height
        )
        if is_small and is_peripheral:
            output[labeled == region.label] = False
    return output


def remove_trachea(mask: np.ndarray) -> np.ndarray:
    """Remove only small, strictly central, anterior airway components."""
    labeled = label(mask)
    output = mask.copy()
    height, width = mask.shape
    image_area = height * width

    for region in regionprops(labeled):
        center_y, center_x = region.centroid
        is_small = region.area / image_area < TRACHEA_AREA_THRESHOLD
        is_central = (
            abs(center_x - width / 2) < TRACHEA_CENTER_HALF_WIDTH_RATIO * width
        )
        is_anterior = center_y < TRACHEA_MAX_Y_RATIO * height
        if is_small and is_central and is_anterior:
            output[labeled == region.label] = False
    return output


def clean_reference(candidate: np.ndarray) -> np.ndarray:
    """Clean one reference candidate and retain at most two lung components."""
    mask = remove_table(candidate)
    mask = remove_trachea(mask)
    mask = keep_largest_components(mask, 2)
    return ndi.binary_fill_holes(mask).astype(bool)


def has_bilateral_lung_components(mask: np.ndarray) -> bool:
    """Return whether substantial components occur on both image halves."""
    height, width = mask.shape
    minimum_area = 0.01 * height * width
    center_x_values = [
        region.centroid[1]
        for region in regionprops(label(mask))
        if region.area >= minimum_area
    ]
    return (
        any(center_x < width / 2 for center_x in center_x_values)
        and any(center_x > width / 2 for center_x in center_x_values)
    )


def select_reference_index(candidates: np.ndarray) -> int:
    """Prefer the nearest-to-middle clean slice containing bilateral lungs."""
    middle = len(candidates) // 2
    search_order = sorted(range(len(candidates)), key=lambda index: abs(index - middle))
    nearest_valid_index = None
    for index in search_order:
        cleaned = clean_reference(candidates[index])
        if not cleaned.any():
            continue
        if nearest_valid_index is None:
            nearest_valid_index = index
        if has_bilateral_lung_components(cleaned):
            return index
    if nearest_valid_index is not None:
        return nearest_valid_index
    raise ValueError("No non-empty clean reference mask could be constructed.")


def match_candidate_components(
    candidate: np.ndarray,
    reference_support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select candidate components with sufficient reference overlap."""
    labeled = label(candidate)
    matched = np.zeros_like(candidate, dtype=bool)
    seed = np.zeros_like(candidate, dtype=bool)

    for region in regionprops(labeled):
        component = labeled == region.label
        component_seed = component & reference_support
        if int(component_seed.sum()) >= MIN_COMPONENT_OVERLAP_PIXELS:
            matched |= component
            seed |= component_seed
    return matched, seed


def propagate_one_slice(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Propagate a protected reference into one candidate slice."""
    for iterations in REFERENCE_DILATION_SCHEDULE:
        support = ndi.binary_dilation(reference, iterations=iterations)
        matched, seed = match_candidate_components(candidate, support)
        if seed.any():
            result = ndi.binary_propagation(seed, mask=matched)
            return ndi.binary_fill_holes(result).astype(bool), False
    return np.zeros_like(candidate, dtype=bool), True


def propagate_bidirectionally(
    candidates: np.ndarray,
    reference_index: int,
    reference_mask: np.ndarray,
    in_place: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Propagate both ways, optionally reusing the candidate-volume buffer."""
    protected = candidates if in_place else np.zeros_like(candidates, dtype=bool)
    protected[reference_index] = reference_mask
    failed = []

    last_valid_reference = reference_mask
    for index in range(reference_index - 1, -1, -1):
        candidate = protected[index] if in_place else candidates[index]
        candidate_is_non_empty = bool(candidate.any())
        propagated, did_fail = propagate_one_slice(
            candidate, last_valid_reference
        )
        protected[index] = propagated
        if did_fail and candidate_is_non_empty:
            failed.append(index)
        if not did_fail:
            last_valid_reference = propagated

    last_valid_reference = reference_mask
    for index in range(reference_index + 1, len(candidates)):
        candidate = protected[index] if in_place else candidates[index]
        candidate_is_non_empty = bool(candidate.any())
        propagated, did_fail = propagate_one_slice(
            candidate, last_valid_reference
        )
        protected[index] = propagated
        if did_fail and candidate_is_non_empty:
            failed.append(index)
        if not did_fail:
            last_valid_reference = propagated

    return protected, sorted(failed)


def _close_component(mask: np.ndarray, structure: np.ndarray, radius: int) -> np.ndarray:
    """Close one component in a padded crop rather than a full 512x512 image."""
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return np.zeros_like(mask, dtype=bool)
    lower = np.maximum(coordinates.min(axis=0) - radius - 1, 0)
    upper = np.minimum(coordinates.max(axis=0) + radius + 2, mask.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    crop = mask[slices]
    repaired_crop = ndi.binary_closing(crop, structure=structure)
    repaired_crop = ndi.binary_fill_holes(repaired_crop)
    repaired = np.zeros_like(mask, dtype=bool)
    repaired[slices] = repaired_crop
    return repaired


def repair_boundary(mask: np.ndarray) -> np.ndarray:
    """Repair bounded concavities independently for each connected component."""
    labeled = label(mask)
    repaired = np.zeros_like(mask, dtype=bool)
    structure = disk(BOUNDARY_REPAIR_RADIUS)
    for region in regionprops(labeled):
        component = labeled == region.label
        repaired |= _close_component(component, structure, BOUNDARY_REPAIR_RADIUS)
    return repaired


def finalize_mask_volume(
    protected: np.ndarray,
    show_progress: bool = False,
    in_place: bool = False,
) -> np.ndarray:
    """Remove trachea and repair boundaries, optionally reusing the input."""
    final = protected if in_place else np.zeros_like(protected, dtype=bool)
    indices = range(len(protected))
    if show_progress:
        indices = tqdm(indices, desc="Final mask repair", unit="slice", leave=False)
    for index in indices:
        trachea_cleaned = remove_trachea(protected[index])
        final[index] = repair_boundary(trachea_cleaned)
    return final


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denominator = int(mask_a.sum() + mask_b.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * np.logical_and(mask_a, mask_b).sum() / denominator


def _metrics(mask: np.ndarray, reference_index: int, failed: list[int]) -> SegmentationMetrics:
    areas = mask.sum(axis=(1, 2))
    non_empty = np.flatnonzero(areas)
    adjacent = [_dice(mask[index - 1], mask[index]) for index in range(1, len(mask))]
    return SegmentationMetrics(
        reference_index=reference_index,
        failed_overlap_count=len(failed),
        failed_overlap_slices=failed,
        non_empty_start=int(non_empty[0]) if len(non_empty) else None,
        non_empty_end=int(non_empty[-1]) if len(non_empty) else None,
        foreground_fraction=float(mask.mean()),
        mean_adjacent_dice=float(np.mean(adjacent)) if adjacent else 1.0,
    )


def segment_volume(
    volume: np.ndarray,
    show_progress: bool = False,
) -> tuple[np.ndarray, SegmentationMetrics]:
    """Return the final uint8 lung mask and segmentation diagnostics."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")
    candidates = build_candidate_volume(volume, show_progress)
    reference_index = select_reference_index(candidates)
    reference = clean_reference(candidates[reference_index])
    protected, failed = propagate_bidirectionally(
        candidates,
        reference_index,
        reference,
        in_place=True,
    )
    final = finalize_mask_volume(protected, show_progress, in_place=True)
    return final.astype(np.uint8, copy=False), _metrics(final, reference_index, failed)
