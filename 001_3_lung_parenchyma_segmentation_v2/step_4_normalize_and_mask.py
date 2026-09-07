"""Apply lung windowing, float32 normalization, and the final binary mask."""

import numpy as np
from tqdm import tqdm

from config import WINDOW_LEVEL, WINDOW_WIDTH
from step_3_median_filter import filter_slice


def window_and_normalize(volume: np.ndarray) -> np.ndarray:
    """Map the configured HU window to float32 values in [0, 1]."""
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    normalized = volume.astype(np.float32, copy=True)
    np.clip(normalized, lower, upper, out=normalized)
    normalized -= lower
    normalized /= upper - lower
    return normalized


def normalize_and_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize one CT volume and set all pixels outside the lung to zero."""
    if volume.shape != mask.shape or volume.ndim != 3:
        raise ValueError(
            f"Expected matching 3D CT/mask shapes, received {volume.shape} and {mask.shape}."
        )
    normalized = window_and_normalize(volume)
    normalized[~mask.astype(bool)] = np.float32(0.0)
    return normalized.astype(np.float32, copy=False)


def filter_normalize_and_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    show_progress: bool = False,
) -> np.ndarray:
    """Create float32 parenchyma one slice at a time to limit peak memory."""
    if volume.shape != mask.shape or volume.ndim != 3:
        raise ValueError(
            f"Expected matching 3D CT/mask shapes, received {volume.shape} and {mask.shape}."
        )

    output = np.empty(volume.shape, dtype=np.float32)
    lower = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
    upper = WINDOW_LEVEL + WINDOW_WIDTH / 2.0
    indices = range(len(volume))
    if show_progress:
        indices = tqdm(indices, desc="Filter and normalize", unit="slice", leave=False)

    for index in indices:
        filtered = filter_slice(volume[index])
        normalized = filtered.astype(np.float32)
        np.clip(normalized, lower, upper, out=normalized)
        normalized -= lower
        normalized /= upper - lower
        normalized[mask[index] == 0] = np.float32(0.0)
        output[index] = normalized

    return output
