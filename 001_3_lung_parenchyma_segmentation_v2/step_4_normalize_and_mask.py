"""Apply lung windowing, float32 normalization, and the final binary mask."""

import numpy as np

from config import WINDOW_LEVEL, WINDOW_WIDTH


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

