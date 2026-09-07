"""Apply a 2D median filter independently to every axial CT slice."""

import numpy as np
from scipy import ndimage as ndi

from config import MEDIAN_FILTER_SIZE


def filter_slice(image: np.ndarray) -> np.ndarray:
    """Median-filter one axial slice using only its in-plane neighborhood."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D slice, received {image.shape}.")
    return ndi.median_filter(image, size=MEDIAN_FILTER_SIZE[1:])


def filter_volume(volume: np.ndarray) -> np.ndarray:
    """Return a median-filtered volume without mixing adjacent slices."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")
    return ndi.median_filter(volume, size=MEDIAN_FILTER_SIZE)
