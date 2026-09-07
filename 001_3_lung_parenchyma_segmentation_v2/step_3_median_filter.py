"""Apply a 2D median filter independently to every axial CT slice."""

import numpy as np
from scipy import ndimage as ndi

from config import MEDIAN_FILTER_SIZE


def filter_volume(volume: np.ndarray) -> np.ndarray:
    """Return a median-filtered volume without mixing adjacent slices."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")
    return ndi.median_filter(volume, size=MEDIAN_FILTER_SIZE)

