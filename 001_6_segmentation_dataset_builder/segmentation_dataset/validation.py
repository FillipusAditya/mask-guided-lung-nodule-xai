"""Validation helpers for CT volumes and consensus masks."""

from pathlib import Path

import numpy as np


def load_volume(path: str | Path) -> np.ndarray:
    """Memory-map and validate a numeric CT volume in ``(N, H, W)`` order."""
    path = Path(path)
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3 or volume.size == 0:
        raise ValueError(
            f"Expected a nonempty 3D CT volume at {path}, received {volume.shape}."
        )
    if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
        raise TypeError(f"Expected a real numeric CT volume at {path}: {volume.dtype}")
    return volume


def load_mask(path: str | Path) -> np.ndarray:
    """Load and validate a finite two-dimensional binary mask."""
    path = Path(path)
    mask = np.load(path, allow_pickle=False)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(
            f"Expected a nonempty 2D mask at {path}, received {mask.shape}."
        )
    is_bool = np.issubdtype(mask.dtype, np.bool_)
    is_numeric = np.issubdtype(mask.dtype, np.number)
    if not (is_bool or is_numeric) or np.iscomplexobj(mask):
        raise TypeError(f"Expected a real numeric or boolean mask: {path}")
    if is_numeric and not np.isfinite(mask).all():
        raise ValueError(f"Mask contains non-finite values: {path}")
    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"Mask is not binary: {path}")
    return mask.astype(bool, copy=False)


def validate_paired_shapes(
    ct_slices: dict[str, np.ndarray],
    mask: np.ndarray,
    context: str,
) -> None:
    """Require all CT representations and the mask to share one shape."""
    shapes = {name: array.shape for name, array in ct_slices.items()}
    shapes["mask"] = mask.shape
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Paired sample shapes differ for {context}: {shapes}")

