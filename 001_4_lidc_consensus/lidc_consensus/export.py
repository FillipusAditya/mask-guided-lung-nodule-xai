"""Filesystem export for full-size LIDC-IDRI consensus-mask slices."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def save_consensus_slices(
    masks: Mapping[int, np.ndarray],
    output_dir: str | Path,
    overwrite: bool = False,
) -> list[Path]:
    """Save ``slice_<index>.npy`` files and return their output paths."""
    if not masks:
        raise ValueError("No nonempty consensus slices were generated.")

    output_dir = Path(output_dir)
    output_paths = {
        int(index): output_dir / f"slice_{int(index)}.npy"
        for index in masks
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Consensus mask output already exists: {existing[0]}. "
            "Use overwrite=True to replace existing slices."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in output_paths.items():
        mask = np.asarray(masks[index])
        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask for slice {index}, received {mask.shape}."
            )
        np.save(path, mask.astype(bool, copy=False))

    return [output_paths[index] for index in sorted(output_paths)]
