"""Strict filename and directory parsing for prepared samples."""

from pathlib import Path
import re


SLICE_PATTERN = re.compile(r"slice_(\d+)\.npy")


def extract_slice_index(mask_path: str | Path) -> int:
    """Extract the non-negative index from ``slice_<index>.npy``."""
    mask_path = Path(mask_path)
    match = SLICE_PATTERN.fullmatch(mask_path.name)
    if match is None:
        raise ValueError(
            f"Expected mask filename 'slice_<index>.npy': {mask_path.name}"
        )
    return int(match.group(1))


def extract_nodule_id(directory: str | Path, prefix: str) -> int:
    """Extract an integer from ``<prefix>_<id>``."""
    directory = Path(directory)
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", directory.name)
    if match is None:
        raise ValueError(
            f"Expected nodule directory '{prefix}_<id>': {directory.name}"
        )
    return int(match.group(1))


def sample_filename(
    study_name: str,
    nodule_name: str,
    slice_index: int,
) -> str:
    """Return the common filename used for CT and mask outputs."""
    return f"{study_name}_{nodule_name}_slice_{slice_index}.npy"

