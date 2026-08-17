"""Apply two-dimensional consensus masks to matching CT volume slices.

Consensus masks are stored by study and nodule directory: ``cluster_N`` for
LIDC-IDRI or ``finding_N`` for LNDb. A mask named ``slice_N.npy`` is applied
to slice ``N`` of the study's windowed CT volume. Each resulting
two-dimensional CT slice retains its input dtype and is saved using the same
directory hierarchy as the consensus mask.
"""

from pathlib import Path
import re

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CT_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "002_windowed_npy"
)

MASK_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "005_mask_consensus_npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "006_masked_consensus_npy"
)

# Existing output files are skipped by default. Set this to True to replace
# them when regenerating the dataset.
OVERWRITE = False

SLICE_FILENAME_PATTERN = re.compile(r"slice_(\d+)\.npy")
NODULE_DIRECTORY_PATTERN = re.compile(r"(?:cluster|finding)_(\d+)")


def extract_slice_index(mask_path: str | Path) -> int:
    """Extract a non-negative CT slice index from ``slice_N.npy``."""
    mask_path = Path(mask_path)
    match = SLICE_FILENAME_PATTERN.fullmatch(mask_path.name)

    if match is None:
        raise ValueError(
            "Expected consensus mask filename 'slice_<index>.npy', "
            f"but received: {mask_path.name}"
        )

    return int(match.group(1))


def validate_ct_volume(
    ct_volume: np.ndarray,
    ct_path: str | Path = "CT volume",
) -> None:
    """Validate a CT volume with shape ``(N, H, W)``."""
    if not isinstance(ct_volume, np.ndarray):
        raise TypeError(
            "Expected CT volume to be a NumPy array, "
            f"but received {type(ct_volume).__name__}."
        )

    if ct_volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D CT volume at {ct_path}, "
            f"but received shape {ct_volume.shape}."
        )

    if ct_volume.size == 0:
        raise ValueError(f"Expected a non-empty CT volume at {ct_path}.")

    if not np.issubdtype(ct_volume.dtype, np.number) or np.iscomplexobj(
        ct_volume
    ):
        raise TypeError(
            f"Expected a real numeric CT volume at {ct_path}, "
            f"but received dtype {ct_volume.dtype}."
        )

    if not np.isfinite(ct_volume).all():
        raise ValueError(
            f"Expected CT volume at {ct_path} to contain only finite values."
        )


def validate_mask(
    mask: np.ndarray,
    mask_path: str | Path = "consensus mask",
) -> None:
    """Validate a two-dimensional binary consensus mask."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            "Expected consensus mask to be a NumPy array, "
            f"but received {type(mask).__name__}."
        )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D consensus mask at {mask_path}, "
            f"but received shape {mask.shape}."
        )

    if mask.size == 0:
        raise ValueError(f"Expected a non-empty consensus mask at {mask_path}.")

    mask_is_boolean = np.issubdtype(mask.dtype, np.bool_)
    mask_is_numeric = np.issubdtype(mask.dtype, np.number)

    if not (mask_is_boolean or mask_is_numeric) or np.iscomplexobj(mask):
        raise TypeError(
            f"Expected a real numeric or boolean mask at {mask_path}, "
            f"but received dtype {mask.dtype}."
        )

    if mask_is_numeric and not np.isfinite(mask).all():
        raise ValueError(
            f"Expected consensus mask at {mask_path} to contain only "
            "finite values."
        )

    if not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(
            f"Expected consensus mask at {mask_path} to contain only "
            "binary values {0, 1}."
        )


def apply_consensus_mask(
    ct_slice: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Preserve CT pixels inside a binary mask and zero all other pixels."""
    if not isinstance(ct_slice, np.ndarray):
        raise TypeError(
            "Expected CT slice to be a NumPy array, "
            f"but received {type(ct_slice).__name__}."
        )

    if ct_slice.ndim != 2:
        raise ValueError(
            "Expected a 2D CT slice, "
            f"but received shape {ct_slice.shape}."
        )

    if ct_slice.size == 0:
        raise ValueError("Expected a non-empty CT slice.")

    if not np.issubdtype(ct_slice.dtype, np.number) or np.iscomplexobj(
        ct_slice
    ):
        raise TypeError(
            "Expected CT slice to have a real numeric dtype, "
            f"but received {ct_slice.dtype}."
        )

    if not np.isfinite(ct_slice).all():
        raise ValueError("Expected CT slice to contain only finite values.")

    validate_mask(mask)

    if ct_slice.shape != mask.shape:
        raise ValueError(
            "CT slice and consensus mask shapes must match, but received "
            f"CT shape {ct_slice.shape} and mask shape {mask.shape}."
        )

    return np.where(mask.astype(bool), ct_slice, 0).astype(
        ct_slice.dtype,
        copy=False,
    )


def process_mask_file(
    ct_volume: np.ndarray,
    mask_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
) -> str:
    """Apply one consensus mask and return ``'created'`` or ``'skipped'``."""
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    if output_path.is_file() and not overwrite:
        return "skipped"

    slice_index = extract_slice_index(mask_path)

    if slice_index >= ct_volume.shape[0]:
        raise IndexError(
            f"Slice index {slice_index} from {mask_path} is outside the CT "
            f"volume with depth {ct_volume.shape[0]}."
        )

    mask = np.load(mask_path, allow_pickle=False)
    validate_mask(mask, mask_path)

    ct_slice = ct_volume[slice_index]
    if ct_slice.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch for {mask_path}: CT slice shape "
            f"{ct_slice.shape}, mask shape {mask.shape}."
        )

    masked_ct = apply_consensus_mask(ct_slice, mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, masked_ct)
    return "created"


def process_study(
    study_dir: str | Path,
    ct_input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Process every nodule directory and mask belonging to one study."""
    study_dir = Path(study_dir)
    ct_input_dir = Path(ct_input_dir)
    output_dir = Path(output_dir)
    study_name = study_dir.name
    ct_path = ct_input_dir / f"{study_name}.npy"

    if not ct_path.is_file():
        raise FileNotFoundError(f"CT volume not found for {study_name}: {ct_path}")

    ct_volume = np.load(ct_path, allow_pickle=False)
    validate_ct_volume(ct_volume, ct_path)

    nodule_dirs = sorted(
        path
        for path in study_dir.iterdir()
        if path.is_dir()
        and NODULE_DIRECTORY_PATTERN.fullmatch(path.name) is not None
    )

    if not nodule_dirs:
        raise FileNotFoundError(
            "No nodule directories named 'cluster_<index>' or "
            f"'finding_<index>' found for study: {study_dir}"
        )

    counts = {"discovered": 0, "created": 0, "skipped": 0}

    for nodule_dir in nodule_dirs:
        mask_paths = sorted(nodule_dir.glob("slice_*.npy"))
        if not mask_paths:
            raise FileNotFoundError(
                f"No consensus mask files found in: {nodule_dir}"
            )

        for mask_path in mask_paths:
            counts["discovered"] += 1
            status = process_mask_file(
                ct_volume=ct_volume,
                mask_path=mask_path,
                output_path=(
                    output_dir
                    / study_name
                    / nodule_dir.name
                    / mask_path.name
                ),
                overwrite=overwrite,
            )
            counts[status] += 1

    return counts


def process_dataset(
    ct_input_dir: str | Path,
    mask_input_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, int]:
    """Apply all consensus masks and return processing statistics."""
    ct_input_dir = Path(ct_input_dir)
    mask_input_dir = Path(mask_input_dir)
    output_dir = Path(output_dir)

    for directory, description in (
        (ct_input_dir, "CT input"),
        (mask_input_dir, "consensus mask input"),
    ):
        if not directory.exists():
            raise FileNotFoundError(
                f"{description} directory does not exist: {directory}"
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Expected {description} directory: {directory}"
            )

    study_dirs = sorted(path for path in mask_input_dir.iterdir() if path.is_dir())
    if not study_dirs:
        raise FileNotFoundError(
            f"No study directories found in: {mask_input_dir}"
        )

    totals = {
        "studies": len(study_dirs),
        "discovered": 0,
        "created": 0,
        "skipped": 0,
    }

    for study_dir in tqdm(
        study_dirs,
        desc="Applying consensus masks",
        unit="study",
    ):
        counts = process_study(
            study_dir=study_dir,
            ct_input_dir=ct_input_dir,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        for key in ("discovered", "created", "skipped"):
            totals[key] += counts[key]

    return totals


def main() -> None:
    """Apply the configured consensus masks to windowed CT slices."""
    totals = process_dataset(
        ct_input_dir=CT_INPUT_DIR,
        mask_input_dir=MASK_INPUT_DIR,
        output_dir=OUTPUT_DIR,
        overwrite=OVERWRITE,
    )

    print("\nConsensus masking completed successfully.")
    print(f"Studies processed : {totals['studies']}")
    print(f"Masks discovered  : {totals['discovered']}")
    print(f"Files created     : {totals['created']}")
    print(f"Files skipped     : {totals['skipped']}")
    print(f"Output directory  : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
