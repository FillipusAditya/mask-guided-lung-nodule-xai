"""Apply binary masks to windowed CT volumes.

Each CT volume is paired with a mask that has the same filename. Voxels in
the CT volume are preserved where the mask is ``1`` and set to ``0`` where
the mask is ``0``. The result is saved as a NumPy ``.npy`` file with the same
shape, dtype, and filename as the input CT volume. The implementation is not
limited to a specific anatomical structure; any shape-compatible binary mask
can be used by changing ``MASK_INPUT_DIR``.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing windowed CT volumes.
CT_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "002_windowed_npy"
)

# Directory containing binary masks. This configuration uses lung-parenchyma
# masks, but it can point to any directory of shape-compatible binary masks.
MASK_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "003_mask_parenchyma_npy"
)

# Directory where masked CT volumes will be stored.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "004_volume_parencyma_npy"
)


def apply_mask(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
) -> np.ndarray:
    """Apply a binary mask to one CT volume.

    Parameters
    ----------
    ct_volume : np.ndarray
        Three-dimensional windowed CT volume with shape ``(N, H, W)``.
    mask_volume : np.ndarray
        Three-dimensional binary mask with the same shape as ``ct_volume``
        and values ``{0, 1}`` or boolean values.

    Returns
    -------
    np.ndarray
        Masked CT volume. Values where the mask is ``0`` become ``0`` and
        values where the mask is ``1`` remain unchanged. Shape and dtype
        match ``ct_volume``.

    Raises
    ------
    TypeError
        If either input is not a NumPy array or has an unsupported dtype.
    ValueError
        If an input is empty or not three-dimensional, the shapes differ,
        the arrays contain non-finite values, or the mask is not binary.
    """
    if not isinstance(ct_volume, np.ndarray):
        raise TypeError(
            "Expected ct_volume to be a NumPy array, "
            f"but received {type(ct_volume).__name__}."
        )

    if not isinstance(mask_volume, np.ndarray):
        raise TypeError(
            "Expected mask_volume to be a NumPy array, "
            f"but received {type(mask_volume).__name__}."
        )

    if ct_volume.ndim != 3:
        raise ValueError(
            "Expected a 3D CT volume with shape (N, H, W), "
            f"but received shape {ct_volume.shape}."
        )

    if mask_volume.ndim != 3:
        raise ValueError(
            "Expected a 3D mask volume with shape (N, H, W), "
            f"but received shape {mask_volume.shape}."
        )

    if ct_volume.shape != mask_volume.shape:
        raise ValueError(
            "CT and mask shapes must match, but received "
            f"CT shape {ct_volume.shape} and mask shape {mask_volume.shape}."
        )

    if ct_volume.size == 0:
        raise ValueError("Expected CT and mask volumes to contain values.")

    if not np.issubdtype(ct_volume.dtype, np.number) or np.iscomplexobj(
        ct_volume
    ):
        raise TypeError(
            "Expected ct_volume to have a real numeric dtype, "
            f"but received {ct_volume.dtype}."
        )

    mask_is_boolean = np.issubdtype(mask_volume.dtype, np.bool_)
    mask_is_numeric = np.issubdtype(mask_volume.dtype, np.number)

    if not (mask_is_boolean or mask_is_numeric) or np.iscomplexobj(
        mask_volume
    ):
        raise TypeError(
            "Expected mask_volume to have a real numeric or boolean dtype, "
            f"but received {mask_volume.dtype}."
        )

    if not np.isfinite(ct_volume).all():
        raise ValueError("Expected ct_volume to contain only finite values.")

    if mask_is_numeric and not np.isfinite(mask_volume).all():
        raise ValueError("Expected mask_volume to contain only finite values.")

    if not np.logical_or(mask_volume == 0, mask_volume == 1).all():
        raise ValueError(
            "Expected mask_volume to be binary with values {0, 1}."
        )

    return np.where(mask_volume.astype(bool), ct_volume, 0).astype(
        ct_volume.dtype,
        copy=False,
    )


def process_ct_file(
    ct_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> None:
    """Load, mask, and save one CT volume."""
    ct_path = Path(ct_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    for path, description in (
        (ct_path, "CT"),
        (mask_path, "mask"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"{description} NumPy file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a {description} NumPy file: {path}"
            )

        if path.suffix.lower() != ".npy":
            raise ValueError(
                f"Expected {description} path to use the .npy suffix: {path}"
            )

    if output_path.suffix.lower() != ".npy":
        raise ValueError(
            f"Expected output path to use the .npy suffix: {output_path}"
        )

    ct_volume = np.load(ct_path, allow_pickle=False)
    mask_volume = np.load(mask_path, allow_pickle=False)

    masked_ct = apply_mask(
        ct_volume=ct_volume,
        mask_volume=mask_volume,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, masked_ct)


def process_ct_directory(
    ct_input_dir: str | Path,
    mask_input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Apply same-named masks to all CT NumPy files in a directory."""
    ct_input_dir = Path(ct_input_dir)
    mask_input_dir = Path(mask_input_dir)
    output_dir = Path(output_dir)

    for directory, description in (
        (ct_input_dir, "CT input"),
        (mask_input_dir, "mask input"),
    ):
        if not directory.exists():
            raise FileNotFoundError(
                f"{description} directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Expected {description} directory: {directory}"
            )

    ct_files = sorted(ct_input_dir.glob("*.npy"))

    if not ct_files:
        raise FileNotFoundError(
            f"No .npy files found in CT input directory: {ct_input_dir}"
        )

    missing_masks = [
        ct_file.name
        for ct_file in ct_files
        if not (mask_input_dir / ct_file.name).is_file()
    ]

    if missing_masks:
        missing_names = ", ".join(missing_masks)
        raise FileNotFoundError(
            "No same-named mask was found for these CT files: "
            f"{missing_names}"
        )

    for ct_file in tqdm(
        ct_files,
        desc="Applying masks",
        unit="volume",
    ):
        process_ct_file(
            ct_path=ct_file,
            mask_path=mask_input_dir / ct_file.name,
            output_path=output_dir / ct_file.name,
        )


def main() -> None:
    """Apply the configured binary masks to windowed CT volumes."""
    process_ct_directory(
        ct_input_dir=CT_INPUT_DIR,
        mask_input_dir=MASK_INPUT_DIR,
        output_dir=OUTPUT_DIR,
    )

    print("\nProcessing completed successfully.")
    print(f"Masked CT directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
