"""Convert binary NumPy mask arrays to JPG images for visualization.

The input arrays must contain only binary values ``{0, 1}`` or boolean
values. JPG output uses ``uint8`` values ``{0, 255}`` and is intended only
for data exploration, visual inspection, debugging, and quality control. JPG
files are not the primary input for deep learning models; the original binary
NumPy arrays remain the training data source. Both 2D masks and 3D mask
volumes are supported. Because JPG compression is lossy, decoded pixels near
nonuniform mask boundaries are not guaranteed to remain exactly binary; the
JPG files must not be used for quantitative mask processing.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories containing binary mask NumPy volumes.
LIDC_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "005_mask_consensus_npy"
)

# LNDB_INPUT_DIR = (
#     PROJECT_ROOT
#     / "000_dataset"
#     / "_lndb"
#     / "005_mask_consensus_npy"
# )

# Directories where mask JPG visualizations will be stored.
LIDC_OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "000_jpgs"
    / "005_mask_consensus_jpg"
)

# LNDB_OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "000_dataset"
#     / "_lndb"
#     / "000_jpgs"
#     / "005_mask_consensus_jpg"
# )

def _summarize_unique_values(
    mask: np.ndarray,
    maximum_values: int = 10,
) -> str:
    """Return a bounded summary of unique mask values for error messages."""

    unique_values = np.unique(mask)
    displayed_values = unique_values[:maximum_values].tolist()

    if unique_values.size <= maximum_values:
        return str(displayed_values)

    return (
        f"{displayed_values} ... "
        f"({unique_values.size} unique values total)"
    )


def binary_mask_to_uint8(
    mask: np.ndarray,
) -> np.ndarray:
    """
    Convert a binary mask image or volume to 8-bit visualization values.

    Parameters
    ----------
    mask : np.ndarray
        Numeric binary mask containing only ``{0, 1}``, or a boolean mask.
        Supported shapes are ``(H, W)`` and ``(N, H, W)``.

    Returns
    -------
    np.ndarray
        A new array with the same shape as ``mask``, dtype ``np.uint8``, and
        values ``{0, 255}``.

    Raises
    ------
    TypeError
        If ``mask`` is not a NumPy array or does not have a real numeric or
        boolean dtype.
    ValueError
        If ``mask`` is empty, has an unsupported number of dimensions,
        contains non-finite values, or contains values other than ``{0, 1}``.
    """

    if not isinstance(mask, np.ndarray):
        raise TypeError(
            "Expected mask to be a NumPy array, "
            f"but received {type(mask).__name__}."
        )

    if mask.ndim not in (2, 3):
        raise ValueError(
            "Expected a 2D mask image or 3D mask volume, "
            f"but received shape {mask.shape}."
        )

    is_boolean = np.issubdtype(mask.dtype, np.bool_)
    is_numeric = np.issubdtype(mask.dtype, np.number)

    if not (is_boolean or is_numeric) or np.iscomplexobj(mask):
        raise TypeError(
            "Expected mask to have a real numeric or boolean dtype, "
            f"but received {mask.dtype}."
        )

    if mask.size == 0:
        raise ValueError("Expected mask to contain at least one value.")

    if is_numeric and not np.isfinite(mask).all():
        raise ValueError("Expected mask to contain only finite values.")

    if not np.logical_or(mask == 0, mask == 1).all():
        unique_values = _summarize_unique_values(mask)
        raise ValueError(
            "Expected a binary mask with values {0, 1}, "
            f"but received unique values {unique_values}."
        )

    return mask.astype(np.uint8) * 255


def save_mask_as_jpg(
    mask: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save a binary mask image or volume as JPG for visualization.

    Parameters
    ----------
    mask : np.ndarray
        Numeric binary mask containing only ``{0, 1}``, or a boolean mask.
        Supported shapes are ``(H, W)`` and ``(N, H, W)``.
    output_path : str | Path
        Output JPG path for a two-dimensional mask. For a three-dimensional
        volume, its suffix is removed and slices are stored in a directory
        using filenames such as ``slice_000.jpg`` and ``slice_001.jpg``.

    Raises
    ------
    TypeError
        If ``mask`` is not a NumPy array or does not have a real numeric or
        boolean dtype.
    ValueError
        If ``mask`` fails binary mask validation.
    """

    output_path = Path(output_path)
    mask_uint8 = binary_mask_to_uint8(mask)

    if mask_uint8.ndim == 2:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(mask_uint8).save(output_path)

    else:
        output_dir = output_path.with_suffix("")

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, slice_mask in enumerate(mask_uint8):
            Image.fromarray(slice_mask).save(
                output_dir / f"slice_{index:03d}.jpg"
            )


def convert_mask_file(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Convert one binary mask NumPy file to JPG visualization output.

    Parameters
    ----------
    input_path : str | Path
        Path to a ``.npy`` file containing a binary mask image ``(H, W)`` or
        volume ``(N, H, W)`` with values ``{0, 1}`` or boolean values.
    output_path : str | Path
        Output JPG path passed to :func:`save_mask_as_jpg`.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    IsADirectoryError
        If ``input_path`` is a directory.
    ValueError
        If ``input_path`` does not use the ``.npy`` suffix or the loaded array
        fails binary mask validation.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Mask NumPy file does not exist: {input_path}"
        )

    if input_path.is_dir():
        raise IsADirectoryError(
            f"Expected a mask NumPy file, but received directory: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Expected a regular mask NumPy file: {input_path}"
        )

    if input_path.suffix.lower() != ".npy":
        raise ValueError(
            "Expected input_path to use the .npy suffix, "
            f"but received: {input_path}"
        )

    mask = np.load(
        input_path,
        allow_pickle=False,
    )

    save_mask_as_jpg(
        mask=mask,
        output_path=output_path,
    )


def convert_mask_directory(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Convert binary mask NumPy files to JPG while preserving directories.

    Files are loaded and converted one at a time. Every input array must
    contain only binary values ``{0, 1}`` or boolean values.

    Parameters
    ----------
    input_dir : str | Path
        Root directory recursively searched for ``.npy`` files.
    output_dir : str | Path
        Root directory where converted JPG images or slice directories are
        written using the relative input directory hierarchy.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist or contains no ``.npy`` files.
    NotADirectoryError
        If ``input_dir`` is not a directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Expected input directory, but received: {input_dir}"
        )

    npy_files = sorted(input_dir.rglob("*.npy"))

    if not npy_files:
        raise FileNotFoundError(
            f"No .npy files found in input directory: {input_dir}"
        )

    for npy_file in tqdm(
        npy_files,
        desc="Converting masks",
        unit="file",
    ):
        relative_path = npy_file.relative_to(input_dir)
        output_path = (
            output_dir
            / relative_path
        ).with_suffix(".jpg")

        convert_mask_file(
            input_path=npy_file,
            output_path=output_path,
        )

def main() -> None:
    """Convert configured LIDC-IDRI and LNDb binary masks to JPG."""

    print("Converting LIDC-IDRI binary masks...")

    convert_mask_directory(
        input_dir=LIDC_INPUT_DIR,
        output_dir=LIDC_OUTPUT_DIR,
    )

    # print("\nConverting LNDb binary masks...")

    # convert_mask_directory(
    #     input_dir=LNDB_INPUT_DIR,
    #     output_dir=LNDB_OUTPUT_DIR,
    # )

    print("\nMask JPG conversion completed successfully.")


if __name__ == "__main__":
    main()
