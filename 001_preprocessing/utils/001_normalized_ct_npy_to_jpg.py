"""Convert normalized CT NumPy arrays to JPG images for visualization.

The input arrays must already be windowed or clipped, normalized to the
``[0.0, 1.0]`` range, and no longer represented in Hounsfield Units. JPG
files are intended only for visualization and data exploration, not as the
primary input for deep learning models. Conversion to ``uint8`` in the
``[0, 255]`` range occurs only during JPG export.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories containing normalized CT NumPy volumes.
LIDC_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "006_masked_consensus_npy"
)

LNDB_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "006_masked_consensus_npy"
)

# Directories where JPG visualization files will be stored.
LIDC_OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "000_jpgs"
    / "006_masked_consensus_jpg"
)

LNDB_OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "000_jpgs"
    / "006_masked_consensus_jpg"
)

def normalized_ct_to_uint8(
    ct: np.ndarray,
) -> np.ndarray:
    """
    Convert a normalized CT image or volume to 8-bit grayscale values.

    Parameters
    ----------
    ct : np.ndarray
        Numeric CT image or volume normalized to ``[0.0, 1.0]``. Supported
        shapes are ``(H, W)`` and ``(N, H, W)``.

    Returns
    -------
    np.ndarray
        A new array with the same shape as ``ct`` and dtype ``np.uint8``.

    Raises
    ------
    TypeError
        If ``ct`` is not a NumPy array or does not have a real numeric dtype.
    ValueError
        If ``ct`` is empty, has an unsupported number of dimensions, contains
        non-finite values, or has values outside ``[0.0, 1.0]``.
    """

    if not isinstance(ct, np.ndarray):
        raise TypeError(
            "Expected ct to be a NumPy array, "
            f"but received {type(ct).__name__}."
        )

    if ct.ndim not in (2, 3):
        raise ValueError(
            "Expected a 2D CT image or 3D CT volume, "
            f"but received shape {ct.shape}."
        )

    if not np.issubdtype(ct.dtype, np.number) or np.iscomplexobj(ct):
        raise TypeError(
            "Expected ct to have a real numeric dtype, "
            f"but received {ct.dtype}."
        )

    if ct.size == 0:
        raise ValueError("Expected ct to contain at least one value.")

    if not np.isfinite(ct).all():
        raise ValueError("Expected ct to contain only finite values.")

    minimum = float(np.min(ct))
    maximum = float(np.max(ct))

    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            "Expected normalized CT values in [0.0, 1.0], "
            f"but received range [{minimum}, {maximum}]."
        )

    return np.rint(ct * 255.0).astype(np.uint8)


def save_ct_as_jpg(
    ct: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Save a normalized CT image or volume as JPG.

    Parameters
    ----------
    ct : np.ndarray
        Numeric CT image or volume normalized to ``[0.0, 1.0]``. Supported
        shapes are ``(H, W)`` and ``(N, H, W)``.
    output_path : str | Path
        Output JPG path for a two-dimensional image. For a three-dimensional
        volume, its suffix is removed and slices are stored in a directory
        using filenames such as ``slice_000.jpg`` and ``slice_001.jpg``.

    Raises
    ------
    TypeError
        If ``ct`` is not a NumPy array or does not have a real numeric dtype.
    ValueError
        If ``ct`` fails normalized CT validation.
    """

    output_path = Path(output_path)
    ct_uint8 = normalized_ct_to_uint8(ct)

    if ct_uint8.ndim == 2:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        Image.fromarray(ct_uint8).save(output_path)

    else:
        output_dir = output_path.with_suffix("")

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, slice_ct in enumerate(ct_uint8):
            Image.fromarray(slice_ct).save(
                output_dir / f"slice_{index:03d}.jpg"
            )


def convert_ct_file(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """
    Convert one normalized CT NumPy file to JPG visualization output.

    Parameters
    ----------
    input_path : str | Path
        Path to a ``.npy`` file containing a normalized CT image ``(H, W)``
        or volume ``(N, H, W)`` with values in ``[0.0, 1.0]``.
    output_path : str | Path
        Output JPG path passed to :func:`save_ct_as_jpg`.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    IsADirectoryError
        If ``input_path`` is a directory.
    ValueError
        If ``input_path`` does not use the ``.npy`` suffix or the loaded array
        fails normalized CT validation.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"CT NumPy file does not exist: {input_path}"
        )

    if input_path.is_dir():
        raise IsADirectoryError(
            f"Expected a CT NumPy file, but received directory: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Expected a regular CT NumPy file: {input_path}"
        )

    if input_path.suffix.lower() != ".npy":
        raise ValueError(
            "Expected input_path to use the .npy suffix, "
            f"but received: {input_path}"
        )

    ct = np.load(
        input_path,
        allow_pickle=False,
    )

    save_ct_as_jpg(
        ct=ct,
        output_path=output_path,
    )


def convert_ct_directory(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """
    Convert normalized CT NumPy files to JPG while preserving directories.

    Files are loaded and converted one at a time. Every input array must
    already be normalized to ``[0.0, 1.0]``.

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
        desc="Converting CT",
        unit="file",
    ):
        relative_path = npy_file.relative_to(input_dir)
        output_path = (
            output_dir
            / relative_path
        ).with_suffix(".jpg")

        convert_ct_file(
            input_path=npy_file,
            output_path=output_path,
        )

def main() -> None:
    """Convert normalized LIDC-IDRI and LNDb CT volumes to JPG images."""

    # print("Converting LIDC-IDRI normalized CT volumes...")

    # convert_ct_directory(
    #     input_dir=LIDC_INPUT_DIR,
    #     output_dir=LIDC_OUTPUT_DIR,
    # )

    print("\nConverting LNDb normalized CT volumes...")

    convert_ct_directory(
        input_dir=LNDB_INPUT_DIR,
        output_dir=LNDB_OUTPUT_DIR,
    )

    print("\nCT JPG conversion completed successfully.")


if __name__ == "__main__":
    main()