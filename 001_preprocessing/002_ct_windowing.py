"""Apply fixed CT windowing and normalization to NumPy CT volumes.

The input arrays must contain CT values represented in Hounsfield Units.
Each volume is clipped using a fixed window level and window width, then
normalized to the ``[0.0, 1.0]`` range.

The output remains a three-dimensional NumPy volume with shape ``(N, H, W)``.
The spatial shape and slice order are preserved. Output arrays are stored
using the ``numpy.float32`` dtype and are intended as model input for
subsequent preprocessing and deep learning stages.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Fixed lung-window parameters shared by LIDC-IDRI and LNDb.
WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1500.0

# Directories containing CT volumes in Hounsfield Units.
LIDC_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "001_volume_npy"
)

LNDB_INPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "001_volume_npy"
)

# Directories where windowed and normalized CT volumes will be stored.
LIDC_OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "002_windowed_npy"
)

LNDB_OUTPUT_DIR = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "002_windowed_npy"
)


def window_and_normalize_ct(
    ct: np.ndarray,
    window_level: float,
    window_width: float,
) -> np.ndarray:
    """Apply CT windowing and normalize the result to ``[0.0, 1.0]``.

    Parameters
    ----------
    ct : np.ndarray
        Three-dimensional CT volume in Hounsfield Units with shape
        ``(N, H, W)``.
    window_level : float
        Center of the CT window in Hounsfield Units.
    window_width : float
        Width of the CT window in Hounsfield Units. It must be greater
        than zero.

    Returns
    -------
    np.ndarray
        Windowed and normalized CT volume with the same shape as ``ct``,
        dtype ``numpy.float32``, and values in ``[0.0, 1.0]``.

    Raises
    ------
    TypeError
        If ``ct`` is not a NumPy array, has a non-numeric dtype, or if
        window parameters are not real numbers.
    ValueError
        If ``ct`` is empty, is not three-dimensional, contains non-finite
        values, or ``window_width`` is not greater than zero.
    """

    if not isinstance(ct, np.ndarray):
        raise TypeError(
            "Expected ct to be a NumPy array, "
            f"but received {type(ct).__name__}."
        )

    if ct.ndim != 3:
        raise ValueError(
            "Expected a 3D CT volume with shape (N, H, W), "
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

    if not isinstance(window_level, (int, float, np.integer, np.floating)):
        raise TypeError(
            "Expected window_level to be a real number, "
            f"but received {type(window_level).__name__}."
        )

    if not isinstance(window_width, (int, float, np.integer, np.floating)):
        raise TypeError(
            "Expected window_width to be a real number, "
            f"but received {type(window_width).__name__}."
        )

    window_level = float(window_level)
    window_width = float(window_width)

    if not np.isfinite(window_level):
        raise ValueError("Expected window_level to be finite.")

    if not np.isfinite(window_width):
        raise ValueError("Expected window_width to be finite.")

    if window_width <= 0.0:
        raise ValueError(
            "Expected window_width to be greater than zero, "
            f"but received {window_width}."
        )

    lower_bound = window_level - (window_width / 2.0)
    upper_bound = window_level + (window_width / 2.0)

    # Convert before normalization to avoid integer arithmetic.
    ct_float = ct.astype(
        np.float32,
        copy=False,
    )

    ct_windowed = np.clip(
        ct_float,
        lower_bound,
        upper_bound,
    )

    ct_normalized = (
        (ct_windowed - lower_bound)
        / (upper_bound - lower_bound)
    )

    return ct_normalized.astype(
        np.float32,
        copy=False,
    )


def save_ct_volume(
    ct: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save one normalized CT volume as a NumPy file.

    Parameters
    ----------
    ct : np.ndarray
        Three-dimensional normalized CT volume with shape ``(N, H, W)``,
        dtype compatible with floating-point conversion, and values in
        ``[0.0, 1.0]``.
    output_path : str | Path
        Destination path using the ``.npy`` suffix.

    Raises
    ------
    TypeError
        If ``ct`` is not a NumPy array or has a non-numeric dtype.
    ValueError
        If ``ct`` is empty, not three-dimensional, contains non-finite
        values, contains values outside ``[0.0, 1.0]``, or ``output_path``
        does not use the ``.npy`` suffix.
    """

    if not isinstance(ct, np.ndarray):
        raise TypeError(
            "Expected ct to be a NumPy array, "
            f"but received {type(ct).__name__}."
        )

    if ct.ndim != 3:
        raise ValueError(
            "Expected a 3D CT volume with shape (N, H, W), "
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

    tolerance = 1e-6

    if minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            "Expected normalized CT values in [0.0, 1.0], "
            f"but received range [{minimum}, {maximum}]."
        )

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".npy":
        raise ValueError(
            "Expected output_path to use the .npy suffix, "
            f"but received: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        ct.astype(np.float32, copy=False),
    )


def preprocess_ct_file(
    input_path: str | Path,
    output_path: str | Path,
    window_level: float,
    window_width: float,
) -> None:
    """Window and normalize one CT NumPy volume.

    Parameters
    ----------
    input_path : str | Path
        Path to a ``.npy`` file containing a three-dimensional CT volume
        in Hounsfield Units with shape ``(N, H, W)``.
    output_path : str | Path
        Destination ``.npy`` path.
    window_level : float
        Center of the CT window in Hounsfield Units.
    window_width : float
        Width of the CT window in Hounsfield Units.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    IsADirectoryError
        If ``input_path`` is a directory.
    ValueError
        If ``input_path`` is not a regular ``.npy`` file or the loaded
        array fails CT-volume validation.
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

    original_shape = ct.shape

    ct_processed = window_and_normalize_ct(
        ct=ct,
        window_level=window_level,
        window_width=window_width,
    )

    if ct_processed.shape != original_shape:
        raise RuntimeError(
            "CT preprocessing unexpectedly changed the volume shape: "
            f"{original_shape} -> {ct_processed.shape}."
        )

    save_ct_volume(
        ct=ct_processed,
        output_path=output_path,
    )


def preprocess_ct_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    window_level: float,
    window_width: float,
) -> None:
    """Window and normalize CT NumPy volumes while preserving directories.

    Files are loaded and processed one at a time. Every input array must
    contain a three-dimensional CT volume in Hounsfield Units with shape
    ``(N, H, W)``.

    Parameters
    ----------
    input_dir : str | Path
        Root directory recursively searched for ``.npy`` files.
    output_dir : str | Path
        Root directory where processed ``.npy`` files are stored using
        the relative input directory hierarchy.
    window_level : float
        Center of the CT window in Hounsfield Units.
    window_width : float
        Width of the CT window in Hounsfield Units.

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

    failed_files: list[str] = []

    for npy_file in tqdm(
        npy_files,
        desc="Preprocessing CT",
        unit="file",
    ):
        relative_path = npy_file.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            preprocess_ct_file(
                input_path=npy_file,
                output_path=output_path,
                window_level=window_level,
                window_width=window_width,
            )

        except Exception as error:
            failed_files.append(str(relative_path))

            tqdm.write(
                f"Failed processing {relative_path}: {error}"
            )

    _print_preprocessing_summary(
        input_dir=input_dir,
        output_dir=output_dir,
        total=len(npy_files),
        failed_files=failed_files,
    )


def _print_preprocessing_summary(
    input_dir: Path,
    output_dir: Path,
    total: int,
    failed_files: list[str],
) -> None:
    """Print a summary for one CT preprocessing directory."""

    successful = total - len(failed_files)

    print()
    print("=" * 60)
    print("CT preprocessing finished")
    print(f"Input       : {input_dir}")
    print(f"Output      : {output_dir}")
    print(f"Total       : {total}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {len(failed_files)}")
    print(f"Window level: {WINDOW_LEVEL}")
    print(f"Window width: {WINDOW_WIDTH}")
    print("=" * 60)

    if failed_files:
        print("\nFailed files:")

        for file_path in failed_files:
            print(f" - {file_path}")


def main() -> None:
    """Preprocess the configured LIDC-IDRI and LNDb CT volumes."""

    preprocess_ct_directory(
        input_dir=LIDC_INPUT_DIR,
        output_dir=LIDC_OUTPUT_DIR,
        window_level=WINDOW_LEVEL,
        window_width=WINDOW_WIDTH,
    )

    preprocess_ct_directory(
        input_dir=LNDB_INPUT_DIR,
        output_dir=LNDB_OUTPUT_DIR,
        window_level=WINDOW_LEVEL,
        window_width=WINDOW_WIDTH,
    )


if __name__ == "__main__":
    main()