"""Convert LIDC-IDRI and LNDb CT scans to NumPy volumes.

LIDC-IDRI scans are loaded through ``pylidc`` from their configured DICOM
storage. LNDb scans are loaded from MetaImage ``.mhd`` files through
``SimpleITK``. Both conversion paths save three-dimensional CT volumes with
shape ``(N, H, W)`` and dtype ``numpy.int16`` in Hounsfield Units.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pylidc as pl
import SimpleITK as sitk
from tqdm import tqdm


DatasetName = Literal["lidc", "lndb"]


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing the original LNDb MetaImage files.
LNDB_INPUT_DIR = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "lndb" 
    / "data"
)

# Directories where converted CT volumes will be stored.
LIDC_OUTPUT_DIR = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "_lidc" 
    / "001_volume_npy_v3"
)

LNDB_OUTPUT_DIR = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "_lndb" 
    / "001_volume_npy_v3"
)


def save_ct_volume(
    volume: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save one CT volume as a NumPy file.

    Parameters
    ----------
    volume : np.ndarray
        Three-dimensional CT volume with shape ``(N, H, W)``. The volume is
        converted to ``numpy.int16`` before being saved.
    output_path : str | Path
        Destination path using the ``.npy`` suffix.

    Raises
    ------
    TypeError
        If ``volume`` is not a NumPy array or has a non-numeric dtype.
    ValueError
        If ``volume`` is empty, not three-dimensional, contains non-finite
        values, or ``output_path`` does not use the ``.npy`` suffix.
    """

    if not isinstance(volume, np.ndarray):
        raise TypeError(
            "Expected volume to be a NumPy array, "
            f"but received {type(volume).__name__}."
        )

    if volume.ndim != 3:
        raise ValueError(
            "Expected a 3D CT volume with shape (N, H, W), "
            f"but received shape {volume.shape}."
        )

    if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
        raise TypeError(
            "Expected volume to have a real numeric dtype, "
            f"but received {volume.dtype}."
        )

    if volume.size == 0:
        raise ValueError("Expected volume to contain at least one value.")

    if not np.isfinite(volume).all():
        raise ValueError("Expected volume to contain only finite values.")

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
        volume.astype(np.int16, copy=False),
    )


def convert_lidc_scan(
    scan: pl.Scan,
    output_path: str | Path,
) -> None:
    """Convert one LIDC-IDRI DICOM scan to a NumPy CT volume.

    Parameters
    ----------
    scan : pylidc.Scan
        LIDC-IDRI scan retrieved from the pylidc database.
    output_path : str | Path
        Destination ``.npy`` path.
    """

    volume = scan.to_volume()

    # pylidc returns (H, W, N); the project convention is (N, H, W).
    volume = np.transpose(volume, (2, 0, 1))

    save_ct_volume(
        volume=volume,
        output_path=output_path,
    )


def convert_lndb_file(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Convert one LNDb MetaImage scan to a NumPy CT volume.

    Parameters
    ----------
    input_path : str | Path
        Path to an LNDb ``.mhd`` file.
    output_path : str | Path
        Destination ``.npy`` path.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    IsADirectoryError
        If ``input_path`` is a directory.
    ValueError
        If ``input_path`` is not a regular ``.mhd`` file.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"LNDb MetaImage file does not exist: {input_path}"
        )

    if input_path.is_dir():
        raise IsADirectoryError(
            f"Expected an LNDb MetaImage file, but received: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Expected a regular LNDb MetaImage file: {input_path}"
        )

    if input_path.suffix.lower() != ".mhd":
        raise ValueError(
            "Expected input_path to use the .mhd suffix, "
            f"but received: {input_path}"
        )

    image = sitk.ReadImage(str(input_path))
    volume = sitk.GetArrayFromImage(image)

    save_ct_volume(
        volume=volume,
        output_path=output_path,
    )


def convert_lidc_dataset(
    output_dir: str | Path,
) -> None:
    """Convert all configured LIDC-IDRI scans to NumPy CT volumes.

    Parameters
    ----------
    output_dir : str | Path
        Directory where converted ``.npy`` volumes are stored.
    """

    output_dir = Path(output_dir)
    scans = pl.query(pl.Scan).all()

    if not scans:
        raise FileNotFoundError(
            "No LIDC-IDRI scans were found in the pylidc database."
        )

    failed_scans: list[str] = []

    for scan in tqdm(
        scans,
        desc="Converting LIDC-IDRI",
        unit="scan",
    ):
        patient_id = scan.patient_id

        try:
            study_uid = scan.study_instance_uid[-5:]
            series_uid = scan.series_instance_uid[-5:]
            filename = f"{patient_id}_{study_uid}_{series_uid}.npy"

            convert_lidc_scan(
                scan=scan,
                output_path=output_dir / filename,
            )

        except Exception as error:
            failed_scans.append(patient_id)
            tqdm.write(f"Failed processing {patient_id}: {error}")

    _print_conversion_summary(
        dataset_name="LIDC-IDRI",
        total=len(scans),
        failed_scans=failed_scans,
    )


def convert_lndb_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Convert all LNDb MetaImage scans to NumPy CT volumes.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing LNDb ``.mhd`` files.
    output_dir : str | Path
        Directory where converted ``.npy`` volumes are stored.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist or contains no ``.mhd`` files.
    NotADirectoryError
        If ``input_dir`` is not a directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"LNDb input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Expected LNDb input directory, but received: {input_dir}"
        )

    mhd_files = sorted(input_dir.rglob("*.mhd"))

    if not mhd_files:
        raise FileNotFoundError(
            f"No .mhd files found in LNDb input directory: {input_dir}"
        )

    failed_scans: list[str] = []

    for mhd_path in tqdm(
        mhd_files,
        desc="Converting LNDb",
        unit="scan",
    ):
        try:
            relative_path = mhd_path.relative_to(input_dir)
            output_path = (output_dir / relative_path).with_suffix(".npy")

            convert_lndb_file(
                input_path=mhd_path,
                output_path=output_path,
            )

        except Exception as error:
            failed_scans.append(mhd_path.stem)
            tqdm.write(f"Failed processing {mhd_path.name}: {error}")

    _print_conversion_summary(
        dataset_name="LNDb",
        total=len(mhd_files),
        failed_scans=failed_scans,
    )


def convert_ct_dataset(
    dataset: DatasetName,
    output_dir: str | Path,
    input_dir: str | Path | None = None,
) -> None:
    """Convert one supported CT dataset to NumPy volumes.

    Parameters
    ----------
    dataset : {"lidc", "lndb"}
        Dataset conversion backend.
    output_dir : str | Path
        Directory where converted NumPy volumes are stored.
    input_dir : str | Path or None, default=None
        LNDb directory containing ``.mhd`` files. This argument must be
        provided for ``dataset="lndb"`` and omitted for ``dataset="lidc"``.

    Raises
    ------
    ValueError
        If ``dataset`` is unsupported or its path arguments are inconsistent.
    """

    dataset = dataset.lower()

    if dataset == "lidc":
        if input_dir is not None:
            raise ValueError(
                "input_dir must be None for LIDC-IDRI because pylidc "
                "resolves the configured DICOM directory."
            )

        convert_lidc_dataset(output_dir=output_dir)
        return

    if dataset == "lndb":
        if input_dir is None:
            raise ValueError(
                "input_dir is required when converting the LNDb dataset."
            )

        convert_lndb_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        return

    raise ValueError(
        "Unsupported dataset. Expected 'lidc' or 'lndb', "
        f"but received: {dataset!r}."
    )


def _print_conversion_summary(
    dataset_name: str,
    total: int,
    failed_scans: list[str],
) -> None:
    """Print a bounded conversion summary for one dataset."""

    successful = total - len(failed_scans)

    print(f"\n{dataset_name} conversion finished")
    print(f"Total      : {total}")
    print(f"Successful : {successful}")
    print(f"Failed     : {len(failed_scans)}")

    if failed_scans:
        print("\nFailed scans:")

        for scan_id in failed_scans:
            print(f" - {scan_id}")


def main() -> None:
    """Convert the configured LIDC-IDRI and LNDb datasets to NumPy."""
    
    convert_ct_dataset(
        dataset="lidc",
        output_dir=LIDC_OUTPUT_DIR,
    )

    convert_ct_dataset(
        dataset="lndb",
        input_dir=LNDB_INPUT_DIR,
        output_dir=LNDB_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
