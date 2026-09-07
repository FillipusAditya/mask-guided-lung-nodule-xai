"""Discover and load locally available LIDC-IDRI DICOM and LNDb MHD scans."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import pylidc as pl
import SimpleITK as sitk

from config import LIDC_INPUT_DIR, LNDB_INPUT_DIR


@dataclass(frozen=True)
class ScanSource:
    """One locally discoverable CT scan."""

    dataset: str
    scan_id: str
    source_path: Path
    payload: object


def lidc_filename(scan: pl.Scan) -> str:
    """Return a unique stable stem for one pylidc scan."""
    return (
        f"{scan.patient_id}_{scan.study_instance_uid[-5:]}_"
        f"{scan.series_instance_uid[-5:]}"
    )


def discover_lidc_scans(
    input_dir: str | Path = LIDC_INPUT_DIR,
    patient_id: str | None = None,
) -> list[ScanSource]:
    """Return database scans whose expected DICOM series exists locally.

    Querying only patient folders found below ``input_dir`` avoids trying all
    entries in a larger pylidc database when only a subset was downloaded.
    Small secondary/localizer series are ignored because they do not correspond
    to the CT series registered by pylidc.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"LIDC input directory was not found: {input_dir}")

    patient_ids = sorted(
        path.name
        for path in input_dir.glob("LIDC-IDRI-*")
        if path.is_dir() and (patient_id is None or path.name == patient_id)
    )
    sources = []

    for local_patient_id in patient_ids:
        scans = (
            pl.query(pl.Scan)
            .filter(pl.Scan.patient_id == local_patient_id)
            .all()
        )
        for scan in scans:
            series_dir = Path(scan.get_path_to_dicom_files())
            dicom_count = len(list(series_dir.glob("*.dcm"))) if series_dir.is_dir() else 0
            expected_count = len(scan.slice_zvals)

            if dicom_count < expected_count or expected_count == 0:
                continue

            sources.append(
                ScanSource(
                    dataset="lidc",
                    scan_id=lidc_filename(scan),
                    source_path=series_dir,
                    payload=scan,
                )
            )

    return sources


def _mhd_payload_path(mhd_path: Path) -> Path:
    """Resolve ElementDataFile from a MetaImage header."""
    data_file = None
    for line in mhd_path.read_text(errors="replace").splitlines():
        if line.strip().lower().startswith("elementdatafile"):
            data_file = line.split("=", 1)[1].strip()
            break

    if not data_file or data_file.upper() == "LOCAL":
        return mhd_path
    return mhd_path.parent / data_file


def discover_lndb_scans(
    input_path: str | Path = LNDB_INPUT_DIR,
) -> list[ScanSource]:
    """Return MHD scans whose referenced raw payload exists."""
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".mhd":
        mhd_files = [input_path]
    elif input_path.is_dir():
        mhd_files = sorted(input_path.rglob("*.mhd"))
    else:
        raise FileNotFoundError(f"LNDb input was not found: {input_path}")

    sources = []
    for mhd_path in mhd_files:
        try:
            payload_path = _mhd_payload_path(mhd_path)
        except OSError:
            continue
        if not payload_path.is_file():
            continue
        sources.append(
            ScanSource(
                dataset="lndb",
                scan_id=mhd_path.stem,
                source_path=mhd_path,
                payload=mhd_path,
            )
        )
    return sources


def _sorted_lidc_dicom_paths(scan: pl.Scan) -> list[Path]:
    """Return the scan's DICOM paths in pylidc-compatible axial order."""
    series_dir = Path(scan.get_path_to_dicom_files())
    by_z = {}

    for path in sorted(series_dir.glob("*.dcm")):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        if (
            str(header.SeriesInstanceUID).strip() != scan.series_instance_uid
            or str(header.StudyInstanceUID).strip() != scan.study_instance_uid
        ):
            continue

        z_position = float(header.ImagePositionPatient[-1])
        instance_number = float(header.InstanceNumber)
        previous = by_z.get(z_position)
        if previous is None or instance_number < previous[0]:
            by_z[z_position] = (instance_number, path)

    return [by_z[z_position][1] for z_position in sorted(by_z)]


def load_lidc_scan(scan: pl.Scan) -> np.ndarray:
    """Load one LIDC scan slice-wise into one int16 volume."""
    paths = _sorted_lidc_dicom_paths(scan)
    if not paths:
        raise FileNotFoundError(f"No matching DICOM images found for {lidc_filename(scan)}.")

    first = pydicom.dcmread(paths[0])
    volume = np.empty(
        (len(paths), int(first.Rows), int(first.Columns)),
        dtype=np.int16,
    )

    for index, path in enumerate(paths):
        image = first if index == 0 else pydicom.dcmread(path)
        if image.pixel_array.shape != volume.shape[1:]:
            raise ValueError(f"Inconsistent DICOM image shape in {path}.")
        scaled = (
            image.pixel_array * float(image.RescaleSlope)
            + float(image.RescaleIntercept)
        )
        volume[index] = scaled.astype(np.int16, copy=False)

    return validate_ct_volume(volume)


def load_lndb_scan(mhd_path: str | Path) -> np.ndarray:
    """Load one MHD scan as int16 with shape (slices, height, width)."""
    image = sitk.ReadImage(str(mhd_path))
    volume = sitk.GetArrayFromImage(image).astype(np.int16, copy=False)
    return validate_ct_volume(volume)


def load_scan(source: ScanSource) -> np.ndarray:
    """Load either supported source type."""
    if source.dataset == "lidc":
        return load_lidc_scan(source.payload)
    if source.dataset == "lndb":
        return load_lndb_scan(source.source_path)
    raise ValueError(f"Unsupported dataset: {source.dataset}")


def validate_ct_volume(volume: np.ndarray) -> np.ndarray:
    """Validate the common CT volume contract."""
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {volume.shape}.")
    if volume.shape[0] < 2 or min(volume.shape[1:]) < 32:
        raise ValueError(f"CT volume is unexpectedly small: {volume.shape}.")
    if not np.issubdtype(volume.dtype, np.integer) and not np.isfinite(volume).all():
        raise ValueError("CT volume contains non-finite values.")
    return volume
