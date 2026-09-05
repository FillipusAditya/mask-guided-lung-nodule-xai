"""Validated conversion of LIDC-IDRI and LNDb CT scans to NumPy volumes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ConversionResult:
    """Result of converting or skipping one CT scan."""

    source_id: str
    volume_path: Path
    metadata_path: Path
    status: str


def validate_volume(volume: np.ndarray) -> None:
    """Validate a CT array before it is converted to ``int16``."""
    if not isinstance(volume, np.ndarray):
        raise TypeError(f"Expected a NumPy array, received {type(volume).__name__}.")
    if volume.ndim != 3:
        raise ValueError(f"Expected shape (slices, height, width), received {volume.shape}.")
    if volume.size == 0:
        raise ValueError("Expected a nonempty CT volume.")
    if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
        raise TypeError(f"Expected a real numeric dtype, received {volume.dtype}.")
    if not np.isfinite(volume).all():
        raise ValueError("Expected finite CT values.")

    info = np.iinfo(np.int16)
    minimum = float(np.min(volume))
    maximum = float(np.max(volume))
    if minimum < info.min or maximum > info.max:
        raise ValueError(
            f"CT values [{minimum}, {maximum}] cannot be represented as int16."
        )


def _validate_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy output path, received: {path}")
    return path


def _as_json_value(value: Any) -> Any:
    """Convert NumPy scalar/container values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json_value(item) for item in value]
    return value


def save_volume_with_metadata(
    volume: np.ndarray,
    output_path: str | Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> ConversionResult:
    """Save an HU volume and its sidecar JSON with overwrite protection."""
    validate_volume(volume)
    output_path = _validate_output_path(output_path)
    metadata_path = output_path.with_suffix(".json")
    volume_exists = output_path.exists()
    metadata_exists = metadata_path.exists()

    source_id = str(metadata["source_id"])
    if volume_exists != metadata_exists and not overwrite:
        raise FileExistsError(
            "Incomplete existing output pair; use --overwrite after inspecting it: "
            f"{output_path}, {metadata_path}"
        )
    if volume_exists and metadata_exists and not overwrite:
        return ConversionResult(source_id, output_path, metadata_path, "skipped")

    converted = volume.astype(np.int16, copy=False)
    full_metadata = {
        **metadata,
        "array_axis_order": ["slice", "row", "column"],
        "shape_zyx": list(converted.shape),
        "dtype": str(converted.dtype),
        "intensity_unit": "Hounsfield unit",
        "volume_file": output_path.name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.save(stream, converted, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(_as_json_value(full_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ConversionResult(source_id, output_path, metadata_path, "written")


def lidc_filename(scan: Any) -> str:
    """Return the project-standard, scan-unique LIDC filename."""
    return (
        f"{scan.patient_id}_{scan.study_instance_uid[-5:]}_"
        f"{scan.series_instance_uid[-5:]}.npy"
    )


def convert_lidc_scan(
    scan: Any,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert one pylidc scan from ``(H, W, N)`` to project order."""
    volume = np.transpose(scan.to_volume(), (2, 0, 1))
    z_positions = np.asarray(scan.slice_zvals, dtype=float)
    if len(z_positions) != volume.shape[0]:
        raise ValueError(
            f"{scan.patient_id} has {volume.shape[0]} slices but "
            f"{len(z_positions)} z positions."
        )

    spacing_xy = float(scan.pixel_spacing)
    metadata = {
        "dataset": "LIDC-IDRI",
        "source_id": scan.patient_id,
        "study_instance_uid": scan.study_instance_uid,
        "series_instance_uid": scan.series_instance_uid,
        "spacing_xyz_mm": [spacing_xy, spacing_xy, float(scan.slice_spacing)],
        "slice_thickness_mm": float(scan.slice_thickness),
        "slice_z_positions_mm": z_positions,
    }
    return save_volume_with_metadata(
        volume, output_path, metadata, overwrite=overwrite
    )


def convert_lndb_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert one LNDb MetaImage scan and preserve its spatial geometry."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"LNDb input does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Expected a regular LNDb input file: {input_path}")
    if input_path.suffix.lower() != ".mhd":
        raise ValueError(f"Expected a .mhd LNDb input, received: {input_path}")

    import SimpleITK as sitk

    image = sitk.ReadImage(str(input_path))
    volume = sitk.GetArrayFromImage(image)
    metadata = {
        "dataset": "LNDb",
        "source_id": input_path.stem,
        "source_file": input_path.name,
        "spacing_xyz_mm": image.GetSpacing(),
        "origin_xyz_mm": image.GetOrigin(),
        "direction_xyz": image.GetDirection(),
    }
    return save_volume_with_metadata(
        volume, output_path, metadata, overwrite=overwrite
    )


def get_lidc_scans(patient_id: str | None = None) -> list[Any]:
    """Query configured pylidc scans, optionally for one patient."""
    # pylidc is intentionally lazy: LNDb conversion and unit tests do not need it.
    import pylidc as pl

    query = pl.query(pl.Scan)
    if patient_id is not None:
        query = query.filter(pl.Scan.patient_id == patient_id)
    scans = query.all()
    if not scans:
        suffix = f" for {patient_id}" if patient_id else ""
        raise FileNotFoundError(f"No LIDC-IDRI scans were found{suffix}.")
    return scans


def discover_lndb_files(input_dir: str | Path) -> list[Path]:
    """Return deterministically ordered LNDb MetaImage headers."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"LNDb input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected an LNDb input directory: {input_dir}")
    files = sorted(input_dir.rglob("*.mhd"))
    if not files:
        raise FileNotFoundError(f"No .mhd files found in {input_dir}.")
    return files


def convert_lidc_dataset(
    output_dir: str | Path,
    *,
    patient_id: str | None = None,
    overwrite: bool = False,
) -> list[ConversionResult]:
    """Convert all matching LIDC scans."""
    output_dir = Path(output_dir)
    scans = get_lidc_scans(patient_id)
    filenames = [lidc_filename(scan) for scan in scans]
    duplicates = sorted(
        name for name in set(filenames) if filenames.count(name) > 1
    )
    if duplicates:
        raise ValueError(
            "Multiple LIDC scans resolve to the same output filename: "
            + ", ".join(duplicates)
        )
    return [
        convert_lidc_scan(
            scan, output_dir / lidc_filename(scan), overwrite=overwrite
        )
        for scan in _progress(scans, "Converting LIDC-IDRI")
    ]


def convert_lndb_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[ConversionResult]:
    """Convert every LNDb MetaImage while preserving relative subdirectories."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = discover_lndb_files(input_dir)
    return [
        convert_lndb_file(
            path,
            (output_dir / path.relative_to(input_dir)).with_suffix(".npy"),
            overwrite=overwrite,
        )
        for path in _progress(files, "Converting LNDb")
    ]


def _progress(items: Sequence[Any], description: str) -> Iterable[Any]:
    """Add a progress bar without making tqdm part of the conversion API."""
    from tqdm import tqdm

    return tqdm(items, desc=description, unit="scan", dynamic_ncols=True)


def summarize(results: Sequence[ConversionResult]) -> str:
    """Create a concise deterministic conversion summary."""
    written = sum(item.status == "written" for item in results)
    skipped = sum(item.status == "skipped" for item in results)
    return f"Total: {len(results)} | Written: {written} | Skipped: {skipped}"
