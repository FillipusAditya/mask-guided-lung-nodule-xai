"""Median filtering, lung windowing, normalization, and validated export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class PreprocessingResult:
    """Result of processing or skipping one CT volume."""

    input_path: Path
    output_path: Path
    metadata_path: Path
    status: str


def validate_ct_volume(ct: np.ndarray) -> None:
    """Validate a 3D numeric CT volume."""
    if not isinstance(ct, np.ndarray):
        raise TypeError(f"Expected a NumPy array, received {type(ct).__name__}.")
    if ct.ndim != 3:
        raise ValueError(f"Expected shape (slices, height, width), received {ct.shape}.")
    if ct.size == 0:
        raise ValueError("Expected a nonempty CT volume.")
    if not np.issubdtype(ct.dtype, np.number) or np.iscomplexobj(ct):
        raise TypeError(f"Expected a real numeric dtype, received {ct.dtype}.")
    if not np.isfinite(ct).all():
        raise ValueError("Expected finite CT values.")


def validate_median_filter_size(size: Sequence[int]) -> tuple[int, int, int]:
    """Require a slice-wise, odd-sized 3D median-filter kernel."""
    if len(size) != 3:
        raise ValueError(f"Expected three median-filter dimensions, received {size}.")
    normalized = tuple(int(value) for value in size)
    if any(value <= 0 or value % 2 == 0 for value in normalized):
        raise ValueError(f"Median-filter dimensions must be positive odd values: {size}.")
    if normalized[0] != 1:
        raise ValueError(
            "The first median-filter dimension must be 1 to avoid mixing slices."
        )
    return normalized


def median_filter_ct(
    ct: np.ndarray,
    size: Sequence[int] = (1, 3, 3),
) -> np.ndarray:
    """Reduce in-plane noise without mixing adjacent axial slices."""
    validate_ct_volume(ct)
    kernel = validate_median_filter_size(size)
    # SciPy's default reflect boundary behavior matches the parenchyma pipeline.
    return ndi.median_filter(ct, size=kernel)


def window_and_normalize_ct(
    ct: np.ndarray,
    window_level: float,
    window_width: float,
) -> np.ndarray:
    """Map a CT window to float32 values in the closed interval [0, 1]."""
    validate_ct_volume(ct)
    level = float(window_level)
    width = float(window_width)
    if not np.isfinite(level):
        raise ValueError("Window level must be finite.")
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"Window width must be finite and positive, received {width}.")

    lower = level - width / 2.0
    upper = level + width / 2.0
    result = ct.astype(np.float32, copy=True)
    np.clip(result, lower, upper, out=result)
    result -= lower
    result /= width
    return result


def preprocess_ct(
    ct: np.ndarray,
    *,
    window_level: float,
    window_width: float,
    median_filter_size: Sequence[int],
) -> np.ndarray:
    """Apply slice-wise median filtering before windowing and normalization."""
    filtered = median_filter_ct(ct, median_filter_size)
    result = window_and_normalize_ct(filtered, window_level, window_width)
    if result.shape != ct.shape:
        raise RuntimeError(f"Preprocessing changed shape {ct.shape} -> {result.shape}.")
    return result


def _load_source_metadata(input_path: Path) -> dict[str, Any]:
    metadata_path = input_path.with_suffix(".json")
    if not metadata_path.exists():
        return {"source_id": input_path.stem}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid input metadata JSON: {metadata_path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    return value


def save_processed_volume(
    ct: np.ndarray,
    input_path: str | Path,
    output_path: str | Path,
    *,
    window_level: float,
    window_width: float,
    median_filter_size: Sequence[int],
    overwrite: bool = False,
) -> PreprocessingResult:
    """Save normalized float32 data and a preprocessing metadata sidecar."""
    validate_ct_volume(ct)
    minimum = float(np.min(ct))
    maximum = float(np.max(ct))
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"Expected normalized values in [0, 1], got [{minimum}, {maximum}].")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy output path, received: {output_path}")
    metadata_path = output_path.with_suffix(".json")
    volume_exists = output_path.exists()
    metadata_exists = metadata_path.exists()
    if volume_exists != metadata_exists and not overwrite:
        raise FileExistsError(
            "Incomplete existing output pair; inspect it or use --overwrite: "
            f"{output_path}, {metadata_path}"
        )
    if volume_exists and metadata_exists and not overwrite:
        return PreprocessingResult(input_path, output_path, metadata_path, "skipped")

    kernel = validate_median_filter_size(median_filter_size)
    metadata = _load_source_metadata(input_path)
    metadata.update(
        {
            "source_volume": input_path.name,
            "volume_file": output_path.name,
            "shape_zyx": list(ct.shape),
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "preprocessing": {
                "order": ["median_filter", "window", "normalize"],
                "median_filter_size_zyx": list(kernel),
                "median_filter_mode": "reflect",
                "window_level_hu": float(window_level),
                "window_width_hu": float(window_width),
                "window_bounds_hu": [
                    float(window_level - window_width / 2.0),
                    float(window_level + window_width / 2.0),
                ],
            },
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.save(stream, ct.astype(np.float32, copy=False), allow_pickle=False)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PreprocessingResult(input_path, output_path, metadata_path, "written")


def preprocess_ct_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    window_level: float,
    window_width: float,
    median_filter_size: Sequence[int],
    overwrite: bool = False,
) -> PreprocessingResult:
    """Load, preprocess, and save one NumPy CT volume."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CT volume does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Expected a regular CT volume file: {input_path}")
    if input_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy input path, received: {input_path}")

    source = np.load(input_path, mmap_mode="r", allow_pickle=False)
    processed = preprocess_ct(
        source,
        window_level=window_level,
        window_width=window_width,
        median_filter_size=median_filter_size,
    )
    return save_processed_volume(
        processed,
        input_path,
        output_path,
        window_level=window_level,
        window_width=window_width,
        median_filter_size=median_filter_size,
        overwrite=overwrite,
    )


def discover_ct_files(input_dir: str | Path) -> list[Path]:
    """Discover NumPy CT volumes recursively in deterministic order."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Expected an input directory: {input_dir}")
    files = sorted(input_dir.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy CT volumes found in {input_dir}.")
    return files


def preprocess_ct_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    window_level: float,
    window_width: float,
    median_filter_size: Sequence[int],
    overwrite: bool = False,
) -> list[PreprocessingResult]:
    """Preprocess every CT volume while retaining relative directories."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = discover_ct_files(input_dir)
    return [
        preprocess_ct_file(
            path,
            output_dir / path.relative_to(input_dir),
            window_level=window_level,
            window_width=window_width,
            median_filter_size=median_filter_size,
            overwrite=overwrite,
        )
        for path in _progress(files)
    ]


def _progress(items: Sequence[Path]) -> Iterable[Path]:
    from tqdm import tqdm

    return tqdm(
        items,
        desc="Filtering and windowing CT",
        unit="volume",
        dynamic_ncols=True,
    )


def summarize(results: Sequence[PreprocessingResult]) -> str:
    """Return counts for written and skipped volumes."""
    written = sum(item.status == "written" for item in results)
    skipped = sum(item.status == "skipped" for item in results)
    return f"Total: {len(results)} | Written: {written} | Skipped: {skipped}"
