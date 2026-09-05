"""Export normalized CT volumes as 8-bit grayscale PNG slices."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocess import validate_ct_volume


@dataclass(frozen=True)
class PngExportResult:
    """Result of exporting or skipping one normalized CT volume."""

    volume_path: Path
    output_dir: Path
    slice_count: int
    status: str


def normalized_to_uint8(volume: np.ndarray) -> np.ndarray:
    """Map a normalized CT volume from [0, 1] to uint8 [0, 255]."""
    validate_ct_volume(volume)
    minimum = float(np.min(volume))
    maximum = float(np.max(volume))
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"Expected normalized values in [0, 1], got [{minimum}, {maximum}]."
        )

    normalized = np.clip(volume.astype(np.float32, copy=False), 0.0, 1.0)
    return np.rint(normalized * 255.0).astype(np.uint8)


def export_volume_png(
    volume: np.ndarray,
    output_dir: str | Path,
    *,
    volume_path: str | Path = "<memory>",
    overwrite: bool = False,
) -> PngExportResult:
    """Export every axial slice from one normalized CT volume."""
    png_volume = normalized_to_uint8(volume)
    output_dir = Path(output_dir)
    padding = max(4, len(str(png_volume.shape[0] - 1)))
    output_paths = [
        output_dir / f"slice_{index:0{padding}d}.png"
        for index in range(png_volume.shape[0])
    ]
    existing = [path for path in output_paths if path.exists()]

    if existing and len(existing) != len(output_paths) and not overwrite:
        raise FileExistsError(
            "Incomplete PNG output; use --overwrite after inspecting it: "
            f"{output_dir}"
        )
    if len(existing) == len(output_paths) and not overwrite:
        return PngExportResult(
            Path(volume_path),
            output_dir,
            len(output_paths),
            "skipped",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for slice_image, output_path in zip(png_volume, output_paths):
        Image.fromarray(slice_image).save(output_path)

    return PngExportResult(
        Path(volume_path),
        output_dir,
        len(output_paths),
        "written",
    )


def export_volume_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PngExportResult:
    """Load one normalized NumPy CT volume and export its PNG slices."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Normalized CT volume was not found: {input_path}")
    if input_path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy CT volume, received: {input_path}")

    volume = np.load(input_path, allow_pickle=False)
    return export_volume_png(
        volume,
        output_dir,
        volume_path=input_path,
        overwrite=overwrite,
    )


def export_volume_files(
    volume_paths: Sequence[str | Path],
    input_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> list[PngExportResult]:
    """Export selected volumes while preserving relative directories."""
    input_root = Path(input_root)
    output_root = Path(output_root)
    results = []

    for volume_path in tqdm(
        volume_paths,
        desc="Exporting windowed PNG",
        unit="volume",
        dynamic_ncols=True,
    ):
        volume_path = Path(volume_path)
        try:
            relative_path = volume_path.relative_to(input_root)
        except ValueError as error:
            raise ValueError(
                f"Volume is outside the input root: {volume_path}"
            ) from error

        output_dir = output_root / relative_path.with_suffix("")
        results.append(
            export_volume_file(
                volume_path,
                output_dir,
                overwrite=overwrite,
            )
        )

    return results


def summarize_png(results: Sequence[PngExportResult]) -> str:
    """Return counts for written and skipped PNG volumes."""
    written = sum(item.status == "written" for item in results)
    skipped = sum(item.status == "skipped" for item in results)
    slices = sum(item.slice_count for item in results if item.status == "written")
    return (
        f"Volumes: {len(results)} | Written: {written} | "
        f"Skipped: {skipped} | PNG slices written: {slices}"
    )


def parse_args() -> argparse.Namespace:
    """Parse standalone PNG export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Export one normalized CT volume from the command line."""
    args = parse_args()
    result = export_volume_file(
        args.input_file,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print(summarize_png([result]))


if __name__ == "__main__":
    main()
