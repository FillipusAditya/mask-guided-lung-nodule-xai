"""Segmented-nodule PNG and consensus quality-control exports."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .visualize import show_agreement_map, show_consensus_mask


@dataclass(frozen=True)
class ArtifactResult:
    """Result of writing or skipping one artifact group."""

    output_paths: tuple[Path, ...]
    status: str


def preprocess_ct_for_display(
    ct_volume: np.ndarray,
    window_level: float = -600.0,
    window_width: float = 1600.0,
    median_filter_size: tuple[int, int, int] = (1, 3, 3),
) -> np.ndarray:
    """Median-filter, window, and normalize a CT volume to float32 [0, 1]."""
    if ct_volume.ndim != 3:
        raise ValueError(f"Expected shape (N, H, W), received {ct_volume.shape}.")
    if median_filter_size[0] != 1:
        raise ValueError("Median filtering must not mix adjacent axial slices.")
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    filtered = ndi.median_filter(ct_volume, size=median_filter_size)
    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    normalized = filtered.astype(np.float32, copy=True)
    np.clip(normalized, lower, upper, out=normalized)
    normalized -= lower
    normalized /= upper - lower
    return normalized


def _resolve_existing_outputs(
    output_paths: list[Path],
    overwrite: bool,
) -> ArtifactResult | None:
    """Skip complete outputs and reject partial outputs unless overwriting."""
    existing = [path for path in output_paths if path.exists()]
    if existing and len(existing) != len(output_paths) and not overwrite:
        raise FileExistsError(
            "Incomplete artifact output; use --overwrite after inspecting it: "
            f"{output_paths[0].parent}"
        )
    if len(existing) == len(output_paths) and not overwrite:
        return ArtifactResult(tuple(output_paths), "skipped")
    return None


def save_segmented_nodule_png(
    scan: dict,
    normalized_ct: np.ndarray,
    output_dir: str | Path,
    overwrite: bool = False,
) -> ArtifactResult:
    """Save cropped consensus-nodule intensities as 8-bit PNG slices."""
    if normalized_ct.shape != scan["ct_volume"].shape:
        raise ValueError("Normalized CT and source CT shapes must match.")

    output_dir = Path(output_dir)
    bbox = scan["consensus_bbox"]
    normalized_crop = normalized_ct[
        bbox["zmin"]:bbox["zmax"] + 1,
        bbox["ymin"]:bbox["ymax"] + 1,
        bbox["xmin"]:bbox["xmax"] + 1,
    ]
    consensus_mask = np.asarray(scan["consensus_mask"], dtype=bool)
    if normalized_crop.shape != consensus_mask.shape:
        raise ValueError("Normalized CT crop and consensus mask shapes must match.")

    local_slices = np.where(np.any(consensus_mask, axis=(1, 2)))[0]
    output_paths = [
        output_dir / f"slice_{int(local_z + bbox['zmin'])}.png"
        for local_z in local_slices
    ]
    if not output_paths:
        raise ValueError("The consensus mask contains no nodule slices.")

    existing_result = _resolve_existing_outputs(output_paths, overwrite)
    if existing_result is not None:
        return existing_result

    output_dir.mkdir(parents=True, exist_ok=True)
    for local_z, output_path in zip(local_slices, output_paths):
        segmented = np.where(
            consensus_mask[local_z],
            normalized_crop[local_z],
            np.float32(0.0),
        )
        image = np.rint(segmented * 255.0).astype(np.uint8)
        Image.fromarray(image).save(output_path)

    return ArtifactResult(tuple(output_paths), "written")


def save_consensus_quality_control(
    scan: dict,
    normalized_ct: np.ndarray,
    output_dir: str | Path,
    overwrite: bool = False,
) -> ArtifactResult:
    """Save agreement-map and consensus-mask canvases for one finding."""
    if normalized_ct.shape != scan["ct_volume"].shape:
        raise ValueError("Normalized CT and source CT shapes must match.")

    output_dir = Path(output_dir)
    finding_id = int(scan["finding_id"])
    output_paths = [
        output_dir / f"finding_{finding_id}_agreement_map.png",
        output_dir / f"finding_{finding_id}_consensus_mask.png",
    ]
    existing_result = _resolve_existing_outputs(output_paths, overwrite)
    if existing_result is not None:
        return existing_result

    bbox = scan["consensus_bbox"]
    display_scan = {
        **scan,
        "ct_crop": normalized_ct[
            bbox["zmin"]:bbox["zmax"] + 1,
            bbox["ymin"]:bbox["ymax"] + 1,
            bbox["xmin"]:bbox["xmax"] + 1,
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    show_agreement_map(display_scan, save_path=output_paths[0])
    show_consensus_mask(display_scan, save_path=output_paths[1])
    return ArtifactResult(tuple(output_paths), "written")
