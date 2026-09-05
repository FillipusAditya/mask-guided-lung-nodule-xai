"""Segmented-nodule PNG and consensus quality-control exports for LIDC."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image
from pylidc.utils import consensus
from scipy import ndimage as ndi

from .compat import enable_pylidc_numpy_compatibility


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


def prepare_cluster_artifact(
    annotation_cluster: Sequence[Any],
    normalized_ct: np.ndarray,
    consensus_level: float = 0.5,
) -> dict[str, Any]:
    """Build a pylidc consensus, agreement map, and aligned normalized CT crop."""
    if not annotation_cluster:
        raise ValueError("annotation_cluster must contain at least one annotation.")
    if not 0 < consensus_level <= 1:
        raise ValueError("consensus_level must be between 0 and 1.")

    enable_pylidc_numpy_compatibility()
    consensus_yxz, bbox_yxz, masks_yxz = consensus(
        annotation_cluster,
        clevel=consensus_level,
        ret_masks=True,
    )
    y_box, x_box, z_box = bbox_yxz
    ct_crop = normalized_ct[z_box, y_box, x_box]
    consensus_mask = np.transpose(consensus_yxz, (2, 0, 1)).astype(bool)
    mask_stack = np.stack(
        [np.transpose(mask, (2, 0, 1)) for mask in masks_yxz],
        axis=0,
    )
    agreement_map = np.sum(mask_stack, axis=0)

    if ct_crop.shape != consensus_mask.shape:
        raise ValueError("Normalized CT crop and pylidc consensus shapes must match.")

    return {
        "ct_crop": ct_crop,
        "consensus_mask": consensus_mask,
        "agreement_map": agreement_map,
        "slice_offset": int(z_box.start),
        "annotation_count": len(annotation_cluster),
    }


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
    artifact: dict[str, Any],
    output_dir: str | Path,
    overwrite: bool = False,
) -> ArtifactResult:
    """Save cropped consensus-nodule intensities as 8-bit PNG slices."""
    consensus_mask = artifact["consensus_mask"]
    ct_crop = artifact["ct_crop"]
    local_slices = np.where(np.any(consensus_mask, axis=(1, 2)))[0]
    output_dir = Path(output_dir)
    output_paths = [
        output_dir / f"slice_{int(local_z + artifact['slice_offset'])}.png"
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
            ct_crop[local_z],
            np.float32(0.0),
        )
        image = np.rint(segmented * 255.0).astype(np.uint8)
        Image.fromarray(image).save(output_path)

    return ArtifactResult(tuple(output_paths), "written")


def show_agreement_map(
    artifact: dict[str, Any],
    title: str,
    save_path: str | Path,
) -> None:
    """Save pylidc annotation agreement over the normalized CT crop."""
    agreement_map = artifact["agreement_map"]
    slices = np.where(np.any(agreement_map > 0, axis=(1, 2)))[0]
    if len(slices) == 0:
        raise ValueError("No agreement voxels were produced by pylidc.")

    count = int(artifact["annotation_count"])
    colors = ["#4DB6AC", "#FFD54F", "#D32F2F", "#7B1FA2"]
    cmap = ListedColormap(colors[:count])
    norm = BoundaryNorm(np.arange(0.5, count + 1.5, 1), cmap.N)
    columns = min(5, len(slices))
    rows = math.ceil(len(slices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    image = None
    for axis, local_z in zip(axes, slices):
        axis.imshow(artifact["ct_crop"][local_z], cmap="gray", vmin=0, vmax=1)
        image = axis.imshow(
            np.ma.masked_where(agreement_map[local_z] == 0, agreement_map[local_z]),
            cmap=cmap,
            norm=norm,
            alpha=0.6,
            interpolation="none",
        )
        axis.set_title(f"Slice {local_z + artifact['slice_offset']}")
        axis.axis("off")
    for axis in axes[len(slices):]:
        axis.axis("off")

    fig.suptitle(f"{title} | Agreement Map", fontsize=14)
    colorbar = fig.colorbar(
        image,
        ax=axes.tolist(),
        ticks=np.arange(1, count + 1),
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Agreement Level")
    fig.subplots_adjust(top=0.88, right=0.90, wspace=0.08, hspace=0.20)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def show_consensus_mask(
    artifact: dict[str, Any],
    title: str,
    save_path: str | Path,
) -> None:
    """Save the pylidc consensus mask over the normalized CT crop."""
    consensus_mask = artifact["consensus_mask"]
    slices = np.where(np.any(consensus_mask, axis=(1, 2)))[0]
    if len(slices) == 0:
        raise ValueError("No consensus voxels were produced by pylidc.")

    columns = min(5, len(slices))
    rows = math.ceil(len(slices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, local_z in zip(axes, slices):
        axis.imshow(artifact["ct_crop"][local_z], cmap="gray", vmin=0, vmax=1)
        axis.imshow(
            np.ma.masked_where(~consensus_mask[local_z], consensus_mask[local_z]),
            cmap="spring",
            alpha=0.5,
            interpolation="none",
        )
        axis.set_title(f"Slice {local_z + artifact['slice_offset']}")
        axis.axis("off")
    for axis in axes[len(slices):]:
        axis.axis("off")

    fig.suptitle(f"{title} | Consensus Mask", fontsize=14)
    fig.subplots_adjust(top=0.88, wspace=0.10, hspace=0.25)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_consensus_quality_control(
    artifact: dict[str, Any],
    title: str,
    output_dir: str | Path,
    cluster_id: int,
    overwrite: bool = False,
) -> ArtifactResult:
    """Save agreement-map and consensus-mask canvases for one LIDC cluster."""
    output_dir = Path(output_dir)
    output_paths = [
        output_dir / f"cluster_{cluster_id}_agreement_map.png",
        output_dir / f"cluster_{cluster_id}_consensus_mask.png",
    ]
    existing_result = _resolve_existing_outputs(output_paths, overwrite)
    if existing_result is not None:
        return existing_result

    show_agreement_map(artifact, title, output_paths[0])
    show_consensus_mask(artifact, title, output_paths[1])
    return ArtifactResult(tuple(output_paths), "written")
