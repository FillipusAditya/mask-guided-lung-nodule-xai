"""Create one segmented-parenchyma and nodule-overlay PNG per CT study."""

import argparse
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from config import (
    LIDC_MASK_OUTPUT_DIR,
    LIDC_NODULE_MASK_DIR,
    LIDC_NODULE_METADATA_CSV,
    LIDC_PARENCHYMA_OUTPUT_DIR,
    LNDB_MASK_OUTPUT_DIR,
    LNDB_NODULE_MASK_DIR,
    LNDB_NODULE_METADATA_CSV,
    LNDB_PARENCHYMA_OUTPUT_DIR,
    QUALITY_CONTROL_DIR,
)


# High-saturation colors remain visible over grayscale lung parenchyma.
OVERLAY_COLORS = (
    "#FF1744", "#00E5FF", "#76FF03", "#D500F9",
    "#FFD600", "#2979FF", "#FF6D00", "#00E676",
    "#F50057", "#651FFF", "#00BFA5", "#C6FF00",
    "#FF3D00", "#00B0FF", "#AA00FF", "#00FF95",
)


def _load_volume(path: str | Path, description: str) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")
    volume = np.load(path, allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(
            f"{description} must have shape (N, H, W), received {volume.shape}."
        )
    return volume


def _load_slice_masks(
    directory: Path,
    expected_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Load full-size consensus masks indexed by axial slice number."""
    masks = {}
    if not directory.is_dir():
        return masks
    for path in sorted(directory.glob("slice_*.npy")):
        match = re.fullmatch(r"slice_(\d+)\.npy", path.name)
        if match is None:
            continue
        mask = np.load(path, allow_pickle=False).astype(bool)
        if mask.shape != expected_shape:
            raise ValueError(
                f"Nodule mask {path} has shape {mask.shape}; expected {expected_shape}."
            )
        masks[int(match.group(1))] = mask
    return masks


def _class_name(value: Any) -> str:
    label = str(value).strip().title()
    return label or "Unknown"


def load_lidc_findings(
    study_id: str,
    image_shape: tuple[int, int],
    metadata_csv: Path = LIDC_NODULE_METADATA_CSV,
    mask_root: Path = LIDC_NODULE_MASK_DIR,
) -> list[dict[str, Any]]:
    """Load all locally available consensus clusters for one LIDC study."""
    metadata = pd.read_csv(metadata_csv, dtype={"patient_id": str})
    study_keys = (
        metadata["patient_id"].astype(str)
        + "_"
        + metadata["study_instance_uid"].astype(str).str[-5:]
        + "_"
        + metadata["series_instance_uid"].astype(str).str[-5:]
    )
    selected = metadata.loc[study_keys == study_id].sort_values("cluster_id")
    findings = []
    for _, row in selected.iterrows():
        cluster_id = int(row["cluster_id"])
        masks = _load_slice_masks(
            Path(mask_root) / study_id / f"cluster_{cluster_id}", image_shape
        )
        if not masks:
            raise FileNotFoundError(
                f"Consensus masks are missing for {study_id} cluster {cluster_id}."
            )
        findings.append(
            {
                "finding_id": cluster_id,
                "display_id": f"Cluster {cluster_id}",
                "class": _class_name(row["label"]),
                "masks": masks,
            }
        )
    return findings


def load_lndb_findings(
    study_id: str,
    image_shape: tuple[int, int],
    metadata_csv: Path = LNDB_NODULE_METADATA_CSV,
    mask_root: Path = LNDB_NODULE_MASK_DIR,
) -> list[dict[str, Any]]:
    """Load all locally available consensus nodule IDs for one LNDb study."""
    match = re.fullmatch(r"LNDb-(\d+)", study_id)
    if match is None:
        raise ValueError(f"Invalid LNDb study ID: {study_id}")
    scan_id = int(match.group(1))
    metadata = pd.read_csv(metadata_csv)
    selected = metadata.loc[
        pd.to_numeric(metadata["lndbid"], errors="coerce") == scan_id
    ].copy()
    selected["findingid"] = pd.to_numeric(
        selected.get("findingid"), errors="coerce"
    )
    selected = selected.dropna(subset=["findingid"]).sort_values("findingid")
    findings = []
    for _, row in selected.iterrows():
        finding_id = int(row["findingid"])
        masks = _load_slice_masks(
            Path(mask_root) / study_id / f"finding_{finding_id}", image_shape
        )
        if not masks:
            raise FileNotFoundError(
                f"Consensus masks are missing for {study_id} nodule {finding_id}."
            )
        findings.append(
            {
                "finding_id": finding_id,
                "display_id": f"Nodule {finding_id}",
                "class": _class_name(row["label"]),
                "masks": masks,
            }
        )
    return findings


def calculate_inclusion(
    lung_mask: np.ndarray,
    nodule_masks: dict[int, np.ndarray],
) -> tuple[float, dict[int, float]]:
    """Return whole-finding and per-slice nodule inclusion percentages."""
    inside_total = 0
    nodule_total = 0
    per_slice = {}
    for slice_index, nodule_mask in nodule_masks.items():
        if not 0 <= slice_index < len(lung_mask):
            raise IndexError(f"Nodule slice {slice_index} is outside the lung volume.")
        nodule_mask = np.asarray(nodule_mask, dtype=bool)
        total = int(nodule_mask.sum())
        inside = int(np.logical_and(lung_mask[slice_index], nodule_mask).sum())
        per_slice[slice_index] = 100.0 * inside / total if total else 0.0
        inside_total += inside
        nodule_total += total
    overall = 100.0 * inside_total / nodule_total if nodule_total else 0.0
    return overall, per_slice


def save_study_overlay(
    study_id: str,
    lung_mask: np.ndarray,
    parenchyma: np.ndarray,
    findings: Sequence[dict[str, Any]],
    output_path: str | Path,
    columns: int = 3,
) -> list[dict[str, Any]]:
    """Save one PNG containing every annotated slice from one CT study."""
    if lung_mask.shape != parenchyma.shape or lung_mask.ndim != 3:
        raise ValueError("Lung mask and parenchyma must share shape (N, H, W).")
    if columns < 1:
        raise ValueError("columns must be at least 1.")

    if not findings:
        raise ValueError("At least one consensus nodule is required for QC.")

    if len(findings) > len(OVERLAY_COLORS):
        raise ValueError(f"Only {len(OVERLAY_COLORS)} overlay colors are available.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    statistics = []
    for finding in findings:
        overall, per_slice = calculate_inclusion(lung_mask, finding["masks"])
        statistics.append(
            {
                "finding_id": finding["finding_id"],
                "display_id": finding["display_id"],
                "class": finding["class"],
                "inclusion_percent": overall,
                "per_slice": per_slice,
            }
        )

    slice_indices = sorted(
        {slice_index for finding in findings for slice_index in finding["masks"]}
    )
    used_columns = min(columns, len(slice_indices))
    rows = (len(slice_indices) + used_columns - 1) // used_columns
    legend_columns = min(3, len(statistics))
    legend_rows = (len(statistics) + legend_columns - 1) // legend_columns
    # Reserve a fixed physical header height. This prevents overlap on short
    # studies without creating an increasingly large gap on tall studies.
    header_inches = 1.10 + 0.32 * legend_rows
    figure_height = 4.6 * rows + header_inches
    fig, axes = plt.subplots(
        rows,
        used_columns,
        figsize=(5.2 * used_columns, figure_height),
        squeeze=False,
    )

    for position, slice_index in enumerate(slice_indices):
        axis = axes[position // used_columns, position % used_columns]
        axis.imshow(parenchyma[slice_index], cmap="gray", vmin=0.0, vmax=1.0)
        visible_labels = []
        for finding_index, finding in enumerate(findings):
            nodule_mask = finding["masks"].get(slice_index)
            if nodule_mask is None:
                continue
            color = OVERLAY_COLORS[finding_index]
            overlay = np.zeros((*nodule_mask.shape, 4), dtype=np.float32)
            overlay[nodule_mask] = (*to_rgb(color), 0.78)
            axis.imshow(overlay)
            axis.contour(
                nodule_mask, levels=[0.5], colors=[color], linewidths=2.2
            )
            inclusion = statistics[finding_index]["per_slice"][slice_index]
            visible_labels.append(
                f"{finding['display_id']} — {finding['class']} — {inclusion:.1f}%"
            )
        # Every panel has exactly two title lines for consistent spacing.
        axis.set_title(
            f"Slice {slice_index}\n" + " | ".join(visible_labels),
            fontsize=10,
            pad=4,
        )
        axis.axis("off")

    for position in range(len(slice_indices), rows * used_columns):
        axes[position // used_columns, position % used_columns].axis("off")

    legend = [
        Patch(
            facecolor=OVERLAY_COLORS[index],
            edgecolor=OVERLAY_COLORS[index],
            alpha=0.78,
            label=(
                f"{stat['display_id']} — {stat['class']} — "
                f"overall inclusion {stat['inclusion_percent']:.1f}%"
            ),
        )
        for index, stat in enumerate(statistics)
    ]
    fig.suptitle(
        f"{study_id} — Segmented Lung Parenchyma + Nodule Overlay",
        fontsize=16,
        y=1.0 - 0.05 / figure_height,
    )
    fig.legend(
        handles=legend,
        title="Nodule ID / diagnosis / inclusion",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - 0.42 / figure_height),
        ncol=legend_columns,
    )
    content_top = 1.0 - header_inches / figure_height
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=content_top,
        wspace=0.08,
        hspace=0.20,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return statistics


def _available_studies(dataset: str) -> list[tuple[str, Path, Path]]:
    if dataset == "lidc":
        mask_dir, parenchyma_dir = LIDC_MASK_OUTPUT_DIR, LIDC_PARENCHYMA_OUTPUT_DIR
    else:
        mask_dir, parenchyma_dir = LNDB_MASK_OUTPUT_DIR, LNDB_PARENCHYMA_OUTPUT_DIR
    studies = []
    for parenchyma_path in sorted(parenchyma_dir.glob("*_parenchyma.npy")):
        study_id = parenchyma_path.name.removesuffix("_parenchyma.npy")
        mask_path = mask_dir / f"{study_id}_mask.npy"
        if mask_path.is_file():
            studies.append((study_id, mask_path, parenchyma_path))
    return studies


def create_study_quality_control(
    dataset: str,
    study_id: str,
    mask_path: Path,
    parenchyma_path: Path,
    output_path: Path,
    columns: int = 3,
) -> list[dict[str, Any]] | None:
    lung_mask = _load_volume(mask_path, "Final lung mask").astype(bool)
    parenchyma = _load_volume(parenchyma_path, "Segmented lung parenchyma")
    if lung_mask.shape != parenchyma.shape:
        raise ValueError(f"Mask and parenchyma shapes differ for {study_id}.")
    if dataset == "lidc":
        findings = load_lidc_findings(study_id, lung_mask.shape[1:])
    else:
        findings = load_lndb_findings(study_id, lung_mask.shape[1:])
    if not findings:
        return None
    return save_study_overlay(
        study_id, lung_mask, parenchyma, findings, output_path, columns
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("lidc", "lndb", "all"))
    parser.add_argument("--patient-id", help="Filter LIDC, e.g. LIDC-IDRI-0001.")
    parser.add_argument("--scan-id", type=int, help="Filter LNDb, e.g. 1.")
    parser.add_argument("--output", type=Path, help="Custom PNG path for one study.")
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    datasets = ("lidc", "lndb") if args.dataset == "all" else (args.dataset,)
    studies = []
    for dataset in datasets:
        for study_id, mask_path, parenchyma_path in _available_studies(dataset):
            if args.patient_id and not study_id.startswith(args.patient_id + "_"):
                continue
            if args.scan_id is not None and study_id != f"LNDb-{args.scan_id:04d}":
                continue
            studies.append((dataset, study_id, mask_path, parenchyma_path))

    if not studies:
        raise FileNotFoundError(
            "No matching saved V2 mask/parenchyma pair was found. Run run_pipeline.py first."
        )
    if args.output is not None and len(studies) != 1:
        parser.error("--output can only be used when exactly one study is selected.")

    statuses = []
    for dataset, study_id, mask_path, parenchyma_path in studies:
        output_path = args.output or (
            QUALITY_CONTROL_DIR / dataset / f"{study_id}_nodule_overlay.png"
        )
        try:
            statistics = create_study_quality_control(
                dataset,
                study_id,
                mask_path,
                parenchyma_path,
                output_path,
                args.columns,
            )
            if statistics is None:
                state = "skipped_no_nodules"
                if args.output is None and output_path.is_file():
                    output_path.unlink()
                    print(f"removed_stale_empty_png: {output_path}")
                print(f"{state}: {study_id}")
                statuses.append(state)
                continue

            state = "written"
            print(f"{state}: {output_path}")
            for statistic in statistics:
                print(
                    f"  {statistic['display_id']}: {statistic['class']}, "
                    f"inclusion={statistic['inclusion_percent']:.1f}%"
                )
        except Exception as error:
            state = "failed"
            print(f"failed: {study_id}: {type(error).__name__}: {error}")
        statuses.append(state)

    counts = Counter(statuses)
    print(f"QC summary: {dict(counts)}")
    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
