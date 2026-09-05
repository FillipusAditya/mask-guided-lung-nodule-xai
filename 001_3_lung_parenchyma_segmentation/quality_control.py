"""Visualize consensus nodules over final lung-parenchyma volumes."""

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pylidc as pl

from config import (
    LIDC_CLUSTER_METADATA_CSV,
    LIDC_OUTPUT_DIR,
    LNDB_FINDING_METADATA_CSV,
    LNDB_INPUT_DIR,
    LNDB_MASK_DIR,
    LNDB_OUTPUT_DIR,
    PROJECT_ROOT,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)
from step_1_ct_to_numpy import load_lidc_scan


# These project directories start with digits, so expose their package folders
# before importing their reusable consensus operations.
for module_dir in (
    PROJECT_ROOT / "001_4_lidc_consensus",
    PROJECT_ROOT / "001_5_lndb_consensus",
):
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from lidc_consensus import compute_consensus_slices, scan_directory_name  # noqa: E402
from lndb_consensus import prepare_scan_data, process_scan  # noqa: E402


OVERLAY_COLORS = (
    "#FF1744",  # Red
    "#00E5FF",  # Cyan
    "#76FF03",  # Lime
    "#D500F9",  # Magenta
    "#FFD600",  # Yellow
    "#2979FF",  # Blue
    "#FF6D00",  # Orange
    "#00E676",  # Green
    "#F50057",  # Pink
    "#651FFF",  # Violet
    "#00BFA5",  # Teal
    "#C6FF00",  # Chartreuse
    "#FF3D00",  # Bright orange-red
    "#00B0FF",  # Light blue
    "#AA00FF",  # Purple
    "#00FF95",  # Spring green
)


def _load_volume(path: str | Path, description: str) -> np.ndarray:
    """Load and validate one 3D NumPy volume."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found: {path}")

    volume = np.load(path, allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(
            f"{description} must have shape (N, H, W), received {volume.shape}."
        )
    return volume


def _selected_nodule_slices(
    findings: Sequence[Mapping[str, Any]],
    slice_indices: Sequence[int] | None,
) -> list[int]:
    """Return all nodule slices or validate a user-selected subset."""
    available = sorted(
        {
            int(slice_index)
            for finding in findings
            for slice_index in finding["masks"]
        }
    )
    if not available:
        raise ValueError("The consensus operations produced no nodule slices.")

    if slice_indices is None:
        return available

    selected = list(dict.fromkeys(int(index) for index in slice_indices))
    invalid = sorted(set(selected).difference(available))
    if invalid:
        raise ValueError(
            f"Selected slices do not contain a consensus nodule: {invalid}. "
            f"Available slices: {available}"
        )
    return selected


def show_nodule_overlays(
    ct_volume: np.ndarray,
    lung_parenchyma: np.ndarray,
    findings: Sequence[Mapping[str, Any]],
    title: str,
    slice_indices: Sequence[int] | None = None,
    output_path: str | Path | None = None,
) -> None:
    """Show full CT and colored nodule overlays on lung parenchyma.

    Each finding must contain ``finding_id``, ``class``, and ``masks``. The
    ``masks`` value maps axial slice indices to full-size two-dimensional masks.
    """
    if ct_volume.ndim != 3 or lung_parenchyma.ndim != 3:
        raise ValueError("CT and lung parenchyma must both have shape (N, H, W).")
    if ct_volume.shape != lung_parenchyma.shape:
        raise ValueError(
            "CT and lung-parenchyma shapes must match: "
            f"{ct_volume.shape} != {lung_parenchyma.shape}."
        )
    if not findings:
        raise ValueError("No nodule findings are available for visualization.")
    if len(findings) > len(OVERLAY_COLORS):
        raise ValueError(
            f"A maximum of {len(OVERLAY_COLORS)} findings can be displayed."
        )

    expected_mask_shape = ct_volume.shape[1:]
    for finding in findings:
        for slice_index, mask in finding["masks"].items():
            if not 0 <= int(slice_index) < ct_volume.shape[0]:
                raise IndexError(f"Nodule slice is outside the CT volume: {slice_index}")
            if np.asarray(mask).shape != expected_mask_shape:
                raise ValueError(
                    f"Finding {finding['finding_id']} mask shape "
                    f"{np.asarray(mask).shape} does not match {expected_mask_shape}."
                )

    selected_slices = _selected_nodule_slices(findings, slice_indices)
    slice_columns = min(3, len(selected_slices))
    rows = (len(selected_slices) + slice_columns - 1) // slice_columns
    figure_height = 4 * rows + 1
    fig, axes = plt.subplots(
        rows,
        2 * slice_columns,
        figsize=(6 * slice_columns, figure_height),
        squeeze=False,
    )

    lower_bound = WINDOW_LEVEL - (WINDOW_WIDTH / 2.0)
    upper_bound = WINDOW_LEVEL + (WINDOW_WIDTH / 2.0)

    for position, slice_index in enumerate(selected_slices):
        row = position // slice_columns
        pair_column = 2 * (position % slice_columns)
        ct_axis = axes[row, pair_column]
        overlay_axis = axes[row, pair_column + 1]

        ct_axis.imshow(
            ct_volume[slice_index],
            cmap="gray",
            vmin=lower_bound,
            vmax=upper_bound,
        )
        overlay_axis.imshow(
            lung_parenchyma[slice_index],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        visible_ids = []
        for finding_index, finding in enumerate(findings):
            nodule_mask = finding["masks"].get(slice_index)
            if nodule_mask is None:
                continue

            nodule_mask = np.asarray(nodule_mask, dtype=bool)
            color = OVERLAY_COLORS[finding_index]
            overlay = np.zeros((*nodule_mask.shape, 4), dtype=np.float32)
            overlay[nodule_mask] = (*to_rgb(color), 0.65)
            overlay_axis.imshow(overlay)
            overlay_axis.contour(
                nodule_mask,
                levels=[0.5],
                colors=[color],
                linewidths=2.0,
            )
            visible_ids.append(str(finding["finding_id"]))

        image_title = f"Slice {slice_index} | Finding {', '.join(visible_ids)}"
        ct_axis.set_title(f"Full CT\n{image_title}")
        overlay_axis.set_title(f"Lung Parenchyma + Nodule Overlay\n{image_title}")
        ct_axis.axis("off")
        overlay_axis.axis("off")

    for position in range(len(selected_slices), rows * slice_columns):
        row = position // slice_columns
        pair_column = 2 * (position % slice_columns)
        axes[row, pair_column].axis("off")
        axes[row, pair_column + 1].axis("off")

    visible_finding_indices = [
        index
        for index, finding in enumerate(findings)
        if any(slice_index in finding["masks"] for slice_index in selected_slices)
    ]
    legend_handles = [
        Patch(
            facecolor=OVERLAY_COLORS[index],
            edgecolor=OVERLAY_COLORS[index],
            alpha=0.65,
            label=(
                f"Finding {findings[index]['finding_id']} — "
                f"{findings[index]['class']}"
            ),
        )
        for index in visible_finding_indices
    ]

    title_y = 1.0 - (0.05 / figure_height)
    legend_y = 1.0 - (0.45 / figure_height)
    content_top = 1.0 - (1.10 / figure_height)
    fig.suptitle(title, fontsize=16, y=title_y)
    fig.legend(
        handles=legend_handles,
        title="Nodule class",
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=min(4, len(legend_handles)),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, content_top))

    if output_path is None:
        plt.show()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Quality-control figure saved: {output_path}")


def prepare_lidc_quality_control(
    patient_id: str,
    metadata_csv: str | Path = LIDC_CLUSTER_METADATA_CSV,
    parenchyma_file: str | Path | None = None,
    consensus_level: float = 0.5,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str]:
    """Load one LIDC scan and calculate its consensus masks through pylidc."""
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"LIDC metadata was not found: {metadata_csv}")

    metadata = pd.read_csv(metadata_csv, dtype={"patient_id": str})
    selected = metadata.loc[metadata["patient_id"] == patient_id].copy()
    if study_instance_uid is not None:
        selected = selected.loc[
            selected["study_instance_uid"].astype(str) == study_instance_uid
        ]
    if series_instance_uid is not None:
        selected = selected.loc[
            selected["series_instance_uid"].astype(str) == series_instance_uid
        ]
    if selected.empty:
        raise ValueError(f"No LIDC consensus metadata found for {patient_id}.")

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    matching_scans = [
        scan
        for scan in scans
        if np.any(
            (selected["study_instance_uid"].astype(str) == str(scan.study_instance_uid))
            & (
                selected["series_instance_uid"].astype(str)
                == str(scan.series_instance_uid)
            )
        )
    ]
    if len(matching_scans) != 1:
        raise ValueError(
            f"Expected one matching LIDC scan for {patient_id}, found "
            f"{len(matching_scans)}. Supply the study and series UIDs if needed."
        )

    scan = matching_scans[0]
    selected = selected.loc[
        (selected["study_instance_uid"].astype(str) == str(scan.study_instance_uid))
        & (
            selected["series_instance_uid"].astype(str)
            == str(scan.series_instance_uid)
        )
    ].sort_values("cluster_id")

    ct_volume = load_lidc_scan(scan)
    if parenchyma_file is None:
        parenchyma_file = LIDC_OUTPUT_DIR / f"{scan_directory_name(scan)}.npy"
    parenchyma = _load_volume(parenchyma_file, "LIDC lung parenchyma")

    annotation_clusters = scan.cluster_annotations()
    findings = []
    for _, row in selected.iterrows():
        cluster_id = int(row["cluster_id"])
        if not 0 <= cluster_id < len(annotation_clusters):
            raise IndexError(
                f"{patient_id} cluster {cluster_id} is unavailable in pylidc."
            )

        masks = compute_consensus_slices(
            annotation_cluster=annotation_clusters[cluster_id],
            image_shape=ct_volume.shape[1:],
            consensus_level=consensus_level,
        )
        findings.append(
            {
                "finding_id": cluster_id,
                "class": str(row["label"]),
                "masks": masks,
            }
        )

    title = f"{patient_id} — Nodule Mask Overlays"
    return ct_volume, parenchyma, findings, title


def prepare_lndb_quality_control(
    scan_id: int,
    metadata_csv: str | Path = LNDB_FINDING_METADATA_CSV,
    data_dir: str | Path = LNDB_INPUT_DIR,
    mask_dir: str | Path = LNDB_MASK_DIR,
    parenchyma_file: str | Path | None = None,
    consensus_level: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], str]:
    """Load one LNDb scan and calculate each finding through its consensus API."""
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"LNDb metadata was not found: {metadata_csv}")

    metadata = pd.read_csv(metadata_csv)
    scan_ids = pd.to_numeric(metadata["lndbid"], errors="raise").astype(int)
    selected = metadata.loc[scan_ids == int(scan_id)].copy()
    if selected.empty:
        raise ValueError(f"No LNDb consensus metadata found for scan {scan_id}.")

    selected["findingid"] = pd.to_numeric(
        selected["findingid"],
        errors="raise",
    ).astype(int)
    selected = selected.sort_values("findingid")

    if parenchyma_file is None:
        parenchyma_file = LNDB_OUTPUT_DIR / f"LNDb-{scan_id:04d}.npy"
    parenchyma = _load_volume(parenchyma_file, "LNDb lung parenchyma")

    ct_volume = None
    findings = []
    for _, row in selected.iterrows():
        scan = prepare_scan_data(
            row=row,
            data_dir=data_dir,
            mask_dir=mask_dir,
        )
        scan = process_scan(scan, clevel=consensus_level)
        if ct_volume is None:
            ct_volume = np.asarray(scan["ct_volume"])

        masks = {
            int(slice_index): np.asarray(
                scan["consensus_mask_full"][slice_index],
                dtype=bool,
            ).copy()
            for slice_index in scan["consensus_slices"]
        }
        findings.append(
            {
                "finding_id": int(scan["finding_id"]),
                "class": str(scan["label"]),
                "masks": masks,
            }
        )

    if ct_volume is None:
        raise RuntimeError(f"LNDb-{scan_id:04d} did not produce a CT volume.")
    title = f"LNDb-{scan_id:04d} — Nodule Mask Overlays"
    return ct_volume, parenchyma, findings, title


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options shared by the LIDC and LNDb QC commands."""
    parser.add_argument("--parenchyma-file", type=Path)
    parser.add_argument(
        "--consensus-level",
        type=float,
        default=0.5,
        help="Required radiologist agreement fraction in the interval (0, 1].",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        help="Optional nodule slice indices. By default all nodule slices are shown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save the figure instead of displaying it interactively.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the quality-control command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    lidc = subparsers.add_parser("lidc", help="Inspect one LIDC-IDRI patient.")
    lidc.add_argument("--patient-id", required=True)
    lidc.add_argument("--metadata-csv", type=Path, default=LIDC_CLUSTER_METADATA_CSV)
    lidc.add_argument("--study-instance-uid")
    lidc.add_argument("--series-instance-uid")
    _add_common_arguments(lidc)

    lndb = subparsers.add_parser("lndb", help="Inspect one LNDb scan.")
    lndb.add_argument("--scan-id", type=int, required=True)
    lndb.add_argument("--metadata-csv", type=Path, default=LNDB_FINDING_METADATA_CSV)
    lndb.add_argument("--data-dir", type=Path, default=LNDB_INPUT_DIR)
    lndb.add_argument("--mask-dir", type=Path, default=LNDB_MASK_DIR)
    _add_common_arguments(lndb)
    return parser


def main() -> None:
    """Prepare consensus masks and display or save one QC canvas."""
    args = build_parser().parse_args()

    if args.dataset == "lidc":
        prepared = prepare_lidc_quality_control(
            patient_id=args.patient_id,
            metadata_csv=args.metadata_csv,
            parenchyma_file=args.parenchyma_file,
            consensus_level=args.consensus_level,
            study_instance_uid=args.study_instance_uid,
            series_instance_uid=args.series_instance_uid,
        )
    else:
        prepared = prepare_lndb_quality_control(
            scan_id=args.scan_id,
            metadata_csv=args.metadata_csv,
            data_dir=args.data_dir,
            mask_dir=args.mask_dir,
            parenchyma_file=args.parenchyma_file,
            consensus_level=args.consensus_level,
        )

    ct_volume, parenchyma, findings, title = prepared
    show_nodule_overlays(
        ct_volume=ct_volume,
        lung_parenchyma=parenchyma,
        findings=findings,
        title=title,
        slice_indices=args.slices,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
