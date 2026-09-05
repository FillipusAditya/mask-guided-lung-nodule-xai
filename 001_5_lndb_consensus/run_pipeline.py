"""Run the LNDb consensus metadata and mask-generation stages."""

import argparse
from pathlib import Path

from config import (
    CONSENSUS_MASK_DIR,
    CONSENSUS_METADATA_CSV,
    CONSENSUS_PATH_METADATA_CSV,
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    LNDB_DATA_DIR,
    LNDB_MASK_DIR,
    QUALITY_CONTROL_DIR,
    SEGMENTED_NODULE_PNG_DIR,
)
from generate_artifacts import generate_consensus_artifacts
from generate_consensus_masks import generate_consensus_masks
from generate_consensus_metadata import generate_consensus_metadata


def run_pipeline(
    stage: str = "all",
    input_csv: str | Path = INPUT_METADATA_CSV,
    consensus_metadata_csv: str | Path = CONSENSUS_METADATA_CSV,
    output_dir: str | Path = CONSENSUS_MASK_DIR,
    output_metadata_csv: str | Path = CONSENSUS_PATH_METADATA_CSV,
    data_dir: str | Path = LNDB_DATA_DIR,
    mask_dir: str | Path = LNDB_MASK_DIR,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    scan_id: int | None = None,
    segmented_png: bool = False,
    segmented_png_dir: str | Path = SEGMENTED_NODULE_PNG_DIR,
    quality_control: bool = False,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
    overwrite: bool = False,
) -> None:
    """Run one stage or the complete two-stage consensus pipeline."""
    if stage not in {"metadata", "masks", "all"}:
        raise ValueError(f"Unknown stage: {stage}")

    consensus_metadata_csv = Path(consensus_metadata_csv)
    output_metadata_csv = Path(output_metadata_csv)

    if stage in {"metadata", "all"}:
        if consensus_metadata_csv.exists() and not overwrite:
            print(f"LNDb metadata | Skipped: {consensus_metadata_csv}")
        else:
            output = generate_consensus_metadata(
                input_csv=input_csv,
                output_csv=consensus_metadata_csv,
                data_dir=data_dir,
                mask_dir=mask_dir,
                consensus_level=consensus_level,
                overwrite=overwrite,
            )
            print(f"LNDb metadata | Findings: {len(output):,}")

    if stage in {"masks", "all"}:
        if output_metadata_csv.exists() and not overwrite:
            print(f"LNDb masks | Skipped: {output_metadata_csv}")
        else:
            output = generate_consensus_masks(
                input_csv=consensus_metadata_csv,
                output_dir=output_dir,
                output_metadata_csv=output_metadata_csv,
                data_dir=data_dir,
                mask_dir=mask_dir,
                consensus_level=consensus_level,
                overwrite=overwrite,
            )
            print(f"LNDb masks | Findings: {len(output):,}")

    if segmented_png or quality_control:
        counts = generate_consensus_artifacts(
            input_csv=input_csv,
            data_dir=data_dir,
            mask_dir=mask_dir,
            segmented_png_dir=segmented_png_dir,
            quality_control_dir=quality_control_dir,
            consensus_level=consensus_level,
            scan_id=scan_id,
            segmented_png=segmented_png,
            quality_control=quality_control,
            overwrite=overwrite,
        )
        print(
            "LNDb artifacts | "
            f"Findings: {counts['findings']} | "
            f"PNG: {counts['segmented_png']} | "
            f"QC: {counts['quality_control']}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("metadata", "masks", "all"),
        default="all",
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_METADATA_CSV)
    parser.add_argument(
        "--consensus-metadata-csv",
        type=Path,
        default=CONSENSUS_METADATA_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=CONSENSUS_MASK_DIR)
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=CONSENSUS_PATH_METADATA_CSV,
    )
    parser.add_argument("--data-dir", type=Path, default=LNDB_DATA_DIR)
    parser.add_argument("--mask-dir", type=Path, default=LNDB_MASK_DIR)
    parser.add_argument(
        "--consensus-level",
        type=float,
        default=DEFAULT_CONSENSUS_LEVEL,
    )
    parser.add_argument(
        "--scan-id",
        type=int,
        help="Generate optional artifacts only for this LNDb scan ID.",
    )
    parser.add_argument(
        "--segmented-png",
        action="store_true",
        help="Save normalized consensus-nodule crops as PNG slices.",
    )
    parser.add_argument(
        "--segmented-png-dir",
        type=Path,
        default=SEGMENTED_NODULE_PNG_DIR,
    )
    parser.add_argument(
        "--quality-control",
        action="store_true",
        help="Save agreement-map and consensus-mask canvases per scan.",
    )
    parser.add_argument(
        "--quality-control-dir",
        type=Path,
        default=QUALITY_CONTROL_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run the selected pipeline stage from command-line arguments."""
    args = build_parser().parse_args()
    run_pipeline(
        stage=args.stage,
        input_csv=args.input_csv,
        consensus_metadata_csv=args.consensus_metadata_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_metadata_csv,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        consensus_level=args.consensus_level,
        scan_id=args.scan_id,
        segmented_png=args.segmented_png,
        segmented_png_dir=args.segmented_png_dir,
        quality_control=args.quality_control,
        quality_control_dir=args.quality_control_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
