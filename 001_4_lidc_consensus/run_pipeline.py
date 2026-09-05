"""Run the complete LIDC-IDRI consensus-mask pipeline."""

import argparse
from pathlib import Path

from config import (
    CONSENSUS_MASK_DIR,
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    OUTPUT_METADATA_CSV,
    QUALITY_CONTROL_DIR,
    SEGMENTED_NODULE_PNG_DIR,
)
from generate_artifacts import generate_consensus_artifacts
from generate_consensus_masks import generate_consensus_masks


def run_pipeline(
    metadata_csv: str | Path = INPUT_METADATA_CSV,
    output_dir: str | Path = CONSENSUS_MASK_DIR,
    output_metadata_csv: str | Path = OUTPUT_METADATA_CSV,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    patient_id: str | None = None,
    segmented_png: bool = False,
    segmented_png_dir: str | Path = SEGMENTED_NODULE_PNG_DIR,
    quality_control: bool = False,
    quality_control_dir: str | Path = QUALITY_CONTROL_DIR,
    overwrite: bool = False,
) -> None:
    """Generate all selected LIDC-IDRI consensus masks."""
    output_metadata_csv = Path(output_metadata_csv)
    if output_metadata_csv.exists() and not overwrite:
        print(f"LIDC consensus | Skipped: {output_metadata_csv}")
    else:
        output = generate_consensus_masks(
            metadata_csv=metadata_csv,
            output_dir=output_dir,
            output_metadata_csv=output_metadata_csv,
            consensus_level=consensus_level,
            overwrite=overwrite,
        )
        print(
            f"LIDC consensus | Clusters: {len(output):,} | "
            f"Masks: {output_dir} | Metadata: {output_metadata_csv}"
        )

    if segmented_png or quality_control:
        counts = generate_consensus_artifacts(
            metadata_csv=metadata_csv,
            segmented_png_dir=segmented_png_dir,
            quality_control_dir=quality_control_dir,
            consensus_level=consensus_level,
            patient_id=patient_id,
            segmented_png=segmented_png,
            quality_control=quality_control,
            overwrite=overwrite,
        )
        print(
            "LIDC artifacts | "
            f"Clusters: {counts['clusters']} | "
            f"PNG: {counts['segmented_png']} | "
            f"QC: {counts['quality_control']}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=INPUT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=CONSENSUS_MASK_DIR)
    parser.add_argument(
        "--output-metadata-csv",
        type=Path,
        default=OUTPUT_METADATA_CSV,
    )
    parser.add_argument(
        "--consensus-level",
        type=float,
        default=DEFAULT_CONSENSUS_LEVEL,
    )
    parser.add_argument(
        "--patient-id",
        help="Generate optional artifacts only for this LIDC patient ID.",
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
    """Run the pipeline from command-line arguments."""
    args = build_parser().parse_args()
    run_pipeline(
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_metadata_csv,
        consensus_level=args.consensus_level,
        patient_id=args.patient_id,
        segmented_png=args.segmented_png,
        segmented_png_dir=args.segmented_png_dir,
        quality_control=args.quality_control,
        quality_control_dir=args.quality_control_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
