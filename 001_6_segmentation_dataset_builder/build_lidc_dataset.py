"""Build the prepared two-dimensional LIDC-IDRI segmentation dataset."""

import argparse
from dataclasses import replace
from pathlib import Path

from config import LIDC_SPEC
from segmentation_dataset import BuildSummary, build_dataset


def build_lidc_dataset(
    output_dir: str | Path = LIDC_SPEC.output_dir,
    overwrite: bool = False,
) -> BuildSummary:
    """Build LIDC-IDRI CT/mask pairs and metadata."""
    spec = replace(LIDC_SPEC, output_dir=Path(output_dir))
    return build_dataset(spec, overwrite=overwrite)


def main() -> None:
    """Build LIDC-IDRI samples from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=LIDC_SPEC.output_dir)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = build_lidc_dataset(args.output_dir, overwrite=args.overwrite)
    print(f"LIDC-IDRI studies: {summary.studies}")
    print(f"Samples: {summary.samples}")
    print(f"Created: {summary.created}; skipped: {summary.skipped}")
    print(f"Metadata: {args.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
