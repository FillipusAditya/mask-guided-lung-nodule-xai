"""Generate per-slice LNDb consensus masks and path metadata."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    CONSENSUS_MASK_DIR,
    CONSENSUS_METADATA_CSV,
    CONSENSUS_PATH_METADATA_CSV,
    DEFAULT_CONSENSUS_LEVEL,
    LNDB_DATA_DIR,
    LNDB_MASK_DIR,
    PROJECT_ROOT,
)
from generate_consensus_metadata import require_input_path, validate_input_metadata
from lndb_consensus import prepare_scan_data, process_scan, save_consensus_slices


def metadata_path_value(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def generate_consensus_masks(
    input_csv: str | Path = CONSENSUS_METADATA_CSV,
    output_dir: str | Path = CONSENSUS_MASK_DIR,
    output_metadata_csv: str | Path = CONSENSUS_PATH_METADATA_CSV,
    data_dir: str | Path = LNDB_DATA_DIR,
    mask_dir: str | Path = LNDB_MASK_DIR,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate masks for every finding and save metadata containing paths.

    Each mask is stored as ``slice_<z>.npy`` below
    ``LNDb-<id>/finding_<id>``. The CSV stores project-relative POSIX paths for
    outputs inside this repository, making it portable across operating systems.
    """
    input_csv = require_input_path(input_csv, "Consensus metadata CSV")
    data_dir = require_input_path(data_dir, "LNDb CT directory")
    mask_dir = require_input_path(mask_dir, "LNDb mask directory")
    output_dir = Path(output_dir)
    output_metadata_csv = Path(output_metadata_csv)

    if output_metadata_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output metadata already exists: {output_metadata_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    metadata = pd.read_csv(input_csv)
    validate_input_metadata(metadata)
    output_metadata = metadata.copy()
    output_metadata["consensus_mask_path"] = None

    for index, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Generating consensus masks",
        unit="finding",
        dynamic_ncols=True,
    ):
        scan = prepare_scan_data(
            row=row,
            data_dir=data_dir,
            mask_dir=mask_dir,
        )
        scan = process_scan(scan, clevel=consensus_level)

        finding_dir = (
            output_dir
            / f"LNDb-{scan['lndb_id']:04d}"
            / f"finding_{scan['finding_id']}"
        )
        save_consensus_slices(
            scan=scan,
            output_dir=finding_dir,
            overwrite=overwrite,
        )
        output_metadata.loc[index, "consensus_mask_path"] = (
            metadata_path_value(finding_dir)
        )

    output_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.to_csv(output_metadata_csv, index=False)
    return output_metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=CONSENSUS_METADATA_CSV)
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
        help="Required radiologist agreement fraction in the interval (0, 1].",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Generate consensus masks and the default path metadata CSV."""
    args = build_parser().parse_args()
    output = generate_consensus_masks(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        output_metadata_csv=args.output_metadata_csv,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        consensus_level=args.consensus_level,
        overwrite=args.overwrite,
    )
    print(
        f"LNDb masks | Findings: {len(output):,} | "
        f"Masks: {args.output_dir} | Metadata: {args.output_metadata_csv}"
    )


if __name__ == "__main__":
    main()
