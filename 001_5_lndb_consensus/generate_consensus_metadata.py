"""Generate bounding-box and slice metadata for LNDb consensus masks."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from config import (
    CONSENSUS_METADATA_CSV,
    DEFAULT_CONSENSUS_LEVEL,
    INPUT_METADATA_CSV,
    LNDB_DATA_DIR,
    LNDB_MASK_DIR,
)
from lndb_consensus import prepare_scan_data, process_scan


REQUIRED_COLUMNS = {
    "lndbid",
    "findingid",
    "radid",
    "radfindingid",
    "label",
}


def require_input_path(path: str | Path, description: str) -> Path:
    """Return an existing input path or raise a descriptive error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found: {path}")
    return path


def validate_input_metadata(metadata: pd.DataFrame) -> None:
    """Validate columns required to locate scans and annotations."""
    missing = sorted(REQUIRED_COLUMNS.difference(metadata.columns))
    if missing:
        raise ValueError(
            "Input metadata is missing required columns: " + ", ".join(missing)
        )
    if metadata.empty:
        raise ValueError("Input metadata contains no findings.")


def consensus_metadata(scan: dict[str, Any]) -> dict[str, Any]:
    """Extract serializable bounding-box and slice fields from a scan state."""
    bbox = scan["consensus_bbox"]
    slices = scan["consensus_slices"]

    height = bbox["ymax"] - bbox["ymin"] + 1
    width = bbox["xmax"] - bbox["xmin"] + 1
    depth = bbox["zmax"] - bbox["zmin"] + 1

    return {
        "bbox_y_min": bbox["ymin"],
        "bbox_y_max": bbox["ymax"],
        "bbox_x_min": bbox["xmin"],
        "bbox_x_max": bbox["xmax"],
        "bbox_z_min": bbox["zmin"],
        "bbox_z_max": bbox["zmax"],
        "bbox_height": height,
        "bbox_width": width,
        "bbox_depth": depth,
        "consensus_height": height,
        "consensus_width": width,
        "consensus_bbox_volume": height * width * depth,
        "consensus_num_slices": len(slices),
        "consensus_slice_list": ",".join(map(str, slices.tolist())),
    }


def generate_consensus_metadata(
    input_csv: str | Path = INPUT_METADATA_CSV,
    output_csv: str | Path = CONSENSUS_METADATA_CSV,
    data_dir: str | Path = LNDB_DATA_DIR,
    mask_dir: str | Path = LNDB_MASK_DIR,
    consensus_level: float = DEFAULT_CONSENSUS_LEVEL,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Process every finding and write enriched consensus metadata.

    The operation is fail-fast: an invalid scan or annotation stops execution,
    preventing a partially populated CSV from being mistaken for a complete
    result. Set ``overwrite=True`` to replace an existing output file.
    """
    input_csv = require_input_path(input_csv, "Input metadata CSV")
    data_dir = require_input_path(data_dir, "LNDb CT directory")
    mask_dir = require_input_path(mask_dir, "LNDb mask directory")
    output_csv = Path(output_csv)

    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output metadata already exists: {output_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    metadata = pd.read_csv(input_csv)
    validate_input_metadata(metadata)
    output = metadata.copy()

    for index, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Generating consensus metadata",
        unit="finding",
        dynamic_ncols=True,
    ):
        scan = prepare_scan_data(
            row=row,
            data_dir=data_dir,
            mask_dir=mask_dir,
        )
        scan = process_scan(scan, clevel=consensus_level)

        for column, value in consensus_metadata(scan).items():
            output.loc[index, column] = value

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_METADATA_CSV)
    parser.add_argument("--output-csv", type=Path, default=CONSENSUS_METADATA_CSV)
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
    """Generate the default or user-configured consensus metadata CSV."""
    args = build_parser().parse_args()
    output = generate_consensus_metadata(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir,
        consensus_level=args.consensus_level,
        overwrite=args.overwrite,
    )
    print(
        f"LNDb metadata | Findings: {len(output):,} | "
        f"Output: {args.output_csv}"
    )


if __name__ == "__main__":
    main()
