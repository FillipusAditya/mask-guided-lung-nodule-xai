"""Convert LIDC-IDRI and/or LNDb CT scans to NumPy HU volumes."""

import argparse

from config import (
    LIDC_OUTPUT_DIR,
    LIDC_PNG_OUTPUT_DIR,
    LNDB_INPUT_DIR,
    LNDB_OUTPUT_DIR,
    LNDB_PNG_OUTPUT_DIR,
)
from convert import (
    convert_lidc_dataset,
    convert_lndb_dataset,
    discover_lndb_files,
    summarize,
)
from export_png import export_volume_files, summarize_png


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("all", "lidc", "lndb"))
    parser.add_argument(
        "--patient-id",
        help="Convert one LIDC patient (valid only for dataset lidc).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing NumPy, metadata, and requested PNG outputs.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export every axial slice as an 8-bit PNG preview.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested dataset conversion."""
    args = parse_args()
    if args.patient_id and args.dataset != "lidc":
        raise SystemExit("--patient-id is only valid with dataset 'lidc'.")

    # For `all`, validate the LNDb source before potentially writing LIDC files.
    if args.dataset == "all":
        discover_lndb_files(LNDB_INPUT_DIR)

    if args.dataset in {"all", "lidc"}:
        results = convert_lidc_dataset(
            LIDC_OUTPUT_DIR,
            patient_id=args.patient_id,
            overwrite=args.overwrite,
        )
        print(f"LIDC-IDRI | {summarize(results)}")
        if args.png:
            png_results = export_volume_files(
                [result.volume_path for result in results],
                LIDC_OUTPUT_DIR,
                LIDC_PNG_OUTPUT_DIR,
                overwrite=args.overwrite,
            )
            print(f"LIDC PNG  | {summarize_png(png_results)}")

    if args.dataset in {"all", "lndb"}:
        results = convert_lndb_dataset(
            LNDB_INPUT_DIR,
            LNDB_OUTPUT_DIR,
            overwrite=args.overwrite,
        )
        print(f"LNDb      | {summarize(results)}")
        if args.png:
            png_results = export_volume_files(
                [result.volume_path for result in results],
                LNDB_OUTPUT_DIR,
                LNDB_PNG_OUTPUT_DIR,
                overwrite=args.overwrite,
            )
            print(f"LNDb PNG  | {summarize_png(png_results)}")


if __name__ == "__main__":
    main()
