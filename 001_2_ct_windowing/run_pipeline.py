"""Median-filter, lung-window, and normalize LIDC-IDRI and/or LNDb CT."""

import argparse

from config import (
    LIDC_INPUT_DIR,
    LIDC_OUTPUT_DIR,
    LIDC_PNG_OUTPUT_DIR,
    LNDB_INPUT_DIR,
    LNDB_OUTPUT_DIR,
    LNDB_PNG_OUTPUT_DIR,
    MEDIAN_FILTER_SIZE,
    WINDOW_LEVEL,
    WINDOW_WIDTH,
)
from export_png import export_volume_files, summarize_png
from preprocess import discover_ct_files, preprocess_ct_directory, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("all", "lidc", "lndb"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing NumPy, metadata, and requested PNG outputs.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also export every normalized axial slice as an 8-bit PNG.",
    )
    return parser.parse_args()


def _run(
    name: str,
    input_dir,
    output_dir,
    png_output_dir,
    overwrite: bool,
    export_png: bool,
) -> None:
    results = preprocess_ct_directory(
        input_dir,
        output_dir,
        window_level=WINDOW_LEVEL,
        window_width=WINDOW_WIDTH,
        median_filter_size=MEDIAN_FILTER_SIZE,
        overwrite=overwrite,
    )
    print(f"{name:<9} | {summarize(results)}")

    if export_png:
        png_results = export_volume_files(
            [result.output_path for result in results],
            output_dir,
            png_output_dir,
            overwrite=overwrite,
        )
        print(f"{name:<9} PNG | {summarize_png(png_results)}")


def main() -> None:
    args = parse_args()
    # Preflight both sources before `all` starts writing either output.
    if args.dataset == "all":
        discover_ct_files(LIDC_INPUT_DIR)
        discover_ct_files(LNDB_INPUT_DIR)
    if args.dataset in {"all", "lidc"}:
        _run(
            "LIDC-IDRI",
            LIDC_INPUT_DIR,
            LIDC_OUTPUT_DIR,
            LIDC_PNG_OUTPUT_DIR,
            args.overwrite,
            args.png,
        )
    if args.dataset in {"all", "lndb"}:
        _run(
            "LNDb",
            LNDB_INPUT_DIR,
            LNDB_OUTPUT_DIR,
            LNDB_PNG_OUTPUT_DIR,
            args.overwrite,
            args.png,
        )


if __name__ == "__main__":
    main()
