"""Run the complete protected lung-parenchyma pipeline on local datasets."""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    LIDC_INPUT_DIR,
    LIDC_MASK_OUTPUT_DIR,
    LIDC_PARENCHYMA_OUTPUT_DIR,
    LNDB_INPUT_DIR,
    LNDB_MASK_OUTPUT_DIR,
    LNDB_PARENCHYMA_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
)
from step_1_ct_to_numpy import (
    ScanSource,
    discover_lidc_scans,
    discover_lndb_scans,
    load_scan,
)
from step_2_segmentation import segment_volume
from step_4_normalize_and_mask import filter_normalize_and_mask


def create_lung_parenchyma(
    volume: np.ndarray,
    show_slice_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return final uint8 mask, float32 parenchyma, and diagnostics."""
    mask, metrics = segment_volume(volume, show_progress=show_slice_progress)
    parenchyma = filter_normalize_and_mask(
        volume,
        mask,
        show_progress=show_slice_progress,
    )
    validate_outputs(volume, mask, parenchyma)
    return mask, parenchyma, metrics.to_dict()


def validate_outputs(
    source: np.ndarray,
    mask: np.ndarray,
    parenchyma: np.ndarray,
) -> None:
    """Validate the final data contract before an output is accepted."""
    if mask.shape != source.shape or parenchyma.shape != source.shape:
        raise ValueError("Source, mask, and parenchyma shapes do not match.")
    if mask.dtype != np.uint8:
        raise TypeError(f"Mask must be uint8, received {mask.dtype}.")
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("Mask contains values other than 0 and 1.")
    if parenchyma.dtype != np.float32:
        raise TypeError(f"Parenchyma must be float32, received {parenchyma.dtype}.")
    if parenchyma.min() < 0.0 or parenchyma.max() > 1.0:
        raise ValueError("Normalized parenchyma is outside [0, 1].")
    for index in range(len(parenchyma)):
        if not np.isfinite(parenchyma[index]).all():
            raise ValueError("Parenchyma contains non-finite values.")
        if np.any(parenchyma[index][mask[index] == 0] != 0.0):
            raise ValueError("Pixels outside the lung mask are not zero.")


def _atomic_save(array: np.ndarray, output_path: Path) -> None:
    """Write an NPY file completely before replacing its destination."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    temporary_path.replace(output_path)


def _output_paths(source: ScanSource) -> tuple[Path, Path]:
    if source.dataset == "lidc":
        mask_dir = LIDC_MASK_OUTPUT_DIR
        parenchyma_dir = LIDC_PARENCHYMA_OUTPUT_DIR
    else:
        mask_dir = LNDB_MASK_OUTPUT_DIR
        parenchyma_dir = LNDB_PARENCHYMA_OUTPUT_DIR
    return (
        mask_dir / f"{source.scan_id}_mask.npy",
        parenchyma_dir / f"{source.scan_id}_parenchyma.npy",
    )


def process_source(
    source: ScanSource,
    save_outputs: bool,
    overwrite: bool,
    show_slice_progress: bool,
) -> dict:
    """Process one scan and return a serializable report row."""
    start = time.perf_counter()
    mask_path, parenchyma_path = _output_paths(source)
    row = {
        "dataset": source.dataset,
        "scan_id": source.scan_id,
        "source_path": str(source.source_path),
        "status": "pending",
        "error": "",
        "mask_path": str(mask_path) if save_outputs else "",
        "parenchyma_path": str(parenchyma_path) if save_outputs else "",
    }

    try:
        if (
            save_outputs
            and not overwrite
            and mask_path.is_file()
            and parenchyma_path.is_file()
        ):
            row["status"] = "skipped_existing"
            return row

        volume = load_scan(source)
        row.update(
            slices=int(volume.shape[0]),
            height=int(volume.shape[1]),
            width=int(volume.shape[2]),
            source_dtype=str(volume.dtype),
            source_min_hu=int(volume.min()),
            source_max_hu=int(volume.max()),
        )
        mask, parenchyma, metrics = create_lung_parenchyma(
            volume,
            show_slice_progress=show_slice_progress,
        )
        row.update(metrics)
        row.update(
            mask_dtype=str(mask.dtype),
            parenchyma_dtype=str(parenchyma.dtype),
            parenchyma_min=float(parenchyma.min()),
            parenchyma_max=float(parenchyma.max()),
        )

        if save_outputs:
            _atomic_save(mask, mask_path)
            _atomic_save(parenchyma, parenchyma_path)
            row["status"] = "written"
        else:
            row["status"] = "validated"
    except Exception as error:  # Batch mode must continue past incomplete scans.
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["seconds"] = round(time.perf_counter() - start, 3)

    return row


def discover_sources(
    dataset: str,
    patient_id: str | None,
    lndb_input: Path,
) -> list[ScanSource]:
    sources = []
    if dataset in ("lidc", "all"):
        sources.extend(discover_lidc_scans(LIDC_INPUT_DIR, patient_id))
    if dataset in ("lndb", "all"):
        sources.extend(discover_lndb_scans(lndb_input))
    return sources


def save_report(rows: list[dict], report_dir: Path, mode: str) -> tuple[Path, Path]:
    """Save CSV and JSON reports, including failures."""
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"pipeline_{mode}_{timestamp}"
    csv_path = report_dir / f"{stem}.csv"
    json_path = report_dir / f"{stem}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("lidc", "lndb", "all"))
    parser.add_argument("--patient-id", help="Optional single LIDC patient ID.")
    parser.add_argument(
        "--lndb-input", type=Path, default=LNDB_INPUT_DIR,
        help="One LNDb MHD file or a directory containing MHD files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run the complete pipeline and validations without writing arrays.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N scans.")
    parser.add_argument("--show-slice-progress", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_OUTPUT_DIR)
    args = parser.parse_args()

    sources = discover_sources(args.dataset, args.patient_id, args.lndb_input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError("No complete local CT scans were discovered.")

    print(
        f"Discovered {len(sources)} complete scan(s): "
        f"{dict(Counter(source.dataset for source in sources))}"
    )
    rows = []
    for source in tqdm(sources, desc="Lung parenchyma", unit="scan", dynamic_ncols=True):
        row = process_source(
            source,
            save_outputs=not args.no_save,
            overwrite=args.overwrite,
            show_slice_progress=args.show_slice_progress,
        )
        rows.append(row)
        if row["status"] == "failed":
            tqdm.write(f"FAILED {source.scan_id}: {row['error']}")
            if args.fail_fast:
                break

    csv_path, json_path = save_report(
        rows,
        args.report_dir,
        mode="validation" if args.no_save else "write",
    )
    counts = Counter(row["status"] for row in rows)
    print(f"Summary: {dict(counts)}")
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")

    if counts.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
