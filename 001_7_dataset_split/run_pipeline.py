"""Create holdout splits and classification cross-validation folds."""

import argparse
from pathlib import Path

from config import (
    COMBINED_ERROR_LOG_JSON,
    COMBINED_HOLDOUT_CSV,
    COMBINED_LABELED_METADATA_CSV,
    CV_METADATA_CSV,
    CV_SUMMARY_CSV,
    DATASET_DIR,
    N_SPLITS,
    RANDOM_SEED,
)
from dataset_split import (
    create_classification_folds,
    run_combined_dataset_split,
    run_single_dataset_split,
)
from dataset_split.pipeline import output_paths


def run_holdout(
    dataset: str,
    output_dir: str | Path = DATASET_DIR,
    overwrite: bool = False,
) -> int:
    """Create a combined or source-specific holdout split."""
    if dataset == "all":
        outputs = (
            COMBINED_LABELED_METADATA_CSV,
            COMBINED_HOLDOUT_CSV,
            COMBINED_ERROR_LOG_JSON,
        )
        if all(path.exists() for path in outputs) and not overwrite:
            print(f"Combined holdout | Skipped: {COMBINED_HOLDOUT_CSV}")
            return 0
        if any(path.exists() for path in outputs) and not overwrite:
            existing = [str(path) for path in outputs if path.exists()]
            raise FileExistsError(
                "Combined holdout outputs are incomplete. Use --overwrite "
                f"to regenerate all files. Existing: {existing}"
            )
        return run_combined_dataset_split()

    outputs = output_paths(dataset, output_dir)
    split_csv = outputs[1]
    if all(path.exists() for path in outputs) and not overwrite:
        print(f"{dataset.upper()} holdout | Skipped: {split_csv}")
        return 0
    if any(path.exists() for path in outputs) and not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            "Single-dataset holdout outputs are incomplete. Use --overwrite "
            f"to regenerate all files. Existing: {existing}"
        )
    return run_single_dataset_split(dataset, output_dir)


def run_cross_validation(overwrite: bool = False) -> int:
    """Create classification folds from the combined holdout metadata."""
    outputs = (CV_METADATA_CSV, CV_SUMMARY_CSV)
    if all(path.exists() for path in outputs) and not overwrite:
        print(f"Classification CV | Skipped: {CV_METADATA_CSV}")
        return 0
    if any(path.exists() for path in outputs) and not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            "Cross-validation outputs are incomplete. Use --overwrite to "
            f"regenerate both files. Existing: {existing}"
        )
    create_classification_folds(
        n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("holdout", "cv", "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "lidc", "lndb"),
        default="all",
        help="Dataset selection for the holdout stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR,
        help="Output directory for a single-dataset holdout.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    """Run the requested pipeline stage."""
    args = build_parser().parse_args()
    if args.stage in {"cv", "all"} and args.dataset != "all":
        raise SystemExit("--dataset is only applicable to stage 'holdout'.")
    if args.dataset == "all" and args.output_dir != DATASET_DIR:
        raise SystemExit("--output-dir is only supported for lidc or lndb.")

    if args.stage in {"holdout", "all"}:
        status = run_holdout(
            args.dataset,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        if status:
            return status
    if args.stage in {"cv", "all"}:
        return run_cross_validation(overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
