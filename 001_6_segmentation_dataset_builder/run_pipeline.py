"""Build LNDb, LIDC-IDRI, or both prepared segmentation datasets."""

import argparse

from config import DATASET_SPECS
from segmentation_dataset import build_dataset
from segmentation_dataset.builder import create_build_plan


def run_pipeline(dataset: str = "all", overwrite: bool = False) -> None:
    """Preflight and build the selected source datasets."""
    slugs = list(DATASET_SPECS) if dataset == "all" else [dataset]
    active_specs = []
    for slug in slugs:
        spec = DATASET_SPECS[slug]
        if spec.metadata_csv.exists() and not overwrite:
            print(f"Skip existing {spec.display_name} dataset: {spec.metadata_csv}")
        else:
            active_specs.append(spec)

    # Validate every selected source before any output is written. This keeps
    # an `all` run from completing one dataset and then failing on the other.
    plans = {
        spec.slug: create_build_plan(spec)
        for spec in active_specs
    }

    for spec in active_specs:
        summary = build_dataset(
            spec,
            overwrite=overwrite,
            plans=plans[spec.slug],
        )
        print(
            f"{spec.display_name}: {summary.samples} samples from "
            f"{summary.studies} studies "
            f"({summary.created} created, {summary.skipped} skipped)"
        )
        print(f"Metadata: {spec.metadata_csv}")


def main() -> None:
    """Build selected datasets from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=("lndb", "lidc", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.dataset, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
