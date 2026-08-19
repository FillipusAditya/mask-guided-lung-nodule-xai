"""Create a patient-level holdout split for either LNDb or LIDC-IDRI.

Examples
--------
Create an LNDb-only split using the default dataset directory::

    python 003_holdout_split_single_dataset.py --dataset LNDb

Create a LIDC-IDRI-only split in another output directory::

    python 003_holdout_split_single_dataset.py \
        --dataset LIDC \
        --output-dir /path/to/output

The default output filenames are dataset-specific, so this utility does not
overwrite ``001_holdout_split_lidc_lndb.csv`` or the other source dataset.
"""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import importlib.util
from pathlib import Path
from types import ModuleType


COMBINED_SPLIT_MODULE_PATH = Path(__file__).with_name(
    "003_holdout_split_lidc_lndb.py"
)


def load_combined_split_module() -> ModuleType:
    """Load the combined split module whose filename starts with a number."""

    module_spec = importlib.util.spec_from_file_location(
        "holdout_split_lidc_lndb",
        COMBINED_SPLIT_MODULE_PATH,
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(
            f"Could not load split utilities from {COMBINED_SPLIT_MODULE_PATH}."
        )

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


split_utils = load_combined_split_module()


def dataset_argument(value: str) -> str:
    """Normalize supported command-line dataset aliases."""

    normalized_value = value.strip().lower().replace("_", "-")
    aliases = {
        "lndb": split_utils.LNDB_DATASET_NAME,
        "lidc": split_utils.LIDC_DATASET_NAME,
        "lidc-idri": split_utils.LIDC_DATASET_NAME,
    }
    if normalized_value not in aliases:
        raise ArgumentTypeError(
            "dataset must be one of: LNDb, LIDC, or LIDC-IDRI."
        )
    return aliases[normalized_value]


def parse_arguments() -> Namespace:
    """Parse command-line configuration."""

    parser = ArgumentParser(
        description=(
            "Create a reproducible 70/15/15 patient-level holdout split for "
            "one source dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=dataset_argument,
        metavar="{LNDb,LIDC-IDRI}",
        help="Source dataset to include. The alias LIDC is also accepted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=split_utils.DATASET_DIR,
        help=(
            "Directory for generated CSV and JSON files. Defaults to the "
            "prepared segmentation dataset directory."
        ),
    )
    return parser.parse_args()


def dataset_configuration(dataset_name: str) -> dict[str, object]:
    """Return source-specific parsing and label configuration."""

    configurations = {
        split_utils.LNDB_DATASET_NAME: {
            "slug": "lndb",
            "filename_pattern": split_utils.LNDB_FILENAME_PATTERN,
            "label_csv": split_utils.LNDB_LABEL_CSV,
            "build_label_lookup": split_utils.build_lndb_label_lookup,
        },
        split_utils.LIDC_DATASET_NAME: {
            "slug": "lidc",
            "filename_pattern": split_utils.LIDC_FILENAME_PATTERN,
            "label_csv": split_utils.LIDC_LABEL_CSV,
            "build_label_lookup": split_utils.build_lidc_label_lookup,
        },
    }
    try:
        return configurations[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from error


def output_paths(
    dataset_name: str,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Return labeled metadata, split metadata, and error-log paths."""

    slug = str(dataset_configuration(dataset_name)["slug"])
    output_dir = Path(output_dir)
    return (
        output_dir / f"labeled_metadata_{slug}.csv",
        output_dir / f"001_holdout_split_{slug}.csv",
        output_dir / f"001_holdout_split_{slug}_errors.json",
    )


def run_single_dataset_split(
    dataset_name: str,
    output_dir: str | Path,
) -> int:
    """Label and split samples belonging to one source dataset."""

    configuration = dataset_configuration(dataset_name)
    labeled_metadata_csv, split_metadata_csv, error_log_json = output_paths(
        dataset_name,
        output_dir,
    )

    errors: list[dict[str, object]] = []
    processed_samples = 0
    saved_samples = 0
    status = "failed"

    try:
        split_utils.validate_split_ratios(
            split_utils.TRAIN_RATIO,
            split_utils.VAL_RATIO,
            split_utils.TEST_RATIO,
        )
        all_filenames = split_utils.collect_candidate_filenames(
            split_utils.DATA_DIRECTORIES
        )
        filename_pattern = configuration["filename_pattern"]
        filenames = [
            filename
            for filename in all_filenames
            if filename_pattern.fullmatch(filename) is not None
        ]
        processed_samples = len(filenames)
        if not filenames:
            raise RuntimeError(
                f"No prepared {dataset_name} sample filenames were found."
            )

        build_label_lookup = configuration["build_label_lookup"]
        label_lookup = build_label_lookup(
            configuration["label_csv"],
            errors=errors,
        )
        lndb_label_lookup = (
            label_lookup
            if dataset_name == split_utils.LNDB_DATASET_NAME
            else {}
        )
        lidc_label_lookup = (
            label_lookup
            if dataset_name == split_utils.LIDC_DATASET_NAME
            else {}
        )

        labeled_metadata_df = split_utils.build_labeled_metadata(
            filenames,
            split_utils.DATA_DIRECTORIES,
            lndb_label_lookup,
            lidc_label_lookup,
            dataset_dir=split_utils.DATASET_DIR,
            errors=errors,
        )
        saved_samples = len(labeled_metadata_df)
        if labeled_metadata_df.empty:
            raise RuntimeError(
                f"No valid labeled {dataset_name} samples remain."
            )

        observed_datasets = set(labeled_metadata_df["dataset"])
        if observed_datasets != {dataset_name}:
            raise RuntimeError(
                "Single-dataset filtering produced unexpected datasets: "
                f"{sorted(observed_datasets)}."
            )
        split_utils.save_metadata(
            labeled_metadata_df,
            labeled_metadata_csv,
        )

        patient_ids = split_utils.select_patients_by_dataset(
            labeled_metadata_df,
            dataset_name,
        )
        if not patient_ids:
            raise RuntimeError(
                f"No labeled {dataset_name} patients were found."
            )

        train_ids, val_ids, test_ids = split_utils.split_patients(patient_ids)
        split_metadata_df = split_utils.add_split_assignments(
            labeled_metadata_df,
            train_ids,
            val_ids,
            test_ids,
        )
        split_utils.save_metadata(
            split_metadata_df,
            split_metadata_csv,
        )
        split_utils.print_summary(
            split_metadata_df,
            {dataset_name: (train_ids, val_ids, test_ids)},
            labeled_metadata_csv=labeled_metadata_csv,
            split_metadata_csv=split_metadata_csv,
        )
        status = "completed_with_errors" if errors else "completed"
    except Exception as error:
        split_utils.record_error(
            errors,
            "run_single_dataset_split",
            error,
            dataset=dataset_name,
        )
        print()
        print(f"Pipeline could not continue: {error}")
    finally:
        split_utils.save_error_log(
            errors,
            status=status,
            processed_samples=processed_samples,
            saved_samples=saved_samples,
            output_json=error_log_json,
        )
        print(f"Error log      : {error_log_json}")
        print(f"Logged errors  : {len(errors)}")

    return 1 if status == "failed" else 0


def main() -> int:
    """Run the selected single-dataset split from command-line arguments."""

    arguments = parse_arguments()
    return run_single_dataset_split(
        arguments.dataset,
        arguments.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
