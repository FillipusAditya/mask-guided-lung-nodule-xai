"""Label prepared samples and create patient-level dataset splits."""

from pathlib import Path
from datetime import datetime, timezone
import json
import re
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "000_dataset" / "_segmentation_dataset"

DATA_DIRECTORIES = {
    "ct_windowed_path": DATASET_DIR / "ct_windowed",
    "ct_parenchyma_path": DATASET_DIR / "ct_parenchyma",
    "mask_path": DATASET_DIR / "mask",
}

LIDC_LABEL_CSV = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "000_metadata"
    / "003_cluster_metadata_cleaned_path.csv"
)
LNDB_LABEL_CSV = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "000_metadata"
    / "004_consensus_clean_path.csv"
)

LABELED_METADATA_CSV = DATASET_DIR / "labeled_metadata.csv"
SPLIT_METADATA_CSV = DATASET_DIR / "split_metadata.csv"
ERROR_LOG_JSON = DATASET_DIR / "split_metadata_errors.json"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

LIDC_DATASET_NAME = "LIDC-IDRI"
LNDB_DATASET_NAME = "LNDb"
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

LNDB_FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>LNDb-(?P<lndbid>\d+))_"
    r"finding_(?P<finding_id>\d+)_slice_(?P<slice_index>\d+)\.npy$"
)
LIDC_FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>LIDC-IDRI-\d+)_"
    r"(?P<study_suffix>\d{5})_(?P<series_suffix>\d{5})_"
    r"cluster_(?P<cluster_id>\d+)_slice_(?P<slice_index>\d+)\.npy$"
)

LABELED_METADATA_COLUMNS = [
    "dataset",
    "patient_id",
    "filename",
    "ct_windowed_path",
    "ct_parenchyma_path",
    "mask_path",
    "label",
]


def record_error(
    errors: list[dict[str, object]],
    stage: str,
    error: Exception,
    **context: object,
) -> None:
    """Append one structured error record without stopping the pipeline."""

    errors.append(
        {
            "stage": stage,
            **context,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )


def save_error_log(
    errors: list[dict[str, object]],
    status: str,
    processed_samples: int = 0,
    saved_samples: int = 0,
) -> None:
    """Save a JSON report for skipped samples and fatal pipeline errors."""

    ERROR_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "processed_samples": processed_samples,
        "saved_samples": saved_samples,
        "error_count": len(errors),
        "errors": errors,
    }
    with ERROR_LOG_JSON.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    tolerance: float = 1e-8,
) -> None:
    """Validate that positive split ratios sum to one."""

    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    for name, ratio in ratios.items():
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError(
                f"Expected {name} to be numeric, "
                f"but received {type(ratio).__name__}."
            )

        if not 0.0 < float(ratio) < 1.0:
            raise ValueError(
                f"Expected {name} to be between 0 and 1, "
                f"but received {ratio}."
            )

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > tolerance:
        raise ValueError(
            "Expected train, validation, and test ratios to sum to 1.0, "
            f"but received {total_ratio:.10f}."
        )


def require_file(path: str | Path, description: str) -> Path:
    """Return an existing file path or raise a descriptive error."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def parse_slice_list(value: object, field_name: str) -> set[int]:
    """Parse comma-separated or bracketed slice indices from CSV metadata."""

    if pd.isna(value):
        raise ValueError(f"Missing {field_name} value.")

    slice_indices = {
        int(match)
        for match in re.findall(r"-?\d+", str(value))
    }
    if not slice_indices:
        raise ValueError(
            f"Could not parse any slice index from {field_name}: {value}"
        )
    return slice_indices


def normalize_label(value: object, context: str) -> str:
    """Return a non-empty classification label."""

    if pd.isna(value):
        raise ValueError(f"Missing label for {context}.")
    label = str(value).strip()
    if not label:
        raise ValueError(f"Empty label for {context}.")
    return label


def integer_value(value: object, field_name: str, context: str) -> int:
    """Convert a metadata value to an integer without silent truncation."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field_name} for {context}: {value}"
        ) from error

    if not numeric_value.is_integer():
        raise ValueError(
            f"Expected integer {field_name} for {context}, but received {value}."
        )
    return int(numeric_value)


def uid_suffix(value: object, field_name: str, context: str) -> str:
    """Extract the last five digits from a DICOM UID."""

    if pd.isna(value):
        raise ValueError(f"Missing {field_name} for {context}.")
    uid = str(value).strip()
    if len(uid) < 5 or not uid[-5:].isdigit():
        raise ValueError(f"Invalid {field_name} for {context}: {value}")
    return uid[-5:]


def add_unique_label(
    lookup: dict[tuple[object, ...], str],
    key: tuple[object, ...],
    label: str,
    context: str,
) -> None:
    """Add one label mapping and reject ambiguous source metadata."""

    if key in lookup:
        raise ValueError(
            f"Duplicate label mapping for {context}. "
            f"Existing: {lookup[key]}; new: {label}."
        )
    lookup[key] = label


def build_lndb_label_lookup(
    metadata_csv: str | Path,
    errors: list[dict[str, object]] | None = None,
) -> dict[tuple[int, int, int], str]:
    """Build ``(lndbid, finding_id, slice_index) -> label`` mappings."""

    metadata_df = pd.read_csv(require_file(metadata_csv, "LNDb label CSV"))
    required_columns = {
        "lndbid",
        "findingid",
        "consensus_slice_list",
        "label",
    }
    missing_columns = required_columns - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            f"LNDb label CSV is missing columns: {sorted(missing_columns)}"
        )

    errors = errors if errors is not None else []
    lookup: dict[tuple[int, int, int], str] = {}
    for row_index, row in metadata_df.iterrows():
        context = f"LNDb metadata row {row_index}"
        try:
            lndbid = integer_value(row["lndbid"], "lndbid", context)
            finding_id = integer_value(row["findingid"], "findingid", context)
            label = normalize_label(row["label"], context)
            slice_indices = parse_slice_list(
                row["consensus_slice_list"],
                "consensus_slice_list",
            )

            for slice_index in slice_indices:
                key = (lndbid, finding_id, slice_index)
                add_unique_label(lookup, key, label, str(key))
        except Exception as error:
            record_error(
                errors,
                "build_lndb_label_lookup",
                error,
                source_csv=str(metadata_csv),
                row_index=int(row_index),
            )
    return lookup


def build_lidc_label_lookup(
    metadata_csv: str | Path,
    errors: list[dict[str, object]] | None = None,
) -> dict[tuple[str, str, str, int, int], str]:
    """Build LIDC scan, cluster, and slice label mappings."""

    metadata_df = pd.read_csv(
        require_file(metadata_csv, "LIDC-IDRI label CSV"),
        dtype={"patient_id": "string"},
    )
    required_columns = {
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "cluster_id",
        "consensus_mask_slice_list",
        "label",
    }
    missing_columns = required_columns - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            f"LIDC-IDRI label CSV is missing columns: {sorted(missing_columns)}"
        )

    errors = errors if errors is not None else []
    lookup: dict[tuple[str, str, str, int, int], str] = {}
    for row_index, row in metadata_df.iterrows():
        context = f"LIDC-IDRI metadata row {row_index}"
        try:
            patient_id = str(row["patient_id"]).strip()
            if not re.fullmatch(r"LIDC-IDRI-\d+", patient_id):
                raise ValueError(
                    f"Invalid patient_id for {context}: {patient_id}"
                )

            study_suffix = uid_suffix(
                row["study_instance_uid"], "study_instance_uid", context
            )
            series_suffix = uid_suffix(
                row["series_instance_uid"], "series_instance_uid", context
            )
            cluster_id = integer_value(
                row["cluster_id"], "cluster_id", context
            )
            label = normalize_label(row["label"], context)
            slice_indices = parse_slice_list(
                row["consensus_mask_slice_list"],
                "consensus_mask_slice_list",
            )

            for slice_index in slice_indices:
                key = (
                    patient_id,
                    study_suffix,
                    series_suffix,
                    cluster_id,
                    slice_index,
                )
                add_unique_label(lookup, key, label, str(key))
        except Exception as error:
            record_error(
                errors,
                "build_lidc_label_lookup",
                error,
                source_csv=str(metadata_csv),
                row_index=int(row_index),
            )
    return lookup


def parse_sample_filename(filename: str) -> dict[str, object]:
    """Parse a supported prepared-sample filename."""

    lndb_match = LNDB_FILENAME_PATTERN.fullmatch(filename)
    if lndb_match is not None:
        return {
            "dataset": LNDB_DATASET_NAME,
            "patient_id": lndb_match.group("patient_id"),
            "label_key": (
                int(lndb_match.group("lndbid")),
                int(lndb_match.group("finding_id")),
                int(lndb_match.group("slice_index")),
            ),
        }

    lidc_match = LIDC_FILENAME_PATTERN.fullmatch(filename)
    if lidc_match is not None:
        return {
            "dataset": LIDC_DATASET_NAME,
            "patient_id": lidc_match.group("patient_id"),
            "label_key": (
                lidc_match.group("patient_id"),
                lidc_match.group("study_suffix"),
                lidc_match.group("series_suffix"),
                int(lidc_match.group("cluster_id")),
                int(lidc_match.group("slice_index")),
            ),
        }

    raise ValueError(f"Unsupported sample filename: {filename}")


def collect_candidate_filenames(
    data_directories: dict[str, Path],
) -> list[str]:
    """Return the union of filenames found in configured data directories."""

    filename_sets: dict[str, set[str]] = {}
    for column_name, directory in data_directories.items():
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"Dataset directory for {column_name} does not exist: {directory}"
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Expected a directory for {column_name}: {directory}"
            )

        filenames = {
            path.name
            for path in directory.glob("*.npy")
            if path.is_file()
        }
        filename_sets[column_name] = filenames

    return sorted(set().union(*filename_sets.values()))


def build_labeled_metadata(
    filenames: Sequence[str],
    data_directories: dict[str, Path],
    lndb_label_lookup: dict[tuple[int, int, int], str],
    lidc_label_lookup: dict[tuple[str, str, str, int, int], str],
    dataset_dir: str | Path = DATASET_DIR,
    errors: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build path and classification-label metadata for prepared samples."""

    dataset_dir = Path(dataset_dir)
    errors = errors if errors is not None else []
    records: list[dict[str, str]] = []
    for filename in tqdm(
        filenames,
        desc="Labeling dataset samples",
        unit="file",
    ):
        try:
            missing_paths = [
                str(Path(directory) / filename)
                for directory in data_directories.values()
                if not (Path(directory) / filename).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"Missing paired sample files: {missing_paths}"
                )

            sample = parse_sample_filename(filename)
            dataset_name = str(sample["dataset"])
            label_key = sample["label_key"]
            label = (
                lndb_label_lookup.get(label_key)
                if dataset_name == LNDB_DATASET_NAME
                else lidc_label_lookup.get(label_key)
            )
            if label is None:
                raise KeyError(
                    f"No classification label found using key {label_key}."
                )

            record = {
                "dataset": dataset_name,
                "patient_id": str(sample["patient_id"]),
                "filename": filename,
            }
            for column_name, directory in data_directories.items():
                record[column_name] = (
                    Path(directory) / filename
                ).relative_to(dataset_dir).as_posix()
            record["label"] = label
            records.append(record)
        except Exception as error:
            record_error(
                errors,
                "build_labeled_metadata",
                error,
                filename=filename,
            )

    metadata_df = pd.DataFrame(records, columns=LABELED_METADATA_COLUMNS)
    if metadata_df.empty:
        return metadata_df
    return metadata_df.sort_values(
        by=["dataset", "patient_id", "filename"],
        kind="stable",
    ).reset_index(drop=True)


def save_metadata(metadata_df: pd.DataFrame, output_csv: str | Path) -> None:
    """Save a non-empty metadata table to CSV."""

    if metadata_df.empty:
        raise ValueError("Cannot save an empty metadata table.")
    output_csv = Path(output_csv)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv output path: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_csv, index=False)


def split_patients(
    patient_ids: Sequence[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[set[str], set[str], set[str]]:
    """Split unique patients into train, validation, and test subsets."""

    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    unique_patient_ids = list(dict.fromkeys(patient_ids))
    if len(unique_patient_ids) != len(patient_ids):
        raise ValueError("Expected unique patient identifiers.")
    if len(unique_patient_ids) < 4:
        raise ValueError(
            "At least four patients are required to create non-empty train, "
            "validation, and test subsets with the configured two-stage split."
        )

    train_ids, temporary_ids = train_test_split(
        unique_patient_ids,
        train_size=train_ratio,
        shuffle=True,
        random_state=random_seed,
    )
    adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temporary_ids,
        train_size=adjusted_val_ratio,
        shuffle=True,
        random_state=random_seed,
    )
    splits = (set(train_ids), set(val_ids), set(test_ids))
    validate_split_membership(*splits, expected_ids=set(unique_patient_ids))
    return splits


def validate_split_membership(
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    expected_ids: set[str],
) -> None:
    """Validate complete and mutually exclusive patient assignments."""

    overlaps = {
        "train/validation": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "validation/test": val_ids & test_ids,
    }
    for split_pair, overlap in overlaps.items():
        if overlap:
            raise RuntimeError(
                f"Patient overlap detected for {split_pair}: {sorted(overlap)}"
            )

    assigned_ids = train_ids | val_ids | test_ids
    if assigned_ids != expected_ids:
        raise RuntimeError(
            "Patient split membership is incomplete or inconsistent. "
            f"Missing: {sorted(expected_ids - assigned_ids)}. "
            f"Unexpected: {sorted(assigned_ids - expected_ids)}."
        )


def select_patients_by_dataset(
    metadata_df: pd.DataFrame,
    dataset_name: str,
) -> list[str]:
    """Return sorted unique patients belonging to one source dataset."""

    return sorted(
        metadata_df.loc[
            metadata_df["dataset"] == dataset_name,
            "patient_id",
        ].unique()
    )


def assign_patient_split(
    patient_id: str,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> str:
    """Return the unique split assigned to a patient."""

    matching_splits = [
        split_name
        for split_name, patient_ids in (
            (TRAIN_SPLIT, train_ids),
            (VAL_SPLIT, val_ids),
            (TEST_SPLIT, test_ids),
        )
        if patient_id in patient_ids
    ]
    if len(matching_splits) != 1:
        raise RuntimeError(
            f"Expected patient {patient_id} in exactly one split, "
            f"but found: {matching_splits}."
        )
    return matching_splits[0]


def add_split_assignments(
    labeled_metadata_df: pd.DataFrame,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    """Add patient-level split assignments to labeled metadata."""

    metadata_df = labeled_metadata_df.copy()
    metadata_df["split"] = metadata_df["patient_id"].map(
        lambda patient_id: assign_patient_split(
            patient_id, train_ids, val_ids, test_ids
        )
    )
    split_order = pd.CategoricalDtype(
        categories=[TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
        ordered=True,
    )
    metadata_df["split"] = metadata_df["split"].astype(split_order)
    metadata_df = metadata_df.sort_values(
        by=["split", "dataset", "patient_id", "filename"],
        kind="stable",
    ).reset_index(drop=True)
    metadata_df["split"] = metadata_df["split"].astype("string")
    return metadata_df


def print_summary(
    metadata_df: pd.DataFrame,
    patient_splits: dict[str, tuple[set[str], set[str], set[str]]],
) -> None:
    """Print label, patient, and sample-level summaries."""

    print()
    print("=" * 72)
    print("Patient summary")
    print("=" * 72)
    for dataset_name, (train_ids, val_ids, test_ids) in patient_splits.items():
        total = len(train_ids | val_ids | test_ids)
        print(f"\n{dataset_name}: {total} patients")
        print(f"  Train : {len(train_ids)}")
        print(f"  Val   : {len(val_ids)}")
        print(f"  Test  : {len(test_ids)}")

    print()
    print("=" * 72)
    print("Sample summary by dataset, split, and label")
    print("=" * 72)
    print()
    summary = (
        metadata_df.groupby(
            ["dataset", "split", "label"],
            observed=True,
        )
        .size()
        .rename("sample_count")
        .reset_index()
    )
    print(summary.to_string(index=False))
    print()
    print("=" * 72)
    print(f"Labeled metadata: {LABELED_METADATA_CSV}")
    print(f"Split metadata  : {SPLIT_METADATA_CSV}")
    print("=" * 72)


def main() -> None:
    """Build labeled metadata first, then generate patient-level splits."""

    errors: list[dict[str, object]] = []
    processed_samples = 0
    saved_samples = 0
    status = "failed"

    try:
        validate_split_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
        filenames = collect_candidate_filenames(DATA_DIRECTORIES)
        processed_samples = len(filenames)
        lndb_label_lookup = build_lndb_label_lookup(
            LNDB_LABEL_CSV,
            errors=errors,
        )
        lidc_label_lookup = build_lidc_label_lookup(
            LIDC_LABEL_CSV,
            errors=errors,
        )

        labeled_metadata_df = build_labeled_metadata(
            filenames,
            DATA_DIRECTORIES,
            lndb_label_lookup,
            lidc_label_lookup,
            errors=errors,
        )
        saved_samples = len(labeled_metadata_df)
        if labeled_metadata_df.empty:
            raise RuntimeError(
                "No valid labeled samples remain after error filtering."
            )
        save_metadata(labeled_metadata_df, LABELED_METADATA_CSV)

        lidc_patients = select_patients_by_dataset(
            labeled_metadata_df, LIDC_DATASET_NAME
        )
        lndb_patients = select_patients_by_dataset(
            labeled_metadata_df, LNDB_DATASET_NAME
        )
        if not lidc_patients:
            raise RuntimeError("No labeled LIDC-IDRI patients were found.")
        if not lndb_patients:
            raise RuntimeError("No labeled LNDb patients were found.")

        patient_splits = {
            LIDC_DATASET_NAME: split_patients(lidc_patients),
            LNDB_DATASET_NAME: split_patients(lndb_patients),
        }
        lidc_train, lidc_val, lidc_test = patient_splits[LIDC_DATASET_NAME]
        lndb_train, lndb_val, lndb_test = patient_splits[LNDB_DATASET_NAME]
        train_ids = lidc_train | lndb_train
        val_ids = lidc_val | lndb_val
        test_ids = lidc_test | lndb_test

        validate_split_membership(
            train_ids,
            val_ids,
            test_ids,
            expected_ids=set(labeled_metadata_df["patient_id"]),
        )
        split_metadata_df = add_split_assignments(
            labeled_metadata_df,
            train_ids,
            val_ids,
            test_ids,
        )
        save_metadata(split_metadata_df, SPLIT_METADATA_CSV)
        print_summary(split_metadata_df, patient_splits)
        status = "completed_with_errors" if errors else "completed"
    except Exception as error:
        record_error(errors, "main", error)
        print()
        print(f"Pipeline could not continue: {error}")
    finally:
        save_error_log(
            errors,
            status=status,
            processed_samples=processed_samples,
            saved_samples=saved_samples,
        )
        print(f"Error log      : {ERROR_LOG_JSON}")
        print(f"Logged errors  : {len(errors)}")


if __name__ == "__main__":
    main()
