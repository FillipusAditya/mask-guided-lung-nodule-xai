"""Generate patient-level train, validation, and test split metadata.

The script scans CT NumPy files from the configured segmentation dataset,
extracts patient identifiers from their filenames, and assigns each patient
to exactly one data split.

LIDC-IDRI and LNDb patients are split independently so that each dataset is
represented in the training, validation, and test subsets. All files that
belong to the same patient are assigned to the same split to prevent patient-
level data leakage.

The resulting metadata are saved as a CSV file with the following columns:

- ``dataset``
- ``patient_id``
- ``filename``
- ``split``
"""

from pathlib import Path
import re
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Root directory of the project.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Root directory containing the prepared segmentation dataset.
DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "_segmentation_dataset"
)

# Directory containing CT NumPy files.
CT_DIR = DATASET_DIR / "ct"

# Destination CSV file containing split assignments.
OUTPUT_CSV = DATASET_DIR / "split_metadata.csv"

# Patient-level split ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed used to produce reproducible split assignments.
RANDOM_SEED = 42

# Supported dataset names.
LIDC_DATASET_NAME = "LIDC-IDRI"
LNDB_DATASET_NAME = "LNDb"

# Supported split names.
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

# Filename patterns used to extract patient identifiers.
LIDC_PATIENT_PATTERN = re.compile(r"^(LIDC-IDRI-\d+)")
LNDB_PATIENT_PATTERN = re.compile(r"^(LNDb-\d+)")


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    tolerance: float = 1e-8,
) -> None:
    """Validate train, validation, and test split ratios.

    Parameters
    ----------
    train_ratio : float
        Proportion of patients assigned to the training split.
    val_ratio : float
        Proportion of patients assigned to the validation split.
    test_ratio : float
        Proportion of patients assigned to the test split.
    tolerance : float, default=1e-8
        Absolute tolerance used when checking whether the ratios sum to one.

    Raises
    ------
    TypeError
        If any ratio is not numeric.
    ValueError
        If any ratio is not strictly between zero and one, or if the ratios
        do not sum to one.
    """

    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    for name, ratio in ratios.items():
        if not isinstance(ratio, (int, float)):
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


def extract_patient_id(
    filename: str,
) -> str:
    """Extract a patient identifier from a segmentation filename.

    Parameters
    ----------
    filename : str
        Filename containing an LIDC-IDRI or LNDb patient identifier.

    Returns
    -------
    str
        Extracted patient identifier.

    Raises
    ------
    TypeError
        If ``filename`` is not a string.
    ValueError
        If the filename does not match a supported patient naming pattern.

    Examples
    --------
    ``LIDC-IDRI-0001_30178_03192_cluster_0_slice_86.npy`` becomes
    ``LIDC-IDRI-0001``.

    ``LNDb-0311_finding_1_slice_86.npy`` becomes ``LNDb-0311``.
    """

    if not isinstance(filename, str):
        raise TypeError(
            "Expected filename to be a string, "
            f"but received {type(filename).__name__}."
        )

    for pattern in (
        LIDC_PATIENT_PATTERN,
        LNDB_PATIENT_PATTERN,
    ):
        match = pattern.match(filename)

        if match is not None:
            return match.group(1)

    raise ValueError(
        "Could not extract a supported patient identifier from filename: "
        f"{filename}"
    )


def get_dataset_name(
    patient_id: str,
) -> str:
    """Return the source dataset associated with a patient identifier.

    Parameters
    ----------
    patient_id : str
        Patient identifier extracted from a filename.

    Returns
    -------
    str
        Either ``"LIDC-IDRI"`` or ``"LNDb"``.

    Raises
    ------
    TypeError
        If ``patient_id`` is not a string.
    ValueError
        If the patient identifier does not belong to a supported dataset.
    """

    if not isinstance(patient_id, str):
        raise TypeError(
            "Expected patient_id to be a string, "
            f"but received {type(patient_id).__name__}."
        )

    if patient_id.startswith(LIDC_DATASET_NAME):
        return LIDC_DATASET_NAME

    if patient_id.startswith(LNDB_DATASET_NAME):
        return LNDB_DATASET_NAME

    raise ValueError(
        f"Unsupported patient identifier: {patient_id}"
    )


def split_patients(
    patient_ids: Sequence[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
) -> tuple[set[str], set[str], set[str]]:
    """Split patient identifiers into train, validation, and test subsets.

    The first split separates the training patients from the temporary
    validation-test subset. The second split divides the temporary subset
    according to the relative validation and test proportions.

    Parameters
    ----------
    patient_ids : Sequence[str]
        Unique patient identifiers from one dataset.
    train_ratio : float, default=TRAIN_RATIO
        Proportion assigned to the training subset.
    val_ratio : float, default=VAL_RATIO
        Proportion assigned to the validation subset.
    test_ratio : float, default=TEST_RATIO
        Proportion assigned to the test subset.
    random_seed : int, default=RANDOM_SEED
        Seed used by scikit-learn to make the split reproducible.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        Training, validation, and test patient identifier sets.

    Raises
    ------
    ValueError
        If the ratios are invalid, patient identifiers are duplicated, or
        there are too few patients to create all three subsets.
    """

    validate_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    unique_patient_ids = list(dict.fromkeys(patient_ids))

    if len(unique_patient_ids) != len(patient_ids):
        raise ValueError(
            "Expected patient_ids to contain unique patient identifiers."
        )

    if len(unique_patient_ids) < 3:
        raise ValueError(
            "At least three patients are required to create train, "
            "validation, and test subsets."
        )

    train_ids, temporary_ids = train_test_split(
        unique_patient_ids,
        train_size=train_ratio,
        shuffle=True,
        random_state=random_seed,
    )

    adjusted_val_ratio = val_ratio / (
        val_ratio + test_ratio
    )

    val_ids, test_ids = train_test_split(
        temporary_ids,
        train_size=adjusted_val_ratio,
        shuffle=True,
        random_state=random_seed,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    validate_split_membership(
        train_ids=train_set,
        val_ids=val_set,
        test_ids=test_set,
        expected_ids=set(unique_patient_ids),
    )

    return train_set, val_set, test_set


def validate_split_membership(
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
    expected_ids: set[str],
) -> None:
    """Validate that split assignments are complete and mutually exclusive.

    Parameters
    ----------
    train_ids : set[str]
        Patient identifiers assigned to training.
    val_ids : set[str]
        Patient identifiers assigned to validation.
    test_ids : set[str]
        Patient identifiers assigned to testing.
    expected_ids : set[str]
        Complete set of patient identifiers expected across all splits.

    Raises
    ------
    RuntimeError
        If patients overlap across splits or if the combined split membership
        does not match ``expected_ids``.
    """

    if train_ids & val_ids:
        raise RuntimeError(
            "Patient overlap detected between train and validation splits."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Patient overlap detected between train and test splits."
        )

    if val_ids & test_ids:
        raise RuntimeError(
            "Patient overlap detected between validation and test splits."
        )

    assigned_ids = train_ids | val_ids | test_ids

    if assigned_ids != expected_ids:
        missing_ids = sorted(expected_ids - assigned_ids)
        unexpected_ids = sorted(assigned_ids - expected_ids)

        raise RuntimeError(
            "Patient split membership is incomplete or inconsistent. "
            f"Missing: {missing_ids}. Unexpected: {unexpected_ids}."
        )


def collect_patient_files(
    ct_dir: str | Path,
) -> dict[str, list[str]]:
    """Group CT NumPy filenames by patient identifier.

    Parameters
    ----------
    ct_dir : str | Path
        Directory containing segmentation CT files in NumPy format.

    Returns
    -------
    dict[str, list[str]]
        Mapping from patient identifiers to sorted CT filenames.

    Raises
    ------
    FileNotFoundError
        If ``ct_dir`` does not exist or contains no ``.npy`` files.
    NotADirectoryError
        If ``ct_dir`` is not a directory.
    """

    ct_dir = Path(ct_dir)

    if not ct_dir.exists():
        raise FileNotFoundError(
            f"CT directory does not exist: {ct_dir}"
        )

    if not ct_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a CT directory, but received: {ct_dir}"
        )

    ct_files = sorted(ct_dir.glob("*.npy"))

    if not ct_files:
        raise FileNotFoundError(
            f"No .npy files found in CT directory: {ct_dir}"
        )

    patient_files: dict[str, list[str]] = {}

    for ct_file in tqdm(
        ct_files,
        desc="Grouping files by patient",
        unit="file",
    ):
        patient_id = extract_patient_id(ct_file.name)

        patient_files.setdefault(
            patient_id,
            [],
        ).append(ct_file.name)

    for filenames in patient_files.values():
        filenames.sort()

    return patient_files


def assign_patient_split(
    patient_id: str,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> str:
    """Return the split assigned to one patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier to classify.
    train_ids : set[str]
        Patient identifiers assigned to training.
    val_ids : set[str]
        Patient identifiers assigned to validation.
    test_ids : set[str]
        Patient identifiers assigned to testing.

    Returns
    -------
    str
        One of ``"train"``, ``"val"``, or ``"test"``.

    Raises
    ------
    RuntimeError
        If the patient does not belong to exactly one split.
    """

    matching_splits: list[str] = []

    if patient_id in train_ids:
        matching_splits.append(TRAIN_SPLIT)

    if patient_id in val_ids:
        matching_splits.append(VAL_SPLIT)

    if patient_id in test_ids:
        matching_splits.append(TEST_SPLIT)

    if len(matching_splits) != 1:
        raise RuntimeError(
            f"Expected patient {patient_id} to belong to exactly one split, "
            f"but found assignments: {matching_splits}."
        )

    return matching_splits[0]


def build_split_metadata(
    patient_files: dict[str, list[str]],
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> pd.DataFrame:
    """Build file-level metadata from patient-level split assignments.

    Parameters
    ----------
    patient_files : dict[str, list[str]]
        Mapping from patient identifiers to CT filenames.
    train_ids : set[str]
        Patient identifiers assigned to training.
    val_ids : set[str]
        Patient identifiers assigned to validation.
    test_ids : set[str]
        Patient identifiers assigned to testing.

    Returns
    -------
    pandas.DataFrame
        File-level split metadata with columns ``dataset``, ``patient_id``,
        ``filename``, and ``split``.
    """

    metadata_records: list[dict[str, str]] = []

    for patient_id in tqdm(
        sorted(patient_files),
        desc="Generating split metadata",
        unit="patient",
    ):
        split_name = assign_patient_split(
            patient_id=patient_id,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        dataset_name = get_dataset_name(patient_id)

        for filename in patient_files[patient_id]:
            metadata_records.append(
                {
                    "dataset": dataset_name,
                    "patient_id": patient_id,
                    "filename": filename,
                    "split": split_name,
                }
            )

    metadata_df = pd.DataFrame(
        metadata_records,
        columns=[
            "dataset",
            "patient_id",
            "filename",
            "split",
        ],
    )

    return metadata_df.sort_values(
        by=[
            "split",
            "dataset",
            "patient_id",
            "filename",
        ],
        kind="stable",
    ).reset_index(drop=True)


def save_split_metadata(
    metadata_df: pd.DataFrame,
    output_csv: str | Path,
) -> None:
    """Save split metadata as a CSV file.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Metadata table to save.
    output_csv : str | Path
        Destination path using the ``.csv`` suffix.

    Raises
    ------
    ValueError
        If the metadata table is empty or ``output_csv`` does not use the
        ``.csv`` suffix.
    """

    if metadata_df.empty:
        raise ValueError(
            "Cannot save an empty split metadata table."
        )

    output_csv = Path(output_csv)

    if output_csv.suffix.lower() != ".csv":
        raise ValueError(
            "Expected output_csv to use the .csv suffix, "
            f"but received: {output_csv}"
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_df.to_csv(
        output_csv,
        index=False,
    )


def select_patients_by_prefix(
    patient_ids: Iterable[str],
    prefix: str,
) -> list[str]:
    """Select and sort patient identifiers beginning with a prefix.

    Parameters
    ----------
    patient_ids : Iterable[str]
        Patient identifiers to filter.
    prefix : str
        Required patient identifier prefix.

    Returns
    -------
    list[str]
        Sorted identifiers matching the prefix.
    """

    return sorted(
        patient_id
        for patient_id in patient_ids
        if patient_id.startswith(prefix)
    )


def print_split_summary(
    metadata_df: pd.DataFrame,
    lidc_patients: Sequence[str],
    lndb_patients: Sequence[str],
    lidc_splits: tuple[set[str], set[str], set[str]],
    lndb_splits: tuple[set[str], set[str], set[str]],
    output_csv: Path,
) -> None:
    """Print patient-level and file-level split summaries.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Generated file-level split metadata.
    lidc_patients : Sequence[str]
        All LIDC-IDRI patient identifiers.
    lndb_patients : Sequence[str]
        All LNDb patient identifiers.
    lidc_splits : tuple[set[str], set[str], set[str]]
        LIDC-IDRI train, validation, and test patient sets.
    lndb_splits : tuple[set[str], set[str], set[str]]
        LNDb train, validation, and test patient sets.
    output_csv : pathlib.Path
        Path where the metadata CSV was saved.
    """

    lidc_train, lidc_val, lidc_test = lidc_splits
    lndb_train, lndb_val, lndb_test = lndb_splits

    print()
    print("=" * 60)
    print("Patient summary")
    print("=" * 60)

    print(f"\n{LIDC_DATASET_NAME}: {len(lidc_patients)} patients")
    print(f"  Train : {len(lidc_train)}")
    print(f"  Val   : {len(lidc_val)}")
    print(f"  Test  : {len(lidc_test)}")

    print(f"\n{LNDB_DATASET_NAME}: {len(lndb_patients)} patients")
    print(f"  Train : {len(lndb_train)}")
    print(f"  Val   : {len(lndb_val)}")
    print(f"  Test  : {len(lndb_test)}")

    print()
    print("=" * 60)
    print("File summary")
    print("=" * 60)
    print()

    file_summary = (
        metadata_df.groupby(
            ["dataset", "split"],
            observed=True,
        )
        .size()
        .rename("file_count")
        .reset_index()
    )

    print(file_summary.to_string(index=False))

    print()
    print("=" * 60)
    print(f"Split metadata saved to:\n{output_csv}")
    print("=" * 60)


def main() -> None:
    """Generate patient-level segmentation dataset split metadata."""

    validate_split_ratios(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    patient_files = collect_patient_files(
        ct_dir=CT_DIR,
    )

    lidc_patients = select_patients_by_prefix(
        patient_ids=patient_files,
        prefix=LIDC_DATASET_NAME,
    )

    lndb_patients = select_patients_by_prefix(
        patient_ids=patient_files,
        prefix=LNDB_DATASET_NAME,
    )

    if not lidc_patients:
        raise RuntimeError(
            "No LIDC-IDRI patients were found in the CT directory."
        )

    if not lndb_patients:
        raise RuntimeError(
            "No LNDb patients were found in the CT directory."
        )

    lidc_splits = split_patients(
        patient_ids=lidc_patients,
    )

    lndb_splits = split_patients(
        patient_ids=lndb_patients,
    )

    lidc_train, lidc_val, lidc_test = lidc_splits
    lndb_train, lndb_val, lndb_test = lndb_splits

    train_patients = lidc_train | lndb_train
    val_patients = lidc_val | lndb_val
    test_patients = lidc_test | lndb_test

    all_patient_ids = set(patient_files)

    validate_split_membership(
        train_ids=train_patients,
        val_ids=val_patients,
        test_ids=test_patients,
        expected_ids=all_patient_ids,
    )

    metadata_df = build_split_metadata(
        patient_files=patient_files,
        train_ids=train_patients,
        val_ids=val_patients,
        test_ids=test_patients,
    )

    save_split_metadata(
        metadata_df=metadata_df,
        output_csv=OUTPUT_CSV,
    )

    print_split_summary(
        metadata_df=metadata_df,
        lidc_patients=lidc_patients,
        lndb_patients=lndb_patients,
        lidc_splits=lidc_splits,
        lndb_splits=lndb_splits,
        output_csv=OUTPUT_CSV,
    )


if __name__ == "__main__":
    main()
