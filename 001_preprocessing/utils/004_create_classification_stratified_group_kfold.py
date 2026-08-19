"""Create reproducible patient-grouped folds for classification training.

The utility reads the existing slice-level
``001_holdout_split_lidc_lndb.csv``, keeps the current test split as an
untouched holdout set, and combines the current train and validation splits
into one development set. Fold assignment is performed on one row per nodule
with ``StratifiedGroupKFold``:

* stratification target: source dataset and nodule label;
* non-overlapping group: source dataset and patient identifier;
* assignment unit: nodule;
* training input unit written to the output: slice.

Assigning folds at nodule level prevents nodules with many slices from having
more influence on fold construction. Grouping at patient level guarantees that
all nodules and slices belonging to one patient remain in the same fold.

For a selected validation fold ``k``, classification training should use:

* train: ``cv_role == "development"`` and ``cv_fold != k``;
* validation: ``cv_role == "development"`` and ``cv_fold == k``;
* final test: ``cv_role == "holdout_test"`` and ``cv_fold == -1``.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------
# PATHS
# ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "000_dataset" / "_segmentation_dataset_v2"

INPUT_METADATA_CSV = DATASET_DIR / "001_holdout_split_lidc_lndb.csv"
OUTPUT_METADATA_CSV = (
    DATASET_DIR / "classification_cv_5fold_seed42.csv"
)
OUTPUT_SUMMARY_CSV = (
    DATASET_DIR / "classification_cv_5fold_seed42_summary.csv"
)


# ---------------------------------
# CROSS-VALIDATION CONFIGURATION
# ---------------------------------
N_SPLITS = 5
RANDOM_SEED = 42
SHUFFLE = True

DEVELOPMENT_SPLITS = ("train", "val")
HOLDOUT_SPLIT = "test"

DEVELOPMENT_ROLE = "development"
HOLDOUT_ROLE = "holdout_test"
HOLDOUT_FOLD = -1

GROUP_SEPARATOR = "::"
SLICE_SUFFIX_PATTERN = re.compile(r"_slice_\d+\.npy$")

REQUIRED_COLUMNS = {
    "dataset",
    "patient_id",
    "filename",
    "ct_windowed_path",
    "ct_parenchyma_path",
    "mask_path",
    "label",
    "split",
}


def require_csv(path: str | Path, description: str) -> Path:
    """Return an existing CSV path or raise a descriptive exception."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected {description} to be a CSV file: {path}")
    return path


def validate_cv_configuration(n_splits: int, random_seed: int) -> None:
    """Validate cross-validation configuration values."""

    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise TypeError("n_splits must be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError("random_seed must be an integer.")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")


def load_metadata(input_csv: str | Path) -> pd.DataFrame:
    """Load and validate the source slice-level metadata."""

    input_csv = require_csv(input_csv, "Input metadata CSV")
    metadata_df = pd.read_csv(input_csv)

    if metadata_df.empty:
        raise ValueError(f"Input metadata is empty: {input_csv}")

    missing_columns = REQUIRED_COLUMNS - set(metadata_df.columns)
    if missing_columns:
        raise ValueError(
            "Input metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    metadata_df = metadata_df.copy()
    string_columns = [
        "dataset",
        "patient_id",
        "filename",
        "label",
        "split",
    ]
    for column in string_columns:
        if metadata_df[column].isna().any():
            missing_count = int(metadata_df[column].isna().sum())
            raise ValueError(
                f"Column {column!r} contains {missing_count} missing values."
            )

        metadata_df[column] = (
            metadata_df[column]
            .astype("string")
            .str.strip()
        )
        empty_mask = metadata_df[column].eq("")
        if empty_mask.any():
            raise ValueError(
                f"Column {column!r} contains "
                f"{int(empty_mask.sum())} empty values."
            )

    allowed_splits = {*DEVELOPMENT_SPLITS, HOLDOUT_SPLIT}
    observed_splits = set(metadata_df["split"].unique())
    unexpected_splits = observed_splits - allowed_splits
    if unexpected_splits:
        raise ValueError(
            f"Unexpected split values: {sorted(unexpected_splits)}"
        )

    missing_splits = allowed_splits - observed_splits
    if missing_splits:
        raise ValueError(
            f"Required split values are missing: {sorted(missing_splits)}"
        )

    duplicate_mask = metadata_df.duplicated(
        subset=["dataset", "filename"],
        keep=False,
    )
    if duplicate_mask.any():
        duplicates = (
            metadata_df.loc[duplicate_mask, ["dataset", "filename"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate dataset/filename pairs were found. "
            f"Examples: {duplicates}"
        )

    return metadata_df


def add_identifiers(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Add collision-safe patient-group and nodule identifiers."""

    metadata_df = metadata_df.copy()
    metadata_df["cv_group_id"] = (
        metadata_df["dataset"]
        + GROUP_SEPARATOR
        + metadata_df["patient_id"]
    )

    metadata_df["nodule_id"] = metadata_df["filename"].str.replace(
        SLICE_SUFFIX_PATTERN,
        "",
        regex=True,
    )
    invalid_nodule_mask = metadata_df["nodule_id"].eq(
        metadata_df["filename"]
    )
    if invalid_nodule_mask.any():
        invalid_examples = (
            metadata_df.loc[invalid_nodule_mask, "filename"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            "Could not derive nodule_id from filenames. Expected filenames "
            "to end with '_slice_<integer>.npy'. Examples: "
            f"{invalid_examples}"
        )

    metadata_df["cv_nodule_id"] = (
        metadata_df["dataset"]
        + GROUP_SEPARATOR
        + metadata_df["nodule_id"]
    )

    return metadata_df


def validate_source_membership(metadata_df: pd.DataFrame) -> None:
    """Ensure holdout patients do not occur in the development set."""

    development_mask = metadata_df["split"].isin(DEVELOPMENT_SPLITS)
    holdout_mask = metadata_df["split"].eq(HOLDOUT_SPLIT)

    development_groups = set(
        metadata_df.loc[development_mask, "cv_group_id"]
    )
    holdout_groups = set(metadata_df.loc[holdout_mask, "cv_group_id"])
    overlap = development_groups & holdout_groups
    if overlap:
        raise RuntimeError(
            "Patient leakage exists between development and holdout data. "
            f"Overlapping groups: {sorted(overlap)[:10]}"
        )


def build_nodule_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse development slices into one validated row per nodule."""

    development_df = metadata_df.loc[
        metadata_df["split"].isin(DEVELOPMENT_SPLITS)
    ].copy()
    if development_df.empty:
        raise ValueError("No development samples were found.")

    consistency = (
        development_df.groupby("cv_nodule_id", sort=False)
        .agg(
            dataset_count=("dataset", "nunique"),
            patient_count=("patient_id", "nunique"),
            group_count=("cv_group_id", "nunique"),
            label_count=("label", "nunique"),
        )
    )
    inconsistent_mask = consistency.ne(1).any(axis=1)
    if inconsistent_mask.any():
        examples = consistency.index[inconsistent_mask].tolist()[:10]
        raise ValueError(
            "Each nodule must map to exactly one dataset, patient, group, "
            f"and label. Inconsistent nodules: {examples}"
        )

    nodule_df = (
        development_df[
            [
                "dataset",
                "patient_id",
                "cv_group_id",
                "nodule_id",
                "cv_nodule_id",
                "label",
            ]
        ]
        .drop_duplicates(subset=["cv_nodule_id"])
        .sort_values(
            by=["dataset", "patient_id", "nodule_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # Combining source and label keeps both domain and class proportions as
    # similar as possible across folds while patient groups remain intact.
    nodule_df["cv_stratum"] = (
        nodule_df["dataset"]
        + GROUP_SEPARATOR
        + nodule_df["label"]
    )

    return nodule_df


def validate_strata(nodule_df: pd.DataFrame, n_splits: int) -> None:
    """Ensure each dataset/label stratum spans enough patient groups."""

    stratum_group_counts = (
        nodule_df.groupby("cv_stratum", observed=True)["cv_group_id"]
        .nunique()
        .sort_index()
    )
    insufficient = stratum_group_counts[stratum_group_counts < n_splits]
    if not insufficient.empty:
        raise ValueError(
            "Every dataset/label stratum must contain at least n_splits "
            "distinct patient groups. Insufficient strata: "
            f"{insufficient.to_dict()}"
        )


def assign_nodule_folds(
    nodule_df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    random_seed: int = RANDOM_SEED,
    shuffle: bool = SHUFFLE,
) -> pd.DataFrame:
    """Assign one validation fold to every development patient group."""

    validate_cv_configuration(n_splits, random_seed)
    validate_strata(nodule_df, n_splits)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_seed if shuffle else None,
    )
    assigned_df = nodule_df.copy()
    assigned_df["cv_fold"] = np.full(len(assigned_df), -1, dtype=np.int64)

    features = np.zeros((len(assigned_df), 1), dtype=np.uint8)
    for fold_index, (train_indices, validation_indices) in enumerate(
        splitter.split(
            X=features,
            y=assigned_df["cv_stratum"],
            groups=assigned_df["cv_group_id"],
        )
    ):
        train_groups = set(
            assigned_df.iloc[train_indices]["cv_group_id"]
        )
        validation_groups = set(
            assigned_df.iloc[validation_indices]["cv_group_id"]
        )
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in fold {fold_index}: "
                f"{sorted(overlap)[:10]}"
            )

        assigned_df.loc[validation_indices, "cv_fold"] = fold_index

    if assigned_df["cv_fold"].lt(0).any():
        raise RuntimeError("Some development nodules were not assigned a fold.")

    folds_per_group = assigned_df.groupby("cv_group_id")["cv_fold"].nunique()
    invalid_groups = folds_per_group[folds_per_group.ne(1)]
    if not invalid_groups.empty:
        raise RuntimeError(
            "Each patient group must belong to exactly one fold. Invalid "
            f"groups: {invalid_groups.index.tolist()[:10]}"
        )

    observed_folds = set(assigned_df["cv_fold"].unique())
    expected_folds = set(range(n_splits))
    if observed_folds != expected_folds:
        raise RuntimeError(
            f"Unexpected fold assignments. Expected {expected_folds}; "
            f"observed {observed_folds}."
        )

    return assigned_df


def add_slice_fold_assignments(
    metadata_df: pd.DataFrame,
    assigned_nodule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map patient folds back to every slice and mark the holdout set."""

    group_fold_counts = assigned_nodule_df.groupby("cv_group_id")[
        "cv_fold"
    ].nunique()
    if group_fold_counts.ne(1).any():
        raise RuntimeError("Patient groups have inconsistent nodule folds.")

    group_to_fold = (
        assigned_nodule_df.groupby("cv_group_id", sort=False)["cv_fold"]
        .first()
        .to_dict()
    )

    output_df = metadata_df.copy()
    development_mask = output_df["split"].isin(DEVELOPMENT_SPLITS)
    holdout_mask = output_df["split"].eq(HOLDOUT_SPLIT)

    output_df["cv_role"] = HOLDOUT_ROLE
    output_df.loc[development_mask, "cv_role"] = DEVELOPMENT_ROLE
    output_df["cv_fold"] = HOLDOUT_FOLD
    output_df.loc[development_mask, "cv_fold"] = (
        output_df.loc[development_mask, "cv_group_id"]
        .map(group_to_fold)
    )

    if output_df.loc[development_mask, "cv_fold"].isna().any():
        missing_groups = (
            output_df.loc[
                development_mask & output_df["cv_fold"].isna(),
                "cv_group_id",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise RuntimeError(
            f"Development groups are missing fold assignments: {missing_groups}"
        )

    output_df["cv_fold"] = output_df["cv_fold"].astype("int64")
    if not output_df.loc[holdout_mask, "cv_fold"].eq(HOLDOUT_FOLD).all():
        raise RuntimeError("Holdout samples must use cv_fold = -1.")

    role_order = pd.CategoricalDtype(
        categories=[DEVELOPMENT_ROLE, HOLDOUT_ROLE],
        ordered=True,
    )
    output_df["cv_role"] = output_df["cv_role"].astype(role_order)
    output_df = output_df.sort_values(
        by=[
            "cv_role",
            "cv_fold",
            "dataset",
            "patient_id",
            "nodule_id",
            "filename",
        ],
        kind="stable",
    ).reset_index(drop=True)
    output_df["cv_role"] = output_df["cv_role"].astype("string")

    return output_df


def validate_output_assignments(
    output_df: pd.DataFrame,
    n_splits: int,
) -> None:
    """Validate patient isolation, complete folds, and validation coverage."""

    development_df = output_df.loc[
        output_df["cv_role"].eq(DEVELOPMENT_ROLE)
    ]
    holdout_df = output_df.loc[output_df["cv_role"].eq(HOLDOUT_ROLE)]

    development_groups = set(development_df["cv_group_id"])
    holdout_groups = set(holdout_df["cv_group_id"])
    if development_groups & holdout_groups:
        raise RuntimeError("Development and holdout patient groups overlap.")

    folds_per_group = development_df.groupby("cv_group_id")[
        "cv_fold"
    ].nunique()
    if folds_per_group.ne(1).any():
        raise RuntimeError("A development patient appears in multiple folds.")

    folds_per_nodule = development_df.groupby("cv_nodule_id")[
        "cv_fold"
    ].nunique()
    if folds_per_nodule.ne(1).any():
        raise RuntimeError("A development nodule appears in multiple folds.")

    expected_strata = set(
        development_df["dataset"]
        + GROUP_SEPARATOR
        + development_df["label"]
    )
    for fold_index in range(n_splits):
        validation_df = development_df.loc[
            development_df["cv_fold"].eq(fold_index)
        ]
        if validation_df.empty:
            raise RuntimeError(f"Fold {fold_index} is empty.")

        observed_strata = set(
            validation_df["dataset"]
            + GROUP_SEPARATOR
            + validation_df["label"]
        )
        if observed_strata != expected_strata:
            raise RuntimeError(
                f"Fold {fold_index} does not cover all dataset/label strata. "
                f"Missing: {sorted(expected_strata - observed_strata)}"
            )


def build_fold_summary(output_df: pd.DataFrame) -> pd.DataFrame:
    """Build fold-level slice, nodule, and patient distribution counts."""

    summary_df = (
        output_df.groupby(
            ["cv_role", "cv_fold", "dataset", "label"],
            observed=True,
        )
        .agg(
            slice_count=("filename", "size"),
            nodule_count=("cv_nodule_id", "nunique"),
            patient_count=("cv_group_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            by=["cv_role", "cv_fold", "dataset", "label"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return summary_df


def save_csv(dataframe: pd.DataFrame, output_csv: str | Path) -> None:
    """Save a non-empty dataframe to a CSV file."""

    if dataframe.empty:
        raise ValueError("Cannot save an empty dataframe.")

    output_csv = Path(output_csv)
    if output_csv.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV output path: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv, index=False)


def print_summary(
    output_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Print the generated CV configuration and distribution summary."""

    development_df = output_df.loc[
        output_df["cv_role"].eq(DEVELOPMENT_ROLE)
    ]
    holdout_df = output_df.loc[output_df["cv_role"].eq(HOLDOUT_ROLE)]

    print()
    print("=" * 88)
    print("Classification Stratified Group K-Fold summary")
    print("=" * 88)
    print(f"Input metadata     : {INPUT_METADATA_CSV}")
    print(f"Output metadata    : {OUTPUT_METADATA_CSV}")
    print(f"Output summary     : {OUTPUT_SUMMARY_CSV}")
    print(f"Folds              : {N_SPLITS}")
    print(f"Random seed        : {RANDOM_SEED}")
    print("Group              : dataset + patient_id")
    print("Stratification     : dataset + nodule label")
    print(f"Development slices : {len(development_df)}")
    print(
        "Development nodules: "
        f"{development_df['cv_nodule_id'].nunique()}"
    )
    print(
        "Development patients: "
        f"{development_df['cv_group_id'].nunique()}"
    )
    print(f"Holdout slices     : {len(holdout_df)}")
    print(
        "Holdout nodules      : "
        f"{holdout_df['cv_nodule_id'].nunique()}"
    )
    print(
        "Holdout patients     : "
        f"{holdout_df['cv_group_id'].nunique()}"
    )
    print()
    print(summary_df.to_string(index=False))
    print("=" * 88)


def main() -> None:
    """Generate and validate the classification cross-validation CSVs."""

    validate_cv_configuration(N_SPLITS, RANDOM_SEED)
    metadata_df = load_metadata(INPUT_METADATA_CSV)
    metadata_df = add_identifiers(metadata_df)
    validate_source_membership(metadata_df)

    nodule_df = build_nodule_metadata(metadata_df)
    assigned_nodule_df = assign_nodule_folds(
        nodule_df,
        n_splits=N_SPLITS,
        random_seed=RANDOM_SEED,
        shuffle=SHUFFLE,
    )
    output_df = add_slice_fold_assignments(
        metadata_df,
        assigned_nodule_df,
    )
    validate_output_assignments(output_df, n_splits=N_SPLITS)

    summary_df = build_fold_summary(output_df)
    save_csv(output_df, OUTPUT_METADATA_CSV)
    save_csv(summary_df, OUTPUT_SUMMARY_CSV)
    print_summary(output_df, summary_df)


if __name__ == "__main__":
    main()
