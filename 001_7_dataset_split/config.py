"""Paths and reproducibility settings for dataset splitting."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "000_dataset" / "_segmentation_dataset_v2"

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

COMBINED_LABELED_METADATA_CSV = (
    DATASET_DIR / "001_labeled_metadata_lidc_lndb.csv"
)
COMBINED_HOLDOUT_CSV = DATASET_DIR / "001_holdout_split_lidc_lndb.csv"
COMBINED_ERROR_LOG_JSON = (
    DATASET_DIR / "001_holdout_split_lidc_lndb.json"
)

SINGLE_OUTPUT_INDEX = {"lidc": "002", "lndb": "003"}

N_SPLITS = 5
RANDOM_SEED = 42
SHUFFLE = True
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CV_METADATA_CSV = (
    DATASET_DIR / f"004_classification_cv_{N_SPLITS}fold_seed{RANDOM_SEED}.csv"
)
CV_SUMMARY_CSV = DATASET_DIR / (
    f"004_classification_cv_{N_SPLITS}fold_seed{RANDOM_SEED}_summary.csv"
)
