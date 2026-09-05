"""Shared paths and parameters for the LNDb consensus pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LNDB_DATA_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "data"
LNDB_MASK_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "masks"

LNDB_METADATA_DIR = PROJECT_ROOT / "000_dataset" / "_lndb" / "000_metadata"
INPUT_METADATA_CSV = LNDB_METADATA_DIR / "002_trainNodules_gt_clean.csv"
CONSENSUS_METADATA_CSV = LNDB_METADATA_DIR / "003_consensus_clean.csv"
CONSENSUS_PATH_METADATA_CSV = (
    LNDB_METADATA_DIR / "004_consensus_clean_path.csv"
)

CONSENSUS_MASK_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lndb" / "005_mask_consensus_npy"
)
SEGMENTED_NODULE_PNG_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lndb" / "006_segmented_nodule_png"
)
QUALITY_CONTROL_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lndb" / "007_consensus_quality_control"
)

WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1600.0
MEDIAN_FILTER_SIZE = (1, 3, 3)

# A voxel belongs to the consensus mask when at least this fraction of the
# participating radiologists annotated it. With three radiologists and 0.5,
# at least two annotations are required.
DEFAULT_CONSENSUS_LEVEL = 0.5
