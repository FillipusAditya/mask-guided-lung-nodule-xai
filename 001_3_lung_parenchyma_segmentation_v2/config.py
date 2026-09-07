"""Paths and parameters for protected lung-parenchyma segmentation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIDC_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "lidc_idri"
LNDB_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "data"

OUTPUT_ROOT = PROJECT_ROOT / "000_dataset_v2" / "lung_parenchyma_v3"
LIDC_MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks" / "lidc"
LNDB_MASK_OUTPUT_DIR = OUTPUT_ROOT / "masks" / "lndb"
LIDC_PARENCHYMA_OUTPUT_DIR = OUTPUT_ROOT / "parenchyma" / "lidc"
LNDB_PARENCHYMA_OUTPUT_DIR = OUTPUT_ROOT / "parenchyma" / "lndb"
REPORT_OUTPUT_DIR = OUTPUT_ROOT / "reports"
QUALITY_CONTROL_DIR = OUTPUT_ROOT / "quality_control"

LIDC_NODULE_METADATA_CSV = (
    PROJECT_ROOT
    / "000_dataset_v2"
    / "_lidc"
    / "000_metadata"
    / "003_cluster_metadata_cleaned_path.csv"
)
LNDB_NODULE_METADATA_CSV = (
    PROJECT_ROOT
    / "000_dataset_v2"
    / "_lndb"
    / "000_metadata"
    / "004_consensus_clean_path.csv"
)
LIDC_NODULE_MASK_DIR = (
    PROJECT_ROOT / "000_dataset_v2" / "_lidc" / "004_mask_consensus_npy"
)
LNDB_NODULE_MASK_DIR = (
    PROJECT_ROOT / "000_dataset_v2" / "_lndb" / "004_mask_consensus_npy"
)

# Candidate mask: threshold -> clear border -> table geometry -> components -> fill.
HU_THRESHOLD = -320
NUM_LARGEST_COMPONENTS = 4

# Wide, flat lower-image table artifacts are removed from every candidate.
TABLE_LOWER_CENTER_Y_RATIO = 0.70
TABLE_MIN_WIDTH_RATIO = 0.40
TABLE_MAX_HEIGHT_WIDTH_RATIO = 0.35

# Additional small-peripheral cleanup is restricted to the reference slice.
TABLE_COMPONENT_AREA_THRESHOLD = 0.02
TABLE_MIN_Y_RATIO = 0.30
TABLE_MAX_Y_RATIO = 0.60
TRACHEA_AREA_THRESHOLD = 0.0069
TRACHEA_CENTER_HALF_WIDTH_RATIO = 0.05
TRACHEA_MAX_Y_RATIO = 0.55

# Protected bidirectional component propagation.
REFERENCE_DILATION_SCHEDULE = (3, 6, 10, 15)
MIN_COMPONENT_OVERLAP_PIXELS = 10

# Post-propagation boundary repair.
BOUNDARY_REPAIR_RADIUS = 16

# Median filtering and lung-window normalization.
MEDIAN_FILTER_SIZE = (1, 3, 3)
WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1600.0
