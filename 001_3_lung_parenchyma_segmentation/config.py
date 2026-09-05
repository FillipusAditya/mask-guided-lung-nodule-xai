"""Shared paths and preprocessing parameters."""

from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LNDB_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "data"
LNDB_MASK_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "masks"

LIDC_CLUSTER_METADATA_CSV = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lidc"
    / "000_metadata"
    / "002_cluster_metadata_cleaned.csv"
)
LNDB_FINDING_METADATA_CSV = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "000_metadata"
    / "002_trainNodules_gt_clean.csv"
)

LIDC_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "lung_parenchyma" / "lidc"
LNDB_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "lung_parenchyma" / "lndb"
QUALITY_CONTROL_DIR = (
    PROJECT_ROOT / "000_dataset" / "lung_parenchyma" / "quality_control"
)

# Lung window: W=1600 and L=-600 gives the range [-1400, 200] HU.
WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1600.0

HU_THRESHOLD = -320
NUM_LARGEST_COMPONENTS = 4
TRACHEA_AREA_THRESHOLD = 0.0069
DILATION_ITERATIONS = 5
EROSION_ITERATIONS = 5
MEDIAN_FILTER_SIZE = (1, 3, 3)

# Disabled stages pass their input mask through unchanged.
STAGE_ENABLED = OrderedDict(
    [
        ("1. Threshold HU", True),
        ("2. Remove border objects", True),
        ("3. Keep largest components", True),
        ("4. Fill holes", True),
        ("5. Remove trachea", True),
        ("6. Remove table/artifacts", False),
        ("7. Binary dilation", True),
        ("8. Fill holes after dilation", True),
    ]
)
