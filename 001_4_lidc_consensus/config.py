"""Shared paths and parameters for the LIDC-IDRI consensus pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIDC_METADATA_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "000_metadata"
INPUT_METADATA_CSV = LIDC_METADATA_DIR / "002_cluster_metadata_cleaned.csv"
OUTPUT_METADATA_CSV = (
    LIDC_METADATA_DIR / "003_cluster_metadata_cleaned_path.csv"
)
CONSENSUS_MASK_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lidc" / "005_mask_consensus_npy"
)
SEGMENTED_NODULE_PNG_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lidc" / "006_segmented_nodule_png"
)
QUALITY_CONTROL_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lidc" / "007_consensus_quality_control"
)

WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1600.0
MEDIAN_FILTER_SIZE = (1, 3, 3)

# A voxel belongs to the consensus mask when at least this fraction of the
# cluster's radiologists annotated it.
DEFAULT_CONSENSUS_LEVEL = 0.5
