"""Project paths and shared CT preprocessing parameters."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIDC_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "001_volume_npy"
LNDB_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lndb" / "001_volume_npy"

# New directories protect the existing outputs produced with W=1500 and no
# median filter. Downstream consumers can migrate after validation.
LIDC_OUTPUT_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lidc" / "002_windowed_median_npy"
)
LNDB_OUTPUT_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lndb" / "002_windowed_median_npy"
)
LIDC_PNG_OUTPUT_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lidc" / "002_windowed_median_png"
)
LNDB_PNG_OUTPUT_DIR = (
    PROJECT_ROOT / "000_dataset" / "_lndb" / "002_windowed_median_png"
)

# Identical to 001_3_lung_parenchyma_segmentation/config.py.
WINDOW_LEVEL = -600.0
WINDOW_WIDTH = 1600.0
MEDIAN_FILTER_SIZE = (1, 3, 3)
