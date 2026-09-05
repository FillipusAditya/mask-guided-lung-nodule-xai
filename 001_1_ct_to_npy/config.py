"""Project paths used by the CT-to-NumPy conversion pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LNDB_INPUT_DIR = PROJECT_ROOT / "000_dataset" / "lndb" / "data"

# The v3 directories intentionally match the latest legacy script. This keeps
# the converter from overwriting the established 001_volume_npy datasets.
LIDC_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "001_volume_npy_v3"
LNDB_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lndb" / "001_volume_npy_v3"

LIDC_PNG_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lidc" / "001_volume_png_v3"
LNDB_PNG_OUTPUT_DIR = PROJECT_ROOT / "000_dataset" / "_lndb" / "001_volume_png_v3"

# PNG files are 8-bit previews; NumPy files retain the original HU values.
PNG_WINDOW_MIN = -1400.0
PNG_WINDOW_MAX = 200.0
