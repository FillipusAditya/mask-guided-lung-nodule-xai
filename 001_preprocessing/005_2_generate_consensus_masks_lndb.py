from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm


# Make packages inside 001_preprocessing importable without installation.
PREPROCESSING_DIR = Path(__file__).resolve().parent

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))

from lndb_consensus import (
    prepare_scan_data,
    process_scan,
    save_consensus_slices,
)


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory of ct.mhd
DATA_DIR = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "lndb" 
    / "data"
)

# Directory of mask.mhd
MASK_DIR = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "lndb" 
    / "masks"
)

# Metadata containing the selected nodule
CSV_PATH = (
    PROJECT_ROOT 
    / "000_dataset" 
    / "_lndb" 
    / "000_metadata" 
    / "003_consensus_clean.csv"
)

# Directory where consensus masks will be stored
OUTPUT_DIR = (
    PROJECT_ROOT 
    / "000_dataset"
    / "_lndb" 
    / "005_mask_consensus_npy"
)

# Metadata with and additional column containing the consensus mask path
OUTPUT_METADATA_CSV = (
    PROJECT_ROOT
    / "000_dataset"
    / "_lndb"
    / "000_metadata"
    / "004_consensus_clean_path.csv"
)


def generate_consensus_masks(
    csv_path: Path,
    data_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    output_metadata_csv: Path,
) -> None:
    """
    Generate consensus segmentation masks for every LNDb finding.

    Workflow
    --------
    1. Load cleaned metadata.
    2. Process every finding.
    3. Generate consensus mask.
    4. Save every consensus slice as an individual NumPy file.
    5. Save updated metadata containing the output directory.
    """

    metadata_df = pd.read_csv(csv_path)

    output_metadata_df = metadata_df.copy()
    output_metadata_df["consensus_mask_path"] = None

    for index, row in tqdm(
        metadata_df.iterrows(),
        total=len(metadata_df),
        desc="Generating consensus masks",
        unit="finding",
    ):
        # Prepare scan
        scan = prepare_scan_data(
            row=row,
            data_dir=data_dir,
            mask_dir=mask_dir,
        )

        # Generate consensus masks
        scan = process_scan(scan)

        # Output directory
        cluster_output_dir = (
            output_dir
            / f"LNDb-{scan['lndb_id']:04d}"
            / f"finding_{scan['finding_id']}"
        )

        cluster_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save output path into metadata
        output_metadata_df.loc[
            index,
            "consensus_mask_path",
        ] = str(cluster_output_dir)

        # Save consensus mask
        save_consensus_slices(
            scan=scan,
            output_dir=cluster_output_dir,
        )

    # Save updated metadata
    output_metadata_df.to_csv(
        output_metadata_csv,
        index=False,
    )


if __name__ == "__main__":
    generate_consensus_masks(
        csv_path=CSV_PATH,
        data_dir=DATA_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        output_metadata_csv=OUTPUT_METADATA_CSV,
    )
