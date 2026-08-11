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
)


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory of ct.mhd
DATA_DIR = PROJECT_ROOT / "dataset" / "lndb" / "data"

# Directory of mask.mhd
MASK_DIR = PROJECT_ROOT / "dataset" / "lndb" / "masks"

# Input CSV
INPUT_CSV = (
    PROJECT_ROOT 
    / "dataset" 
    / "_lndb" 
    / "000_metadata" 
    / "trainNodules_gt_clean_v2.csv"
)

# Output CSV
OUTPUT_CSV = (
    PROJECT_ROOT 
    / "dataset" 
    / "_lndb" 
    / "000_metadata" 
    / "consensus_clean.csv"
)


def main():
    """Generate consensus metadata for all LNDb findings."""

    # Load Metadata
    df = pd.read_csv(INPUT_CSV)

    print(f"Total findings : {len(df)}")

    success_count = 0
    failed_count = 0

    # Process Each Finding
    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing",
    ):
        lndbid = int(row["lndbid"])
        findingid = int(row["findingid"])

        try:
            print(
                f"Processing LNDb-{lndbid:04d} "
                f"Finding {findingid}"
            )

            # Prepare Scan
            scan = prepare_scan_data(
                row=row,
                data_dir=DATA_DIR,
                mask_dir=MASK_DIR,
            )

            # Generate Consensus
            scan = process_scan(scan)

            bbox = scan["consensus_bbox"]
            slices = scan["consensus_slices"]

            # Bounding Box Dimensions
            ymin = bbox["ymin"]
            ymax = bbox["ymax"]

            xmin = bbox["xmin"]
            xmax = bbox["xmax"]

            zmin = bbox["zmin"]
            zmax = bbox["zmax"]

            height = ymax - ymin + 1
            width = xmax - xmin + 1
            depth = zmax - zmin + 1

            # Save Bounding Box Metadata
            df.loc[idx, "bbox_y_min"] = ymin
            df.loc[idx, "bbox_y_max"] = ymax

            df.loc[idx, "bbox_x_min"] = xmin
            df.loc[idx, "bbox_x_max"] = xmax

            df.loc[idx, "bbox_z_min"] = zmin
            df.loc[idx, "bbox_z_max"] = zmax

            df.loc[idx, "bbox_height"] = height
            df.loc[idx, "bbox_width"] = width
            df.loc[idx, "bbox_depth"] = depth

            # Save Consensus Metadata
            df.loc[idx, "consensus_height"] = height
            df.loc[idx, "consensus_width"] = width

            df.loc[idx, "consensus_bbox_volume"] = (
                height * width * depth
            )

            df.loc[idx, "consensus_num_slices"] = len(slices)

            df.loc[idx, "consensus_slice_list"] = ",".join(
                map(str, slices.tolist())
            )

            success_count += 1
        except Exception as e:
            failed_count += 1

            print(
                f"\nFailed: LNDb-{lndbid:04d} "
                f"Finding {findingid}"
            )
            print(e)

    # Save Metadata
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Summary
    print()
    print("=" * 60)
    print("Consensus metadata generation completed")
    print("=" * 60)

    print(f"Total findings         : {len(df)}")
    print(f"Successfully processed : {success_count}")
    print(f"Failed                 : {failed_count}")

    print()
    print(f"Output CSV : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
