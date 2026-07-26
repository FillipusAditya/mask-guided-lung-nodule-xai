from pathlib import Path

import pandas as pd

from lndb_consensus import (
    prepare_scan_data,
    process_scan,
    save_ct_slices,
    save_consensus_slices,
    save_visualizations,
)

# Configuration
DATA_DIR = Path("../dataset_sample/lndb/data")
MASK_DIR = Path("../dataset_sample/lndb/masks")
CSV_PATH = Path("../dataset_sample/trainNodules_gt_clean.csv")

OUTPUT_DIR = Path("output")

CT_DIR = OUTPUT_DIR / "ct"
MASK_OUT_DIR = OUTPUT_DIR / "mask"
VIS_DIR = OUTPUT_DIR / "visualization"

SAVE_VISUALIZATION = True

# Create Output Directories
CT_DIR.mkdir(parents=True, exist_ok=True)
MASK_OUT_DIR.mkdir(parents=True, exist_ok=True)

if SAVE_VISUALIZATION:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

# Load Metadata
df = pd.read_csv(CSV_PATH)

# Select One Finding
row = df[
    (df["lndbid"] == 1) &
    (df["findingid"] == 1)
].iloc[0]

print(f"Processing LNDb-{row['lndbid']:04.0f} Finding {row['findingid']}")

# Prepare Scan
scan = prepare_scan_data(
    row=row,
    data_dir=DATA_DIR,
    mask_dir=MASK_DIR,
)

# Run Consensus Pipeline
scan = process_scan(scan)

# Export CT
save_ct_slices(
    scan=scan,
    output_dir=CT_DIR,
)

# Export Consensus Mask
save_consensus_slices(
    scan=scan,
    output_dir=MASK_OUT_DIR,
)

# Export Visualization
if SAVE_VISUALIZATION:
    save_visualizations(
        scan=scan,
        output_dir=VIS_DIR,
    )

# Create Metadata
records = []

for z in scan["consensus_slices"]:
    records.append(
        {
            "lndb_id": scan["lndb_id"],
            "finding_id": scan["finding_id"],
            "slice": int(z),
            "ct_path": (
                f"ct/"
                f"LNDb-{scan['lndb_id']:04d}"
                f"_finding{scan['finding_id']}"
                f"_slice{z}.npy"
            ),
            "mask_path": (
                f"mask/"
                f"LNDb-{scan['lndb_id']:04d}"
                f"_finding{scan['finding_id']}"
                f"_slice{z}.npy"
            ),
        }
    )

metadata_df = pd.DataFrame(records)
metadata_path = OUTPUT_DIR / "metadata.csv"
metadata_df.to_csv(
    metadata_path,
    index=False,
)

# Summary
print()
print("=" * 50)
print("Dataset generation completed.")
print("=" * 50)

print(f"LNDb ID       : {scan['lndb_id']}")
print(f"Finding ID    : {scan['finding_id']}")
print(f"Total slices  : {len(scan['consensus_slices'])}")
print(f"Metadata      : {metadata_path}")
print(f"CT folder     : {CT_DIR}")
print(f"Mask folder   : {MASK_OUT_DIR}")

if SAVE_VISUALIZATION:
    print(f"Visualization : {VIS_DIR}")