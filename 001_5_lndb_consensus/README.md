# LNDb Multi-Radiologist Nodule Consensus

This directory builds one binary nodule mask from the annotations supplied by
multiple LNDb radiologists. It is a standalone, modular pipeline: paths and
parameters are centralized, each processing operation is reusable from Python,
and both dataset stages are available as command-line programs.

## Processing flow

```text
002_trainNodules_gt_clean.csv + LNDb CT/mask MHD files
                           │
                           ▼
              prepare and validate scan geometry
                           │
                           ▼
        extract each radiologist's target finding mask
                           │
                           ▼
       bounding boxes → common crop → agreement map
                           │
                           ▼
              threshold voxel-wise agreement
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
003_consensus_clean.csv       per-slice consensus masks
                                         │
                                         ▼
                           004_consensus_clean_path.csv
```

The default consensus level is `0.5`. The integer threshold is calculated as
`ceil(consensus_level × number_of_radiologists)`. For three radiologists, this
means that a voxel must be annotated by at least two radiologists.

## Directory structure

| File or directory | Responsibility |
|---|---|
| `config.py` | Project paths and the default consensus level. |
| `generate_consensus_metadata.py` | Stage 1: calculate bounding-box and consensus-slice metadata. |
| `generate_consensus_masks.py` | Stage 2: save each binary consensus slice and its directory path. |
| `generate_artifacts.py` | Export optional segmented-nodule PNG and per-scan quality-control images. |
| `run_pipeline.py` | Run Stage 1, Stage 2, or the complete pipeline. |
| `lndb_consensus/loader.py` | Prepare finding metadata, load CT/masks, and verify image geometry. |
| `lndb_consensus/bbox.py` | Calculate individual and enclosing bounding boxes. |
| `lndb_consensus/crop.py` | Crop CT and annotation masks to the enclosing box. |
| `lndb_consensus/consensus.py` | Stack masks, calculate agreement, threshold, and restore the mask. |
| `lndb_consensus/pipeline.py` | Orchestrate the in-memory processing stages. |
| `lndb_consensus/export.py` | Export mask slices, CT slices, and diagnostic figures. |
| `lndb_consensus/artifacts.py` | Preprocess CT intensities and save the optional image products. |
| `lndb_consensus/visualize.py` | Visualize annotations, agreement, bounding boxes, and consensus. |
| `consesus_development.ipynb` | Step-by-step development and visual verification notebook. |
| `tests/` | Fast unit tests for core array and export behavior. |

## Inputs

Default paths are defined in `config.py`:

```text
000_dataset/lndb/data/*.mhd
000_dataset/lndb/masks/*_rad*.mhd
000_dataset/_lndb/000_metadata/002_trainNodules_gt_clean.csv
```

The input CSV must contain at least:

| Column | Meaning |
|---|---|
| `lndbid` | LNDb scan identifier. |
| `findingid` | Consensus finding identifier within the scan. |
| `radid` | Comma-separated radiologist identifiers. |
| `radfindingid` | Matching comma-separated finding labels inside each mask volume. |
| `label` | Classification label retained in the output metadata. |

`radid` and `radfindingid` are paired in order. Each radiologist mask must have
the same size, spacing, origin, and direction as its CT volume.

## Outputs

Stage 1 creates:

```text
000_dataset/_lndb/000_metadata/003_consensus_clean.csv
```

It adds bounding-box coordinates and dimensions, consensus bounding-box
volume, number of consensus slices, and a comma-separated slice list.

Stage 2 creates:

```text
000_dataset/_lndb/005_mask_consensus_npy/
└── LNDb-0001/
    └── finding_1/
        ├── slice_257.npy
        ├── slice_258.npy
        └── ...

000_dataset/_lndb/000_metadata/004_consensus_clean_path.csv
```

Each `.npy` file is a two-dimensional binary mask aligned with the
corresponding axial CT slice. `consensus_mask_path` is stored as a portable,
project-relative POSIX path when the output is inside the repository.

Optional segmented-nodule and quality-control images are written per scan:

```text
000_dataset/_lndb/006_segmented_nodule_png/
└── LNDb-0001/
    └── finding_1/
        └── slice_257.png

000_dataset/_lndb/007_consensus_quality_control/
└── LNDb-0001/
    ├── finding_1_agreement_map.png
    └── finding_1_consensus_mask.png
```

Before rendering, the CT volume is processed in this fixed order: median
filter `(1, 3, 3)`, lung window `W=1600/L=-600`, then float32 normalization to
`[0, 1]`. A segmented PNG is cropped to the consensus bounding box; pixels
outside the consensus mask are black. Both QC functions automatically display
`LNDb-XXXX | Finding N` in the canvas title.

## Installation

Use the project environment, or install the direct dependencies:

```bash
python -m pip install -r 001_5_lndb_consensus/requirements.txt
```

## Run the complete pipeline

Run commands from the project root:

Metadata, mask, and optional artifact generation display finding-level progress
bars and one compact result line per operation.

```bash
python 001_5_lndb_consensus/run_pipeline.py all
```

Existing final metadata files are skipped by default. To regenerate and
replace existing metadata and mask slices:

```bash
python 001_5_lndb_consensus/run_pipeline.py all --overwrite
```

Use a different agreement fraction:

```bash
python 001_5_lndb_consensus/run_pipeline.py all \
    --consensus-level 0.67 \
    --overwrite
```

Generate both optional image products for one scan:

```bash
python 001_5_lndb_consensus/run_pipeline.py all \
    --scan-id 1 \
    --segmented-png \
    --quality-control
```

`--scan-id` limits optional artifact generation to one LNDb scan. Omit it to
process every finding in the input metadata. Existing artifact files are
skipped; use `--overwrite` only when they must be replaced.

## Run individual stages

Generate only consensus metadata:

```bash
python 001_5_lndb_consensus/generate_consensus_metadata.py --overwrite
```

Generate only mask slices and path metadata:

```bash
python 001_5_lndb_consensus/generate_consensus_masks.py --overwrite
```

The unified runner provides the same stage selection:

```bash
python 001_5_lndb_consensus/run_pipeline.py metadata --overwrite
python 001_5_lndb_consensus/run_pipeline.py masks --overwrite
```

Every path can be overridden from the CLI. Run a command with `--help` for the
complete argument list.

## Python API

Process one finding entirely in memory:

```python
import pandas as pd

from lndb_consensus import prepare_scan_data, process_scan

row = pd.read_csv(
    "000_dataset/_lndb/000_metadata/002_trainNodules_gt_clean.csv"
).iloc[0]

scan = prepare_scan_data(
    row=row,
    data_dir="000_dataset/lndb/data",
    mask_dir="000_dataset/lndb/masks",
)
scan = process_scan(scan, clevel=0.5)

mask = scan["consensus_mask_full"]
slices = scan["consensus_slices"]
```

When importing from the project root, add `001_5_lndb_consensus` to
`PYTHONPATH`, or run code from that directory:

```bash
PYTHONPATH=001_5_lndb_consensus python your_program.py
```

Dataset-level functions are also reusable:

```python
from generate_consensus_metadata import generate_consensus_metadata
from generate_consensus_masks import generate_consensus_masks
```

## Safety and reproducibility

- Existing outputs are not replaced unless `--overwrite` is supplied.
- Optional PNG and quality-control files follow the same overwrite protection.
- Dataset processing is fail-fast, so an incomplete CSV is not silently saved.
- Image geometry is verified before annotation arrays are combined.
- Consensus masks are boolean arrays and retain original CT dimensions after
  restoration.
- Filesystem paths are based on the location of `config.py`, not the shell's
  current working directory.

## Tests

Run the test suite from the project root:

```bash
python -m unittest discover -s 001_5_lndb_consensus/tests -v
```

The tests use synthetic arrays and temporary directories; they do not modify
the LNDb dataset.
