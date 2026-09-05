# LIDC-IDRI Multi-Radiologist Nodule Consensus

This directory generates full-size, two-dimensional consensus masks for the
LIDC-IDRI annotation clusters selected in the cleaned metadata. It is the
LIDC-IDRI counterpart of `001_5_lndb_consensus`, with centralized configuration,
reusable array operations, a dataset runner, output safeguards, and tests.

## Processing flow

```text
002_cluster_metadata_cleaned.csv + pylidc database/DICOM files
                              │
                              ▼
        match patient + study + series + cluster identifiers
                              │
                              ▼
            pylidc.utils.consensus(annotation cluster)
                              │
                              ▼
              cropped boolean mask in (y, x, z)
                              │
                              ▼
           restore each nonempty slice to CT dimensions
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
     per-slice `.npy` masks    003_cluster_metadata_cleaned_path.csv
```

The default consensus level is `0.5`, meaning a voxel must be included by at
least half of the annotations in a cluster according to `pylidc` consensus
semantics.

## Directory structure

| File or directory | Responsibility |
|---|---|
| `config.py` | Project-relative metadata/output paths and consensus level. |
| `generate_consensus_masks.py` | Match selected clusters, calculate masks, save slices, and update metadata. |
| `generate_artifacts.py` | Export optional segmented-nodule PNG and per-scan quality-control images. |
| `run_pipeline.py` | Safe top-level runner that skips an existing output unless overwrite is requested. |
| `lidc_consensus/artifacts.py` | Preprocess CT intensities and render optional artifacts using `pylidc` consensus. |
| `lidc_consensus/identifiers.py` | Construct stable cluster and scan directory names. |
| `lidc_consensus/consensus.py` | Calculate and restore consensus masks. |
| `lidc_consensus/export.py` | Validate and save per-slice NumPy masks. |
| `tests/` | Unit tests using synthetic masks and fake scan identifiers. |

## Inputs

The default metadata input is:

```text
000_dataset/_lidc/000_metadata/002_cluster_metadata_cleaned.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `patient_id` | LIDC-IDRI patient identifier. |
| `study_instance_uid` | DICOM study identifier used to match the exact scan. |
| `series_instance_uid` | DICOM series identifier used to match the exact scan. |
| `cluster_id` | Zero-based position returned by `scan.cluster_annotations()`. |
| `cluster_uid` | Stable value `<patient_id>_cluster_<cluster_id>`. |

All selected scans must exist in both the local pylidc database and configured
DICOM directory. The pipeline validates scan identifiers and cluster indices
before exporting each selected cluster.

## Outputs

Masks are written to:

```text
000_dataset/_lidc/005_mask_consensus_npy/
└── LIDC-IDRI-0001_30178_03192/
    └── cluster_0/
        ├── slice_86.npy
        ├── slice_87.npy
        └── ...
```

Each `.npy` file contains one full-size, two-dimensional boolean mask aligned
with the original CT slice. Empty consensus slices are omitted.

The corresponding metadata is written to:

```text
000_dataset/_lidc/000_metadata/003_cluster_metadata_cleaned_path.csv
```

This is a copy of the cleaned metadata with `consensus_mask_path` added. Paths
inside the repository are stored as portable project-relative POSIX paths.

Optional segmented-nodule and quality-control images are written per scan:

```text
000_dataset/_lidc/006_segmented_nodule_png/
└── LIDC-IDRI-0001_30178_03192/
    └── cluster_0/
        └── slice_86.png

000_dataset/_lidc/007_consensus_quality_control/
└── LIDC-IDRI-0001_30178_03192/
    ├── cluster_0_agreement_map.png
    └── cluster_0_consensus_mask.png
```

Before rendering, the CT volume is processed in this fixed order: median
filter `(1, 3, 3)`, lung window `W=1600/L=-600`, then float32 normalization to
`[0, 1]`. A segmented PNG is cropped to the `pylidc` consensus bounding box;
pixels outside the consensus mask are black.

## pylidc configuration

`pylidc` reads the DICOM root from `~/.pylidcrc`:

```ini
[dicom]
path = /absolute/path/to/LIDC-IDRI
```

The patient data must be available below that path and registered in the
pylidc SQLite database.

## Installation

Use the project environment, or install the direct dependencies:

```bash
python -m pip install -r 001_4_lidc_consensus/requirements.txt
```

## Run the pipeline

Run from the project root:

Mask and optional artifact generation display scan-level progress bars and one
compact result line per operation.

```bash
python 001_4_lidc_consensus/run_pipeline.py
```

Existing output metadata is skipped by default. Regenerate all selected masks
and replace the output metadata with:

```bash
python 001_4_lidc_consensus/run_pipeline.py --overwrite
```

Set another consensus level:

```bash
python 001_4_lidc_consensus/run_pipeline.py \
    --consensus-level 0.75 \
    --overwrite
```

Generate both optional image products for one scan:

```bash
python 001_4_lidc_consensus/run_pipeline.py \
    --patient-id LIDC-IDRI-0001 \
    --segmented-png \
    --quality-control
```

`--patient-id` limits optional artifact generation to one patient. Omit it to
process all scans selected by the input metadata. Existing artifact files are
skipped; use `--overwrite` only when they must be replaced.

The generation module can also be run directly:

```bash
python 001_4_lidc_consensus/generate_consensus_masks.py --overwrite
```

All paths can be overridden. Run either command with `--help` for its complete
argument list.

## Python API

The core restoration operation can be reused without querying the database:

```python
from lidc_consensus import compute_consensus_slices, save_consensus_slices

masks = compute_consensus_slices(
    annotation_cluster=annotation_cluster,
    image_shape=(512, 512),
    consensus_level=0.5,
)
save_consensus_slices(masks, "output/cluster_0")
```

Add the module directory to `PYTHONPATH` when importing from the project root:

```bash
PYTHONPATH=001_4_lidc_consensus python your_program.py
```

## Safety and reproducibility

- Existing output metadata and mask slices are protected unless `--overwrite`
  is supplied.
- Optional PNG and quality-control files follow the same overwrite protection.
- Metadata is matched using patient, study, and series identifiers rather than
  patient ID alone.
- Duplicate or inconsistent `cluster_uid` values are rejected.
- Missing scans and out-of-range cluster indices stop processing with a
  descriptive error.
- DICOM availability is checked before mask export begins, preventing a
  partially regenerated dataset when local patient files are incomplete.
- Exported masks are normalized to boolean dtype.
- Output paths are derived from `config.py`, not the shell working directory.
- A local compatibility helper supplies the legacy NumPy aliases still used by
  `pylidc 0.2.3`, allowing the pipeline to run with current NumPy releases.

## Tests

Run from the project root:

```bash
python -m unittest discover -s 001_4_lidc_consensus/tests -v
```

Tests use synthetic arrays and temporary directories. They do not change the
LIDC-IDRI dataset or query the pylidc database.
