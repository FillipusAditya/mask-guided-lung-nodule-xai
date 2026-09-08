# Prepared 2D Segmentation Dataset Builder

This directory builds aligned two-dimensional CT and consensus-mask samples
for LNDb and LIDC-IDRI. It replaces the two duplicated dataset builders that
previously lived in `001_preprocessing` with one shared, validated engine and
small dataset-specific entry points.

## Processing flow

```text
3D windowed CT ───────┐
3D parenchyma CT ─────┼── select every consensus slice ──┐
consensus mask slices ┘                                  │
                                                         ▼
                                        validate shape and binary mask
                                                         │
                                                         ▼
                         matching 2D CT + CT + mask files and metadata.csv
```

The builder does not apply a mask to a CT image. It extracts the same axial
slice from each configured CT representation and copies the corresponding
binary consensus mask, producing aligned inputs and targets for segmentation.

## Directory structure

| File or directory | Responsibility |
|---|---|
| `config.py` | LNDb and LIDC-IDRI input/output specifications. |
| `build_lndb_dataset.py` | Build only the LNDb prepared dataset. |
| `build_lidc_dataset.py` | Build only the LIDC-IDRI prepared dataset. |
| `run_pipeline.py` | Build LNDb, LIDC-IDRI, or both with a shared preflight. |
| `segmentation_dataset/spec.py` | Immutable source-specific configuration model. |
| `segmentation_dataset/naming.py` | Strict mask, nodule-directory, and sample naming. |
| `segmentation_dataset/validation.py` | CT volume, binary mask, and paired-shape validation. |
| `segmentation_dataset/builder.py` | Discovery, preflight, extraction, export, and metadata. |
| `tests/` | Synthetic end-to-end and unit tests. |

## Inputs

### LNDb

```text
000_dataset/_lndb/002_windowed_npy/*.npy
000_dataset/_lndb/004_volume_parenchyma_npy/*.npy
000_dataset/_lndb/005_mask_consensus_npy/
└── LNDb-XXXX/finding_N/slice_Z.npy
```

### LIDC-IDRI

```text
000_dataset/_lidc/002_windowed_npy/*.npy
000_dataset/_lidc/004_volume_parencyma_npy/*.npy
000_dataset/_lidc/005_mask_consensus_npy/
└── LIDC-IDRI-XXXX_<study>_<series>/cluster_N/slice_Z.npy
```

The LIDC-IDRI parenchyma directory intentionally uses the existing legacy
spelling `parencyma`. The normalized output column and directory remain
`ct_parenchyma`.

Every CT file must be a nonempty numeric array in `(slices, height, width)`
order. Masks must be finite, binary, two-dimensional arrays named
`slice_<index>.npy`.

## Outputs

Default output roots:

```text
000_dataset/_lndb/007_segmentation_dataset_npy/
000_dataset/_lidc/007_segmentation_dataset_npy/
```

Each has the same layout:

```text
007_segmentation_dataset_npy/
├── ct_windowed/
│   └── <study>_<finding-or-cluster>_slice_<index>.npy
├── ct_parenchyma/
│   └── <same-filename>.npy
├── mask/
│   └── <same-filename>.npy
└── metadata.csv
```

The metadata includes:

- common filename and study/patient identifier;
- `finding_id` for LNDb or `cluster_id` for LIDC-IDRI;
- axial slice index;
- relative POSIX paths for every CT representation and mask;
- height and width for every array;
- foreground mask-pixel count.

## Installation

```bash
python -m pip install -r 001_segmentation_dataset/requirements.txt
```

## Run the pipeline

Run from the project root:

```bash
# Preflight and build both datasets.
python 001_segmentation_dataset/run_pipeline.py all

# Build one source dataset.
python 001_segmentation_dataset/run_pipeline.py lndb
python 001_segmentation_dataset/run_pipeline.py lidc
```

Existing output metadata is protected. To regenerate matching files:

```bash
python 001_segmentation_dataset/run_pipeline.py lndb --overwrite
```

Dataset-specific entry points provide an optional output directory:

```bash
python 001_segmentation_dataset/build_lndb_dataset.py \
    --output-dir /path/to/lndb-output

python 001_segmentation_dataset/build_lidc_dataset.py \
    --output-dir /path/to/lidc-output
```

Run any command with `--help` for its complete argument list.

## Python API

```python
from config import LNDB_SPEC
from segmentation_dataset import build_dataset

summary = build_dataset(LNDB_SPEC)
print(summary.samples)
print(summary.metadata.head())
```

When importing from the project root, add the module directory to
`PYTHONPATH`:

```bash
PYTHONPATH=001_segmentation_dataset python your_program.py
```

Additional CT representations can be added to `ct_volume_dirs` in `config.py`.
The engine automatically creates corresponding output files, path columns,
and dimension columns.

## Safety and reproducibility

- Every source directory, study, volume, nodule directory, and mask filename
  is checked during preflight before sample files are written.
- The `all` runner preflights both datasets before writing either dataset.
- Volumes are memory-mapped to avoid loading multiple multi-gigabyte datasets
  fully into RAM.
- CT slices and masks must have identical dimensions.
- Consensus masks are validated as binary and saved as boolean arrays.
- If only part of an output pair exists, the builder stops instead of silently
  accepting an inconsistent dataset.
- Existing complete pairs and metadata are protected unless `--overwrite` is
  supplied.
- Output metadata is deterministically sorted and uses relative POSIX paths.

The builder does not delete stale files that are no longer represented by
source masks. Use a clean output directory when producing a release dataset.

## Tests

```bash
python -m unittest discover -s 001_segmentation_dataset/tests -v
```

Tests use synthetic NumPy volumes, masks, and temporary directories. They do
not modify project datasets.

