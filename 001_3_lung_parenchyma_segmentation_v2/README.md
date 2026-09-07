# Protected Lung Parenchyma Segmentation V2

This directory provides an end-to-end pipeline for LIDC-IDRI DICOM series and
LNDb MHD/RAW volumes. It produces two NumPy volumes for each study:

- a binary final lung mask (`uint8`, values 0 and 1); and
- segmented lung parenchyma (`float32`, range `[0, 1]`, zero background).

Nodule overlays are not embedded in these NumPy outputs. They are generated as
separate quality-control PNG files by `quality_control.py`.

## Processing workflow

```text
Original CT in Hounsfield Units
├── candidate masks for every slice
│   threshold → clear border → remove wide/flat table
│             → largest components → fill holes
│                              │
│                              ▼
│   nearest clean bilateral reference
│   remove table → remove trachea → keep two lungs
│                              │
│                              ▼
│   bidirectional matching with retained last-valid reference
│                              │
│                              ▼
│   residual trachea removal → radius-16 boundary repair → final mask
│
└── 3×3 median filter → lung window [-1400, 200] HU
                         → float32 normalization [0, 1]
                         → apply final mask
                         → segmented lung parenchyma
```

## Files

| File | Responsibility |
|---|---|
| `step_by_step_process.ipynb` | Detailed explanation and visualizations. |
| `config.py` | Input paths, output paths, and processing parameters. |
| `step_1_ct_to_numpy.py` | Discover and load complete DICOM/MHD studies. |
| `step_2_segmentation.py` | Protected bidirectional segmentation and repair. |
| `step_3_median_filter.py` | Slice-wise 3×3 median filtering. |
| `step_4_normalize_and_mask.py` | Window, normalize, and apply the final mask. |
| `run_pipeline.py` | Run one dataset or the complete batch and write reports. |
| `quality_control.py` | Create one nodule-overlay PNG per study. |
| `test_pipeline.py` | Fast synthetic regression tests. |

## Before running

All commands below assume that the terminal is opened at the project root:

```bash
cd "/run/media/dityanugroho/New Volume/mask-guided-lung-nodule-xai"
```

Confirm that the Conda environment exists:

```bash
conda env list
```

The examples use the existing `nodule_py310` environment. If its dependencies
are incomplete, install them once:

```bash
conda run -n nodule_py310 python -m pip install -r \
  001_3_lung_parenchyma_segmentation_v2/requirements.txt
```

Expected source locations are configured in `config.py`:

```text
000_dataset/lidc_idri/LIDC-IDRI-*/.../*.dcm
000_dataset/lndb/data/*.mhd
000_dataset/lndb/data/*.raw
```

For LNDb, every `.mhd` header must point to an available payload through its
`ElementDataFile` entry. For LIDC, the local DICOM path must also correspond to
a scan registered in the local `pylidc` database.

## Recommended run order

Follow these four steps for a new setup.

### Step 1 — Run the fast tests

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/test_pipeline.py
```

Expected ending:

```text
Ran 11 tests
OK
```

## Batch memory behavior

The batch runner processes one study at a time. Within a study, the candidate
mask buffer is reused for propagation and final boundary repair. Median
filtering, normalization, and masking are performed one axial slice at a time
and written directly into the final `float32` parenchyma array. This avoids
holding separate full-volume candidate, protected, repaired, median-filtered,
and normalized arrays in memory at the same time.

### Step 2 — Validate all available studies without saving arrays

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py all --no-save
```

This executes the complete segmentation, median filtering, windowing,
normalization, masking, and output validation path. It writes only CSV/JSON
reports and does not write the large mask/parenchyma arrays.

Use this step to confirm that the source data can be read before starting the
production run.

### Step 3 — Process and save all available studies

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py all
```

This is the production command. By default, a study is skipped when both its
mask and parenchyma output already exist. Therefore, the same command can be
used to resume an interrupted batch safely.

To recompute and replace every existing result:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py all --overwrite
```

Do not add `--no-save` when the outputs are needed for quality control or later
model training.

### Step 4 — Generate nodule-overlay quality-control PNGs

Run this only after Step 3 has created the mask/parenchyma pairs:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py all
```

Exactly one PNG is produced per saved study that has local consensus nodule
data. Studies without consensus nodules are skipped and do not produce empty
PNG files. All annotated slices from one study are placed on the same canvas.
The PNG includes:

- segmented lung parenchyma as the grayscale background;
- a high-contrast color for each nodule;
- `Cluster N` for LIDC or `Nodule N` for LNDb;
- `Benign` or `Malignant` classification;
- per-slice inclusion in each panel title; and
- whole-nodule inclusion in the legend.

The terminal reports `skipped_no_nodules` when a study is intentionally skipped
because its consensus nodule data was removed by filtering.

## Process only selected data

### All available LIDC studies

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py lidc
```

### One LIDC patient

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py lidc \
  --patient-id LIDC-IDRI-0015
```

The patient ID must use the complete form `LIDC-IDRI-NNNN`.

### All available LNDb studies

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py lndb
```

### One LNDb MHD study

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py lndb \
  --lndb-input 000_dataset/lndb/data/LNDb-0001.mhd
```

The RAW payload referenced by that MHD file must also exist.

### Quick smoke test on the first discovered study

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py all \
  --limit 1 --no-save
```

`--limit` is intended for a quick check only. Discovery is sorted, so this does
not select a random study.

## Quality control for one study

### One LIDC patient

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py lidc \
  --patient-id LIDC-IDRI-0001
```

### One LNDb study

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py lndb \
  --scan-id 1
```

### Use a custom PNG path

`--output` is valid only when exactly one study is selected:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py lndb \
  --scan-id 1 --output /tmp/LNDb-0001_qc.png
```

Change the number of panels per row with `--columns`, for example:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py lndb \
  --scan-id 1 --columns 4
```

## Output locations

The actual root directory is controlled by `OUTPUT_ROOT` in `config.py`.
Always check that value before a large production run. At the time of writing,
it is configured as:

```text
000_dataset_v2/lung_parenchyma_v3
```

The generated structure is:

```text
<OUTPUT_ROOT>/
├── masks/
│   ├── lidc/<study_id>_mask.npy
│   └── lndb/LNDb-NNNN_mask.npy
├── parenchyma/
│   ├── lidc/<study_id>_parenchyma.npy
│   └── lndb/LNDb-NNNN_parenchyma.npy
├── quality_control/
│   ├── lidc/<study_id>_nodule_overlay.png
│   └── lndb/LNDb-NNNN_nodule_overlay.png
└── reports/
    ├── pipeline_validation_<timestamp>.csv
    ├── pipeline_validation_<timestamp>.json
    ├── pipeline_write_<timestamp>.csv
    └── pipeline_write_<timestamp>.json
```

The pipeline does not save an additional CT volume and does not modify the
source DICOM, MHD, or RAW files.

## Understanding terminal status

| Status | Meaning |
|---|---|
| `validated` | Full processing succeeded in `--no-save` mode. |
| `written` | Mask and parenchyma were successfully saved. |
| `skipped_existing` | Both outputs already existed and were not overwritten. |
| `failed` | The study could not be loaded, processed, validated, or saved. |
| `skipped_no_nodules` | QC was skipped and no PNG was written because no consensus nodule was available. |

Every pipeline run writes a CSV and JSON report. Inspect the `error` column for
failed studies. Useful segmentation diagnostics include:

- `failed_overlap_count` and `failed_overlap_slices`;
- `non_empty_start` and `non_empty_end`;
- `foreground_fraction`;
- `mean_adjacent_dice`; and
- `parenchyma_min` and `parenchyma_max`.

## Useful command-line options

| Option | Effect |
|---|---|
| `--no-save` | Run everything but write only validation reports. |
| `--overwrite` | Recompute studies even when both outputs already exist. |
| `--limit N` | Process only the first `N` discovered studies. |
| `--show-slice-progress` | Display candidate/repair progress for individual slices. |
| `--fail-fast` | Stop immediately after the first failed study. |
| `--report-dir PATH` | Write CSV/JSON reports to another directory. |
| `--lndb-input PATH` | Use one MHD file or another directory of MHD files. |

Display the complete CLI help at any time:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py --help

conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py --help
```

## Run the step-by-step notebook

The notebook contains relative paths beginning with `../000_dataset`, so start
Jupyter from the pipeline directory:

```bash
cd "/run/media/dityanugroho/New Volume/mask-guided-lung-nodule-xai/001_3_lung_parenchyma_segmentation_v2"
conda run -n nodule_py310 jupyter lab step_by_step_process.ipynb
```

Run the cells from top to bottom. Change `LIDC_PATIENT_ID`, `LNDB_MHD_PATH`, or
`REFERENCE_SLICE_INDICES` in the parameter cell to inspect another example.

## Incomplete datasets

The local dataset does not need to contain every official LIDC or LNDb study.
The pipeline processes only studies that can be discovered locally.

- LIDC discovery ignores small localizer series and requires the expected local
  DICOM slice count for the `pylidc` scan.
- LNDb discovery excludes an MHD header when its referenced RAW payload is
  missing.
- A processing error in one discovered study is recorded as `failed`; the
  remaining studies continue unless `--fail-fast` is used.

## Troubleshooting

### `No complete local CT scans were discovered`

Check the input paths in `config.py`. For LIDC, also confirm that the local
`pylidc` database and DICOM paths refer to the same patient/series. For LNDb,
open the MHD header and verify that `ElementDataFile` points to an existing RAW
file.

### `No matching saved V2 mask/parenchyma pair was found`

Run the production command without `--no-save` first:

```bash
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/run_pipeline.py all
```

QC requires both `<study_id>_mask.npy` and `<study_id>_parenchyma.npy`.

### The pipeline reports `skipped_existing`

This is normal resume behavior. Add `--overwrite` only if the existing result
must be recomputed.

### Matplotlib reports that its configuration directory is not writable

This warning does not invalidate the PNG. To avoid it, use a writable temporary
configuration directory:

```bash
MPLCONFIGDIR=/tmp/matplotlib-lung-qc \
conda run -n nodule_py310 python \
  001_3_lung_parenchyma_segmentation_v2/quality_control.py all
```

### The QC image reports low nodule inclusion

Inspect the affected slice titles and colors in that study's PNG. Inclusion is
calculated from the final binary lung mask, not from nonzero parenchyma
intensity, so zero-valued normalized air does not bias this measurement.

## Local validation performed

The complete production path was previously validated on all complete local
sources: 15 LIDC-IDRI DICOM series and 15 LNDb MHD/RAW volumes (30/30 passed).
The previous QC batch found consensus nodules in 24 of 30 locally saved studies;
the remaining six studies are now intentionally skipped. The regression suite
covers component matching, protection of a small off-center lung, boundary
repair, median filtering, normalization, zero background, nodule inclusion,
single-study PNG generation, and suppression of empty QC PNG files.
