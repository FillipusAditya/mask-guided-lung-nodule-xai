# 3D Lung Parenchyma Segmentation

This directory implements the processing flow documented in
`001_3_lung_parenchyma_segmentation/step_by_step_process.ipynb`.

The final output is a masked CT volume with:

- shape `(N, H, W)`;
- dtype `float32`;
- intensity range `[0, 1]`;
- background outside the lung mask set to `0`.

## Processing flow

```text
Original CT in Hounsfield Units
├── Step 2: lung segmentation ── fill post-dilation holes ── Step 3: erosion
│                                                                    │
│                                                                    ▼
│                                                               lung mask
└── Step 4: median filtering ── Step 5: window and normalization
                                                │
                              apply lung mask ──┘
                                                ↓
                              final 3D lung parenchyma
```

Step 1 loads LIDC-IDRI DICOM scans through `pylidc` or LNDb MHD scans
through `SimpleITK`. Every segmentation and erosion operation is applied
independently to each axial slice.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Input paths, output paths, and preprocessing parameters. |
| `step_1_ct_to_numpy.py` | Convert LIDC-IDRI or LNDb scans to `(N, H, W)` arrays. |
| `step_2_segmentation.py` | Generate a binary 3D lung mask. |
| `step_3_erosion.py` | Reduce the outer mask boundary. |
| `step_4_median_filter.py` | Apply a `(1, 3, 3)` median filter. |
| `step_5_normalize_and_mask.py` | Normalize the lung window and apply the mask. |
| `run_pipeline.py` | Run every step in memory and save only the final volumes. |
| `quality_control.py` | Display or save full CT and consensus-nodule overlays on the final lung parenchyma. |

## Final dataset

Run from the project root:

Every batch command displays one progress bar and a compact summary. Individual
filenames are not printed, keeping terminal output readable for large datasets.

```bash
# Process every LIDC scan available on disk and every LNDb MHD file.
python 001_3_lung_parenchyma_segmentation/run_pipeline.py all

# Use this command when only LIDC-IDRI-0001 is stored locally.
python 001_3_lung_parenchyma_segmentation/run_pipeline.py \
    all \
    --patient-id LIDC-IDRI-0001
```

The outputs are written to:

```text
000_dataset/lung_parenchyma/lidc
000_dataset/lung_parenchyma/lndb
```

Process only one LIDC-IDRI patient:

```bash
python 001_3_lung_parenchyma_segmentation/run_pipeline.py \
    lidc \
    --patient-id LIDC-IDRI-0001
```

Process one LNDb scan:

```bash
python 001_3_lung_parenchyma_segmentation/run_pipeline.py \
    lndb \
    --lndb-input 000_dataset/lndb/data/LNDb-0001.mhd
```

Use `--overwrite` to replace existing final files.

## Nodule overlay quality control

Add `--quality-control` to generate a QC canvas immediately after each selected
scan is processed:

```bash
python 001_3_lung_parenchyma_segmentation/run_pipeline.py \
    all \
    --patient-id LIDC-IDRI-0001 \
    --quality-control
```

The images are written to:

```text
000_dataset/lung_parenchyma/quality_control/
├── lidc/
└── lndb/
```

Use `--quality-control-dir PATH` to change this root directory. If a final
`.npy` file already exists but its QC image does not, `--quality-control` still
creates the QC image. Existing QC images are protected unless `--overwrite`
is also supplied.

Quality control can also be run separately after a final lung-parenchyma file
has been generated. The LIDC command obtains masks with
`compute_consensus_slices` from `001_4_lidc_consensus`. The LNDb command uses
`prepare_scan_data` and `process_scan` from `001_5_lndb_consensus`.

Display all consensus-nodule slices for one LIDC-IDRI patient:

```bash
python 001_3_lung_parenchyma_segmentation/quality_control.py \
    lidc \
    --patient-id LIDC-IDRI-0001
```

Display all consensus-nodule slices for one LNDb scan:

```bash
python 001_3_lung_parenchyma_segmentation/quality_control.py \
    lndb \
    --scan-id 20
```

Use `--slices` to inspect only selected nodule slices:

```bash
python 001_3_lung_parenchyma_segmentation/quality_control.py \
    lndb \
    --scan-id 20 \
    --slices 248 249
```

Use `--output` to save the canvas instead of opening it interactively:

```bash
python 001_3_lung_parenchyma_segmentation/quality_control.py \
    lidc \
    --patient-id LIDC-IDRI-0001 \
    --output quality_control/lidc_0001.png
```

Every displayed nodule slice contains two panels: the full CT and the final
lung parenchyma with a colored consensus-mask overlay. Finding classes appear
in the legend below the main canvas title. Use `--parenchyma-file` when the
final `.npy` file is stored outside the default output directory.

## Run individual steps

Each stage accepts one file with `single` or all `.npy` files in a directory
with `batch`.

```bash
# Step 1: one LNDb MHD scan to NumPy
python 001_3_lung_parenchyma_segmentation/step_1_ct_to_numpy.py \
    lndb-one input.mhd volume.npy

# Step 1: all LNDb MHD scans to NumPy
python 001_3_lung_parenchyma_segmentation/step_1_ct_to_numpy.py \
    lndb-batch input_mhd_dir output_npy_dir

# Step 2: CT volume to lung mask
python 001_3_lung_parenchyma_segmentation/step_2_segmentation.py \
    single volume.npy mask.npy

# Step 3: erode the mask
python 001_3_lung_parenchyma_segmentation/step_3_erosion.py \
    single mask.npy eroded_mask.npy

# Step 4: median-filter the original CT volume
python 001_3_lung_parenchyma_segmentation/step_4_median_filter.py \
    single volume.npy filtered.npy

# Step 5: normalize the filtered volume and apply the eroded mask
python 001_3_lung_parenchyma_segmentation/step_5_normalize_and_mask.py \
    single filtered.npy eroded_mask.npy lung_parenchyma.npy
```

Replace `single` with `batch` and provide input/output directories to process
multiple same-named files. Run a script with `--help` for its exact arguments.

## LIDC-IDRI configuration

`pylidc` reads the DICOM root from `~/.pylidcrc`:

```ini
[dicom]
path = /absolute/path/to/LIDC-IDRI
```

Only scans registered in the local `pylidc` database and available below this
path can be processed.

## Parameters

The defaults match the V2 notebook:

- HU threshold: `-320`;
- largest components: `4`;
- dilation iterations: `5`;
- fill holes after dilation: enabled;
- erosion iterations: `5`;
- median filter size: `(1, 3, 3)`;
- lung window: width `1600`, level `-600` (`[-1400, 200] HU`);
- table/artifact removal: disabled.

Edit `config.py` once to change these values for every module.
