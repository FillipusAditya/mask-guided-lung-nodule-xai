# CT Median Filtering and Windowing

This compact pipeline replaces `001_preprocessing/002_ct_windowing.py`. It
prepares LIDC-IDRI and LNDb HU volumes for model input using the same lung
window as `001_3_lung_parenchyma_segmentation`, with an added slice-wise median
filter.

## Processing contract

```text
int16 HU volume (N, H, W)
    -> median filter (1, 3, 3), reflect boundary
    -> clip to [-1400, 200] HU (level=-600, width=1600)
    -> normalize to [0, 1]
    -> float32 volume (N, H, W)
```

The kernel's slice dimension is `1`, so anatomy from neighboring axial slices
is never mixed. The order matches the lung-parenchyma pipeline: median
filtering is performed on HU data before windowing and normalization.

## Directory structure

| File | Responsibility |
|---|---|
| `config.py` | Input/output paths, lung window, and median kernel. |
| `preprocess.py` | Validation, filtering, normalization, export, and discovery. |
| `export_png.py` | Convert normalized volumes into 8-bit axial PNG slices. |
| `run_pipeline.py` | CLI for LNDb, LIDC-IDRI, or both. |
| `requirements.txt` | Direct Python dependencies. |
| `tests/test_preprocess.py` | Isolated synthetic tests. |

## Default paths

```text
LIDC input:   000_dataset/_lidc/001_volume_npy/
LNDb input:   000_dataset/_lndb/001_volume_npy/

LIDC output:  000_dataset/_lidc/002_windowed_median_npy/
LNDb output:  000_dataset/_lndb/002_windowed_median_npy/

LIDC PNG:     000_dataset/_lidc/002_windowed_median_png/
LNDb PNG:     000_dataset/_lndb/002_windowed_median_png/
```

The output is intentionally separate from legacy `002_windowed_npy`, which
used width 1500 and no median filter. Downstream configuration is not changed
automatically; migrate it after checking the new outputs.

Every output array has a same-stem JSON sidecar recording its processing
order, kernel, window parameters, shape, dtype, and value range. Geometry from
an input JSON sidecar produced by `001_1_ct_to_npy` is propagated.

## Usage

From the project root:

Windowing and optional PNG export show one progress bar per operation, followed
by a compact total/written/skipped summary.

```bash
python 001_2_ct_windowing/run_pipeline.py lndb
python 001_2_ct_windowing/run_pipeline.py lidc
python 001_2_ct_windowing/run_pipeline.py all
```

Add `--png` to save the normalized NumPy/JSON output and all axial PNG slices
in one run:

```bash
python 001_2_ct_windowing/run_pipeline.py lidc --png
python 001_2_ct_windowing/run_pipeline.py lndb --png
python 001_2_ct_windowing/run_pipeline.py all --png
```

Each processed volume receives its own directory:

```text
002_windowed_median_png/
└── <volume-name>/
    ├── slice_0000.png
    ├── slice_0001.png
    └── ...
```

The normalized `float32 [0,1]` values are mapped directly to grayscale
`uint8 [0,255]`. No additional windowing or normalization is applied during
PNG export.

Complete existing `.npy`/`.json` pairs are skipped. Partial pairs stop the
pipeline. To explicitly regenerate outputs:

```bash
python 001_2_ct_windowing/run_pipeline.py lndb --overwrite
```

The same `--overwrite` option replaces requested PNG slices. Complete PNG
directories are otherwise skipped, while incomplete directories raise an
error for inspection.

Export PNG slices separately from an existing normalized NumPy volume:

```bash
python 001_2_ct_windowing/export_png.py \
    windowed_volume.npy \
    output_png_directory
```

The runner validates both input directories before `all` writes either
dataset. It is fail-fast and does not delete stale output files.

## Python API

```bash
PYTHONPATH=001_2_ct_windowing python your_program.py
```

```python
from preprocess import preprocess_ct

result = preprocess_ct(
    ct_volume,
    window_level=-600.0,
    window_width=1600.0,
    median_filter_size=(1, 3, 3),
)
```

## Tests

```bash
python -m unittest discover -s 001_2_ct_windowing/tests -v
```

Tests use temporary directories and do not modify project datasets.
