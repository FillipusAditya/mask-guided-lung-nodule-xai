# CT to NumPy

This small pipeline converts source CT scans from LIDC-IDRI and LNDb into
three-dimensional NumPy arrays. It replaces
`001_preprocessing/001_ct_to_npy.py` while deliberately keeping the design
compact: configuration, conversion logic, one runner, and tests.

## Directory structure

| File | Responsibility |
|---|---|
| `config.py` | Project-relative input and output paths. |
| `convert.py` | Discovery, validation, conversion, export, and metadata. |
| `export_png.py` | Convert a NumPy CT volume into 8-bit axial PNG slices. |
| `run_pipeline.py` | Command-line entry point for either or both datasets. |
| `requirements.txt` | Direct Python dependencies. |
| `tests/test_convert.py` | Synthetic unit and end-to-end conversion tests. |

## Data contract

Every `.npy` output contains:

- axis order `(slice, row, column)` or `(z, y, x)`;
- dtype `numpy.int16`;
- original CT intensities in Hounsfield units;
- no windowing, normalization, resampling, or segmentation.

Optional PNG files are visualization copies, not replacements for the NumPy
volumes. The fixed `[-1400, 200]` HU display range is clipped and mapped to
8-bit grayscale `[0, 255]`. Therefore, PNG values must not be interpreted as
the original Hounsfield Units.

Each array has a same-stem `.json` sidecar because NumPy arrays do not retain
medical-image geometry. Common metadata includes shape, dtype, axis order,
dataset, and source identifier. LNDb metadata also stores spacing, origin, and
direction from MetaImage. LIDC metadata stores spacing, slice thickness,
slice-z positions, and DICOM study/series UIDs available through pylidc.

## Inputs and outputs

Defaults are centralized in `config.py`:

```text
LNDb input:   000_dataset/lndb/data/**/*.mhd
LNDb output:  000_dataset/_lndb/001_volume_npy_v3/
LNDb PNG:     000_dataset/_lndb/001_volume_png_v3/
LIDC input:   pylidc's configured DICOM path and SQLite database
LIDC output:  000_dataset/_lidc/001_volume_npy_v3/
LIDC PNG:     000_dataset/_lidc/001_volume_png_v3/
```

The `_v3` output names preserve the behavior of the latest legacy script and
avoid overwriting the established `001_volume_npy` directories. Downstream
scripts currently reading `001_volume_npy` do **not** automatically switch to
these outputs; update their configuration only after validating the new data.

LNDb relative subdirectories are retained. LIDC filenames follow the existing
project convention:

```text
LIDC-IDRI-XXXX_<last-5-study-uid>_<last-5-series-uid>.npy
```

## Installation

```bash
python -m pip install -r 001_1_ct_to_npy/requirements.txt
```

LIDC conversion additionally requires a valid pylidc configuration and local
DICOM files. LNDb conversion does not import pylidc.

## Usage

Run commands from the project root:

Batch conversion and optional PNG export show one progress bar per operation,
followed by a compact total/written/skipped summary.

```bash
python 001_1_ct_to_npy/run_pipeline.py lndb
python 001_1_ct_to_npy/run_pipeline.py lidc
python 001_1_ct_to_npy/run_pipeline.py all
python 001_1_ct_to_npy/run_pipeline.py lidc --patient-id LIDC-IDRI-0001
```

Add `--png` to produce the NumPy/JSON output and all axial PNG slices in one
run:

```bash
python 001_1_ct_to_npy/run_pipeline.py \
    lidc \
    --patient-id LIDC-IDRI-0001 \
    --png

python 001_1_ct_to_npy/run_pipeline.py lndb --png
```

Each scan receives its own directory:

```text
001_volume_png_v3/
└── <scan-name>/
    ├── slice_0000.png
    ├── slice_0001.png
    └── ...
```

Existing complete `.npy`/`.json` pairs are skipped. A partial pair raises an
error to prevent inconsistent output. Explicit replacement requires:

```bash
python 001_1_ct_to_npy/run_pipeline.py lndb --overwrite
```

The same `--overwrite` option also replaces requested PNG slices. A complete
PNG directory is otherwise skipped, while an incomplete directory raises an
error for inspection.

Export PNG slices separately from an existing NumPy volume with:

```bash
python 001_1_ct_to_npy/export_png.py \
    volume.npy \
    output_png_directory
```

The standalone command accepts `--window-min`, `--window-max`, and
`--overwrite` when a different display range or explicit replacement is
needed.

The runner is fail-fast: a bad or unavailable scan stops the command and
preserves the error rather than reporting an apparently successful dataset.
It never deletes stale output files.

## Python API

When calling the converter from project-root code, add the module directory to
`PYTHONPATH`:

```bash
PYTHONPATH=001_1_ct_to_npy python your_program.py
```

```python
from config import LNDB_INPUT_DIR, LNDB_OUTPUT_DIR
from convert import convert_lndb_dataset, summarize

results = convert_lndb_dataset(LNDB_INPUT_DIR, LNDB_OUTPUT_DIR)
print(summarize(results))
```

## Validation

Before saving, the converter checks that every volume is nonempty, finite,
numeric, three-dimensional, and representable as `int16`. For LIDC, the number
of z positions must equal the number of slices. These checks prevent silent
axis, dtype, and geometry corruption.

Run the isolated synthetic tests with:

```bash
python -m unittest discover -s 001_1_ct_to_npy/tests -v
```

The tests write only to temporary directories.
