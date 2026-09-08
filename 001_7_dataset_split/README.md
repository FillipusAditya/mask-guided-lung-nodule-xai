# Patient-Level Dataset Splitting

This directory creates reproducible holdout splits and classification folds
for the prepared LIDC-IDRI and LNDb slice dataset. It replaces the three
standalone scripts previously stored in `001_preprocessing/utils` with one
modular pipeline.

## Processing flow

```text
prepared CT/mask slices + source label metadata
                         |
                         v
             validate pairs and attach labels
                         |
                         v
        split patients independently per source (70/15/15)
                         |
              +----------+----------+
              |                     |
              v                     v
       untouched test set    train + validation
                                    |
                                    v
                    stratified patient-grouped folds
```

The holdout assignment is made at patient level. Classification folds are
assigned from one row per nodule and then mapped back to all slices, so neither
patients nor slices from a nodule can leak between partitions.

## Directory structure

| File or directory | Responsibility |
|---|---|
| `config.py` | Input paths, output paths, ratios, fold count, and random seed. |
| `run_pipeline.py` | Unified CLI for holdout, CV, or both stages. |
| `dataset_split/holdout.py` | Filename parsing, label joins, validation, and patient splitting. |
| `dataset_split/pipeline.py` | Dataset-specific holdout orchestration and output naming. |
| `dataset_split/cross_validation.py` | Stratified grouped fold assignment and summaries. |
| `tests/` | Fast tests for naming, parsing, split isolation, and fold metadata. |

## Outputs

The default output directory is
`000_dataset/_segmentation_dataset_v2`:

```text
001_labeled_metadata_lidc_lndb.csv
001_holdout_split_lidc_lndb.csv
001_holdout_split_lidc_lndb.json
002_labeled_metadata_lidc.csv
002_holdout_split_lidc.csv
002_holdout_split_lidc_errors.json
003_labeled_metadata_lndb.csv
003_holdout_split_lndb.csv
003_holdout_split_lndb_errors.json
004_classification_cv_5fold_seed42.csv
004_classification_cv_5fold_seed42_summary.csv
```

The CV file preserves the original `split` column and adds:

- `cv_group_id`: source dataset plus patient identifier;
- `nodule_id` and `cv_nodule_id`: slice-independent nodule identifiers;
- `cv_role`: `development` or `holdout_test`;
- `cv_fold`: validation fold `0..4`, or `-1` for the untouched test set.

For validation fold `k`, train on development rows where `cv_fold != k`,
validate where `cv_fold == k`, and test only on `holdout_test` rows.

## Run

Run commands from the project root:

```bash
# Combined LIDC-IDRI + LNDb holdout, followed by classification CV.
python 001_dataset_split/run_pipeline.py all

# Only the combined holdout.
python 001_dataset_split/run_pipeline.py holdout

# A source-specific holdout.
python 001_dataset_split/run_pipeline.py holdout --dataset lidc
python 001_dataset_split/run_pipeline.py holdout --dataset lndb

# Only cross-validation, using the existing combined holdout.
python 001_dataset_split/run_pipeline.py cv
```

Existing complete outputs are skipped. Use `--overwrite` to regenerate them.
`--output-dir` is supported for a source-specific holdout.

Paths and reproducibility parameters are centralized in `config.py`.

## Python API

```python
from dataset_split import (
    create_classification_folds,
    run_single_dataset_split,
    split_patients,
)

train_ids, val_ids, test_ids = split_patients(patient_ids)
run_single_dataset_split("lidc")
output, summary = create_classification_folds()
```

When importing from the project root, include the pipeline directory:

```bash
PYTHONPATH=001_dataset_split python your_program.py
```

## Tests

```bash
python -m unittest discover -s 001_dataset_split/tests -v
```
