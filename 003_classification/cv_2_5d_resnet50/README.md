# 2.5D ResNet-50 five-fold classification

This package classifies lung nodules from adjacent axial CT slices. For every
central slice `z`, the model receives `[z-1, z, z+1]` as the three ResNet-50
input channels. Missing neighbours are replaced by the central slice by
default, so one metadata row remains one classification window.

The implementation is self-contained inside this package and only imports
shared metric, checkpoint, prediction, and seed helpers from
`003_classification.utils`. It does not import another experiment package.

## Training

Run from the repository root:

```bash
conda run -n deep_learning python -m 003_classification.cv_2_5d_resnet50.train
```

Use a custom configuration or output directory with:

```bash
conda run -n deep_learning python -m 003_classification.cv_2_5d_resnet50.train \
  --config 003_classification/configs/cv_2_5d_resnet50.json \
  --output-dir experiment_results/my_run/classification/cv_2_5d_resnet50
```

Training performs patient-grouped five-fold cross-validation. The holdout
partition is never used for optimization or model selection. Best checkpoints
are selected using validation loss. Both window-level and nodule-level metrics
are recorded; nodule probabilities are the arithmetic mean of all window
probabilities belonging to that nodule.

## Holdout test

```bash
conda run -n deep_learning python -m 003_classification.cv_2_5d_resnet50.test \
  experiment_results/<experiment-id>/classification/cv_2_5d_resnet50
```

The test program averages probabilities from all five fold models. Useful
options include:

```text
--device auto|cpu|cuda
--batch-size N
--num-workers N
--max-samples N
--skip-xai
--force-inference
--dpi N
--rows-per-figure N
```

Use `--skip-xai` for a faster metrics-only evaluation. XAI inference is much
more expensive because Grad-CAM and LRP are calculated for every fold model.

## Explainability semantics

- Standard Grad-CAM produces one 2D map for the complete three-slice window.
  It is displayed over the central slice.
- LRP preserves the input-channel dimension and therefore produces three maps:
  one for each input slice.
- When a physical slice occurs in multiple windows, its LRP maps are averaged
  and written to `test/lrp_slice_aggregated_npy`.
- Ground-truth masks are used only in figures and are never passed to the
  classifier.

## Main outputs

```text
cv_2_5d_resnet50/
├── fold_0 ... fold_4/
├── cv_summary.csv
├── cv_summary.json
├── out_of_fold_predictions.csv
├── out_of_fold_nodule_predictions.csv
└── test/
    ├── test_predictions.csv
    ├── test_nodule_predictions.csv
    ├── test_results.json
    ├── gradcam_npy/
    ├── lrp_window_npy/
    ├── lrp_slice_aggregated_npy/
    └── visualization/
```

The English Google Colab workflow is available in
`cv_2_5d_resnet50_colab.ipynb`. Its parameter cell exposes every configuration,
training, testing, and XAI option in one place.

