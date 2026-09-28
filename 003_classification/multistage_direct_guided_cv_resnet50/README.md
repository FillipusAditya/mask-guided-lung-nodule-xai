# Multi-stage direct-guided ResNet-50

This package implements an independent ResNet-50 classifier that applies the
U-Net probability map directly after `layer1`, `layer2`, and `layer3`:

```text
guided_features = features * (1 + alpha * resized_probability_map)
```

The model does not import or use the learned-attention model. By default, each
alpha is fixed at `1.0`, probability maps are reduced with adaptive max
pooling, and the classifier returns two raw logits.

Run the local architecture checks from this directory:

```bash
python test.py
```

Install evaluation and visualization dependencies when needed:

```bash
python -m pip install -r requirements.txt
```

Run the offline shape-only smoke test in the model file:

```bash
python direct_guided_resnet50.py
```

Run the fair-comparison five-fold training workflow from the repository root:

```bash
conda run -n deep_learning python \
  003_classification/multistage_direct_guided_cv_resnet50/train.py
```

The package configuration supplies the established fair-comparison data paths,
splits, preprocessing, and training hyperparameters. Results are written to
the same experiment UUID under the separate component
`classification/multistage_direct_guided_cv_resnet50`.

The default configuration is stored at:

`003_classification/configs/multistage_direct_guided_cv_resnet50.json`

After training, run the complete holdout ensemble and multi-stage XAI:

```bash
python test.py /path/to/classification/multistage_direct_guided_cv_resnet50 \
  --dataset-root /path/to/000_dataset_v2/_segmentation_dataset \
  --metadata-path /path/to/004_classification_cv_5fold_seed42.csv \
  --probability-root /path/to/probability_npy
```

The test output contains separate `gradcam_npy/layer1..layer4` and
`lrp_npy/layer1..layer4` directories. Study-level figures combine CT,
ground-truth mask, U-Net probability, four Grad-CAM panels, and four LRP
panels.
