# VGG-16 5-fold baseline

This baseline uses full fine-tuning of VGG-16 with three-channel CT input.
It does not use a mask or probability map as model input. The data splits,
preprocessing, optimizer, early stopping, and evaluation are consistent with
`cv_resnet50` to support a fair comparison.

`train.py` is a standalone training program. It only uses shared functions
from `003_classification/utils` and does not depend on another experiment package.

The main configuration is stored in
`003_classification/configs/cv_vgg16.json`. With the default configuration,
the output is stored under the same experiment UUID as the U-Net and guided
classifier components:

```text
experiment_results/a0d90f9e-3dd4-4de0-98af-12858696f613/classification/
├── baseline_resnet50/
└── cv_vgg16/
```

Run the programs from the repository root:

```bash
conda run -n deep_learning python -m 003_classification.cv_vgg16.train
conda run -n deep_learning python -m pip install -r 003_classification/cv_vgg16/requirements.txt
conda run -n deep_learning python -m 003_classification.cv_vgg16.test
```

Training copies a snapshot of the effective JSON configuration to the output
directory. Testing evaluates the five-fold ensemble and produces metrics,
predictions, Grad-CAM, LRP, and one PNG for each study with a section for each
nodule. Grad-CAM targets `features[28]`, the final convolution in VGG-16, and
LRP uses Zennit `VGGCanonizer`. The ground-truth mask is only used for visual
interpretation and is never passed to the baseline model.

The plateau scheduler is enabled by default and advances through the explicit
learning-rate sequence `1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6`. Set
`scheduler.enabled` to `false` for a constant learning rate. The default
`scheduler.patience` is 3, while `early_stopping.patience` is 30 so every
configured learning rate can be attempted during a long validation plateau.
The Colab notebook exposes the same values through `SCHEDULER_LEARNING_RATES`,
`SCHEDULER_PATIENCE`, and `EARLY_STOPPING_PATIENCE`. Scheduler progress and the
number of learning-rate reductions are saved in each latest checkpoint.

The Google Colab notebook is available as `cv_vgg16_colab.ipynb`.
