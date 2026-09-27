# DenseNet-121 5-fold baseline

This baseline uses a pretrained DenseNet-121 with three-channel CT input. Feature
modules through `transition3` are frozen, while `denseblock4`, `norm5`, and the
classifier are trainable. It does not use a mask or probability map as model
input. The data splits, preprocessing, and evaluation are consistent with
`cv_resnet50` to support a fair comparison.

`train.py` is a standalone training program. It only uses shared functions
from `003_classification/utils` and does not depend on another experiment
package.

The main configuration is stored in
`003_classification/configs/cv_densenet121.json`. With the default configuration,
the output is stored under the same experiment UUID as the U-Net and guided
classifier components:

```text
experiment_results/a0d90f9e-3dd4-4de0-98af-12858696f613/classification/
├── baseline_resnet50/
└── cv_densenet121/
```

Run the programs from the repository root:

```bash
conda run -n deep_learning python -m pip install \
  -r 003_classification/cv_densenet121/requirements.txt
conda run -n deep_learning python -m 003_classification.cv_densenet121.train
conda run -n deep_learning python -m 003_classification.cv_densenet121.test
```

Training copies a snapshot of the effective JSON configuration to the output
directory. Testing evaluates the five-fold ensemble and produces metrics,
predictions, Grad-CAM, LRP, and one PNG for each study with a section for each
nodule. Grad-CAM targets `features.norm5`, the final normalized feature map in
DenseNet-121, and LRP uses Zennit `EpsilonPlusFlat` without an architecture
canonizer. The ground-truth mask is only used for visual
interpretation and is never passed to the baseline model.

The plateau scheduler is enabled by default and advances through the explicit
learning-rate sequence `1e-5, 5e-6, 1e-6`. Set `scheduler.enabled` to `false`
for a constant learning rate. The default `scheduler.patience` is 3, while
`early_stopping.patience` is 12 so the lower learning rates can be attempted
without unnecessarily extending a validation plateau.
The Colab notebook exposes the same values through `SCHEDULER_LEARNING_RATES`,
`SCHEDULER_PATIENCE`, and `EARLY_STOPPING_PATIENCE`. Scheduler progress and the
number of learning-rate reductions are saved in each latest checkpoint.

The Google Colab notebook is available as `cv_densenet121_colab.ipynb`. It reuses
`a0d90f9e-3dd4-4de0-98af-12858696f613_cv_resnet50_parenchyma.tar.gz` because
the metadata, parenchyma CT arrays, and masks are identical model-independent
inputs.
