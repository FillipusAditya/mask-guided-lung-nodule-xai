# ResNet-50 5-fold baseline

This baseline uses a pretrained ResNet-50 with three-channel CT input. The stem
and residual stages `layer1` through `layer3` are frozen, while `layer4` and the
entire `fc` classifier are trainable. It does not use a mask or probability map
as model input. The data splits,
preprocessing, optimizer, early stopping, and evaluation are consistent with
`segmentation_guided_cv_resnet50` to support a fair comparison.

`train.py` is a standalone training program. It only uses shared functions
from `003_classification/utils` and does not depend on
`fulltuning_cv_resnet50/train.py`.

The main configuration is stored in
`003_classification/configs/cv_resnet50.json`. With the default configuration,
the output is stored under the same experiment UUID as the U-Net and guided
classifier components:

```text
experiment_results/a0d90f9e-3dd4-4de0-98af-12858696f613/
├── segmentation/unet/
└── classification/
    ├── guided_resnet50/
    └── baseline_resnet50/
```

Run the programs from the repository root:

```bash
conda run -n deep_learning python -m 003_classification.cv_resnet50.train
conda run -n deep_learning python -m pip install -r 003_classification/cv_resnet50/requirements.txt
conda run -n deep_learning python -m 003_classification.cv_resnet50.test
```

Training copies a snapshot of the effective JSON configuration to the output
directory. Testing evaluates the five-fold ensemble and produces metrics,
predictions, Grad-CAM, LRP, and one PNG for each study with a section for each
nodule. Each visualization contains four panels: full CT, ground-truth mask,
Grad-CAM overlay, and LRP overlay. The ground-truth mask is only used for visual
interpretation and is never passed to the baseline model.

The plateau scheduler is enabled by default and advances through `1e-5`, `5e-6`,
and `1e-6` when validation loss plateaus. `scheduler.patience` is 3 and
`early_stopping.patience` is 12. Frozen BatchNorm modules remain in evaluation
mode, while BatchNorm modules in `layer4` continue to learn.

The Google Colab notebook is available as `cv_resnet50_colab.ipynb`.
