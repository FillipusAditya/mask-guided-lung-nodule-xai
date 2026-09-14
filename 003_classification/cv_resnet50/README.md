# ResNet-50 5-fold baseline

This baseline uses full fine-tuning of ResNet-50 with three-channel CT input.
It does not use a mask or probability map as model input. The data splits,
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
experiment_results/dc730a13-5813-4d87-b15c-3b630deb32b5/
├── segmentation/unet/
└── classification/
    ├── guided_resnet50/
    └── cv_resnet50/
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

The Google Colab notebook is available as `cv_resnet50_colab.ipynb`.
