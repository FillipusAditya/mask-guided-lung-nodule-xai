# ResNet-50 5-fold baseline

This baseline uses a fully fine-tuned ImageNet-pretrained ResNet-50 with
three-channel CT input. Every backbone stage, BatchNorm layer, and classifier
parameter is trainable. It does not use a mask or probability map as model
input. The data splits, preprocessing, optimizer, early stopping, and
evaluation are aligned with `multistage_direct_guided_cv_resnet50` so that the
guidance mechanism remains the main experimental difference.

`train.py` is a standalone training program. It only uses shared functions
from `003_classification/utils` and does not depend on another classification
architecture.

The main configuration is stored in
`003_classification/configs/cv_resnet50.json`. With the default configuration,
the output is stored under the same experiment UUID as the U-Net and guided
classifier components:

```text
experiment_results/fa028196-fd4c-441f-a5d3-3efb9707e7f9/
├── segmentation/unet/
└── classification/
    ├── multistage_direct_guided_cv_resnet50/
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
nodule. Grad-CAM and LRP are generated for `layer1`, `layer2`, `layer3`, and
`layer4`. Each visualization contains CT, ground-truth mask, four Grad-CAM
panels, and four LRP panels. The ground-truth mask is only used for visual
interpretation and is never passed to the baseline model.

The default experiment uses SGD with learning rate `1e-3`, no learning-rate
scheduler, and early-stopping patience `20`, matching the direct-guided
comparison. All BatchNorm modules remain in training mode during training.

The Google Colab notebook is available as `cv_resnet50_colab.ipynb`.
