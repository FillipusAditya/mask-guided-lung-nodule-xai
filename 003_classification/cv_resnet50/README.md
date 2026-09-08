# ResNet-50 5-fold baseline

Baseline ini memakai full fine-tuning ResNet-50 dengan input CT tiga kanal,
tanpa mask atau probability map sebagai input model. Split, preprocessing,
optimizer, early stopping, dan evaluasinya dibuat konsisten dengan
`segmentation_guided_cv_resnet50` agar perbandingan tetap adil.

Konfigurasi utama berada di `003_classification/configs/cv_resnet50.json`.
Dengan konfigurasi default, output disimpan bersama komponen U-Net dan guided
classifier dalam UUID yang sama:

```text
experiment_results/dc730a13-5813-4d87-b15c-3b630deb32b5/
├── segmentation/unet/
└── classification/
    ├── guided_resnet50/
    └── cv_resnet50/
```

Jalankan dari root repository:

```bash
conda run -n deep_learning python -m 003_classification.cv_resnet50.train
conda run -n deep_learning python -m pip install -r 003_classification/cv_resnet50/requirements.txt
conda run -n deep_learning python -m 003_classification.cv_resnet50.test
```

Training menyalin snapshot JSON efektif ke direktori output. Test melakukan
ensemble lima fold dan menghasilkan metrik, prediksi, Grad-CAM, LRP, serta satu
PNG per study dengan section per nodule. Visualisasi memiliki empat panel:
full CT, ground-truth mask, Grad-CAM overlay, dan LRP overlay. Ground-truth mask
hanya digunakan untuk interpretasi visual dan tidak masuk ke model.

Notebook Google Colab tersedia sebagai `cv_resnet50_colab.ipynb`.
