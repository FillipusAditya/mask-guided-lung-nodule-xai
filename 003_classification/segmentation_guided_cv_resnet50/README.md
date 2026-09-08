# Segmentation-guided ResNet-50

Model ini mempertahankan konfigurasi full fine-tuning dan 5-fold CV dari
`fulltuning_cv_resnet50`, lalu menambahkan U-Net probability map sebagai
residual multiplicative attention setelah ResNet `layer3`:

![Diagram arsitektur](../architecture_diagrams/segmentation_guided_resnet50.png)

```text
CT (3x224x224) -> ResNet stem/layer1-3 -> F (1024x14x14) ----(*)----> layer4 -> GAP -> classifier
                                                         /    |
Probability map -> resize -> Conv3x3 -> Conv1x1 -> sigmoid     |
                              A (1024x14x14)             F * (1 + alpha*A)
```

`alpha` dipelajari dan diinisialisasi ke nol. Dengan demikian, model dimulai
sebagai ResNet-50 biasa dan belajar seberapa kuat probability map perlu
memandu fitur. CT dan probability map menerima transform geometris yang sama;
augmentasi intensitas hanya diterapkan pada CT. Interpolasi linear dipakai
untuk mempertahankan nilai probability map yang kontinu.

Setiap fold dijalankan maksimal 100 epoch dengan early stopping berdasarkan
`val_loss`. Training fold dihentikan setelah 20 epoch berturut-turut tanpa
penurunan `val_loss`, kemudian bobot dari epoch terbaik dimuat kembali untuk
evaluasi dan pembuatan prediksi validation.

## Data

Konfigurasi default membaca:

- metadata: `000_dataset/_segmentation_dataset_v2/004_classification_cv_5fold_seed42.csv`
- CT: kolom `ct_windowed_path`
- probability map: `segmentation_results/unet_holdout_split/242d4058-fee2-47cb-b1f2-6608348300f5/inference/probability_npy`

Nama setiap probability `.npy` harus sama dengan kolom `filename`. Dataset
memvalidasi keberadaan semua pasangan, dimensi array, nilai finite, dan rentang
probabilitas `[0, 1]` sebelum training/inference. Resolusi awal CT dan
probability map boleh berbeda; keduanya di-resize langsung ke `224 x 224`
oleh transform berpasangan sebelum augmentasi geometris berikutnya.

## Menjalankan

Jalankan dari root repository pada environment `deep_learning`:

```bash
conda run -n deep_learning python -m 003_classification.segmentation_guided_cv_resnet50.train
```

Setelah kelima fold selesai, gunakan direktori `cv_result_*` yang dihasilkan:

```bash
conda run -n deep_learning python -m pip install -r 003_classification/segmentation_guided_cv_resnet50/requirements.txt
conda run -n deep_learning python -m 003_classification.segmentation_guided_cv_resnet50.test \
  classification_results/segmentation_guided_cv_resnet50/cv_result_YYYYMMDD_HHMMSS_ID
```

`test.py` melakukan ensemble atas lima model fold pada holdout test dan
menyimpan prediksi, metrik, confusion matrix, ROC, serta Grad-CAM dan LRP dalam
`<result_dir>/test/`. LRP menggunakan Zennit `EpsilonPlusFlat` dengan
`ResNetCanonizer` dan menjelaskan kanal CT dengan probability map yang dibuat
tetap untuk sampel tersebut.

Struktur utama hasil test:

```text
test/
├── gradcam/ dan gradcam_npy/
├── lrp/ dan lrp_npy/
├── xai/                       # panel CT, probability, Grad-CAM, dan LRP
├── test_predictions.csv
├── test_results.json
├── confusion_matrix.*
└── roc_curve.png
```

## Catatan validasi

Untuk estimasi CV yang benar-benar bebas kebocoran, probability map pada fold
validasi sebaiknya dihasilkan oleh U-Net yang tidak pernah dilatih memakai
pasien pada fold tersebut (out-of-fold segmentation). Run U-Net default di
atas tetap dipakai sesuai konfigurasi yang diminta, tetapi provenance split
segmentasinya perlu diperhitungkan ketika melaporkan hasil eksperimen.
