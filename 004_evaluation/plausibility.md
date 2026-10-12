# Evaluasi plausibility Grad-CAM dan LRP

Program `evaluate_xai_plausibility.py` mengukur kesesuaian spasial antara
explanation map dan ground-truth nodule mask menggunakan intersection over
union (IoU).

Program mengevaluasi:

- metode XAI `gradcam` dan `lrp`;
- `layer1`, `layer2`, `layer3`, dan `layer4`;
- top attribution 1%, 5%, 10%, dan 20%; serta
- IoU@10% sebagai primary plausibility metric.

IoU dihitung terlebih dahulu pada setiap CT slice. Nilai slice kemudian
dirata-ratakan untuk setiap nodule. Ringkasan akhir merupakan mean dan sample
standard deviation dari IoU antar-nodule. Dengan demikian, nodule yang memiliki
lebih banyak slice tidak memperoleh bobot lebih besar.

## Kebutuhan program

Program menggunakan:

- Python 3;
- NumPy; dan
- Pillow.

Explanation map harus sudah dibuat oleh proses test/inference dengan struktur:

```text
nama_model/
└── test/
    ├── test_predictions.csv
    ├── gradcam_npy/
    │   ├── layer1/
    │   ├── layer2/
    │   ├── layer3/
    │   └── layer4/
    └── lrp_npy/
        ├── layer1/
        ├── layer2/
        ├── layer3/
        └── layer4/
```

Jalankan semua perintah berikut dari root repository.

## Menjalankan semua model

```bash
python3 004_evaluation/evaluate_xai_plausibility.py
```

Secara default, program mencari model berikut di dalam
`experiment_results/fa028196-fd4c-441f-a5d3-3efb9707e7f9/classification`:

- `cv_resnet50`;
- `multistage_direct_guided_cv_resnet50`; dan
- model guided dengan nama `guided_cv_resnet_50`, `guided_cv_resnet50`,
  `segmentation_guided_cv_resnet50`, atau `guided_resnet50`.

Model default yang belum memiliki folder hasil akan dilewati dan dicatat di
`model_status.csv`.

Hasil gabungan semua model disimpan di:

```text
experiment_results/fa028196-fd4c-441f-a5d3-3efb9707e7f9/
└── evaluation/xai_plausibility/
```

## Menjalankan satu folder model tertentu

Gunakan `--model-dir` diikuti nama folder model:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir cv_resnet50
```

Contoh untuk multistage direct-guided:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir multistage_direct_guided_cv_resnet50
```

Contoh untuk guided setelah hasil inference tersedia:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir guided_cv_resnet_50
```

Nama folder relatif tersebut otomatis dicari di dalam folder `classification`.
Untuk satu model yang dipilih secara eksplisit, output default disimpan secara
terpisah agar tidak menimpa hasil gabungan:

```text
evaluation/xai_plausibility/nama_model/
```

Program juga menerima path model absolut:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir "/path/ke/classification/cv_resnet50"
```

Jika model dipilih secara eksplisit tetapi foldernya belum ada, program akan
berhenti dan menampilkan `FileNotFoundError`. Hal ini membantu mendeteksi salah
ketik pada nama folder.

## Menjalankan beberapa folder model tertentu

Tuliskan beberapa nama folder setelah `--model-dir`:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir \
  cv_resnet50 \
  multistage_direct_guided_cv_resnet50
```

Hasil model-model pilihan tersebut digabungkan dan secara default disimpan di:

```text
evaluation/xai_plausibility/selected_models/
```

Kolom `model` pada setiap CSV menunjukkan model asal setiap baris.

Nama lama `--model-dirs` tetap dapat digunakan dan memiliki fungsi yang sama
dengan `--model-dir`.

## Menentukan direktori eksperimen lain

Gunakan `--experiment-dir` jika ingin mengevaluasi experiment ID yang berbeda:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --experiment-dir \
  experiment_results/EXPERIMENT_ID/classification \
  --model-dir cv_resnet50
```

## Menentukan lokasi output

Gunakan `--output-dir` untuk menentukan sendiri lokasi CSV:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir cv_resnet50 \
  --output-dir 004_evaluation/results/cv_resnet50
```

## Smoke test

Gunakan `--max-samples` untuk memastikan seluruh path dan file dapat dibaca
sebelum menjalankan evaluasi penuh:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir cv_resnet50 \
  --max-samples 5
```

Jika `--output-dir` tidak diberikan, nama direktori smoke test memperoleh suffix
`_smoke_5` sehingga tidak menimpa evaluasi penuh.

## Mengubah top-k attribution

Nilai default adalah 1%, 5%, 10%, dan 20%, dengan IoU@10% sebagai primary
metric. Contoh pengaturan lain:

```bash
python3 004_evaluation/evaluate_xai_plausibility.py \
  --model-dir cv_resnet50 \
  --top-k 1 5 15 20 \
  --primary-top-k 15
```

Nilai `--primary-top-k` harus terdapat di dalam daftar `--top-k`.

## File CSV hasil evaluasi

Program menghasilkan lima CSV:

1. `plausibility_slice.csv`
   - Satu baris untuk setiap kombinasi model, slice, metode XAI, layer, dan
     top-k.
   - Memuat ukuran intersection, union, dan IoU.

2. `plausibility_nodule.csv`
   - Agregasi seluruh slice yang termasuk dalam nodule yang sama.
   - `mean_slice_iou` merupakan nilai plausibility satu nodule.

3. `plausibility_summary.csv`
   - Ringkasan setiap kombinasi model, metode XAI, layer, dan top-k.
   - `mean_iou_across_nodules` dan `std_iou_across_nodules` adalah hasil utama
     pada level dataset.

4. `plausibility_primary_iou_at_10.csv`
   - Ringkasan khusus primary metric IoU@10%.
   - Nama file mengikuti nilai `--primary-top-k` jika nilainya diubah.

5. `model_status.csv`
   - Mencatat model yang berhasil dievaluasi, belum memiliki hasil inference,
     atau belum tersedia.

Jika beberapa model dievaluasi sekaligus, kelima CSV merupakan file gabungan.
Gunakan kolom `model` untuk membedakan hasil setiap model.

## Catatan target explanation

Explanation map yang saat ini tersimpan oleh program inference dibuat untuk
kelas hasil prediksi model. Oleh karena itu:

- `target_class` adalah kelas yang dijelaskan oleh Grad-CAM atau LRP;
- `true_class` adalah label ground truth; dan
- `correct_prediction` menunjukkan apakah keduanya sama.

Untuk LRP, hanya relevance positif (`relevance > 0`) yang digunakan. Nilai
negatif dibuang sebelum resize dan normalisasi. Grad-CAM juga dibatasi pada
positive attribution. Explanation map kemudian di-resize secara bilinear ke
resolusi ground-truth mask dan dinormalisasi ke rentang 0--1.
