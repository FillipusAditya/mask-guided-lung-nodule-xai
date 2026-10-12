# Evaluasi fidelity Grad-CAM dan LRP

Program `evaluate_xai_fidelity.py` mengukur fidelity explanation map dengan
**deletion test**. Program tidak melakukan training ulang dan tidak membuat
Grad-CAM atau LRP kembali. Explanation map yang sudah disimpan oleh proses
test/inference digunakan untuk menentukan urutan pixel yang dihapus.

Untuk setiap CT slice, program melakukan langkah berikut:

1. menjalankan inference pada CT asli;
2. memakai kelas prediksi CT asli sebagai `target_class`;
3. mengurutkan pixel dari attribution tertinggi ke terendah;
4. mengganti top 5%, 10%, 20%, 30%, 40%, dan 50% pixel secara kumulatif
   dengan nilai dari CT yang sudah di-Gaussian blur;
5. menjalankan inference ulang tanpa mengubah `target_class`; dan
6. menghitung area under the deletion curve atau Deletion AUC.

Semakin cepat probability target turun setelah pixel penting dihapus, semakin
rendah Deletion AUC dan semakin baik fidelity explanation tersebut.

Program mengevaluasi:

- metode XAI `gradcam` dan `lrp`;
- `layer1`, `layer2`, `layer3`, dan `layer4`;
- weak blur: sigma 3 dan kernel 19 x 19;
- medium blur: sigma 5 dan kernel 31 x 31; serta
- strong blur: sigma 10 dan kernel 61 x 61.

Medium blur merupakan konfigurasi utama. Weak dan strong blur digunakan untuk
sensitivity analysis.

## Kebutuhan program

Program membutuhkan:

- Python 3;
- PyTorch dan Torchvision;
- OpenCV;
- NumPy;
- pandas;
- Matplotlib; dan
- tqdm.

Explanation map dan metadata inference harus memiliki struktur berikut:

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

Jalankan perintah berikut dari root repository. Gunakan environment Python
yang sama dengan proses training agar versi PyTorch dan arsitektur model
sesuai dengan checkpoint.

## Smoke test yang disarankan

Lakukan smoke test sebelum evaluasi penuh:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir cv_resnet50 \
  --methods gradcam \
  --layers layer4 \
  --blur-config medium \
  --max-samples 5 \
  --device cuda
```

Hasil smoke test memperoleh suffix `_smoke_5`, sehingga tidak menimpa hasil
evaluasi penuh.

## Melanjutkan proses yang terhenti

Program menulis deletion curve secara bertahap ke
`fidelity_curve.csv.tmp`. Jika proses terhenti setelah sebagian sample selesai,
jalankan kembali perintah yang sama dan tambahkan `--resume`:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir \
  cv_resnet50 \
  multistage_direct_guided_cv_resnet50 \
  --device cuda \
  --resume
```

Konfigurasi metode, layer, blur, deletion percentage, model, dan output harus
sama dengan run yang terhenti. Program memvalidasi temporary curve sebelum
melanjutkan. Sample yang sudah lengkap dilewati, sedangkan temporary rows dari
sample yang belum lengkap dibuang dan dihitung ulang. Jika curve suatu model
sudah lengkap, inference model tersebut tidak diulang; program hanya membangun
kembali file agregasinya.

## Menjalankan semua model dan seluruh sensitivity analysis

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir \
  cv_resnet50 \
  multistage_direct_guided_cv_resnet50 \
  --device cuda
```

Pilihan default sudah mencakup kedua metode, keempat layer, ketiga blur, dan
deletion percentage 5, 10, 20, 30, 40, serta 50.

Hasil default disimpan di:

```text
experiment_results/fa028196-fd4c-441f-a5d3-3efb9707e7f9/
└── evaluation/xai_fidelity/
```

## Menjalankan satu folder model tertentu

Baseline ResNet-50:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir cv_resnet50 \
  --device cuda
```

Multistage direct-guided ResNet-50:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir multistage_direct_guided_cv_resnet50 \
  --device cuda
```

Nama folder relatif otomatis dicari di dalam:

```text
experiment_results/fa028196-fd4c-441f-a5d3-3efb9707e7f9/classification
```

Path absolut juga dapat diberikan pada `--model-dir`.

## Menjalankan hanya konfigurasi utama

Untuk menghemat waktu, evaluasi awal dapat dijalankan hanya dengan medium
blur:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir \
  cv_resnet50 \
  multistage_direct_guided_cv_resnet50 \
  --blur-config medium \
  --device cuda
```

Setelah itu, jalankan weak dan strong blur sebagai sensitivity analysis.

## Mengubah metode, layer, dan deletion percentage

Contoh evaluasi Grad-CAM pada layer3 dan layer4:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir cv_resnet50 \
  --methods gradcam \
  --layers layer3 layer4 \
  --deletion-percentages 5 10 20 30 40 50 \
  --device cuda
```

Deletion percentage adalah persentase seluruh pixel input, bukan persentase
ground-truth nodule mask. Top 10%, misalnya, berarti 10% pixel input dengan
attribution tertinggi diganti dengan nilai dari blurred CT.

## Pengaturan performa

`--inference-batch-size` mengatur jumlah citra hasil perturbasi yang diproses
bersamaan. Nilai default adalah 64. Turunkan nilainya jika GPU kehabisan
memori:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir cv_resnet50 \
  --inference-batch-size 16 \
  --device cuda
```

Tambahkan `--amp` untuk memakai automatic mixed precision pada CUDA. Karena
AMP dapat sedikit mengubah nilai probability, jalankan semua model dengan
pengaturan AMP yang sama agar perbandingan adil.

Program juga dapat dijalankan dengan `--device cpu`, tetapi evaluasi penuh akan
jauh lebih lambat.

## Menentukan lokasi lain

Experiment ID lain:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --experiment-dir experiment_results/EXPERIMENT_ID/classification \
  --model-dir cv_resnet50
```

Lokasi output lain:

```bash
python3 004_evaluation/evaluate_xai_fidelity.py \
  --model-dir cv_resnet50 \
  --output-dir 004_evaluation/results/xai_fidelity \
  --device cuda
```

Untuk direct-guided model, lokasi probability map segmentasi biasanya
terdeteksi otomatis. Jika diperlukan, tentukan secara eksplisit dengan
`--probability-root`.

## File hasil untuk setiap model

Setiap folder model menghasilkan:

1. `fidelity_curve.csv`
   - Satu baris untuk setiap deletion point pada setiap slice, metode, layer,
     dan blur.
   - Memuat probability asli, probability setelah perturbasi, probability
     drop, jumlah pixel yang dihapus, dan metadata target.

2. `fidelity_slice.csv`
   - Satu baris untuk setiap deletion curve pada level slice.
   - Memuat Deletion AUC dan probability pada setiap deletion percentage.

3. `fidelity_nodule.csv`
   - Rata-rata hasil seluruh slice yang termasuk dalam nodule yang sama.
   - Agregasi ini mencegah nodule dengan lebih banyak slice mendominasi hasil.

4. `fidelity_summary.csv`
   - Ringkasan antar-nodule untuk setiap model, metode, layer, dan blur.
   - Kolom utama adalah `mean_deletion_auc_across_nodules`.

5. `fidelity_primary_medium_blur.csv`
   - Bagian dari `fidelity_summary.csv` yang hanya menggunakan medium blur.
   - File ini adalah primary fidelity result.

6. `fidelity_curve_summary.csv`
   - Mean deletion curve antar-nodule untuk membuat atau memeriksa grafik.

7. `figures/*.png`
   - Grafik deletion curve untuk setiap metode dan blur.

8. `model_status.csv`
   - Konfigurasi evaluasi, jumlah sampel, jumlah nodule, device, dan waktu
     proses untuk model tersebut.

9. `inference_consistency_issues.csv`
   - Mencatat slice yang kelas atau probability hasil inference ulangnya tidak
     sama dengan cached inference.
   - File tetap dibuat dengan header meskipun tidak ada masalah.

Jika beberapa model dijalankan bersamaan, root output juga berisi:

- `fidelity_summary_combined.csv`;
- `fidelity_primary_medium_blur_combined.csv`; dan
- `model_status.csv`.

Gunakan kolom `model` untuk membandingkan model dalam file gabungan.

## Cara membaca Deletion AUC

Program menyimpan tiga bentuk AUC pada level slice:

- `deletion_auc_raw`: luas parsial dengan sumbu x dalam rentang 0 sampai 0.5;
- `deletion_auc_normalized`: AUC dibagi panjang rentang deletion, sehingga
  nilainya setara dengan rata-rata probability sepanjang kurva; dan
- `relative_deletion_auc`: probability setiap titik dibagi probability asli
  sebelum AUC dihitung.

Ringkasan utama menggunakan `deletion_auc_normalized` dan menamakannya
`mean_deletion_auc_across_nodules`. Nilai yang **lebih rendah lebih baik**.
Probability drop yang lebih besar juga menunjukkan fidelity yang lebih baik.

Perbandingan sebaiknya dilakukan pada blur yang sama. Kesimpulan dianggap
lebih kuat apabila urutan model, metode, dan layer tetap serupa pada weak,
medium, serta strong blur.

Probability tidak selalu turun secara monoton pada setiap sample. Menghapus
region tertentu dapat menaikkan probability karena interaksi nonlinear model.
Program mempertahankan nilai tersebut apa adanya agar hasil tidak bias.

## Catatan direct-guided model

Untuk `multistage_direct_guided_cv_resnet50`, deletion hanya diterapkan pada CT
asli. Probability map segmentasi tetap sama pada seluruh titik deletion curve.
Dengan demikian, pengujian mengukur apakah attribution pada cabang klasifikasi
menunjukkan region CT yang benar-benar dipakai model, tanpa sekaligus merusak
input guidance.

## Pemeriksaan konsistensi inference

Program menghitung ulang probability citra asli dan membandingkannya dengan
`test_predictions.csv`. Jika kelas prediksi berubah atau selisih probability
melebihi `--probability-tolerance` (default 0.005), masalah dicatat di
`inference_consistency_issues.csv` dan program tetap berjalan. Target fidelity
tetap `predicted_index` yang tersimpan karena Grad-CAM dan LRP dibuat untuk
kelas tersebut.

Perbedaan kelas dapat terjadi ketika probability sangat dekat dengan threshold
0.5 dan terdapat perbedaan numerik kecil antara runtime inference lama dan
runtime evaluasi. Gunakan `--strict-original-check` hanya jika program memang
harus berhenti pada ketidaksesuaian tersebut.
