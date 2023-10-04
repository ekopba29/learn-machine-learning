# Laporan Proyek Machine Learning - Eko Purnomo

## Domain Proyek
Diabetes adalah penyakit kronis yang memengaruhi jutaan orang di seluruh dunia. Salah satu tantangan utama dalam menghadapi diabetes adalah deteksi dini dan manajemen yang efektif. Saat ini, banyak kasus diabetes didiagnosis pada tahap lanjut, yang dapat meningkatkan risiko komplikasi serius. Oleh karena itu, ada kebutuhan untuk mengidentifikasi cara yang lebih cepat dan efisien untuk mendeteksi diabetes pada tahap awal.

- Perawatan diabetes pada tahap lanjut dapat memakan biaya yang signifikan, termasuk biaya perawatan medis dan obat-obatan. Deteksi dini dan manajemen yang tepat waktu dapat mengurangi beban finansial yang ditimbulkan oleh penyakit ini
  Format Referensi: 

## Business Understanding
- Semakin banyak penderita diabetes tentu tidak baik, dan jika diabetes tidak dideteksi lebih awal tentu akan lebih parah lagi karena diabetes bisa menyebabkan banyak komplikasi ke penyakit lainya. Oleh karena itu dengan ada nya ML yang dapat membantu masyarakat mendeteksi dini diabetes mereka bisa mengunjungi fasilitas kesehatan untuk mendapatkan penanganan sebelum penyakit menjadi lebih buruk.

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, perusahaan akan mengembangkan sebuah sistem prediksi harga diamonds untuk menjawab permasalahan berikut.
- Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap positif diabetes ?
- Seberapa akurat deteksi yang dihasilkan oleh machine learning ?

### Goals
Untuk  menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Mengetahui fitur yang paling berkorelasi dengan positif diabetes.
- Membuat beberapa model dan membandingkanya untuk mendapatkan akurasi yang terbaik.

    ### Solution statements
    - Menggunakan beberapa algoritma dan membandingkanya. Diantaranya adalah algoritma DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier.
    - Dari hasil perbandingan antara akurasi, presisi, recall dan f1. Maka algoritma GradientBoostingClassifier dipilih karena memiliki F1 tertinggi diantara lainya

## Data Understanding
Dataset berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Tujuannya adalah untuk memprediksi apakah seorang pasien memiliki diabetes berdasarkan pengukuran diagnostik.

Ada beberapa batasan yang diterapkan pada pemilihan sampel-sampel ini dari basis data yang lebih besar. Secara khusus, semua pasien di sini adalah wanita yang setidaknya berusia 21 tahun dan keturunan Pima Indian.
Repository : https://www.kaggle.com/datasets/mathchi/diabetes-data-set

EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Dalam penerapan kasus ini meliputi memastikan data tidak ada yang null dan NA dan hasilnya tidak ada yang null/NA, kemudian pengecekan adanya outlier atau tidak, selain itu mencari korelasi masing-masing fitur terhadap fitur outcome (label positif atau negatif diabetes).

Untuk EDA Univariate Analysis sendiri ada beberapa hasil yang didapat dari dataset antara lain semakin tinggi gula semakin berpotensi positif diabet dan umur 20an lebih banyak yang negatid diabet. 

EDA Multivariate  correlation matrix diperoleh dari analisis heatmap dan diperoleh urutan dari yang paling berpengaruh yaitu Glucosa, BMI, Age, Pregnancies, BloodPressure, DiabetesPedingFreeFunction, Insulin, dan yang paling jauh Insulin

### Variabel-variabel pada dataset adalah sebagai berikut:
- Glucose: Konsentrasi glukosa plasma 2 jam setelah uji toleransi glukosa oral
- BloodPressure: Tekanan darah diastolik (mm Hg)
- SkinThickness: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- DiabetesPedigreeFunction: Fungsi pewaris diabetes
- Age: Usia (tahun)
- Outcome: Variabel kelas (0 atau 1)

## Data Preparation
- Membagi train dan test 80:20, untuk memastikan model berjalan dengan baik di data selain data train
- Mengurangi outlier dengan menggunakan metode IQR, Hal pertama yang perlu Anda lakukan adalah membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3. , setelah perhitungan selesai kemudian terapkan ke dataset agar tidak lebih dari batas atas dan bawah

## Modeling
- Untuk komparasi antar model menggunakan nilai default bawaan dari library, kemudian setelah komparasi dipilihlah model GradientBoostingClassifier kemudian dilakukan evaluasi dengan menerapkan gridsearch dan mendapat hasil parameter berikut learning_rate=0.01, max_depth=4, min_samples_leaf=8,min_samples_split=10, n_estimators=300,subsample=0.8

- Random Forest Classifier: Pengambilan dataset diabetes dengan subset acak pada setiap iterasi.  Hasil prediksi dari semua pohon digabungkan melalui mayoritas suara untuk menentukan diabet atau tidak.

- Decision Tree Classifier: Hampir sama dengan Random Forest Classifier, bedanya dalam Decision Tree Classifier, pohon keputusan tersebut adalah model tunggal yang didasarkan pada pemilihan fitur terbaik untuk memisahkan data selama proses pembelajaran. Setiap cabang dalam pohon keputusan mewakili keputusan positif diabet atau tidak atau prediksi berdasarkan aturan-aturan yang telah dipelajari dari data pelatihan.


- AdaBoost Classifier: AdaBoost akan mengambil sampel data secara acak dari dataset diabetes pada setiap iterasi pelatihan, dan kemudian melatih model lemah (biasanya Decision Trees) pada sampel tersebut. Algoritma Adaboost bekerja dengan cara secara iteratif melatih weak learners, seperti decision tree, pada dataset diabetes dan memberikan bobot pada setiap instance training berdasarkan kesalahan klasifikasinya. AdaBoost secara iteratif memperbaiki kemampuan model dalam mengklasifikasikan pasien diabetes dan non-diabetes.

- Gradient Boosting Classifier: Gradient Boosting tidak menggunakan pendekatan pengambilan sampel data secara acak dan pelatihan model pada setiap iterasi seperti yang dilakukan oleh AdaBoost. Sebaliknya, Gradient Boosting berfokus pada pengoptimalan model secara berurutan dengan meminimalkan kesalahan prediksi pada setiap iterasi.

- Untuk algoritma yang terpilih adalah GradientBoostingClassifier dengan bantuan gridsearch untuk mendapatkan hypermeter yang bagus sehingga menghasilkan F1 yang tinggi 

## Evaluation
- kesulitan dalam case ini adalah minimnya data set dan mencari parameter yang dapat menghasilkan score f1 tinggi
- Decision Tree Classifier: Dari algoritma ini diperoleh score F1 sebesar 0.540541 yang mana paling kecil dari lainya.
- Random Forest Classifier: F1 score dari algoritma ini menempati peringkat ke dua setelah GradientBoostingClassifier dengan perolehan F1 sebesar 0.714286
- Gradient Boosting Classifier: Ini merupakan algoritma dengan perolehan paling besar diantara lainya dengan F1 0.795290
- AdaBoost Classifier: . Perolehan F1 sebesar 0.625000 dan menempati urutan ke 3.
- Dari model yang telah dibuat diperoleh scoring f1-macro sebesar 81 persen.
- Matrix yang digunakan adalah F1, F1 adalah metrik evaluasi yang mengukur kualitas klasifikasi pada masalah klasifikasi biner (dua kelas) berdasarkan presisi (precision) dan recall. F1 adalah harmonic mean dari presisi dan recall, yang memberikan bobot yang seimbang pada kedua metrik tersebut. Hasil penerapan F1 score dalam kasus diabetes memberikan  gambaran tentang seberapa baik model klasifikasi dapat mengidentifikasi pasien diabetes (kelas positif) tanpa mengabaikan keakuratan hasil.Nilai F1 score berkisar antara 0 hingga 1, di mana:
F1 score = 1: Model sempurna yang memiliki presisi dan recall yang sempurna.
F1 score = 0: Model yang sangat buruk yang tidak memiliki presisi atau recall.

Berikut tabel hasil masing masing metrik untuk setiap algoritma dengan hypermeter default.
|------------------------------|----------|-----------|----------|----------|----------|
| Model                        | Accuracy | Precision | Recall   | F1       | ROC AUC  |
|------------------------------|----------|-----------|----------|----------|----------|
| DecisionTreeClassifier       | 0.734375 | 0.526316  | 0.555556 | 0.540541 | 0.679952 |
| RandomForestClassifier       | 0.843750 | 0.735294  | 0.694444 | 0.714286 | 0.798309 |
| GradientBoostingClassifier   | 0.851562 | 0.774194  | 0.666667 | 0.716418 | 0.795290 |
| AdaBoostClassifier           | 0.812500 | 0.714286  | 0.555556 | 0.625000 | 0.734300 |
|------------------------------|----------|-----------|----------|----------|----------|

- Nilai F1 dapat diinterpretasikan sebagai mean harmonik dari presisi dan recall, di mana nilai F1 mencapai nilai terbaiknya pada 1 dan nilai terburuk pada 0. Kontribusi relatif presisi dan recall terhadap nilai F1 adalah sama. Rumus untuk nilai F1 adalah:

**---Ini adalah bagian akhir laporan---**
  - Judul : [Cost analysis of diabetes mellitus](https://journal.ugm.ac.id/jmpf/article/view/29634) 
  - Author : Elny Fitri(1*), Tri Murti Andayani(2), Endang Suparniati(3)
(1) Fakultas Farmasi, Universitas Gadjah Mada, Yogyakarta
(2) Fakultas Farmasi, Universitas Gadjah Mada, Yogyakarta
(3) RSUP Dr. Sardjito Yogyakarta, Yogyakarta