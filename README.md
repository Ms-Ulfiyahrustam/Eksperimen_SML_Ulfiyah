# Heart Disease Prediction Project - MLOps Pipeline - Eksperimen

Proyek ini bertujuan untuk membangun sistem prediksi penyakit jantung menggunakan pendekatan MLOps. Eksperimen mencakup siklus hidup pengembangan model mulai dari pemrosesan data, pelatihan, hingga pelacakan model menggunakan MLflow.

## 1. Deskripsi Eksperimen

Eksperimen ini berfokus pada klasifikasi biner untuk menentukan apakah seorang pasien memiliki risiko penyakit jantung berdasarkan data klinis (Cleveland Dataset). 

### Alur Kerja Eksperimen:
1.  **Data Preprocessing**: 
    * Penanganan data duplikat dan nilai yang hilang.
    * Scaling fitur numerik (seperti `age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
    * Encoding pada fitur kategorikal (seperti `sex`, `cp`, `restecg`, `thal`).
2.  **Model Building**: Menggunakan algoritma klasifikasi (seperti Random Forest atau Decision Tree) untuk melatih model.
3.  **Experiment Tracking**: Seluruh parameter model, metrik evaluasi (Accuracy, Precision, Recall), dan artefak model dicatat secara otomatis menggunakan **MLflow**.
4.  **Monitoring**: Visualisasi performa sistem dan metrik model diintegrasikan melalui **Grafana**.

## 2. Dataset Information
* **Nama**: Heart Disease Predictions (Cleveland Dataset)
* **Sumber**: [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
* **Jumlah Fitur**: 13 Fitur klinis.
* **Target**: 1 (Penyakit Jantung), 0 (Normal).

## 3. Pelacakan Model (MLflow)
Eksperimen ini menggunakan `mlflow.autolog()` untuk memastikan reproduksibilitas. Artefak yang dihasilkan meliputi:
* **Folder Model**: Berisi model yang telah dilatih dalam format serialisasi.
* **Estimator**: File `estimator.html` yang merinci struktur model.
* **Metrik**: Pencatatan otomatis akurasi dan loss pada setiap run.

## 4. Cara Menjalankan
1. Pastikan library terinstall: `pip install pandas scikit-learn mlflow`.
2. Jalankan notebook `preprocessing_steps.ipynb`.
3. Buka MLflow UI untuk melihat hasil eksperimen: `mlflow ui`.
