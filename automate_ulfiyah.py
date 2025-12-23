import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def perform_preprocessing(input_path, output_path):
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
        print(f"[INFO] Data berhasil dimuat. Shape awal: {df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] File tidak ditemukan di: {input_path}")
        return None

    # 2. Handling Missing Values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("[INFO] Missing values ditemukan dan telah di-drop.")
    else:
        print("[INFO] Tidak ada missing values.")

    # 3. Encoding Categorical Variables

    object_cols = df.select_dtypes(include=['object']).columns

    # Simpan nama kolom untuk referensi
    print(f"[INFO] Kolom kategorikal yang di-encode: {list(object_cols)}")

    for col in object_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # 4. Feature Scaling (Opsional, tapi bagus untuk Heart Disease)
    target_col = 'HeartDisease'

    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Scale fitur X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Gabungkan kembali menjadi DataFrame utuh untuk disimpan
        df_clean = pd.DataFrame(X_scaled, columns=X.columns)
        df_clean[target_col] = y.values # Masukkan kembali target

        # 5. Simpan Hasil ke CSV
        df_clean.to_csv(output_path, index=False)
        print(f"[SUCCESS] Preprocessing selesai! File disimpan di: {output_path}")
        print(f"Shape akhir: {df_clean.shape}")

        return df_clean
    else:
        print(f"[WARNING] Kolom target '{target_col}' tidak ditemukan. Scaling dilewati.")
        df.to_csv(output_path, index=False)
        return df

if __name__ == "__main__":
    input_file = "heart_raw/heart.csv"  # File mentah (INPUT)
    output_file = "preprocessing/heart_preprocessing.csv" # File matang (OUTPUT)

    perform_preprocessing(input_file, output_file)
