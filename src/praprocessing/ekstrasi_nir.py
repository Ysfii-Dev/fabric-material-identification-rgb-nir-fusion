import cv2
import os
import numpy as np
import pandas as pd

# Fungsi untuk mengekstraksi fitur NIR dengan pertukaran kanal


def extract_nir_from_image(image):
    # Pisahkan kanal warna
    B, G, R = cv2.split(image)

    # Pertukaran kanal untuk ekstraksi NIR
    # Asumsi: Saluran Red asli mengandung informasi NIR
    nir_proxy = R  # Kanal NIR Proxy = Saluran Red dari gambar asli

    # Hitung nilai rata-rata NIR
    mean_NIR = np.mean(nir_proxy)

    # # Juga ekstrak nilai RGB asli untuk referensi
    # mean_R = np.mean(R)
    # mean_G = np.mean(G)
    # mean_B = np.mean(B)

    # return mean_NIR, mean_R, mean_G, mean_B
    return mean_NIR

# Fungsi untuk memproses semua gambar dalam folder dan menghasilkan CSV


def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # mean_NIR, mean_R, mean_G, mean_B = extract_nir_from_image(
                mean_NIR = extract_nir_from_image(image)
                results.append({
                    'filename': filename,
                    'mean_NIR': mean_NIR,
                })
            else:
                print(f"Failed to read {image_path}")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Hasil telah disimpan ke {output_csv}')

# Fungsi untuk memproses semua folder dalam folder utama


def process_all_folders(main_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)

    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            output_csv = f'{folder_name}_nir_features.csv'
            output_csv_path = os.path.join(output_folder_path, output_csv)
            process_images_in_folder(folder_path, output_csv_path)


# Path folder utama dan output
main_folder_path = r"D:\Semester7\CV\identifikasi_kain\data\raw\NIR"
output_folder_path = r"D:\Semester7\CV\identifikasi_kain\outputs\nir_features"

# Eksekusi
if __name__ == "__main__":
    process_all_folders(main_folder_path, output_folder_path)
