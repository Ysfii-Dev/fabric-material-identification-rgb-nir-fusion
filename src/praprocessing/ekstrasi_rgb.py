import cv2
import os
import numpy as np
import pandas as pd


# Fungsi untuk menghitung rata-rata nilai RGB langsung dari gambar
def extract_rgb_from_image(image):
    B, G, R = cv2.split(image)
    mean_R = np.mean(R)
    mean_G = np.mean(G)
    mean_B = np.mean(B)
    return mean_R, mean_G, mean_B

# Fungsi untuk memproses semua gambar dalam folder dan menghasilkan CSV


def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".JPG")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                mean_R, mean_G, mean_B = extract_rgb_from_image(image)
                results.append({
                    'filename': filename,
                    'mean_R': mean_R,
                    'mean_G': mean_G,
                    'mean_B': mean_B
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
            output_csv = f'{folder_name}_rgb_means.csv'
            output_csv_path = os.path.join(output_folder_path, output_csv)
            process_images_in_folder(folder_path, output_csv_path)


# Path folder utama dan output
main_folder_path = r"D:\Semester7\CV\identifikasi_kain\data\raw\RGB"
output_folder_path = r"D:\Semester7\CV\identifikasi_kain\outputs\rgb_means"

# Eksekusi
process_all_folders(main_folder_path, output_folder_path)
