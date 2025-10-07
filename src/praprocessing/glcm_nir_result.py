import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

# Fungsi untuk mengekstrak nilai NIR (menggunakan channel Red sebagai proxy)


def extract_nir_from_image(image):
    _, _, R = cv2.split(image)  # Ambil channel Red saja
    mean_intensity = np.mean(R)
    return mean_intensity

# Fungsi untuk menghitung fitur GLCM dari channel NIR


def calculate_glcm_features(image_channel):
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128,
                     144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(image_channel, bins)
    max_value = inds.max() + 1
    glcm = feature.graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=max_value, normed=True, symmetric=True)

    contrast = feature.graycoprops(glcm, 'contrast')
    dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')
    energy = feature.graycoprops(glcm, 'energy')
    correlation = feature.graycoprops(glcm, 'correlation')
    asm = feature.graycoprops(glcm, 'ASM')

    # Hitung Entropi
    glcm = glcm + 1e-10
    entropy = -np.sum(glcm * np.log2(glcm))

    return np.mean(contrast), np.mean(dissimilarity), np.mean(homogeneity), np.mean(energy), np.mean(correlation), np.mean(asm), entropy

# Fungsi untuk memproses semua gambar dalam satu folder


def process_images_in_folder(folder_path, output_csv):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Ekstrak channel Red sebagai proxy NIR
            nir_channel = img_as_ubyte(image[:, :, 2])
            mean_intensity = extract_nir_from_image(image)

            # Hitung fitur GLCM untuk NIR
            contrast, dissimilarity, homogeneity, energy, correlation, asm, entropy = calculate_glcm_features(
                nir_channel)

            results.append({
                'filename': filename,
                'mean_nir': mean_intensity,
                'contrast': contrast,
                'dissimilarity': dissimilarity,
                'homogeneity': homogeneity,
                'energy': energy,
                'correlation': correlation,
                'asm': asm,
                'entropy': entropy
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'Hasil telah disimpan ke {output_csv}')

# Fungsi untuk memproses semua subfolder


def process_all_folders(main_folder_path, output_folder_path):
    os.makedirs(output_folder_path, exist_ok=True)
    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
        if os.path.isdir(folder_path):
            output_csv = f'{folder_name}_nir_features.csv'
            output_csv_path = os.path.join(output_folder_path, output_csv)
            process_images_in_folder(folder_path, output_csv_path)


# Jalankan proses
# Gunakan folder RGB yang sama
main_folder_path = r"D:\Semester7\CV\identifikasi_kain\data\raw\NIR"
output_folder_path = r"D:\Semester7\CV\identifikasi_kain\outputs\glcm_nir_result"
process_all_folders(main_folder_path, output_folder_path)
