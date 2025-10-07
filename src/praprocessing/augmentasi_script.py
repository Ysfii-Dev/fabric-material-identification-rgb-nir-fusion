import os
import cv2
import numpy as np
import albumentations as A

# Path ke folder dataset
dataset_path = r"D:\Semester7\CV\identifikasi_kain\data\proses_augmentasi\RGB"

# Daftar semua subfolder yang perlu diaugmentasi
subfolders = ["katun", "polyester", "rayon", "sutra"]

# Mendefinisikan augmentasi yang akan digunakan
transform = A.Compose([
    A.Rotate(limit=25, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(
        brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=0, p=0.5),
    A.CropAndPad(percent=(-0.1, 0.1), p=0.5),
    A.Blur(blur_limit=3, p=0.3)
])

# Fungsi untuk melakukan augmentasi pada sebuah folder


def augment_folder(folder_path, target_count=250):
    # Membuat daftar semua file gambar di folder
    image_files = [f for f in os.listdir(
        folder_path) if f.endswith('.jpg') or f.endswith('.JPG')]
    print(f"\nMemproses folder: {os.path.basename(folder_path)}")
    print(f"Jumlah gambar awal: {len(image_files)}")

    # Membaca semua gambar
    images = []
    valid_image_files = []
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            valid_image_files.append(image_file)
        else:
            print(f"Gagal membaca gambar: {image_file}")

    print(f"Jumlah gambar yang berhasil dibaca: {len(images)}")

    # Jika tidak ada gambar yang berhasil dibaca, skip folder ini
    if len(images) == 0:
        print(f"Tidak ada gambar yang dapat diproses di folder {folder_path}")
        return

    # Menghitung berapa banyak augmentasi yang diperlukan per gambar
    current_count = len(images)
    augmentations_per_image = max(
        1, (target_count - current_count) // current_count + 1)

    print(f"Melakukan {augmentations_per_image} augmentasi per gambar")

    # Melakukan augmentasi dan menyimpan gambar
    augmented_count = 0
    for i, image in enumerate(images):
        # Melakukan augmentasi sebanyak yang diperlukan
        for j in range(augmentations_per_image):
            if current_count + augmented_count >= target_count:
                break

            # Mengaplikasikan augmentasi
            augmented = transform(image=image)
            augmented_image = augmented["image"]

            # Membuat nama file baru
            base_name = os.path.splitext(valid_image_files[i])[0]
            new_filename = f"{base_name}_aug_{augmented_count+1}.jpg"
            new_filepath = os.path.join(folder_path, new_filename)

            # Menyimpan gambar hasil augmentasi
            cv2.imwrite(new_filepath, augmented_image)
            augmented_count += 1

            if (current_count + augmented_count) % 50 == 0:  # Update setiap 50 gambar
                print(f"Telah dibuat {current_count + augmented_count} gambar")

        if current_count + augmented_count >= target_count:
            break

    print(
        f"Augmentasi selesai! Total gambar sekarang: {current_count + augmented_count}")


# Memproses setiap subfolder
for folder_name in subfolders:
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        augment_folder(folder_path)
    else:
        print(f"Folder {folder_path} tidak ditemukan atau bukan direktori!")

print("\nAugmentasi selesai untuk semua folder!")
