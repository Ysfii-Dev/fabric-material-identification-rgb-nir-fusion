# 🧵 Fabric Material Identification using RGB–NIR Data Fusion and Deep CNN

### 📘 Project Title

**Identifikasi Material Tekstil: Klasifikasi Kain Alami vs. Sintetis Berbasis Fusi Data RGB dan NIR dengan Deep Convolutional Neural Network (CNN)**  
**(Objek: Kain Katun & Sutra vs. Polyester & Rayon)**

---

## 📖 Deskripsi Proyek

Proyek ini bertujuan untuk **mengidentifikasi material tekstil alami dan sintetis** melalui pendekatan **fusi data multispektral (RGB dan NIR)** menggunakan **model Deep CNN**.  
Kain alami seperti _katun_ dan _sutra_ dibandingkan dengan kain sintetis seperti _polyester_ dan _rayon_, untuk menguji kemampuan jaringan saraf dalam mengenali pola visual dan spektral dari dua sumber data berbeda.

Penelitian ini merupakan langkah awal menuju sistem **otomatisasi klasifikasi kain** berbasis pengolahan citra dan kecerdasan buatan, yang dapat digunakan pada industri tekstil dan bidang kontrol kualitas material.

---

## 🧩 Struktur Direktori

```markdown
fabric-material-identification-rgb-nir-fusion/
│
├── data/
│ └── raw/
│ ├── RGB/
│ │ ├── katun/
│ │ ├── polyester/
│ │ ├── rayon/
│ │ └── sutra/
│ └── NIR/
│ ├── katun/
│ ├── polyester/
│ ├── rayon/
│ └── sutra/
│
├── outputs/
│ ├── analysis_results/
│ ├── glcm_nir_results/
│ ├── glcm_rgb_results/
│ └── models/
│
├── src/
│ ├── analysis/
│ │ ├── analysis_kain_result.ipynb
│ │ ├── fabric_channel_analysis.ipynb
│ │ └── fabric_classification_analysis.ipynb
│ │
│ ├── modeling/
│ │ └── model_classification_image_fiks.ipynb
│ │
│ └── praprocessing/
│ ├── augmentasi_script.py
│ ├── ekstrasi_rgb.py
│ ├── ekstrasi_nir.py
│ ├── ekstraksi_cnn_features.ipynb
│ ├── glcm_rgb_result.py
│ ├── glcm_nir_result.py
│ └── rename_files.py
│
└── README.md
```

---

## ⚙️ Tahapan Utama

1. **Praproses Data**

   - Penghapusan background citra
   - Normalisasi dan augmentasi gambar
   - Ekstraksi fitur GLCM (tekstur)
   - Ekstraksi fitur CNN (deep feature extraction)

2. **Fusi Data RGB & NIR**

   - Kombinasi fitur spasial (RGB) dan spektral (NIR)
   - Reduksi dimensi dan normalisasi antar kanal

3. **Klasifikasi Deep CNN**

   - Eksperimen dengan beberapa arsitektur CNN (VGG16, MobileNetV2, customCNN dsb.)
   - Pelatihan model berbasis data fusi
   - Evaluasi performa dengan metrik akurasi, presisi, recall, dan F1-score

4. **Analisis dan Interpretasi**
   - Visualisasi feature map dan confusion matrix
   - Analisis hasil klasifikasi tiap jenis kain
   - Pembandingan performa antar pendekatan (RGB vs NIR vs Fusion)

---

## 🧠 Teknologi yang Digunakan

| Komponen                | Teknologi                              |
| ----------------------- | -------------------------------------- |
| Bahasa Pemrograman      | Python 3.10+                           |
| Lingkungan              | Jupyter Notebook, Virtualenv           |
| Framework Deep Learning | TensorFlow / Keras                     |
| Analisis Tekstur        | GLCM (Gray Level Co-occurrence Matrix) |
| Visualisasi             | Matplotlib, Seaborn                    |
| Data Management         | Pandas, NumPy                          |
| Citra                   | OpenCV, Pillow                         |

---

## 🚀 Cara Menjalankan Proyek

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Ysfii-Dev/fabric-material-identification-rgb-nir-fusion.git
cd fabric-material-identification-rgb-nir-fusion
```

### 2️⃣ Buat Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate  # (Linux/Mac)
myenv\Scripts\activate     # (Windows)
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Jalankan Notebook

Masuk ke folder `src/analysis` atau `src/modeling`, lalu buka file `.ipynb` melalui Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📊 Hasil dan Analisis

Output utama proyek meliputi:

- Akurasi model klasifikasi kain alami vs sintetis
- Visualisasi fitur CNN dan hasil fusi RGB-NIR
- Tabel perbandingan hasil klasifikasi antar model
- GLCM feature map untuk masing-masing jenis kain

Hasil ini akan dijelaskan lebih rinci dalam laporan analisis di folder:

```
src/analysis/
```

---

---

## 👤 Author

**Yusfi Syawali**
📧 [LinkedIn](https://linkedin.com/in/yusfisyawali) | 🌐 [GitHub](https://github.com/Ysfii-Dev)
**&**
**Tim 6 Computer Vision**

---

## 🧾 License

This project is licensed under the **MIT License** — feel free to use and modify for research and educational purposes.

---

### 🌈 Acknowledgements

Proyek ini merupakan bagian dari riset akademik yang berfokus pada **penerapan pengolahan citra multispektral dan deep learning** untuk klasifikasi material tekstil.
Terima kasih kepada semua pihak yang mendukung penelitian ini.
