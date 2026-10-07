# 📊 Sentilytics-AI — Automated NLP Sentiment Analytics & Model Training Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](#)
[![Django](https://img.shields.io/badge/Django-5.x-092E20.svg?logo=django&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-AutoML%20Engine-F7931E.svg?logo=scikit-learn&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Mining-150458.svg?logo=pandas&logoColor=white)](#)
[![Explainable AI](https://img.shields.io/badge/XAI-Narrative%20Insights-8B5CF6.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

**Sentilytics-AI** adalah platform web cerdas berbasis Django untuk analisis sentimen teks otomatis (*Automated NLP Sentiment Analysis*), pelatihan dan evaluasi model Machine Learning dinamis (*On-the-fly Model Training & Evaluation*), visualisasi Explainable AI (XAI), perbandingan multi-dataset, serta ekspor model pipeline (`.pkl`) siap pakai.

Didesain untuk data scientist, product manager, dan analis bisnis yang membutuhkan solusi cepat untuk mengevaluasi sentimen ulasan konsumen, mengidentifikasi faktor penentu kepuasan pelanggan, dan menghasilkan model klasifikasi siap produksi tanpa perlu menulis kode berulang kali.

---

## 🌟 Mengapa Menggunakan Sentilytics-AI?

- 🚫 **Tanpa Pelatihan Model Manual (*Zero-Code AutoML*)**: Cukup unggah berkas CSV berisi teks ulasan dan rating, sistem secara otomatis melakukan preprocessing, penentuan label sentimen, ekstraksi fitur TF-IDF, dan melatih model klasifikasi `RandomForest` secara instan.
- ⚡ **Evaluasi Komprehensif Real-Time**: Menyajikan metrik performa model terperinci (Akurasi, Presisi, Recall, F1-Score) disertai visualisasi Confusion Matrix heatmap resolusi tinggi.
- 🎯 **Explainable AI (XAI) & Narasi Sentimen Otomatis**: Tidak hanya angka, platform ini menyusun narasi insight otomatis serta mengekstrak 5 kata kunci utama (*top predictive words*) yang paling menentukan sentimen ulasan.
- 📦 **Multi-Dataset Benchmarking & Model Export**: Membandingkan distribusi sentimen dari berbagai dataset dalam satu sesi visual dan memungkinkan ekspor model machine learning (`.pkl` berisi classifier + TF-IDF vectorizer) untuk deployment produksi.

---

## 🚀 Fitur Utama

1. **Pembersihan & Pelabelan Teks Otomatis (`Text Sanitization & Labeling`)**:
   - Pembersihan teks ulasan menggunakan Regex sanitization dan pemetaan label otomatis berdasarkan rating (Rating 1–2: *Negative*, Rating 3: *Neutral*, Rating 4–5: *Positive*).
2. **Pelatihan Model Machine Learning Dinamis (`Random Forest + TF-IDF`)**:
   - Ekstraksi 1.000 fitur semantik dengan `TfidfVectorizer` dan pelatihan model `RandomForestClassifier` dengan konfigurasi parameter yang dapat disesuaikan (`n_estimators`, `sample_size`).
3. **Visualisasi Confusion Matrix Heatmap (`Seaborn + In-Memory Rendering`)**:
   - Visualisasi matriks klasifikasi interaktif yang digenerate langsung di sisi server dalam format Base64 tanpa meninggalkan temporary file.
4. **Narasi Insight & Diagnosis Reputasi AI (`Automated Sentiment Verdict`)**:
   - Analisis sentimen otomatis berbasis probabilitas dengan status diagnosis instan (*Favorit Pengguna*, *Krisis Reputasi*, *Cukup Positif*, *Perlu Evaluasi*).
5. **Komparasi Multi-Dataset Interaktif (`Multi-Dataset Benchmarking`)**:
   - Grafik batang perbandingan sentimen antar dataset untuk memantau tren performa produk/layanan yang berbeda dalam satu dashboard.
6. **Ekspor Model Pipeline Siap Pakai (`Downloadable .pkl Bundle`)**:
   - Mengunduh artefak model terlatih lengkap dengan vectorizer terkalibrasi untuk integrasi langsung ke microservice atau aplikasi backend lain.

---

## 🛠️ Tata Cara Instalasi

### 1. Prasyarat Sistem
Pastikan perangkat Anda telah terinstal:
- **Python 3.9+** ([Unduh Python](https://www.python.org/downloads/))
- **Git** ([Unduh Git](https://git-scm.com/))
- **pip** (Python Package Installer)

---

### 2. Clone Repository
```bash
git clone https://github.com/MasterPandaa/Sentilytics-AI.git
cd Sentilytics-AI
```

---

### 3. Setup Virtual Environment (Disarankan)

#### 🪟 Windows (PowerShell / Command Prompt)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### 🐧 Linux / 🍎 macOS (Bash / Zsh)
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

---

### 5. Menjalankan Server Django
```bash
cd dashboard_sentimen
python manage.py migrate
python manage.py runserver
```
Setelah server aktif, buka browser dan akses:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📖 Panduan Penggunaan

1. **Buka Dashboard**: Akses [http://127.0.0.1:8000](http://127.0.0.1:8000) pada browser Anda.
2. **Siapkan Format Dataset CSV**:
   Pastikan dataset CSV Anda memiliki kolom wajib berikut:
   | Kolom | Tipe Data | Deskripsi |
   |---|---|---|
   | `Ulasan` (atau `ulasan`) | Teks / String | Kalimat ulasan atau feedback pengguna |
   | `Rating` (atau `rating`) | Numerik (1-5) | Skor rating ulasan |
3. **Konfigurasi Parameter & Upload**:
   - Pilih file CSV ulasan Anda.
   - Tentukan parameter `Jumlah Pohon (n_estimators)` (default: 50).
   - Tentukan `Ukuran Sampel (sample_size)` (default: 5.000 data).
   - Klik tombol **"Analisis Dataset"**.
4. **Evaluasi Hasil & Narasi AI**:
   - Tinjau metrik evaluasi (Akurasi, Presisi, Recall, F1-Score, Rating Bintang).
   - Analisis grafik **Confusion Matrix** dan **Kata Kunci Pembeda Sentimen**.
   - Baca ringkasan kesimpulan dan diagnosis reputasi dari AI.
5. **Bandingkan & Download Model**:
   - Unggah dataset lain untuk membandingkan distribusi sentimen secara visual pada grafik komparasi.
   - Klik tombol **"Download Model (.pkl)"** untuk mengunduh bundle pipeline model terlatih.

---

## 📦 Struktur Project

```text
Sentilytics-AI/
├── dashboard_sentimen/
│   ├── analyzer/
│   │   ├── migrations/        # Skema migrasi database Django
│   │   ├── templates/
│   │   │   └── analyzer/
│   │   │       └── index.html # Antarmuka UI dashboard analisis interaktif
│   │   ├── admin.py           # Konfigurasi modul admin Django
│   │   ├── apps.py            # Konfigurasi registry aplikasi analyzer
│   │   ├── forms.py           # Validasi form upload CSV & parameter ML
│   │   ├── models.py          # Definisi ORM model data
│   │   ├── urls.py            # Routing endpoint URL analyzer
│   │   └── views.py           # Core engine: NLP preprocessing, ML training, XAI & export
│   ├── dashboard_sentimen/
│   │   ├── asgi.py            # Konfigurasi interface ASGI
│   │   ├── settings.py        # Pengaturan konfigurasi utama Django
│   │   ├── urls.py            # Root URL routing
│   │   └── wsgi.py            # Konfigurasi interface WSGI
│   ├── db.sqlite3             # Database SQLite lokal
│   └── manage.py              # CLI management script Django
├── .gitattributes             # Konfigurasi format line ending Git
├── Procfile                   # Definisi proses web server untuk deployment cloud
├── README.md                  # Dokumentasi & panduan penggunaan komprehensif
└── requirements.txt           # Daftar dependensi library Python proyek
```

---

## 🔒 Privasi & Keamanan

- **100% On-Premise & Local Processing**: Seluruh dataset, tokenisasi NLP, dan training model diproses secara lokal di server Anda tanpa dependensi API eksternal pihak ketiga.
- **In-Memory Graph & Model Rendering**: Visualisasi dan serialisasi model diproses langsung di memori (RAM) tanpa meninggalkan residu file sementara di media penyimpanan server.
- **Session-Isolated Storage**: Riwayat analisis tersimpan aman di dalam sesi pengguna yang dapat di-reset kapan saja.

---

## 📄 Lisensi

Didistribusikan di bawah lisensi [MIT](LICENSE). Bebas digunakan, dimodifikasi, dan didistribusikan untuk kebutuhan riset akademik, personal, maupun komersial.
