# Cats vs Dogs Classifier 🐱🐶

**Deskripsi singkat**

Proyek ini adalah aplikasi klasifikasi gambar _Cats vs Dogs_ berbasis Convolutional Neural Network (CNN) yang dibuat dengan TensorFlow/Keras dan di-deploy menggunakan Streamlit. Aplikasi menerima upload gambar (single atau multiple), menampilkan prediksi kelas (Cat / Dog) beserta confidence dan visualisasi probabilitas.

---

## Fitur utama ✅

- Prediksi gambar (single dan batch)
- Visualisasi confidence dengan gauge & bar chart (Plotly)
- Download hasil prediksi sebagai CSV
- UI interaktif menggunakan Streamlit

---

## Struktur proyek 🔧

- `app.py` — Streamlit app (UI + inference)
- `cats_vs_dogs_cnn_final.keras` — model Keras hasil training (dipakai oleh `app.py`)
- `best_model.h5` — model alternatif / checkpoint
- `requirements.txt` — daftar dependensi
- `uts-deepl-mfikrizaelani078.ipynb` — notebook training dan eksperimen

---

## Instalasi (Windows) 💡

1. Buat virtual environment (opsional tapi direkomendasikan):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependensi:

```bash
pip install -r requirements.txt
```

> Catatan: Versi utama yang dipakai adalah TensorFlow >= 2.14, Streamlit >= 1.28.

---

## Menjalankan aplikasi 🚀

Jalankan perintah berikut di folder proyek:

```bash
streamlit run app.py
```

Lalu buka URL yang ditampilkan (biasanya `http://localhost:8501`).

**Tips:** Pastikan file model `cats_vs_dogs_cnn_final.keras` ada di folder yang sama dengan `app.py`. Jika tidak ada, aplikasi akan menampilkan pesan kesalahan saat loading model.

---

## Cara pakai aplikasi 🧭

- Pilih mode: _Single Image_ atau _Multiple Images_ di sidebar
- Upload gambar format `jpg`, `jpeg`, atau `png`
- Klik tombol **Predict** (atau **Predict All** untuk multiple)
- Lihat hasil, confidence, dan download CSV bila diperlukan

Parameter penting: **Confidence Threshold** di sidebar (default 80%). Prediksi dengan confidence di bawah threshold akan diberi peringatan.

---

## Melatih / Membangun model 🔬

Training dan eksperimen dilakukan pada notebook `uts-deepl-mfikrizaelani078.ipynb`. Langkah umum:

1. Buka notebook di Jupyter / Colab
2. Jalankan semua cell untuk memproses data, augmentasi, serta training
3. Simpan model yang terlatih sebagai `cats_vs_dogs_cnn_final.keras` atau `best_model.h5`

---

## Troubleshooting ⚠️

- Error saat load model: pastikan `cats_vs_dogs_cnn_final.keras` berada di direktori kerja yang sama dan kompatibel dengan versi TensorFlow yang terinstall.
- Jika ada masalah kompatibilitas package, coba buat ulang virtual environment dan install versi di `requirements.txt`.

---

## Kontribusi & Lisensi 📝

Kontribusi diterima melalui fork & pull request. Untuk penggunaan bebas, sertakan atribusi. (Anda dapat menambahkan lisensi resmi seperti MIT jika ingin)

---

Jika Anda ingin, saya bisa menambahkan contoh screenshot, GIF singkat, atau petunjuk deploy (Heroku/Streamlit Cloud). 🔧

---

© Project by [Your Name or Team]
