<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
</head>
<body>

<h1>📊 Mini Project Analisis Big Data: Analisis Cuaca Menggunakan API BMKG dan PySpark</h1>

<p>
    Proyek ini dikembangkan sebagai bagian dari <strong>proyek akhir mata kuliah Big Data</strong>. Tujuannya adalah untuk melakukan proses <strong>akuisisi data, preprocessing dan penyimpanan, hingga analisis dan visualisasi</strong> menggunakan tools Big Data. Proyek ini menggunakan data cuaca dari public API BMKG dan menganalisis tren dan prediksi cuaca menggunakan <code>PySpark</code> dan <code>scikit-learn</code>.
</p>

<hr>

<h2>👥 Pengembang</h2>
<ul>
    <li>Yonathan Sihotang (220211060127)</li>
    <li>David Haniko (220211060212)</li>
    <li>Natalio Tumuahi (220211060042)</li>
</ul>

<h3>👨‍🏫 Dosen Pengampu</h3>
<ul>
    <li>AGUSTINUS JACOBUS ST, M.Cs</li>
</ul>

<hr>

<h2>📌 Lingkup Proyek</h2>

<h3>1. Akuisisi Data</h3>
<ul>
    <li>Data diambil dari <strong>public API BMKG</strong></li>
    <li>Mendukung ekspansi ke data IoT atau data streaming menggunakan simulator (opsional)</li>
</ul>

<h3>2. Preprocessing dan Penyimpanan Data</h3>
<ul>
    <li>Melakukan proses cleaning dan transformasi data cuaca</li>
    <li>Menyimpan data ke dalam <code>Parquet</code> atau <code>CSV</code> format</li>
    <li>Penyimpanan mendukung media lokal atau berbasis cloud</li>
    <li>Preferensi penggunaan <strong>tools Big Data</strong> seperti Apache Spark</li>
</ul>

<h3>3. Analisis dan Visualisasi</h3>
<ul>
    <li>Melakukan analisis deskriptif dan prediktif terhadap data cuaca</li>
    <li>Menyajikan hasil analisis dalam bentuk visualisasi yang informatif</li>
    <li>Model dapat berupa custom model atau model publik (dari Kaggle, GitHub, HuggingFace)</li>
</ul>

<hr>

<h2>🚀 Fitur Proyek</h2>
<ul>
    <li>Akuisisi data cuaca harian dari API BMKG</li>
    <li>Preprocessing data untuk standarisasi dan pembersihan</li>
    <li>Penyimpanan dalam format <code>.parquet</code> untuk efisiensi dan skalabilitas</li>
    <li>Analisis data menggunakan <code>PySpark</code></li>
    <li>Model prediksi cuaca dengan algoritma <code>RandomForest</code></li>
    <li>Visualisasi hasil analisis (distribusi suhu, curah hujan, dll)</li>
</ul>

<hr>

<h2>🛠️ Cara Menjalankan</h2>
<pre><code># Clone repositori
git clone https://github.com/natalio123/analisis_cuaca.git
cd analisis_cuaca </code></pre>

<pre><code># Aktifkan virtual environment
python -m venv .env
source .env/bin/activate  # atau .env\Scripts\activate di Windows </code></pre>

<pre><code># Install dependencies
pip install -r requirements.txt</code></pre>

<pre><code># Jalankan proyek
python main.py
</code></pre>

<pre><code># Jalankan dashboard
streamlit run app.py
</code></pre>


<hr>

<h2>🧪 Pengujian dan Coverage</h2>
<pre><code>pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html
start htmlcov/index.html  # Windows
</code></pre>

<hr>

<h2>📂 Struktur Proyek</h2>
<pre><code>
analisis_cuaca/
├── data/
├── plots/
├── tests/
│   ├── test_main.py
│   └── test_akuisisiData.py
│   └── test_preprocessing.py
│   └── test_visualisasi.py
│   └── test_dashboard.py
├── utils/
│   ├── akuisisi_data.py
│   ├── preprocessing.py
│   ├── visualisasi.py
│   ├── dashboard.py
│   └── main.py
├── requirements.txt
└── README.md
</code></pre>

<hr>

</body>
</html>
