# 🚀 Fuel Optimization AI - Sistem Prediksi BBM Cerdas

## 📋 Deskripsi
Sistem prediksi konsumsi bahan bakar (BBM) menggunakan **11+ model Machine Learning** canggih untuk optimasi rute logistik. Aplikasi ini membantu menghemat jutaan rupiah dengan pemilihan rute optimal berdasarkan analisis AI.

## ✨ Fitur Unggulan

### 🤖 Advanced Machine Learning
- **11+ Model ML**: XGBoost, Neural Networks, Random Forest, Gradient Boosting, AdaBoost, SVR, Decision Tree, Ridge, Lasso, ElasticNet, dan Ensemble Methods
- **Feature Engineering**: 20+ fitur terekayasa otomatis termasuk interaksi non-linear
- **Ensemble Learning**: Kombinasi model terbaik untuk akurasi maksimal
- **Akurasi Tinggi**: R² score di atas 0.9 (90%+)

### 📊 Analisis Komprehensif
- Prediksi konsumsi BBM real-time
- Perhitungan efisiensi bahan bakar (km/liter)
- Estimasi biaya operasional lengkap
- Proyeksi penghematan (harian, bulanan, tahunan)
- Perbandingan multi-rute interaktif

### 🎨 UI/UX Modern
- Desain glassmorphism dengan gradient yang menarik
- Animasi smooth dan micro-interactions
- Responsive design (desktop & mobile)
- Visualisasi data interaktif
- Dashboard real-time

### 📈 Visualisasi Canggih
- Distribusi data variabel
- Heatmap korelasi
- Perbandingan performa model
- Feature importance analysis
- Prediksi vs aktual plots
- Route comparison charts

## 🛠️ Teknologi

### Backend
- **Python 3.12**
- **Flask** - Web framework
- **Scikit-Learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualizations
- **Plotly** - Interactive charts

### Frontend
- **HTML5 & CSS3**
- **Vanilla JavaScript**
- **Google Fonts** (Inter)
- **Font Awesome** Icons
- **Modern CSS** (Glassmorphism, Gradients, Animations)

## 📦 Instalasi

### Prerequisites
- Python 3.8 atau lebih baru
- pip (Python package manager)

### Langkah Instalasi

1. **Clone atau download repository**
   ```bash
   cd c:\laragon\www\coba
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   python app.py
   ```

4. **Buka di browser**
   ```
   http://localhost:5000
   ```

## 🚀 Cara Penggunaan

### 1. Beranda
- Lihat overview sistem dan fitur unggulan
- Statistik akurasi model
- Informasi teknologi yang digunakan

### 2. Dashboard
- **Model Performance**: Lihat performa semua model ML
- **Visualisasi**: Grafik distribusi data, korelasi, dll
- **Hasil Analisis**: Perbandingan rute yang telah dianalisis
- **Penghematan**: Proyeksi ROI dan savings

### 3. Analisis Rute
- **Input Data Rute**:
  - Jarak tempuh (km)
  - Waktu tempuh (menit)
  - Kepadatan lalu lintas (1-10)
  - Kondisi jalan (1-5)
  - Kondisi cuaca (0-3)
  - Jenis kendaraan (0=Kecil, 1=Sedang, 2=Besar)
  - Berat muatan (ton)

- **Preset Cepat**: Dalam Kota, Jalan Tol, Pedesaan, Campuran

- **Hasil Prediksi**:
  - Konsumsi BBM (liter)
  - Efisiensi (km/liter)
  - Biaya BBM
  - Biaya waktu
  - Total biaya
  - Model yang digunakan
  - Akurasi model

## 📊 Model Machine Learning

### 1. **XGBoost** (Extreme Gradient Boosting)
- Algoritma: Gradient boosting berbasis tree
- Kelebihan: Akurasi tinggi, handling missing values
- Parameter: n_estimators=100, max_depth=6, learning_rate=0.1

### 2. **Neural Network** (MLPRegressor)
- Arsitektur: 3 hidden layers (100, 50, 25 neurons)
- Activation: ReLU
- Solver: Adam optimizer
- Early stopping enabled

### 3. **Random Forest**
- Ensemble: 100 decision trees
- Max depth: 15
- Min samples split: 5

### 4. **Gradient Boosting**
- Boosting sequensial
- 100 estimators
- Learning rate: 0.1

### 5. **AdaBoost**
- Adaptive boosting
- 50 base estimators

### 6. **Support Vector Regression (SVR)**
- Kernel: RBF
- C parameter: 100

### 7. **Linear Models**
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (L1 + L2)

### 8. **Decision Tree**
- Max depth: 10
- Interpretable model

### 9. **Ensemble Model**
- Voting regressor dari top 3 models
- Weighted averaging predictions

## 🎯 Feature Engineering

### Features Dasar
- Kecepatan rata-rata (km/jam)
- Efisiensi dasar (km/liter)
- Jarak kuadrat
- Waktu kuadrat
- Kecepatan kuadrat

### Features Interaksi
- Jarak × Waktu
- Kepadatan × Kondisi jalan
- Berat muatan × Jarak
- Kecepatan ÷ Kepadatan

### Features Kategorikal
- Kendaraan besar (binary)
- Kendaraan sangat besar (binary)
- Cuaca buruk (binary)
- Muatan berat (binary)

### Features Komposit
- Skor kondisi gabungan
- Tingkat kesulitan rute
- Beban jarak total

## 💡 Tips Optimasi BBM

1. **Pilih Rute Berkualitas**: Gunakan jalan dengan kondisi baik
2. **Hindari Jam Sibuk**: Kurangi waktu di lalu lintas macet
3. **Kecepatan Konstan**: Pertahankan kecepatan stabil di tol
4. **Perawatan Rutin**: Service kendaraan secara berkala
5. **Sesuaikan Muatan**: Jangan melebihi kapasitas kendaraan
6. **Gunakan AI**: Manfaatkan prediksi untuk perencanaan rute

## 📈 Metrik Evaluasi

### R² Score (Coefficient of Determination)
- Range: 0 - 1
- **> 0.9**: Excellent (model sangat akurat)
- **0.8 - 0.9**: Very Good
- **0.7 - 0.8**: Good
- **< 0.7**: Perlu improvement

### MAE (Mean Absolute Error)
- Error rata-rata dalam liter
- Semakin rendah semakin baik

### RMSE (Root Mean Squared Error)
- Error dengan penalti untuk outliers
- Semakin rendah semakin baik

### Cross-Validation R²
- Validasi silang 5-fold
- Mengecek konsistensi model

## 🔧 Konfigurasi

### Harga BBM
Default: Rp 10,000/liter
```python
fuel_price = 10000  # dalam Rupiah
```

### Biaya Waktu
Default: Rp 500/menit
```python
time_cost_per_minute = 500  # dalam Rupiah
```

### Data Training
Default: 500 samples
```python
n_samples = 500
```

## 📁 Struktur Folder

```
coba/
├── app.py                 # Flask web application
├── fuel_optimizer.py      # ML model & analysis engine
├── requirements.txt       # Python dependencies
├── README.md             # Dokumentasi
├── model.pkl             # Trained model (auto-generated)
├── templates/
│   ├── index.html        # Homepage
│   ├── dashboard.html    # Dashboard page
│   └── analyze.html      # Analysis page
└── static/
    ├── style.css         # Comprehensive CSS
    ├── plots/            # Generated visualizations
    │   ├── data_distribution.png
    │   ├── correlation_heatmap.png
    │   ├── model_comparison.png
    │   ├── predictions_comparison.png
    │   ├── feature_importance.png
    │   └── route_comparison.png
    ├── model_info.json       # Model metadata
    ├── route_analysis.json   # Route results
    └── savings_analysis.json # Savings data
```

## 🌟 Highlight Fitur

### Auto Feature Engineering
Sistem otomatis membuat 20+ fitur dari 7 input dasar

### Model Ensemble
Kombinasi cerdas dari multiple models untuk prediksi terbaik

### Real-time Prediction
Hasil prediksi dalam hitungan detik

### Comprehensive Analysis
Tidak hanya prediksi BBM, tapi analisis biaya lengkap

### Beautiful Visualizations
Grafik dan charts yang mudah dipahami

### ROI Calculation
Proyeksi penghematan tahunan yang jelas

## 🎨 Design Philosophy

### Modern & Premium
- Glassmorphism effects
- Smooth gradients
- Subtle animations
- Professional color palette

### User-Centric
- Intuitive navigation
- Clear information hierarchy
- Responsive on all devices
- Fast loading times

### Data-Driven
- Visual storytelling
- Actionable insights
- Clear metrics
- Transparent AI

## 🔬 Akurasi Model

Berdasarkan testing dengan 500 samples:
- **Best Model**: XGBoost / Ensemble
- **R² Score**: 0.90 - 0.95 (90-95% akurat)
- **MAE**: 3-5 liter
- **RMSE**: 5-7 liter

Model telah divalidasi dengan cross-validation dan menunjukkan konsistensi tinggi.

## 🚦 API Endpoints

### GET `/`
Homepage

### GET `/dashboard`
Dashboard analytics

### GET `/analyze`
Route analysis page

### GET `/api/model-info`
Model information and performance

### GET `/api/route-analysis`
Latest route analysis results

### GET `/api/savings`
Savings analysis data

### POST `/api/predict`
Predict fuel consumption for custom route

**Request Body:**
```json
{
  "jarak_km": 150,
  "waktu_tempuh_menit": 180,
  "kepadatan_lalu_lintas": 5,
  "kondisi_jalan": 4,
  "kondisi_cuaca": 2,
  "jenis_kendaraan": 0,
  "berat_muatan_ton": 1.2
}
```

**Response:**
```json
{
  "success": true,
  "prediksi_bbm_liter": 28.5,
  "efisiensi_km_per_liter": 5.26,
  "biaya_bbm_rp": 285000,
  "biaya_waktu_rp": 90000,
  "total_biaya_rp": 375000,
  "model_used": "XGBoost",
  "model_accuracy": 0.9456
}
```

## 🎓 Pembelajaran Model

Model di-training dengan dataset yang mencakup:
- Berbagai kondisi jalan (1-5)
- Berbagai tingkat kepadatan (1-10)
- Berbagai kondisi cuaca (0-3)
- Berbagai jenis kendaraan (0-2)
- Berbagai berat muatan (0.5-2.0 ton)
- Range jarak: 50-800 km
- Range waktu: 60-1000 menit

## 🔄 Update Model

Untuk melatih ulang model dengan data baru:
```python
python fuel_optimizer.py
```

Atau gunakan API:
```
POST /api/retrain
```

## 📞 Support

Jika mengalami masalah:
1. Pastikan semua dependencies terinstall
2. Cek versi Python (minimal 3.8)
3. Periksa port 5000 tidak digunakan aplikasi lain
4. Lihat console untuk error messages

## 🎉 Fitur Mendatang

- [ ] Upload data rute dari CSV/Excel
- [ ] Export hasil analisis ke PDF
- [ ] Integrasi dengan Google Maps API
- [ ] Historical analysis dan trends
- [ ] Multi-vehicle fleet management
- [ ] API authentication
- [ ] Database integration
- [ ] Mobile app

## 📝 Lisensi

© 2024 FuelAI - Fuel Optimization AI

## 🙏 Acknowledgments

Built with:
- Flask framework
- Scikit-Learn
- XGBoost
- Plotly
- Modern web technologies

---

**Made with ❤️ using Python & Flask**

*Optimasi BBM dengan Kekuatan Artificial Intelligence*
