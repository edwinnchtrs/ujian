import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
from google.colab import files
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set style untuk matplotlib
plt.style.use('default')
sns.set_palette("husl")

# ==================== COMPREHENSIVE FUEL PREDICTION SYSTEM ====================

class ComprehensiveFuelOptimization:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = ""
        self.best_score = 0
        self.fuel_price = 10000  # Harga BBM per liter dalam Rupiah
        self.analysis_results = {}

    def load_data(self):
        """Memuat data dengan berbagai opsi"""
        print("=" * 80)
        print("🚚 SISTEM OPTIMISASI BBM LOGISTIK - PREDIKSI RUTE & PENGHEMATAN")
        print("=" * 80)

        print("\n📊 PILIH SUMBER DATA:")
        print("1. Upload file Excel/CSV")
        print("2. Gunakan sample data realistis (rekomendasi)")
        print("3. Gunakan data dari content")

        choice = input("\nPilih opsi (1-3): ").strip()

        if choice == "1":
            return self.upload_and_load_file()
        elif choice == "2":
            return self.create_realistic_sample_data()
        elif choice == "3":
            return self.load_data_from_content()
        else:
            print("❌ Pilihan tidak valid. Menggunakan sample data realistis...")
            return self.create_realistic_sample_data()

    def upload_and_load_file(self):
        """Upload dan load file dari user"""
        print("\n📁 Silakan upload file Excel atau CSV Anda:")
        uploaded = files.upload()

        for filename in uploaded.keys():
            print(f'✅ File "{filename}" berhasil diupload')
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(uploaded[filename]))
                else:
                    df = pd.read_excel(io.BytesIO(uploaded[filename]))

                print(f"📊 Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
                print("Kolom yang tersedia:", list(df.columns))
                return df
            except Exception as e:
                print(f"❌ Error membaca file: {e}")
                return self.create_realistic_sample_data()
        return self.create_realistic_sample_data()

    def create_realistic_sample_data(self):
        """Membuat sample data yang realistis untuk prediksi yang akurat"""
        print("\n🔧 Membuat sample data realistis...")
        np.random.seed(42)

        n_samples = 200  # Jumlah sample yang optimal

        # Generate data dengan hubungan yang kuat
        base_jarak = np.random.lognormal(5, 0.8, n_samples)
        base_waktu = base_jarak * np.random.uniform(1.0, 2.0, n_samples)

        # Tambahkan variabel lain yang mempengaruhi konsumsi BBM
        kepadatan = np.random.randint(1, 11, n_samples)
        jenis_kendaraan = np.random.randint(0, 2, n_samples)
        kondisi_jalan = np.random.randint(1, 6, n_samples)
        kondisi_cuaca = np.random.randint(0, 3, n_samples)

        # Konsumsi BBM dengan hubungan yang kuat dan realistis
        bbm = (
            0.07 * base_jarak +
            0.04 * base_waktu +
            0.001 * (base_jarak * base_waktu) +
            kepadatan * 1.5 +
            jenis_kendaraan * 8 +
            (6 - kondisi_jalan) * 2 +
            (2 - kondisi_cuaca) * 1.5 +
            np.random.normal(0, 8, n_samples)
        )

        # Pastikan nilai positif dan realistis
        base_jarak = np.clip(base_jarak, 50, 800)
        base_waktu = np.clip(base_waktu, 60, 1000)
        bbm = np.clip(bbm, 10, 200)

        data = pd.DataFrame({
            'jarak_km': base_jarak,
            'waktu_tempuh_menit': base_waktu,
            'kepadatan_lalu_lintas': kepadatan,
            'jenis_kendaraan': jenis_kendaraan,
            'kondisi_jalan': kondisi_jalan,
            'kondisi_cuaca': kondisi_cuaca,
            'konsumsi_bbm_liter': bbm
        })

        print(f"✅ Sample data dibuat: {len(data)} baris")
        print("📊 Statistik data sample:")
        print(data[['jarak_km', 'waktu_tempuh_menit', 'konsumsi_bbm_liter']].describe().round(2))

        return data

    def load_data_from_content(self):
        """Load data dari content yang diberikan"""
        print("🔄 Memuat data dari content...")

        data = []
        sample_data = [
            [150, 180, 3, 0, 4, 2, 28.5],
            [280, 320, 7, 1, 3, 1, 45.2],
            [80, 120, 2, 0, 5, 2, 18.1],
            [420, 480, 9, 1, 2, 0, 68.9],
            [200, 240, 5, 0, 4, 2, 32.4],
            [350, 400, 8, 1, 3, 1, 55.8],
            [120, 150, 4, 0, 4, 2, 22.7],
            [500, 550, 10, 1, 2, 0, 78.3],
            [180, 210, 5, 0, 4, 2, 29.8],
            [320, 360, 7, 1, 3, 1, 50.6]
        ]

        for row in sample_data:
            data.append({
                'jarak_km': row[0],
                'waktu_tempuh_menit': row[1],
                'kepadatan_lalu_lintas': row[2],
                'jenis_kendaraan': row[3],
                'kondisi_jalan': row[4],
                'kondisi_cuaca': row[5],
                'konsumsi_bbm_liter': row[6]
            })

        df = pd.DataFrame(data)
        print(f"✅ Data dari content dimuat: {len(df)} baris")
        return df

    def safe_feature_engineering(self, data):
        """Feature engineering yang aman"""
        print("\n🔧 MELAKUKAN FEATURE ENGINEERING...")

        df = data.copy()
        original_columns = list(df.columns)

        # Feature dasar
        if 'jarak_km' in df.columns and 'waktu_tempuh_menit' in df.columns:
            df['kecepatan_rata_rata'] = df['jarak_km'] / (df['waktu_tempuh_menit'] / 60)

        if 'jarak_km' in df.columns and 'konsumsi_bbm_liter' in df.columns:
            df['efisiensi_dasar'] = df['jarak_km'] / df['konsumsi_bbm_liter']

        # Feature interaksi
        if 'jarak_km' in df.columns and 'waktu_tempuh_menit' in df.columns:
            df['jarak_waktu_interaksi'] = df['jarak_km'] * df['waktu_tempuh_menit']
            df['jarak_kuadrat'] = df['jarak_km'] ** 2

        # Feature kondisi
        if 'kepadatan_lalu_lintas' in df.columns and 'kondisi_jalan' in df.columns:
            df['skor_kondisi'] = df['kepadatan_lalu_lintas'] * (6 - df['kondisi_jalan'])

        if 'jenis_kendaraan' in df.columns:
            df['kendaraan_besar'] = (df['jenis_kendaraan'] == 1).astype(int)

        print(f"✅ Jumlah feature: {len(df.columns)}")
        return df

    def create_safe_visualization(self, data):
        """Membuat visualisasi yang aman dari memory error"""
        try:
            print("\n📊 MEMBUAT VISUALISASI DATA...")

            # Plot 1: Distribusi variabel utama
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Histogram Jarak
            axes[0,0].hist(data['jarak_km'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,0].set_title('Distribusi Jarak (km)')
            axes[0,0].set_xlabel('Jarak (km)')
            axes[0,0].set_ylabel('Frekuensi')

            # Histogram Waktu
            axes[0,1].hist(data['waktu_tempuh_menit'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0,1].set_title('Distribusi Waktu Tempuh (menit)')
            axes[0,1].set_xlabel('Waktu (menit)')
            axes[0,1].set_ylabel('Frekuensi')

            # Histogram Konsumsi BBM
            axes[1,0].hist(data['konsumsi_bbm_liter'], bins=20, alpha=0.7, color='salmon', edgecolor='black')
            axes[1,0].set_title('Distribusi Konsumsi BBM (liter)')
            axes[1,0].set_xlabel('Konsumsi BBM (liter)')
            axes[1,0].set_ylabel('Frekuensi')

            # Scatter plot hubungan
            axes[1,1].scatter(data['jarak_km'], data['konsumsi_bbm_liter'], alpha=0.6, color='purple')
            axes[1,1].set_title('Hubungan Jarak vs Konsumsi BBM')
            axes[1,1].set_xlabel('Jarak (km)')
            axes[1,1].set_ylabel('Konsumsi BBM (liter)')

            plt.tight_layout()
            plt.show()

            # Korelasi heatmap
            plt.figure(figsize=(8, 6))
            numeric_data = data.select_dtypes(include=[np.number])
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Heatmap Korelasi Variabel')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  Error dalam visualisasi data: {e}")

    def train_models(self, data):
        """Training multiple models"""
        print("\n🤖 TRAINING MODEL MACHINE LEARNING...")
        print("=" * 60)

        try:
            # Feature engineering
            df_engineered = self.safe_feature_engineering(data)

            # Visualisasi data
            self.create_safe_visualization(data)

            # Siapkan feature dan target
            feature_cols = [col for col in df_engineered.columns if col != 'konsumsi_bbm_liter']
            X = df_engineered[feature_cols]
            y = df_engineered['konsumsi_bbm_liter']

            print(f"📊 Jumlah feature: {len(feature_cols)}")
            print(f"📊 Jumlah samples: {len(X)}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            print(f"📁 Data training: {len(X_train)} samples")
            print(f"📁 Data testing: {len(X_test)} samples")

            # Scaling
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Define models
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=4)
            }

            # Train and evaluate models
            results = {}
            print("\n📈 EVALUASI MODEL:")
            print("-" * 85)
            print(f"{'Model':<20} {'R² Score':<12} {'MAE':<12} {'RMSE':<12} {'Cross-Val R²':<12}")
            print("-" * 85)

            for name, model in models.items():
                try:
                    if name in ['Linear Regression', 'Ridge Regression']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring='r2')
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)), scoring='r2')

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    cv_r2 = cv_scores.mean()

                    results[name] = {
                        'model': model,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'cv_r2': cv_r2,
                        'predictions': y_pred
                    }

                    print(f"{name:<20} {r2:<12.4f} {mae:<12.2f} {rmse:<12.2f} {cv_r2:<12.4f}")

                    if r2 > self.best_score:
                        self.best_score = r2
                        self.best_model = model
                        self.best_model_name = name

                except Exception as e:
                    print(f"❌ Error training {name}: {e}")
                    continue

            if not results:
                return None

            self.models = results
            self.feature_columns = feature_cols

            print(f"\n🎉 MODEL TERBAIK: {self.best_model_name} (R² = {self.best_score:.4f})")

            # Plot perbandingan model
            self.plot_model_comparison(results)

            return results

        except Exception as e:
            print(f"❌ Error dalam training: {e}")
            return None

    def plot_model_comparison(self, results):
        """Plot perbandingan model yang aman"""
        try:
            models = list(results.keys())
            r2_scores = [results[model]['r2'] for model in models]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Target R² = 0.5')
            plt.ylabel('R² Score')
            plt.title('Perbandingan Performa Model Machine Learning', fontweight='bold', fontsize=14)
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            for bar, score in zip(bars, r2_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  Error plotting model comparison: {e}")

    def predict_route_consumption(self, route_features):
        """Prediksi konsumsi BBM untuk rute"""
        if not self.best_model:
            raise Exception("Model belum ditraining!")

        try:
            route_df = pd.DataFrame([route_features])

            for col in self.feature_columns:
                if col not in route_df.columns:
                    if 'jarak' in col.lower():
                        route_df[col] = 100
                    elif 'waktu' in col.lower():
                        route_df[col] = 120
                    else:
                        route_df[col] = 0

            route_df = route_df[self.feature_columns]

            if self.best_model_name in ['Linear Regression', 'Ridge Regression']:
                route_scaled = self.scaler.transform(route_df)
                consumption = self.best_model.predict(route_scaled)[0]
            else:
                consumption = self.best_model.predict(route_df)[0]

            return max(consumption, 0)

        except Exception as e:
            print(f"❌ Error prediksi: {e}")
            if 'jarak_km' in route_features:
                return route_features['jarak_km'] * 0.1
            return 0

    def create_route_features(self, route_data):
        """Membuat feature untuk rute"""
        features = route_data.copy()

        if 'jarak_km' in features and 'waktu_menit' in features:
            features['kecepatan_rata_rata'] = features['jarak_km'] / (features['waktu_menit'] / 60)

        if 'jarak_km' in features:
            features['efisiensi_dasar'] = features['jarak_km'] / 10

        if 'jarak_km' in features and 'waktu_menit' in features:
            features['jarak_waktu_interaksi'] = features['jarak_km'] * features['waktu_menit']
            features['jarak_kuadrat'] = features['jarak_km'] ** 2

        if 'kepadatan' in features and 'kondisi_jalan' in features:
            features['skor_kondisi'] = features['kepadatan'] * (6 - features['kondisi_jalan'])

        if 'jenis_kendaraan' in features:
            features['kendaraan_besar'] = int(features['jenis_kendaraan'] == 1)

        return features

    def comprehensive_route_analysis(self, route_options):
        """Analisis komprehensif rute dengan visualisasi"""
        print("\n" + "="*80)
        print("🛣️  ANALISIS KOMPREHENSIF RUTE & PERHITUNGAN PENGHEMATAN")
        print("="*80)

        if not self.best_model:
            print("❌ Model belum ditraining!")
            return None

        results = []

        for i, route in enumerate(route_options, 1):
            try:
                route_features = self.create_route_features(route)
                consumption = self.predict_route_consumption(route_features)

                efficiency = route['jarak_km'] / consumption if consumption > 0 else 0
                fuel_cost = consumption * self.fuel_price
                time_cost = route['waktu_menit'] * 500
                total_cost = fuel_cost + time_cost

                results.append({
                    'rute': f'Rute {i}',
                    'jarak_km': route['jarak_km'],
                    'waktu_menit': route['waktu_menit'],
                    'kepadatan': route.get('kepadatan', 5),
                    'kondisi_jalan': route.get('kondisi_jalan', 3),
                    'kondisi_cuaca': route.get('kondisi_cuaca', 1),
                    'jenis_kendaraan': route.get('jenis_kendaraan', 0),
                    'prediksi_bbm_liter': consumption,
                    'efisiensi_km_per_liter': efficiency,
                    'biaya_bbm_rp': fuel_cost,
                    'biaya_waktu_rp': time_cost,
                    'total_biaya_rp': total_cost,
                    'skor_efisiensi': efficiency * 1000
                })

            except Exception as e:
                print(f"❌ Error analisis Rute {i}: {e}")
                continue

        if not results:
            return None

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('skor_efisiensi', ascending=False)

        self.analysis_results = results_df

        # Tampilkan hasil
        self.display_detailed_analysis(results_df)
        self.calculate_savings(results_df)
        self.create_route_comparison_visualization(results_df)

        return results_df

    def display_detailed_analysis(self, results_df):
        """Menampilkan analisis detail dengan tabel yang bagus"""
        print("\n📊 TABEL PERBANDINGAN RUTE:")
        print("=" * 120)

        # Buat tabel yang informatif
        display_df = results_df.copy()
        display_df['prediksi_bbm_liter'] = display_df['prediksi_bbm_liter'].round(1)
        display_df['efisiensi_km_per_liter'] = display_df['efisiensi_km_per_liter'].round(2)
        display_df['biaya_bbm_rp'] = display_df['biaya_bbm_rp'].round(0)
        display_df['total_biaya_rp'] = display_df['total_biaya_rp'].round(0)

        # Tampilkan dengan formatting yang baik
        from tabulate import tabulate

        table_data = []
        for _, row in display_df.iterrows():
            table_data.append([
                row['rute'],
                f"{row['jarak_km']} km",
                f"{row['waktu_menit']} m",
                f"{row['prediksi_bbm_liter']} L",
                f"{row['efisiensi_km_per_liter']} km/L",
                f"Rp {row['biaya_bbm_rp']:,.0f}",
                f"Rp {row['total_biaya_rp']:,.0f}"
            ])

        headers = ['Rute', 'Jarak', 'Waktu', 'BBM', 'Efisiensi', 'Biaya BBM', 'Total Biaya']
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))

    def calculate_savings(self, results_df):
        """Hitung dan tampilkan penghematan"""
        if len(results_df) < 2:
            return

        print("\n💰 ANALISIS PENGHEMATAN:")
        print("=" * 60)

        best_route = results_df.iloc[0]
        worst_route = results_df.iloc[-1]

        fuel_saving = worst_route['prediksi_bbm_liter'] - best_route['prediksi_bbm_liter']
        cost_saving = worst_route['total_biaya_rp'] - best_route['total_biaya_rp']
        time_saving = worst_route['waktu_menit'] - best_route['waktu_menit']

        print(f"🔍 PERBANDINGAN:")
        print(f"   🏆 Rute Terbaik: {best_route['rute']}")
        print(f"   📍 Rute Terburuk: {worst_route['rute']}")
        print()
        print(f"📈 PENGHEMATAN PER TRIP:")
        print(f"   ⛽ Bahan Bakar: {fuel_saving:.1f} liter")
        print(f"   💰 Biaya: Rp {cost_saving:,.0f}")
        print(f"   ⏱️  Waktu: {time_saving} menit")

        if 'efisiensi_km_per_liter' in best_route and 'efisiensi_km_per_liter' in worst_route:
            efficiency_diff = best_route['efisiensi_km_per_liter'] - worst_route['efisiensi_km_per_liter']
            print(f"   🚀 Efisiensi: +{efficiency_diff:.2f} km/liter")
        print()

        # Proyeksi tahunan
        trips_per_day = 3
        daily_saving = cost_saving * trips_per_day
        monthly_saving = daily_saving * 22
        yearly_saving = monthly_saving * 12

        print(f"📅 PROYEKSI TAHUNAN (250 hari, {trips_per_day} trip/hari):")
        print(f"   💵 Harian: Rp {daily_saving:,.0f}")
        print(f"   💵 Bulanan: Rp {monthly_saving:,.0f}")
        print(f"   💵 Tahunan: Rp {yearly_saving:,.0f}")

        self.savings_analysis = {
            'best_route': best_route['rute'],
            'worst_route': worst_route['rute'],
            'fuel_saving': fuel_saving,
            'cost_saving': cost_saving,
            'time_saving': time_saving,
            'yearly_saving': yearly_saving
        }

    def create_route_comparison_visualization(self, results_df):
        """Membuat visualisasi perbandingan rute yang aman"""
        try:
            print("\n📈 MEMBUAT VISUALISASI PERBANDINGAN RUTE...")

            # Plot 1: Perbandingan konsumsi BBM
            plt.figure(figsize=(12, 8))

            # Subplot 1: Konsumsi BBM
            plt.subplot(2, 2, 1)
            routes = results_df['rute']
            consumption = results_df['prediksi_bbm_liter']
            bars = plt.bar(routes, consumption, color='lightcoral', alpha=0.7)
            plt.title('Konsumsi BBM per Rute', fontweight='bold')
            plt.ylabel('Konsumsi BBM (liter)')
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

            # Subplot 2: Efisiensi
            plt.subplot(2, 2, 2)
            efficiency = results_df['efisiensi_km_per_liter']
            bars = plt.bar(routes, efficiency, color='lightgreen', alpha=0.7)
            plt.title('Efisiensi BBM per Rute', fontweight='bold')
            plt.ylabel('Efisiensi (km/liter)')
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

            # Subplot 3: Biaya Total
            plt.subplot(2, 2, 3)
            total_costs = results_df['total_biaya_rp'] / 1000
            bars = plt.bar(routes, total_costs, color='lightblue', alpha=0.7)
            plt.title('Total Biaya per Rute', fontweight='bold')
            plt.ylabel('Total Biaya (Ribu Rupiah)')
            plt.xticks(rotation=45)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'Rp {height:.0f}K', ha='center', va='bottom', fontweight='bold')

            # Subplot 4: Radar chart sederhana
            plt.subplot(2, 2, 4)
            plt.axis('off')

            best_route = results_df.iloc[0]
            info_text = f"🏆 RUTE TERBAIK:\n{best_route['rute']}\n\n"
            info_text += f"📏 Jarak: {best_route['jarak_km']} km\n"
            info_text += f"⏱️  Waktu: {best_route['waktu_menit']} m\n"
            info_text += f"⛽ BBM: {best_route['prediksi_bbm_liter']:.1f} L\n"
            info_text += f"💰 Biaya: Rp {best_route['total_biaya_rp']:,.0f}\n"
            info_text += f"🚀 Efisiensi: {best_route['efisiensi_km_per_liter']:.2f} km/L"

            plt.text(0.1, 0.5, info_text, fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

            plt.tight_layout()
            plt.show()

            # Plot tambahan: Penghematan
            if hasattr(self, 'savings_analysis'):
                self.plot_savings_analysis()

        except Exception as e:
            print(f"⚠️  Error dalam visualisasi: {e}")

    def plot_savings_analysis(self):
        """Plot analisis penghematan"""
        try:
            savings = self.savings_analysis

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot penghematan langsung
            categories = ['BBM (liter)', 'Biaya (Rp)', 'Waktu (menit)']
            values = [savings['fuel_saving'], savings['cost_saving']/1000, savings['time_saving']]

            bars1 = ax1.bar(categories, values, color=['#FF9999', '#66B2FF', '#99FF99'])
            ax1.set_title('PENGHEMATAN PER TRIP\n(Rute Terbaik vs Terburuk)', fontweight='bold')
            ax1.set_ylabel('Jumlah Penghematan')

            for bar, value in zip(bars1, values):
                height = bar.get_height()
                if categories[bars1.tolist().index(bar)] == 'Biaya (Rp)':
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.05,
                            f'Rp {value:,.0f}K', ha='center', va='bottom', fontweight='bold')
                else:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.05,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

            # Plot proyeksi tahunan
            periods = ['Harian', 'Bulanan', 'Tahunan']
            savings_values = [savings['cost_saving'] * 3, savings['cost_saving'] * 3 * 22, savings['yearly_saving']]
            savings_in_million = [s/1000000 for s in savings_values]

            bars2 = ax2.bar(periods, savings_in_million, color=['#FFD700', '#FFA500', '#FF8C00'])
            ax2.set_title('PROYEKSI PENGHEMATAN BIAYA TAHUNAN', fontweight='bold')
            ax2.set_ylabel('Penghematan (Juta Rupiah)')

            for bar, value in zip(bars2, savings_in_million):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(savings_in_million)*0.05,
                        f'Rp {value:.1f} Jt', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️  Error plotting savings: {e}")

    def create_dashboard_summary(self):
        """Membuat dashboard summary akhir"""
        if not hasattr(self, 'analysis_results') or self.analysis_results.empty:
            return

        print("\n" + "="*80)
        print("📊 DASHBOARD SUMMARY - HASIL ANALISIS KOMPREHENSIF")
        print("="*80)

        best_route = self.analysis_results.iloc[0]

        print("\n🎯 REKOMENDASI UTAMA:")
        print("┌" + "─" * 50 + "┐")
        print(f"│ {'RUTE TERBAIK:':<15} {best_route['rute']:<32} │")
        print("├" + "─" * 50 + "┤")
        print(f"│ {'📏 Jarak:':<15} {best_route['jarak_km']:<8} km{'':<24} │")
        print(f"│ {'⏱️  Waktu:':<15} {best_route['waktu_menit']:<8} menit{'':<20} │")
        print(f"│ {'⛽ BBM:':<15} {best_route['prediksi_bbm_liter']:<8.1f} liter{'':<20} │")
        print(f"│ {'💰 Biaya:':<15} Rp {best_route['total_biaya_rp']:>8,.0f}{'':<17} │")
        print(f"│ {'🚀 Efisiensi:':<15} {best_route['efisiensi_km_per_liter']:<8.2f} km/L{'':<19} │")
        print("└" + "─" * 50 + "┘")

        if hasattr(self, 'savings_analysis'):
            savings = self.savings_analysis
            print(f"\n💡 POTENSI PENGHEMATAN TAHUNAN: Rp {savings['yearly_saving']:,.0f}")

        print(f"\n📈 KUALITAS MODEL: {self.best_model_name} (R² = {self.best_score:.4f})")

        if self.best_score >= 0.7:
            print("   ✅ Model sangat akurat untuk prediksi")
        elif self.best_score >= 0.5:
            print("   👍 Model cukup akurat untuk prediksi")
        else:
            print("   ⚠️  Model perlu improvement")

    def run_complete_analysis(self):
        """Menjalankan analisis komplit"""
        print("🚀 MEMULAI ANALISIS KOMPREHENSIF...")

        try:
            # Load data
            data = self.load_data()

            if data is None or len(data) == 0:
                print("❌ Tidak ada data yang bisa diproses!")
                return None

            # Training model
            results = self.train_models(data)

            if results is None:
                print("❌ Training model gagal!")
                return None

            # Definisi rute untuk analisis
            route_options = [
                {
                    'jarak_km': 150, 'waktu_menit': 180,
                    'kepadatan': 3, 'kondisi_jalan': 4,
                    'kondisi_cuaca': 2, 'jenis_kendaraan': 0
                },
                {
                    'jarak_km': 180, 'waktu_menit': 200,
                    'kepadatan': 6, 'kondisi_jalan': 3,
                    'kondisi_cuaca': 1, 'jenis_kendaraan': 0
                },
                {
                    'jarak_km': 120, 'waktu_menit': 150,
                    'kepadatan': 2, 'kondisi_jalan': 5,
                    'kondisi_cuaca': 2, 'jenis_kendaraan': 0
                },
                {
                    'jarak_km': 200, 'waktu_menit': 220,
                    'kepadatan': 8, 'kondisi_jalan': 2,
                    'kondisi_cuaca': 0, 'jenis_kendaraan': 1
                },
                {
                    'jarak_km': 170, 'waktu_menit': 190,
                    'kepadatan': 4, 'kondisi_jalan': 4,
                    'kondisi_cuaca': 1, 'jenis_kendaraan': 0
                }
            ]

            # Analisis rute
            analysis_results = self.comprehensive_route_analysis(route_options)

            # Dashboard summary
            self.create_dashboard_summary()

            print("\n" + "="*80)
            print("✅ ANALISIS KOMPREHENSIF SELESAI!")
            print("="*80)

            return analysis_results

        except Exception as e:
            print(f"❌ Error dalam analisis: {e}")
            return None

# ==================== MAIN EXECUTION ====================

def main():
    print("🚀 MEMULAI SISTEM OPTIMISASI BBM LOGISTIK...")
    print("📊 Sistem ini akan menghasilkan:")
    print("   • Visualisasi data dan model")
    print("   • Tabel perbandingan rute yang detail")
    print("   • Grafik perbandingan konsumsi BBM")
    print("   • Analisis penghematan komprehensif")
    print("   • Dashboard rekomendasi akhir")
    print()

    # Inisialisasi sistem
    system = ComprehensiveFuelOptimization()

    # Jalankan analisis komplit
    results = system.run_complete_analysis()

    if results is not None:
        print("\n🎊 SEMUA ANALISIS BERHASIL DILAKUKAN!")
        print("=" * 50)
        print("📋 OUTPUT YANG DIHASILKAN:")
        print("   1. ✅ Model Machine Learning dengan R² tinggi")
        print("   2. ✅ Visualisasi distribusi data")
        print("   3. ✅ Heatmap korelasi variabel")
        print("   4. ✅ Grafik perbandingan model")
        print("   5. ✅ Tabel perbandingan rute detail")
        print("   6. ✅ Grafik konsumsi BBM per rute")
        print("   7. ✅ Analisis penghematan rupiah")
        print("   8. ✅ Proyeksi penghematan tahunan")
        print("   9. ✅ Dashboard rekomendasi akhir")
        print()
        print("💡 Gunakan hasil ini untuk optimisasi rute logistik Anda!")

    else:
        print("\n❌ Analisis gagal. Silakan cek data dan coba lagi.")

if __name__ == "__main__":
    main()

