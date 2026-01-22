import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class AdvancedFuelOptimization:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.best_model = None
        self.best_model_name = ""
        self.best_score = 0
        self.fuel_price = 10000  # Harga BBM per liter dalam Rupiah
        self.analysis_results = {}
        self.feature_importance = {}
        
    def create_realistic_sample_data(self, n_samples=500):
        """Membuat sample data yang realistis dengan jumlah lebih besar untuk deep learning"""
        print("\n [*] Membuat sample data realistis untuk training...")
        np.random.seed(42)

        # Generate data dengan hubungan yang kompleks
        base_jarak = np.random.lognormal(5, 0.8, n_samples)
        base_waktu = base_jarak * np.random.uniform(1.0, 2.0, n_samples)

        # Variabel yang mempengaruhi konsumsi BBM
        kepadatan = np.random.randint(1, 11, n_samples)
        jenis_kendaraan = np.random.randint(0, 3, n_samples)  # 0: kecil, 1: sedang, 2: besar
        kondisi_jalan = np.random.randint(1, 6, n_samples)
        kondisi_cuaca = np.random.randint(0, 4, n_samples)  # 0: hujan, 1: mendung, 2: cerah, 3: panas
        berat_muatan = np.random.uniform(0.5, 2.0, n_samples)  # Faktor berat muatan
        kecepatan_avg = base_jarak / (base_waktu / 60)
        
        # Konsumsi BBM dengan hubungan non-linear yang kompleks
        bbm = (
            0.05 * base_jarak +
            0.03 * base_waktu +
            0.0008 * (base_jarak * base_waktu) +
            kepadatan * 1.2 +
            jenis_kendaraan * 10 +
            (6 - kondisi_jalan) * 2.5 +
            (3 - kondisi_cuaca) * 1.2 +
            berat_muatan * 8 +
            0.002 * (kecepatan_avg ** 2) +  # Efek non-linear kecepatan
            np.random.normal(0, 5, n_samples)
        )

        # Pastikan nilai positif dan realistis
        base_jarak = np.clip(base_jarak, 50, 800)
        base_waktu = np.clip(base_waktu, 60, 1000)
        bbm = np.clip(bbm, 10, 250)

        data = pd.DataFrame({
            'jarak_km': base_jarak,
            'waktu_tempuh_menit': base_waktu,
            'kepadatan_lalu_lintas': kepadatan,
            'jenis_kendaraan': jenis_kendaraan,
            'kondisi_jalan': kondisi_jalan,
            'kondisi_cuaca': kondisi_cuaca,
            'berat_muatan_ton': berat_muatan,
            'konsumsi_bbm_liter': bbm
        })

        print(f"✅ Sample data dibuat: {len(data)} baris")
        print("📊 Statistik data sample:")
        print(data[['jarak_km', 'waktu_tempuh_menit', 'konsumsi_bbm_liter']].describe().round(2))

        return data

    def advanced_feature_engineering(self, data):
        """Feature engineering yang lebih kompleks"""
        print("\n🔧 MELAKUKAN ADVANCED FEATURE ENGINEERING...")

        df = data.copy()

        # Feature dasar
        if 'jarak_km' in df.columns and 'waktu_tempuh_menit' in df.columns:
            df['kecepatan_rata_rata'] = df['jarak_km'] / (df['waktu_tempuh_menit'] / 60)
            df['kecepatan_kuadrat'] = df['kecepatan_rata_rata'] ** 2
            
        if 'jarak_km' in df.columns and 'konsumsi_bbm_liter' in df.columns:
            df['efisiensi_dasar'] = df['jarak_km'] / (df['konsumsi_bbm_liter'] + 0.1)

        # Feature interaksi kompleks
        if 'jarak_km' in df.columns and 'waktu_tempuh_menit' in df.columns:
            df['jarak_waktu_interaksi'] = df['jarak_km'] * df['waktu_tempuh_menit']
            df['jarak_kuadrat'] = df['jarak_km'] ** 2
            df['waktu_kuadrat'] = df['waktu_tempuh_menit'] ** 2

        # Feature kondisi gabungan
        if 'kepadatan_lalu_lintas' in df.columns and 'kondisi_jalan' in df.columns:
            df['skor_kondisi'] = df['kepadatan_lalu_lintas'] * (6 - df['kondisi_jalan'])
            df['tingkat_kesulitan'] = (df['kepadatan_lalu_lintas'] * 0.5) + ((6 - df['kondisi_jalan']) * 0.5)

        if 'jenis_kendaraan' in df.columns:
            df['kendaraan_besar'] = (df['jenis_kendaraan'] >= 1).astype(int)
            df['kendaraan_sangat_besar'] = (df['jenis_kendaraan'] == 2).astype(int)

        # Feature cuaca dan kondisi
        if 'kondisi_cuaca' in df.columns:
            df['cuaca_buruk'] = (df['kondisi_cuaca'] <= 1).astype(int)
            
        # Feature berat muatan
        if 'berat_muatan_ton' in df.columns:
            df['muatan_berat'] = (df['berat_muatan_ton'] > 1.5).astype(int)
            if 'jarak_km' in df.columns:
                df['beban_jarak'] = df['berat_muatan_ton'] * df['jarak_km']

        # Rasio dan proporsi
        if 'kecepatan_rata_rata' in df.columns and 'kepadatan_lalu_lintas' in df.columns:
            df['kecepatan_per_kepadatan'] = df['kecepatan_rata_rata'] / (df['kepadatan_lalu_lintas'] + 1)

        print(f"✅ Total features setelah engineering: {len(df.columns)}")
        
        return df

    def create_visualizations(self, data):
        """Visualizations disabled in Lite version"""
        pass

    def train_advanced_models(self, data):
        """Training multiple advanced models dengan hyperparameter tuning"""
        print("\n🤖 TRAINING ADVANCED MACHINE LEARNING MODELS...")
        print("=" * 80)

        try:
            # Feature engineering
            df_engineered = self.advanced_feature_engineering(data)

            # Visualisasi data
            self.create_visualizations(data)

            # Siapkan feature dan target
            feature_cols = [col for col in df_engineered.columns if col != 'konsumsi_bbm_liter']
            X = df_engineered[feature_cols]
            y = df_engineered['konsumsi_bbm_liter']

            print(f"📊 Jumlah features: {len(feature_cols)}")
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

            # Define advanced models
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
                'Random Forest': RandomForestRegressor(
                    n_estimators=50, 
                    max_depth=10, 
                    random_state=42
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=50, 
                    max_depth=5,
                    random_state=42
                ),
                'AdaBoost': AdaBoostRegressor(
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=42
                ),
                'Neural Network': MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                'SVR': SVR(kernel='rbf', C=100, gamma='scale')
            }

            # Train and evaluate models
            results = {}
            print("\n📈 EVALUASI MODEL:")
            print("-" * 100)
            print(f"{'Model':<25} {'R² Score':<12} {'MAE':<12} {'RMSE':<12} {'Cross-Val R²':<15} {'Status':<15}")
            print("-" * 100)

            for name, model in models.items():
                try:
                    # Tentukan apakah model perlu scaling
                    needs_scaling = name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                                            'ElasticNet', 'SVR', 'Neural Network']
                    
                    if needs_scaling:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                                   cv=min(5, len(X_train)), scoring='r2')
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        cv_scores = cross_val_score(model, X_train, y_train, 
                                                   cv=min(5, len(X_train)), scoring='r2')

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    cv_r2 = cv_scores.mean()

                    # Status berdasarkan performa
                    if r2 >= 0.9:
                        status = "🌟 Excellent"
                    elif r2 >= 0.8:
                        status = "✅ Very Good"
                    elif r2 >= 0.7:
                        status = "👍 Good"
                    elif r2 >= 0.5:
                        status = "⚠️ Fair"
                    else:
                        status = "❌ Poor"

                    results[name] = {
                        'model': model,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'cv_r2': cv_r2,
                        'predictions': y_pred,
                        'needs_scaling': needs_scaling
                    }

                    print(f"{name:<25} {r2:<12.4f} {mae:<12.2f} {rmse:<12.2f} {cv_r2:<15.4f} {status:<15}")

                    if r2 > self.best_score:
                        self.best_score = r2
                        self.best_model = model
                        self.best_model_name = name

                    # Extract feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)

                except Exception as e:
                    print(f"{name:<25} ❌ Error: {str(e)[:50]}")
                    continue

            print("-" * 100)

            if not results:
                return None

            self.models = results
            self.feature_columns = feature_cols
            self.X_test = X_test
            self.y_test = y_test

            print(f"\n🎉 MODEL TERBAIK: {self.best_model_name} (R² = {self.best_score:.4f})")

            self.models = results
            self.feature_columns = feature_cols
            self.X_test = X_test
            self.y_test = y_test

            print(f"\n🎉 MODEL TERBAIK: {self.best_model_name} (R² = {self.best_score:.4f})")

            # Create ensemble model from top 3 models
            self.create_ensemble_model(results, X_train, y_train, X_test, y_test)

            return results

        except Exception as e:
            print(f"❌ Error dalam training: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_ensemble_model(self, results, X_train, y_train, X_test, y_test):
        """Membuat ensemble model dari top 3 models"""
        try:
            print("\n🔮 Membuat Ensemble Model dari Top 3 Models...")
            
            # Sort by R² score
            sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
            
            estimators = []
            for name, model_info in sorted_models:
                estimators.append((name, model_info['model']))
            
            # Create voting regressor
            ensemble = VotingRegressor(estimators=estimators)
            
            # Fit ensemble (gunakan scaled data jika semua model needs_scaling)
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"✅ Ensemble Model Created:")
            print(f"   Combining: {', '.join([name for name, _ in sorted_models])}")
            print(f"   R² Score: {r2:.4f}")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            
            # Update best model if ensemble is better
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = ensemble
                self.best_model_name = "Ensemble (Top 3)"
                self.models['Ensemble (Top 3)'] = {
                    'model': ensemble,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'cv_r2': r2,
                    'predictions': y_pred,
                    'needs_scaling': False
                }
                print(f"🎉 Ensemble model is now the BEST MODEL!")
                
        except Exception as e:
            print(f"⚠️ Error creating ensemble: {e}")

    def plot_model_comparison(self, results):
        pass

    def plot_predictions(self, results):
        pass

    def plot_feature_importance(self):
        pass

    def predict_route_consumption(self, route_features):
        """Prediksi konsumsi BBM untuk rute"""
        if not self.best_model:
            raise Exception("Model belum ditraining!")

        try:
            # Create DataFrame from route features
            route_df = pd.DataFrame([route_features])
            
            # Apply feature engineering
            route_engineered = self.advanced_feature_engineering(route_df)
            
            # Ensure all features match training features
            for col in self.feature_columns:
                if col not in route_engineered.columns:
                    route_engineered[col] = 0
            
            route_engineered = route_engineered[self.feature_columns]
            
            # Get model info
            model_info = self.models.get(self.best_model_name, {})
            needs_scaling = model_info.get('needs_scaling', False)
            
            # Make prediction
            if needs_scaling:
                route_scaled = self.scaler.transform(route_engineered)
                consumption = self.best_model.predict(route_scaled)[0]
            else:
                consumption = self.best_model.predict(route_engineered)[0]

            return max(consumption, 0)

        except Exception as e:
            print(f"❌ Error prediksi: {e}")
            # Fallback: simple estimation
            if 'jarak_km' in route_features:
                return route_features['jarak_km'] * 0.12
            return 0

    def analyze_routes(self, route_options):
        """Analisis komprehensif rute"""
        print("\n" + "="*100)
        print("🛣️  ANALISIS KOMPREHENSIF RUTE & PERHITUNGAN PENGHEMATAN")
        print("="*100)

        if not self.best_model:
            print("❌ Model belum ditraining!")
            return None

        results = []

        for i, route in enumerate(route_options, 1):
            try:
                consumption = self.predict_route_consumption(route)

                efficiency = route['jarak_km'] / consumption if consumption > 0 else 0
                fuel_cost = consumption * self.fuel_price
                time_cost = route['waktu_tempuh_menit'] * 500  # Rp 500/menit
                total_cost = fuel_cost + time_cost

                # Calculate efficiency score (weighted)
                efficiency_score = (efficiency * 40) + ((1000 - route['waktu_tempuh_menit']) * 0.3) + \
                                 ((6 - route.get('kepadatan', 5)) * 200)

                results.append({
                    'rute': f'Rute {i}',
                    'jarak_km': route['jarak_km'],
                    'waktu_menit': route['waktu_tempuh_menit'],
                    'kepadatan': route.get('kepadatan', 5),
                    'kondisi_jalan': route.get('kondisi_jalan', 3),
                    'kondisi_cuaca': route.get('kondisi_cuaca', 2),
                    'jenis_kendaraan': route.get('jenis_kendaraan', 0),
                    'berat_muatan_ton': route.get('berat_muatan_ton', 1.0),
                    'prediksi_bbm_liter': consumption,
                    'efisiensi_km_per_liter': efficiency,
                    'biaya_bbm_rp': fuel_cost,
                    'biaya_waktu_rp': time_cost,
                    'total_biaya_rp': total_cost,
                    'skor_efisiensi': efficiency_score
                })

            except Exception as e:
                print(f"❌ Error analisis Rute {i}: {e}")
                continue

        if not results:
            return None

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('skor_efisiensi', ascending=False)

        self.analysis_results = results_df

        # Save results to JSON for web interface
        results_df.to_json('static/route_analysis.json', orient='records', indent=2)

        # Display and visualize
        self.display_route_analysis(results_df)
        self.calculate_savings(results_df)
        self.create_route_visualizations(results_df)

        return results_df

    def display_route_analysis(self, results_df):
        """Menampilkan analisis rute"""
        print("\n📊 TABEL PERBANDINGAN RUTE:")
        print("=" * 120)
        
        for idx, row in results_df.iterrows():
            print(f"\n{row['rute']} {'🏆' if idx == results_df.index[0] else ''}")
            print(f"  📏 Jarak: {row['jarak_km']} km")
            print(f"  ⏱️  Waktu: {row['waktu_menit']} menit")
            print(f"  ⛽ BBM Prediksi: {row['prediksi_bbm_liter']:.1f} liter")
            print(f"  🚀 Efisiensi: {row['efisiensi_km_per_liter']:.2f} km/liter")
            print(f"  💰 Total Biaya: Rp {row['total_biaya_rp']:,.0f}")
            print(f"  📊 Skor Efisiensi: {row['skor_efisiensi']:.0f}")

    def calculate_savings(self, results_df):
        """Hitung dan tampilkan penghematan"""
        if len(results_df) < 2:
            return

        print("\n💰 ANALISIS PENGHEMATAN:")
        print("=" * 80)

        best_route = results_df.iloc[0]
        worst_route = results_df.iloc[-1]

        fuel_saving = worst_route['prediksi_bbm_liter'] - best_route['prediksi_bbm_liter']
        cost_saving = worst_route['total_biaya_rp'] - best_route['total_biaya_rp']
        time_saving = worst_route['waktu_menit'] - best_route['waktu_menit']

        print(f"\n🔍 PERBANDINGAN:")
        print(f"   🏆 Rute Terbaik: {best_route['rute']}")
        print(f"   📍 Rute Terburuk: {worst_route['rute']}")
        print()
        print(f"📈 PENGHEMATAN PER TRIP:")
        print(f"   ⛽ Bahan Bakar: {fuel_saving:.1f} liter (Rp {fuel_saving * self.fuel_price:,.0f})")
        print(f"   💰 Total Biaya: Rp {cost_saving:,.0f}")
        print(f"   ⏱️  Waktu: {time_saving:.0f} menit")

        # Proyeksi
        trips_per_day = 3
        days_per_month = 22
        months_per_year = 12
        
        daily_saving = cost_saving * trips_per_day
        monthly_saving = daily_saving * days_per_month
        yearly_saving = monthly_saving * months_per_year

        print(f"\n📅 PROYEKSI PENGHEMATAN:")
        print(f"   💵 Per Hari ({trips_per_day} trip): Rp {daily_saving:,.0f}")
        print(f"   💵 Per Bulan ({days_per_month} hari): Rp {monthly_saving:,.0f}")
        print(f"   💵 Per Tahun: Rp {yearly_saving:,.0f}")

        self.savings_data = {
            'best_route': best_route['rute'],
            'worst_route': worst_route['rute'],
            'fuel_saving': float(fuel_saving),
            'cost_saving': float(cost_saving),
            'time_saving': float(time_saving),
            'daily_saving': float(daily_saving),
            'monthly_saving': float(monthly_saving),
            'yearly_saving': float(yearly_saving)
        }
        
        # Save to JSON
        with open('static/savings_analysis.json', 'w') as f:
            json.dump(self.savings_data, f, indent=2)

    def create_route_visualizations(self, results_df):
        pass
            ax7 = fig.add_subplot(gs[2, :])
            ax7.axis('off')
            
            best = results_df.iloc[0]
            worst = results_df.iloc[-1]
            
            info_text = f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                  📊 RINGKASAN ANALISIS RUTE                                    ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                               ║
║  🏆 RUTE TERBAIK: {best['rute']:<15}                     📍 RUTE TERBURUK: {worst['rute']:<15}          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ║
║  📏 Jarak: {best['jarak_km']:<6.0f} km                              📏 Jarak: {worst['jarak_km']:<6.0f} km                    ║
║  ⏱️  Waktu: {best['waktu_menit']:<6.0f} menit                         ⏱️  Waktu: {worst['waktu_menit']:<6.0f} menit               ║
║  ⛽ BBM: {best['prediksi_bbm_liter']:<6.1f} liter                        ⛽ BBM: {worst['prediksi_bbm_liter']:<6.1f} liter              ║
║  🚀 Efisiensi: {best['efisiensi_km_per_liter']:<6.2f} km/L                    🚀 Efisiensi: {worst['efisiensi_km_per_liter']:<6.2f} km/L          ║
║  💰 Biaya: Rp {best['total_biaya_rp']:<10,.0f}                    💰 Biaya: Rp {worst['total_biaya_rp']:<10,.0f}        ║
║                                                                                               ║
║  💡 PENGHEMATAN dengan memilih rute terbaik:                                                  ║
║     • BBM: {(worst['prediksi_bbm_liter'] - best['prediksi_bbm_liter']):.1f} liter/trip                                                                    ║
║     • Biaya: Rp {(worst['total_biaya_rp'] - best['total_biaya_rp']):,.0f}/trip                                                           ║
║     • Waktu: {(worst['waktu_menit'] - best['waktu_menit']):.0f} menit/trip                                                                 ║
║                                                                                               ║
║  📈 Model Prediksi: {self.best_model_name:<25}   Akurasi (R²): {self.best_score:.4f}                    ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
            """
            
            ax7.text(0.5, 0.5, info_text, fontsize=9, fontfamily='monospace',
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
            
            plt.savefig('static/plots/route_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Visualisasi perbandingan rute berhasil dibuat")
            
        except Exception as e:
            print(f"⚠️ Error creating route visualizations: {e}")

    def save_model_info(self):
        """Save model information to JSON"""
        model_info = {
            'best_model': self.best_model_name,
            'best_score': float(self.best_score),
            'fuel_price': self.fuel_price,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for name, info in self.models.items():
            model_info['models'][name] = {
                'r2': float(info['r2']),
                'mae': float(info['mae']),
                'rmse': float(info['rmse']),
                'cv_r2': float(info['cv_r2'])
            }
        
        with open('static/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

# Main execution function
def run_analysis():
    """Menjalankan analisis lengkap"""
    print("🚀 MEMULAI ADVANCED FUEL OPTIMIZATION SYSTEM")
    print("=" * 100)
    
    # Create output directories
    os.makedirs('static/plots', exist_ok=True)
    
    # Initialize system
    system = AdvancedFuelOptimization()
    
    # Generate data
    data = system.create_realistic_sample_data(n_samples=500)
    
    # Train models
    results = system.train_advanced_models(data)
    
    if results:
        # Save model info
        system.save_model_info()
        
        # Define routes for analysis
        route_options = [
            {
                'jarak_km': 150, 'waktu_tempuh_menit': 180,
                'kepadatan_lalu_lintas': 3, 'kondisi_jalan': 4,
                'kondisi_cuaca': 2, 'jenis_kendaraan': 0, 'berat_muatan_ton': 1.2
            },
            {
                'jarak_km': 180, 'waktu_tempuh_menit': 200,
                'kepadatan_lalu_lintas': 6, 'kondisi_jalan': 3,
                'kondisi_cuaca': 1, 'jenis_kendaraan': 0, 'berat_muatan_ton': 1.5
            },
            {
                'jarak_km': 120, 'waktu_tempuh_menit': 150,
                'kepadatan_lalu_lintas': 2, 'kondisi_jalan': 5,
                'kondisi_cuaca': 2, 'jenis_kendaraan': 0, 'berat_muatan_ton': 0.8
            },
            {
                'jarak_km': 200, 'waktu_tempuh_menit': 220,
                'kepadatan_lalu_lintas': 8, 'kondisi_jalan': 2,
                'kondisi_cuaca': 0, 'jenis_kendaraan': 1, 'berat_muatan_ton': 1.8
            },
            {
                'jarak_km': 170, 'waktu_tempuh_menit': 190,
                'kepadatan_lalu_lintas': 4, 'kondisi_jalan': 4,
                'kondisi_cuaca': 1, 'jenis_kendaraan': 0, 'berat_muatan_ton': 1.0
            }
        ]
        
        # Analyze routes
        route_results = system.analyze_routes(route_options)
        
        print("\n" + "="*100)
        print("✅ ANALISIS SELESAI! Semua hasil telah disimpan di folder 'static'")
        print("="*100)
        
        return system, route_results
    
    return None, None

if __name__ == "__main__":
    system, results = run_analysis()
