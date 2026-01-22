# -*- coding: utf-8 -*-
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import os
from fuel_optimizer import AdvancedFuelOptimization
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variable for fuel system
fuel_system = None

def initialize_system():
    """Initialize the fuel optimization system"""
    global fuel_system
    
    try:
        # Try to load existing model
        if os.path.exists('model.pkl'):
            print("Loading existing model...")
            with open('model.pkl', 'rb') as f:
                fuel_system = pickle.load(f)
            print(f"✓ Model loaded: {fuel_system.best_model_name}")
        else:
            print("No existing model found. Using basic system (training disabled on server)...")
            fuel_system = AdvancedFuelOptimization()
                
    except Exception as e:
        print(f"Error initializing system: {e}")
        # Create a basic system anyway
        fuel_system = AdvancedFuelOptimization()

@app.route('/')
def index():
    """Main page - redirect to dashboard"""
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/analyze')
def analyze_page():
    """Analysis page"""
    return render_template('analyze.html')

@app.route('/tips')
def tips_page():
    """Tips page"""
    return render_template('tips.html')

@app.route('/upload')
def upload_page():
    """Upload page"""
    return render_template('upload.html')

@app.route('/api/model-info')
def get_model_info():
    """Get model information"""
    try:
        if os.path.exists('static/model_info.json'):
            with open('static/model_info.json', 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Model info not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route-analysis')
def get_route_analysis():
    """Get route analysis results"""
    try:
        if os.path.exists('static/route_analysis.json'):
            with open('static/route_analysis.json', 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Route analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/savings')
def get_savings():
    """Get savings analysis"""
    try:
        if os.path.exists('static/savings_analysis.json'):
            with open('static/savings_analysis.json', 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Savings analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fuel consumption for a custom route"""
    try:
        global fuel_system
        
        if not fuel_system or not fuel_system.best_model:
            return jsonify({'error': 'Model not initialized'}), 500
        
        data = request.json
        
        # Extract route features
        route_features = {
            'jarak_km': float(data.get('jarak_km', 100)),
            'waktu_tempuh_menit': float(data.get('waktu_tempuh_menit', 120)),
            'kepadatan_lalu_lintas': int(data.get('kepadatan_lalu_lintas', 5)),
            'kondisi_jalan': int(data.get('kondisi_jalan', 3)),
            'kondisi_cuaca': int(data.get('kondisi_cuaca', 2)),
            'jenis_kendaraan': int(data.get('jenis_kendaraan', 0)),
            'berat_muatan_ton': float(data.get('berat_muatan_ton', 1.0))
        }
        
        # Make prediction
        consumption = fuel_system.predict_route_consumption(route_features)
        
        # Calculate costs
        efficiency = route_features['jarak_km'] / consumption if consumption > 0 else 0
        fuel_cost = consumption * fuel_system.fuel_price
        time_cost = route_features['waktu_tempuh_menit'] * 500
        total_cost = fuel_cost + time_cost
        
        result = {
            'success': True,
            'prediksi_bbm_liter': round(consumption, 2),
            'efisiensi_km_per_liter': round(efficiency, 2),
            'biaya_bbm_rp': round(fuel_cost, 0),
            'biaya_waktu_rp': round(time_cost, 0),
            'total_biaya_rp': round(total_cost, 0),
            'model_used': fuel_system.best_model_name,
            'model_accuracy': round(fuel_system.best_score, 4)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-routes', methods=['POST'])
def analyze_routes():
    """Analyze multiple routes"""
    try:
        global fuel_system
        
        if not fuel_system or not fuel_system.best_model:
            return jsonify({'error': 'Model not initialized'}), 500
        
        routes = request.json.get('routes', [])
        
        if not routes:
            return jsonify({'error': 'No routes provided'}), 400
        
        # Analyze routes
        results = fuel_system.analyze_routes(routes)
        
        if results is not None:
            return jsonify({
                'success': True,
                'message': 'Routes analyzed successfully',
                'results_file': '/api/route-analysis'
            })
        else:
            return jsonify({'error': 'Analysis failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serve plot images"""
    return send_from_directory('static/plots', filename)

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain model with new data"""
    try:
        global fuel_system
        
        data_size = request.json.get('data_size', 500)
        
        print(f"🔄 Retraining model with {data_size} samples...")
        
        fuel_system = AdvancedFuelOptimization()
        data = fuel_system.create_realistic_sample_data(n_samples=data_size)
        results = fuel_system.train_advanced_models(data)
        
        if results:
            # Save model
            with open('model.pkl', 'wb') as f:
                pickle.dump(fuel_system, f)
            
            return jsonify({
                'success': True,
                'message': 'Model retrained successfully',
                'best_model': fuel_system.best_model_name,
                'best_score': round(fuel_system.best_score, 4)
            })
        else:
            return jsonify({'error': 'Training failed'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-predict', methods=['POST'])
def upload_predict():
    """Process uploaded file and predict fuel consumption"""
    try:
        global fuel_system
        
        if not fuel_system or not fuel_system.best_model:
            return jsonify({'error': 'Model not initialized'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file based on extension
        import pandas as pd
        import io
        
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Invalid file format. Use CSV or Excel'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Validate required columns
        required_cols = ['jarak_km', 'waktu_tempuh_menit', 'kepadatan_lalu_lintas', 
                        'kondisi_jalan', 'kondisi_cuaca', 'jenis_kendaraan', 'berat_muatan_ton']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400
        
        # Process each row
        results = []
        for idx, row in df.iterrows():
            try:
                route_features = {
                    'jarak_km': float(row['jarak_km']),
                    'waktu_tempuh_menit': float(row['waktu_tempuh_menit']),
                    'kepadatan_lalu_lintas': int(row['kepadatan_lalu_lintas']),
                    'kondisi_jalan': int(row['kondisi_jalan']),
                    'kondisi_cuaca': int(row['kondisi_cuaca']),
                    'jenis_kendaraan': int(row['jenis_kendaraan']),
                    'berat_muatan_ton': float(row['berat_muatan_ton'])
                }
                
                # Predict
                consumption = fuel_system.predict_route_consumption(route_features)
                
                # Calculate costs
                efficiency = route_features['jarak_km'] / consumption if consumption > 0 else 0
                fuel_cost = consumption * fuel_system.fuel_price
                time_cost = route_features['waktu_tempuh_menit'] * 500
                total_cost = fuel_cost + time_cost
                
                results.append({
                    'jarak_km': route_features['jarak_km'],
                    'waktu_menit': route_features['waktu_tempuh_menit'],
                    'prediksi_bbm_liter': round(consumption, 2),
                    'efisiensi_km_per_liter': round(efficiency, 2),
                    'biaya_bbm_rp': round(fuel_cost, 0),
                    'total_biaya_rp': round(total_cost, 0)
                })
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        if not results:
            return jsonify({'error': 'No valid data to process'}), 400
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download-template')
def download_template():
    """Download CSV template"""
    try:
        import pandas as pd
        from io import BytesIO as StringIO
        
        # Create template data
        template_data = {
            'jarak_km': [150, 200, 120],
            'waktu_tempuh_menit': [180, 220, 150],
            'kepadatan_lalu_lintas': [5, 7, 3],
            'kondisi_jalan': [4, 3, 5],
            'kondisi_cuaca': [2, 1, 2],
            'jenis_kendaraan': [0, 1, 0],
            'berat_muatan_ton': [1.2, 1.8, 0.8]
        }
        
        df = pd.DataFrame(template_data)
        
        # Create CSV in memory
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        from flask import Response
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=fuel_prediction_template.csv'}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    print("Starting Fuel Optimization Web Application...")
    print("=" * 80)
    
    # Initialize system
    initialize_system()
    
    print("\nSystem ready!")
    print("Opening web interface at http://localhost:5000")
    print("=" * 80)
    
    # Hugging Face Spaces uses port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=True, host='0.0.0.0', port=port)
