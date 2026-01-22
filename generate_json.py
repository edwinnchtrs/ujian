# Generate model_info.json from existing model
import pickle
import json
import os
import datetime

# Load the model
with open('model.pkl', 'rb') as f:
    fuel_system = pickle.load(f)

# Generate model info
model_info = {
    'best_model': fuel_system.best_model_name,
    'best_score': round(fuel_system.best_score, 4),
    'timestamp': datetime.datetime.now().isoformat(),
    'models': {}
}

for name, info in fuel_system.models.items():
    model_info['models'][name] = {
        'r2': round(info.get('r2', 0), 4),
        'mae': round(info.get('mae', 0), 2),
        'rmse': round(info.get('rmse', 0), 2),
        'cv_r2': round(info.get('cv_r2', 0), 4)
    }

# Ensure directory exists
os.makedirs('static', exist_ok=True)

# Save to JSON
with open('static/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print('Model info JSON generated!')
print(f'Best model: {model_info["best_model"]}')
print(f'R² Score: {model_info["best_score"]}')
print(f'Total models: {len(model_info["models"])}')
