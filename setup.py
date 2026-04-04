"""
setup.py — One-click setup for AI Smart Waste Management System
===============================================================
Run: python setup.py
This script:
  1. Installs all dependencies
  2. Generates the Jupyter notebook
  3. Trains ML models
  4. Initializes the SQLite database
  5. Starts the Flask server
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))

def run(cmd, cwd=None):
    print(f"\n▶ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or BASE)
    return result.returncode

def banner(msg):
    print(f"\n{'='*55}")
    print(f"  {msg}")
    print(f"{'='*55}")

if __name__ == '__main__':
    banner("🌿 AI Smart Waste Management System — Setup")

    # Step 1: Install dependencies
    banner("Step 1/4 · Installing dependencies")
    code = run(f"{sys.executable} -m pip install -r requirements.txt -q")
    if code != 0:
        print("❌ pip install failed. Check requirements.txt")
        sys.exit(1)
    print("✅ Dependencies installed")

    # Step 2: Generate Jupyter notebook
    banner("Step 2/4 · Generating Jupyter notebook")
    code = run(f"{sys.executable} generate_notebook.py")
    if code == 0:
        print("✅ Notebook ready at notebooks/waste_ml_model.ipynb")
    else:
        print("⚠️  Notebook generation failed (non-critical, continuing)")

    # Step 3: Train ML models
    banner("Step 3/4 · Training ML models")
    train_script = """
import sys, os
sys.path.insert(0, os.path.join('backend'))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

csv_path = os.path.join('data', 'waste_dataset.csv')
df = pd.read_csv(csv_path)

# Encode
le_waste    = LabelEncoder()
le_material = LabelEncoder()
le_area     = LabelEncoder()

df['waste_type_enc']    = le_waste.fit_transform(df['waste_type'])
df['material_type_enc'] = le_material.fit_transform(df['material_type'])
df['area_enc']          = le_area.fit_transform(df['area'])
df['day_enc']           = LabelEncoder().fit_transform(df['day_of_week'])

# Classification
X_clf = df[['weight_kg','moisture_pct','recyclable','material_type_enc']]
y_clf = df['waste_type_enc']
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_clf, y_clf)

os.makedirs('backend', exist_ok=True)
joblib.dump({'model': clf, 'encoder': le_waste}, 'backend/classifier.pkl', compress=3)
print('  ✅ Classifier trained & saved (backend/classifier.pkl)')

# Regression
X_reg = df[['area_enc','day_enc','month']]
y_reg = df['volume_liters']
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg.fit(X_reg, y_reg)
joblib.dump(reg, 'backend/regressor.pkl', compress=3)
print('  ✅ Regressor trained & saved  (backend/regressor.pkl)')
"""
    with open('_train_models.py', 'w') as f:
        f.write(train_script)
    code = run(f"{sys.executable} _train_models.py")
    os.remove('_train_models.py')
    if code != 0:
        print("⚠️  Model training failed — server will use fallback training on startup")
    else:
        print("✅ ML models ready")

    # Step 4: Start Flask server
    banner("Step 4/4 · Starting Flask Server")
    print("📊 Dashboard: http://localhost:5000/dashboard.html")
    print("🏠 Home:      http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    run(f"{sys.executable} backend/app.py")
