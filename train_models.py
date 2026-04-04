"""
train_models.py - Train and save ML models to backend/
Run this once before starting the Flask server.
"""
import os, sys

# Add project root to path
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

print("=" * 50)
print("  AI Waste Management - Model Training")
print("=" * 50)

# ── Load dataset ──────────────────────────────────
csv_path = os.path.join(BASE, 'data', 'waste_dataset.csv')
df = pd.read_csv(csv_path)
print(f"\n[1/4] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── Encode categorical columns ─────────────────────
le_waste    = LabelEncoder()
le_material = LabelEncoder()
le_area     = LabelEncoder()
le_day      = LabelEncoder()

df['waste_type_enc']    = le_waste.fit_transform(df['waste_type'])
df['material_type_enc'] = le_material.fit_transform(df['material_type'])
df['area_enc']          = le_area.fit_transform(df['area'])
df['day_enc']           = le_day.fit_transform(df['day_of_week'])

print("[2/4] Feature encoding complete")
print(f"      Waste classes : {list(le_waste.classes_)}")
print(f"      Areas         : {list(le_area.classes_)}")

# ── Classification (Random Forest) ────────────────
FEATURES_CLF = ['weight_kg', 'moisture_pct', 'recyclable', 'material_type_enc']
TARGET_CLF   = 'waste_type_enc'

X_clf = df[FEATURES_CLF]
y_clf = df[TARGET_CLF]

X_tr, X_te, y_tr, y_te = train_test_split(X_clf, y_clf,
                                            test_size=0.2,
                                            random_state=42,
                                            stratify=y_clf)

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))

print(f"\n[3/4] Classifier trained")
print(f"      Algorithm : Random Forest (100 trees)")
print(f"      Test Acc  : {acc*100:.1f}%")

os.makedirs('backend', exist_ok=True)
clf_path = os.path.join(BASE, 'backend', 'classifier.pkl')
joblib.dump({'model': clf, 'encoder': le_waste}, clf_path, compress=3)
print(f"      Saved     -> backend/classifier.pkl ({os.path.getsize(clf_path)//1024} KB)")

# ── Regression (Random Forest Regressor) ──────────
FEATURES_REG = ['area_enc', 'day_enc', 'month']
TARGET_REG   = 'volume_liters'

X_reg = df[FEATURES_REG]
y_reg = df[TARGET_REG]

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg,
                                                     test_size=0.2,
                                                     random_state=42)

reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
reg.fit(X_tr_r, y_tr_r)

y_pred_r = reg.predict(X_te_r)
rmse = mean_squared_error(y_te_r, y_pred_r, squared=False) if hasattr(mean_squared_error, '_called') else \
       float(np.sqrt(mean_squared_error(y_te_r, y_pred_r)))
r2   = r2_score(y_te_r, y_pred_r)

print(f"\n[4/4] Regressor trained")
print(f"      Algorithm : Random Forest Regressor (100 trees)")
print(f"      RMSE      : {rmse:.2f} liters")
print(f"      R2 Score  : {r2:.4f}")

reg_path = os.path.join(BASE, 'backend', 'regressor.pkl')
joblib.dump(reg, reg_path, compress=3)
print(f"      Saved     -> backend/regressor.pkl ({os.path.getsize(reg_path)//1024} KB)")

print("\n" + "=" * 50)
print("  All models saved. Ready to start Flask!")
print("  Next: python backend/app.py")
print("=" * 50)
