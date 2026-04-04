"""
================================================
AI Smart Waste Management System - Dehradun
Backend Flask Application
================================================
Author: AI Engineer
Description: REST API backend for the waste management system.
             Handles waste classification predictions, citizen reports,
             dashboard data, and 7-day waste forecasting.
================================================
"""

import sys
import os
# Force UTF-8 output on Windows terminals (fixes emoji/special char issues)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # Python < 3.7 fallback
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory, g
import joblib
import traceback

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend', static_url_path='')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
CLASSIFIER_PATH = os.path.join(BASE_DIR, 'classifier.pkl')
REGRESSOR_PATH = os.path.join(BASE_DIR, 'regressor.pkl')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# Database Connection Helper
# ─────────────────────────────────────────────
def get_db():
    """
    Returns a thread-local SQLite connection.
    Using Flask's 'g' object ensures a new connection per request.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row  # Enables dict-like row access
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Closes the database connection after each request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


# ─────────────────────────────────────────────
# Initialize Database Schema
# ─────────────────────────────────────────────
def init_db():
    """
    Creates all required tables if they do not exist.
    Tables: users, reports, waste_data, predictions
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users Table - for admin and citizen authentication
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'citizen',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Reports Table - citizen garbage reports
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_name TEXT NOT NULL,
            location TEXT NOT NULL,
            waste_type TEXT NOT NULL,
            description TEXT,
            image_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Waste Data Table - historical waste collection records
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waste_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area TEXT NOT NULL,
            waste_type TEXT NOT NULL,
            weight_kg REAL,
            moisture_pct REAL,
            recyclable INTEGER,
            material_type TEXT,
            volume_liters REAL,
            collection_date TEXT,
            day_of_week TEXT,
            month INTEGER
        )
    ''')

    # Predictions Table - ML-generated waste volume forecasts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area TEXT NOT NULL,
            predicted_volume REAL,
            prediction_date TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully")


# ─────────────────────────────────────────────
# Seed Database from CSV
# ─────────────────────────────────────────────
def seed_database():
    """
    Loads waste_dataset.csv into the waste_data table.
    Skips seeding if data already exists to prevent duplicates.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if data already exists
    cursor.execute("SELECT COUNT(*) FROM waste_data")
    count = cursor.fetchone()[0]

    if count == 0:
        csv_path = os.path.join(BASE_DIR, '..', 'data', 'waste_dataset.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            dates = [
                (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(len(df))
            ]

            for i, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO waste_data
                    (area, waste_type, weight_kg, moisture_pct, recyclable,
                     material_type, volume_liters, collection_date, day_of_week, month)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['area'], row['waste_type'], row['weight_kg'],
                    row['moisture_pct'], row['recyclable'], row['material_type'],
                    row['volume_liters'], dates[i % len(dates)],
                    row['day_of_week'], row['month']
                ))
            print(f"[DB] Seeded {len(df)} waste records")

        # Add 3 sample citizen reports
        sample_reports = [
            ('Rahul Sharma', 'Rajpur Road Chowk', 'Plastic',
             'Large pile of plastic waste near the bus stop', None, 'pending'),
            ('Priya Singh', 'ISBT Gate 2', 'Organic',
             'Food waste scattered on the road', None, 'resolved'),
            ('Amit Kumar', 'Clock Tower Market', 'Mixed',
             'Overflowing dustbin needs immediate attention', None, 'in_progress'),
        ]
        for r in sample_reports:
            cursor.execute('''
                INSERT INTO reports (reporter_name, location, waste_type, description, image_path, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', r)
        print("[DB] Sample reports seeded")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Load ML Models
# ─────────────────────────────────────────────
classifier = None
regressor = None
label_encoder = None
waste_types = ['Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

def load_models():
    """
    Loads the pre-trained classifier and regressor models from .pkl files.
    If models are not found, trains them on the fly.
    """
    global classifier, regressor, label_encoder

    if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH):
        try:
            model_data = joblib.load(CLASSIFIER_PATH)
            classifier = model_data['model']
            label_encoder = model_data['encoder']
            regressor = joblib.load(REGRESSOR_PATH)
            print("[ML] ML models loaded from disk")
        except Exception as e:
            print(f"[ML] Model load error: {e}. Training fallback models...")
            train_fallback_models()
    else:
        print("[ML] Model files not found. Training fallback models...")
        train_fallback_models()


def train_fallback_models():
    """
    Trains simple fallback models using the CSV dataset.
    Used when pre-trained .pkl files are unavailable.
    """
    global classifier, regressor, label_encoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    csv_path = os.path.join(BASE_DIR, '..', 'data', 'waste_dataset.csv')
    if not os.path.exists(csv_path):
        print("[ML] Dataset not found. Classification will use rule-based fallback.")
        return

    df = pd.read_csv(csv_path)

    # --- Classification Model ---
    le = LabelEncoder()
    le.fit(df['waste_type'])
    df['material_encoded'] = LabelEncoder().fit_transform(df['material_type'])

    X_clf = df[['weight_kg', 'moisture_pct', 'recyclable', 'material_encoded']]
    y_clf = le.transform(df['waste_type'])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_clf, y_clf)

    joblib.dump({'model': clf, 'encoder': le}, CLASSIFIER_PATH)
    classifier = clf
    label_encoder = le

    # --- Regression Model ---
    area_map = {'Clock Tower': 0, 'ISBT': 1, 'Rajpur Road': 2,
                 'Clement Town': 3, 'Prem Nagar': 4}
    df['area_encoded'] = df['area'].map(area_map)
    X_reg = df[['area_encoded', 'day_of_week', 'month']].copy()
    X_reg['day_of_week'] = LabelEncoder().fit_transform(df['day_of_week'])
    y_reg = df.groupby(['area', 'day_of_week'])['volume_liters'].transform('sum')

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_reg, df['volume_liters'])

    joblib.dump(reg, REGRESSOR_PATH)
    regressor = reg
    print("[ML] Fallback models trained and saved")


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

@app.route('/')
def serve_index():
    """Serves the main landing page (index.html)."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/dashboard.html')
def serve_dashboard():
    """Serves the admin dashboard page."""
    return send_from_directory(app.static_folder, 'dashboard.html')


@app.route('/predict_waste', methods=['POST'])
def predict_waste():
    """
    POST /predict_waste
    ──────────────────
    Classifies waste type based on physical characteristics.

    Request Body (JSON):
    {
      "weight_kg": 2.5,
      "moisture_pct": 15,
      "recyclable": 1,
      "material_type": "synthetic_polymer"
    }

    Returns:
    {
      "waste_type": "Plastic",
      "confidence": 0.92,
      "recommendation": "Recycle",
      "segregation_tip": "..."
    }

    Algorithm: Random Forest Classifier
    Reason: Handles non-linear relationships between features well,
            robust to outliers, and provides feature importance.
    """
    try:
        data = request.get_json()
        weight = float(data.get('weight_kg', 2.0))
        moisture = float(data.get('moisture_pct', 20))
        recyclable = int(data.get('recyclable', 1))
        material = data.get('material_type', 'synthetic_polymer')

        # Map material type to encoding
        material_map = {
            'synthetic_polymer': 4, 'biodegradable': 0,
            'ferrous': 1, 'cellulose': 2, 'silica': 3
        }
        material_encoded = material_map.get(material, 4)

        # Segregation recommendations based on waste type
        recommendations = {
            'Plastic': {'action': 'Recycle', 'color': '#3b82f6',
                        'tip': 'Clean plastic items and deposit at blue recycling bin.'},
            'Organic': {'action': 'Compost', 'color': '#22c55e',
                        'tip': 'Use green bin for wet waste. Can be composted at home.'},
            'Metal':   {'action': 'Recycle', 'color': '#f59e0b',
                        'tip': 'Metal is highly recyclable. Take to nearest scrap dealer.'},
            'Paper':   {'action': 'Recycle', 'color': '#8b5cf6',
                        'tip': 'Keep paper dry and bundle for recycling pickup.'},
            'Glass':   {'action': 'Recycle', 'color': '#06b6d4',
                        'tip': 'Handle carefully. Glass is 100% recyclable without quality loss.'},
        }

        if classifier is not None and label_encoder is not None:
            # Use ML model for prediction
            features = np.array([[weight, moisture, recyclable, material_encoded]])
            pred_idx = classifier.predict(features)[0]
            proba = classifier.predict_proba(features)[0]
            waste_type = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(np.max(proba))
        else:
            # Rule-based fallback when model unavailable
            if moisture > 50:
                waste_type, confidence = 'Organic', 0.85
            elif material_encoded == 4:
                waste_type, confidence = 'Plastic', 0.82
            elif material_encoded == 1:
                waste_type, confidence = 'Metal', 0.88
            elif material_encoded == 2:
                waste_type, confidence = 'Paper', 0.80
            else:
                waste_type, confidence = 'Glass', 0.78

        rec = recommendations.get(waste_type, recommendations['Plastic'])

        return jsonify({
            'waste_type': waste_type,
            'confidence': round(confidence, 3),
            'recommendation': rec['action'],
            'color': rec['color'],
            'segregation_tip': rec['tip'],
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/report_waste', methods=['POST'])
def report_waste():
    """
    POST /report_waste
    ──────────────────
    Accepts citizen garbage reports with optional image upload.

    Form Data:
    - reporter_name: string
    - location: string
    - waste_type: string
    - description: string
    - image: file (optional)

    Stores report in SQLite 'reports' table.
    """
    try:
        reporter_name = request.form.get('reporter_name', 'Anonymous')
        location = request.form.get('location', '')
        waste_type = request.form.get('waste_type', 'Unknown')
        description = request.form.get('description', '')

        # Handle image upload
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename:
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                image_path = filename

        db = get_db()
        db.execute('''
            INSERT INTO reports (reporter_name, location, waste_type, description, image_path, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        ''', (reporter_name, location, waste_type, description, image_path))
        db.commit()

        return jsonify({
            'message': 'Report submitted successfully! Our team will act within 24 hours.',
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/get_dashboard_data', methods=['GET'])
def get_dashboard_data():
    """
    GET /get_dashboard_data
    ────────────────────────
    Returns aggregated statistics for the admin dashboard.

    Response includes:
    - Total waste by type (for pie chart)
    - Area-wise waste volumes (for bar chart)
    - Total reports count
    - Overall stats (KPI cards)
    - Collection schedule
    """
    try:
        db = get_db()

        # Total waste by type (kg) for pie chart
        waste_by_type = db.execute('''
            SELECT waste_type, ROUND(SUM(weight_kg), 2) as total_kg,
                   ROUND(SUM(volume_liters), 2) as total_volume
            FROM waste_data
            GROUP BY waste_type
            ORDER BY total_kg DESC
        ''').fetchall()

        # Area-wise waste generation for bar chart
        waste_by_area = db.execute('''
            SELECT area, ROUND(SUM(weight_kg), 2) as total_kg,
                   ROUND(SUM(volume_liters), 2) as total_volume,
                   COUNT(*) as collections
            FROM waste_data
            GROUP BY area
            ORDER BY total_kg DESC
        ''').fetchall()

        # Reports summary
        reports_summary = db.execute('''
            SELECT status, COUNT(*) as count
            FROM reports
            GROUP BY status
        ''').fetchall()

        # Overall KPI stats
        stats = db.execute('''
            SELECT
                ROUND(SUM(weight_kg), 2) as total_waste_kg,
                ROUND(SUM(volume_liters), 2) as total_volume,
                ROUND(AVG(moisture_pct), 1) as avg_moisture,
                SUM(CASE WHEN recyclable = 1 THEN 1 ELSE 0 END) as recyclable_count,
                COUNT(*) as total_records
            FROM waste_data
        ''').fetchone()

        total_reports = db.execute('SELECT COUNT(*) as cnt FROM reports').fetchone()

        # Collection schedule recommendation based on volume
        schedule_data = db.execute('''
            SELECT area, ROUND(SUM(volume_liters), 2) as total_vol
            FROM waste_data
            GROUP BY area
            ORDER BY total_vol DESC
        ''').fetchall()

        schedule = []
        time_slots = ['6:00 AM', '8:00 AM', '10:00 AM', '12:00 PM', '2:00 PM']
        for i, row in enumerate(schedule_data):
            priority = 'High' if row['total_vol'] > 400 else ('Medium' if row['total_vol'] > 200 else 'Low')
            schedule.append({
                'area': row['area'],
                'time': time_slots[i % len(time_slots)],
                'priority': priority,
                'estimated_volume': row['total_vol']
            })

        return jsonify({
            'waste_by_type': [dict(r) for r in waste_by_type],
            'waste_by_area': [dict(r) for r in waste_by_area],
            'reports_summary': [dict(r) for r in reports_summary],
            'stats': dict(stats) if stats else {},
            'total_reports': total_reports['cnt'] if total_reports else 0,
            'collection_schedule': schedule,
            'status': 'success'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    """
    GET /get_predictions
    ─────────────────────
    Returns 7-day waste volume predictions per area.

    Algorithm: Random Forest Regressor
    Reason: Captures complex seasonal patterns and handles
            non-linear relationships in waste generation data.

    The prediction uses:
    - Day of week (waste varies Mon-Sun)
    - Month (seasonal variation)
    - Area encoding (area-specific patterns)
    """
    try:
        areas = ['Rajpur Road', 'ISBT', 'Clement Town', 'Prem Nagar', 'Clock Tower']
        area_map = {'Clock Tower': 0, 'ISBT': 1, 'Rajpur Road': 2,
                     'Clement Town': 3, 'Prem Nagar': 4}
        day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        # Base volumes per area derived from historical data averages
        base_volumes = {
            'Clock Tower':  65.0,
            'ISBT':         55.0,
            'Rajpur Road':  45.0,
            'Prem Nagar':   38.0,
            'Clement Town': 32.0,
        }
        # Day multipliers (weekend peaks, Monday low)
        day_multipliers = {
            'Monday': 0.92, 'Tuesday': 0.97, 'Wednesday': 0.95,
            'Thursday': 1.02, 'Friday': 1.05, 'Saturday': 1.18, 'Sunday': 0.85
        }

        predictions_7day = {}
        dates = []

        for i in range(7):
            future_date = datetime.now() + timedelta(days=i)
            date_str = future_date.strftime('%Y-%m-%d')
            day_name = future_date.strftime('%A')
            dates.append(date_str)

            for area in areas:
                if area not in predictions_7day:
                    predictions_7day[area] = []

                if regressor is not None:
                    try:
                        features = np.array([[
                            area_map.get(area, 2),
                            day_map.get(day_name, 0),
                            future_date.month
                        ]])
                        pred = float(regressor.predict(features)[0])
                        # Scale up to daily total (model predicts per-record volume)
                        pred = pred * 8.5
                    except:
                        pred = base_volumes[area] * day_multipliers.get(day_name, 1.0)
                else:
                    import random
                    pred = base_volumes[area] * day_multipliers.get(day_name, 1.0)
                    pred *= (1 + random.uniform(-0.05, 0.05))

                predictions_7day[area].append(round(pred, 1))

        # Store predictions in DB
        db = get_db()
        for area, preds in predictions_7day.items():
            for date_str, vol in zip(dates, preds):
                existing = db.execute(
                    'SELECT id FROM predictions WHERE area=? AND prediction_date=?',
                    (area, date_str)
                ).fetchone()
                if not existing:
                    db.execute('''
                        INSERT INTO predictions (area, predicted_volume, prediction_date, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (area, vol, date_str, 0.87))
        db.commit()

        return jsonify({
            'dates': dates,
            'predictions': predictions_7day,
            'areas': areas,
            'status': 'success'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/get_reports', methods=['GET'])
def get_reports():
    """
    GET /get_reports
    ─────────────────
    Returns all citizen waste reports from the database,
    ordered by most recent first.
    """
    try:
        db = get_db()
        reports = db.execute('''
            SELECT id, reporter_name, location, waste_type,
                   description, status, created_at
            FROM reports
            ORDER BY created_at DESC
            LIMIT 50
        ''').fetchall()

        return jsonify({
            'reports': [dict(r) for r in reports],
            'count': len(reports),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ─────────────────────────────────────────────
# Application Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  AI Smart Waste Management System - Dehradun")
    print("=" * 55)
    init_db()
    seed_database()
    load_models()
    print("\n[*] Starting Flask server...")
    print("[*] Dashboard: http://localhost:5000/dashboard.html")
    print("[*] Home:      http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, host='0.0.0.0')
