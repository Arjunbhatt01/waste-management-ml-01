# 🌿 AI Smart Waste Management System — Dehradun

> **Hackathon-level** full-stack project combining Machine Learning, Flask REST API, SQLite database, and a premium dark-themed dashboard with real-time Chart.js visualizations and Leaflet map.

---

## 📸 Project Overview

| Feature | Details |
|---------|---------|
| **ML Models** | Random Forest, Decision Tree, Logistic Regression |
| **Backend** | Python Flask REST API |
| **Database** | SQLite (4 tables) |
| **Frontend** | HTML + CSS + JS + Bootstrap 5 |
| **Charts** | Chart.js (Pie, Bar, Line, Doughnut) |
| **Map** | Leaflet.js interactive Dehradun map |
| **Notebook** | Jupyter (10 figures, EDA, model comparison) |

---

## 🗂️ Folder Structure

```
project/
├── data/
│   └── waste_dataset.csv          ← 200-record synthetic dataset
│
├── notebooks/
│   └── waste_ml_model.ipynb       ← Full ML notebook (auto-generated)
│
├── backend/
│   ├── app.py                     ← Flask server + all API endpoints
│   ├── classifier.pkl             ← Trained RF classifier (auto-generated)
│   ├── regressor.pkl              ← Trained RF regressor (auto-generated)
│   ├── database.db                ← SQLite DB (auto-generated)
│   └── uploads/                   ← Citizen report images
│
├── frontend/
│   ├── index.html                 ← Landing page
│   ├── dashboard.html             ← Admin dashboard (7 tabs)
│   ├── style.css                  ← Premium dark theme
│   └── script.js                  ← API calls + Chart.js + Leaflet
│
├── generate_notebook.py           ← Generates the .ipynb file
├── setup.py                       ← One-click setup & run
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Option A — One-Click Setup (Recommended)
```bash
python setup.py
```
This automatically:
1. Installs all pip dependencies
2. Generates the Jupyter notebook
3. Trains and saves ML models
4. Initialises SQLite database
5. Seeds sample data
6. Starts the Flask server

### Option B — Manual Setup

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Generate Jupyter notebook**
```bash
python generate_notebook.py
```

**Step 3 — Run the Jupyter notebook** *(optional but recommended)*
```bash
jupyter notebook notebooks/waste_ml_model.ipynb
```
Run all cells → models get trained and saved to `backend/`

**Step 4 — Start Flask server**
```bash
cd backend
python app.py
```

**Step 5 — Open in browser**
- 🏠 Home page:  http://localhost:5000
- 📊 Dashboard:  http://localhost:5000/dashboard.html

---

## 🤖 Machine Learning

### Classification Task
**Goal:** Classify waste into 5 types from physical measurements.

| Feature | Description |
|---------|-------------|
| `weight_kg` | Weight of the waste sample |
| `moisture_pct` | Moisture content (%) — organic waste is high |
| `recyclable` | Binary flag (1 = recyclable) |
| `material_type` | Material category (synthetic_polymer, biodegradable, etc.) |

**Target Classes:** `Plastic` · `Organic` · `Metal` · `Paper` · `Glass`

| Algorithm | Why Used | Accuracy |
|-----------|----------|----------|
| **Random Forest** ⭐ | Ensemble, non-linear, feature importance | ~95%+ |
| Decision Tree | Interpretable baseline | ~90-95% |
| Logistic Regression | Linear baseline | ~85-90% |

**Winner: Random Forest** — Aggregates 100 trees. Best for non-linear waste feature relationships.

### Regression Task
**Goal:** Predict daily waste volume (liters) per area for the next 7 days.

| Feature | Description |
|---------|-------------|
| `area_enc` | Encoded area identifier |
| `day_enc` | Day of week (weekends = peak waste) |
| `month` | Month (seasonal variation) |

**Algorithm:** Random Forest Regressor → High R² score, captures area-specific seasonal patterns.

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the home page |
| `GET` | `/dashboard.html` | Serves the admin dashboard |
| `POST` | `/predict_waste` | ML waste type classification |
| `POST` | `/report_waste` | Submit citizen garbage report |
| `GET` | `/get_dashboard_data` | All stats for charts (waste by type, area, schedule) |
| `GET` | `/get_predictions` | 7-day volume forecast per area |
| `GET` | `/get_reports` | Get all citizen reports |

### Example: Predict Waste Type
```bash
curl -X POST http://localhost:5000/predict_waste \
  -H "Content-Type: application/json" \
  -d '{"weight_kg": 2.5, "moisture_pct": 15, "recyclable": 1, "material_type": "synthetic_polymer"}'
```

**Response:**
```json
{
  "waste_type": "Plastic",
  "confidence": 0.923,
  "recommendation": "Recycle",
  "segregation_tip": "Clean plastic items and deposit at blue recycling bin.",
  "status": "success"
}
```

### Example: Submit Report
```bash
curl -X POST http://localhost:5000/report_waste \
  -F "reporter_name=Rahul Sharma" \
  -F "location=Rajpur Road Chowk" \
  -F "waste_type=Plastic" \
  -F "description=Large pile near bus stop"
```

---

## 🗄️ Database Schema

### `waste_data`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| area | TEXT | Dehradun area name |
| waste_type | TEXT | Plastic/Organic/Metal/Paper/Glass |
| weight_kg | REAL | Weight in kg |
| moisture_pct | REAL | Moisture percentage |
| recyclable | INTEGER | 0 or 1 |
| volume_liters | REAL | Volume in liters |
| collection_date | TEXT | Date of collection |

### `reports`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| reporter_name | TEXT | Citizen name |
| location | TEXT | Garbage location |
| waste_type | TEXT | Reported waste type |
| description | TEXT | Detailed description |
| image_path | TEXT | Uploaded image filename |
| status | TEXT | pending / resolved / in_progress |

### `predictions`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| area | TEXT | Dehradun area |
| predicted_volume | REAL | ML-predicted liters |
| prediction_date | TEXT | Forecast date |
| confidence | REAL | Model confidence |

### `users`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| name | TEXT | User full name |
| email | TEXT UNIQUE | Email address |
| role | TEXT | citizen / admin |

---

## 📊 Dashboard Features

| Tab | Content |
|-----|---------|
| **Overview** | 5 KPI cards + Pie + Bar + Schedule + Reports donut |
| **Analytics** | Volume bar + Recyclable ratio + Waste×Area heatmap |
| **Predictions** | 7-day forecast line chart + per-area predictions table |
| **Collection Schedule** | ML-optimised pickup schedule with priority |
| **Citizen Reports** | Full reports table with status badges |
| **Waste Map** | Interactive Leaflet map of all 5 Dehradun zones |
| **AI Classifier** | Live waste classification tool with model info |

---

## 📍 Coverage Areas (Dehradun)

| Area | Coordinates | Waste Level |
|------|-------------|-------------|
| Clock Tower | 30.3247°N, 78.0413°E | 🔴 High |
| ISBT | 30.2921°N, 78.0492°E | 🔴 High |
| Rajpur Road | 30.3559°N, 78.0637°E | 🟡 Medium |
| Prem Nagar | 30.2875°N, 78.0161°E | 🟡 Medium |
| Clement Town | 30.2701°N, 78.0213°E | 🟢 Low |

---

## ♻️ Segregation Recommendations

| Waste Type | Action | Bin Color | Tip |
|-----------|--------|-----------|-----|
| Plastic | ♻️ Recycle | 🔵 Blue | Clean before disposal |
| Organic | 🌱 Compost | 🟢 Green | Home composting ideal |
| Metal | ♻️ Recycle | 🟡 Yellow | Scrap dealer or yellow bin |
| Paper | ♻️ Recycle | 🟣 Purple | Keep dry, bundle for pickup |
| Glass | ♻️ Recycle | 🔷 Teal | Handle with care, 100% recyclable |

---

## 📓 Jupyter Notebook Sections

1. **Setup & Imports** — Libraries, plotting theme
2. **Data Loading** — CSV → DataFrame
3. **EDA** — 5 visualization figures (distributions, correlations, grouped bars)
4. **Feature Engineering** — Label encoding, train/test split
5. **Classification** — Train 3 models, compare accuracy, confusion matrix
6. **Feature Importance** — Random Forest insight graph
7. **Regression** — Volume prediction, actual vs predicted, residuals
8. **7-Day Forecast** — Multi-line forecast chart per area
9. **Save Models** — joblib serialization to `backend/`

---

## 🛠️ Tech Stack

```
Machine Learning    │ scikit-learn, pandas, numpy, joblib
Visualization (ML)  │ matplotlib, seaborn
Backend             │ Python Flask 3.0
Database            │ SQLite (built-in)
Frontend            │ HTML5, CSS3, JavaScript (ES6+)
UI Framework        │ Bootstrap 5.3
Charts (Frontend)   │ Chart.js 4.4
Map                 │ Leaflet.js 1.9
Icons               │ Font Awesome 6.4
Fonts               │ Google Fonts (Inter, Space Grotesk)
```

---

## 🏆 Interview / Hackathon Highlights

- **End-to-End ML Pipeline** — data → EDA → training → evaluation → deployment
- **REST API Design** — clean endpoints, proper HTTP methods, JSON responses
- **Real Database** — SQLite with 4 normalised tables, no data loss on restart
- **Responsive UI** — works on mobile, tablet, desktop
- **Model Comparison** — 3 algorithms benchmarked with cross-validation
- **Production-Ready Code** — modular, commented, error-handled throughout

---

## 📝 Code Quality

- Every function has a docstring explaining **what**, **why**, and **how**
- Algorithm choices justified with comments
- Error handling on all API routes (try/except + meaningful messages)
- Fallback model training if `.pkl` files are missing
- Frontend shows loading skeletons while fetching data

---

*Built for Dehradun's Smart City initiative · AI-powered waste management*
