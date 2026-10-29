# Delhi NCR Fog Prediction & Monitoring System

> **Regional Early Warning Network for Delhi National Capital Region (NCR)**  
> **Deployment Portal:** `delhifog.org`

---

## 🌟 Executive Summary

The **Delhi NCR Fog Prediction System** is a real-time, multi-modal machine learning platform designed to predict fog and winter smog onset and intensity across the **National Capital Region (Delhi, Noida, Gurugram, and Ghaziabad)**.

The system addresses both:
1. **Binary Classification**: Real-time detection of **Fog** vs. **No-Fog**.
2. **Continuous Regression**: Quantitative estimation of **Fog Index (0–100)** and Runway Visual Range (RVR) / atmospheric visibility in meters.

---

## 🏗️ Architecture & Core Components

```
                           +-----------------------------------------------+
                           |          Data Ingestion & Sensors             |
                           +-----------------------------------------------+
                             /                     |                     \
                            /                      |                      \
                           v                       v                       v
               [Delhi Meteorological Grid]   [Delhi NCR Traffic & CCTV]  [INSAT-3D Satellite]
               Safdarjung, IGI Palam,        Key nodes: IGI Runway,      Multi-spectral channel
               Anand Vihar, Akshardham,      Anand Vihar, India Gate,    Delhi & Indo-Gangetic
               PM2.5/PM10, Temp, RH, Wind    Yamuna belt, Dwarka Expy    fog corridor
                           |                       |                       |
                           |                       +-----------+-----------+
                           |                                   |
                           v                                   v
             +---------------------------+           +---------------------------+
             | Classical ML & Time-Series|           | Computer Vision & Deep ML |
             |---------------------------|           |---------------------------|
             | • Random Forest (Cls/Reg) |           | • Dark Channel Prior (DCP)|
             | • Support Vector Machine  |           | • Atmospheric Light & t(x)|
             | • K-Nearest Neighbors     |           | • Multi-task FogCNN       |
             | • PyTorch 2-Layer LSTM    |           |   (Cls + Fog Index Head)  |
             +---------------------------+           +---------------------------+
                           \                                   /
                            \                                 /
                             v                               v
                           +-----------------------------------+
                           | Multi-Modal Fusion Advisory Engine |
                           +-----------------------------------+
                                             |
                                             v
                           +-----------------------------------+
                           |   Live Portal: delhifog.org UI     |
                           |   & High-Throughput FastAPI Server|
                           +-----------------------------------+
```

### 1. Classical Sensor Models (`src/sensor_models.py`)
- **Random Forest**: Ensemble decision trees with feature importance for dew point depression, relative humidity, and aerosol concentration.
- **Support Vector Machine (SVM)**: Non-linear RBF kernel for optimal boundary separation in normalized atmospheric state space.
- **K-Nearest Neighbors (KNN)**: Non-parametric local density estimator reflecting regional micro-climate clusters.

### 2. Time-Series Forecasting (`src/lstm_model.py`)
- Exploit **PyTorch 2-layer LSTM** neural networks to capture diurnal cycles and temporal momentum over past 24-hour meteorological sequences.
- Multi-head output predicting immediate next-hour fog status and a 24-hour horizon fog index trajectory curve.

### 3. Satellite & CCTV Image Pipeline (`src/image_processing.py` & `src/cnn_model.py`)
- **Dark Channel Prior (DCP)**: Physics-based optical haze analysis ($I(x) = J(x)t(x) + A(1-t(x))$) to calculate atmospheric transmission maps.
- **FogCNN**: Deep convolutional architecture (32 $\rightarrow$ 64 $\rightarrow$ 128 channels) with shared representations and dual classification/regression heads.
- Evaluated on image streams from key strategic transportation nodes:
  - *Indira Gandhi International Airport (T3 / Runway 28)*
  - *Anand Vihar ISBT & Railway Hub*
  - *India Gate & Central Vista Observatory*
  - *Akshardham & Yamuna River Floodplain Corridor*
  - *Dwarka Expressway Junction (Sector 8)*

---

## 📁 Repository Structure

```
fog-prediction-system/
├── app/
│   └── dashboard.py               # Streamlit live web portal (delhifog.org)
├── data/
│   ├── raw/                       # Telemetry data, CCTV frames & satellite tiles
│   └── sample/                    # Sample camera frames for quick testing
├── models/
│   ├── sensor_models.joblib       # Saved bundle (RF, SVM, KNN, Scalers)
│   ├── lstm_fog_model.pt          # PyTorch LSTM checkpoint
│   ├── cnn_fog_model.pt           # PyTorch FogCNN checkpoint
│   └── benchmark_metrics.json     # Quantitative validation metrics
├── src/
│   ├── api.py                     # FastAPI REST API endpoints
│   ├── cnn_model.py               # FogCNN architecture & training
│   ├── data_generator.py          # Delhi winter weather & smog generator
│   ├── image_processing.py        # Dark Channel Prior & transmission maps
│   ├── lstm_model.py              # Sequence LSTM architecture & forecaster
│   ├── predictor.py               # Unified multi-modal fusion pipeline
│   ├── sensor_models.py           # Random Forest, SVM, KNN models
│   └── train_all.py               # End-to-end training orchestrator
├── tests/
│   └── test_models.py             # PyTest test suite
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Setup Virtual Environment & Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train All Models
Execute the end-to-end training pipeline:
```bash
python src/data_generator.py
python src/train_all.py
```

### 3. Launch the Web Portal (`delhifog.org` Dashboard)
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the REST API Backend
```bash
uvicorn src.api:app --reload --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### 5. Run Tests
```bash
python -m pytest tests -p no:cacheprovider
```

---

## 🔐 Authentication

The FastAPI prediction endpoints and Streamlit dashboard now require a configured single-admin login.

### Configure credentials

Set these environment variables **before starting either app**:

- `AUTH_USERNAME`: the login username
- `AUTH_PASSWORD`: a long, unique password
- `JWT_SECRET_KEY`: a random secret of at least 32 characters used to sign API access tokens

Generate a secret with Python:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

On Windows PowerShell, set the variables for the current terminal (replace the example values):

```powershell
$env:AUTH_USERNAME = "admin"
$env:AUTH_PASSWORD = "replace-with-a-long-unique-password"
$env:JWT_SECRET_KEY = "paste-the-generated-secret-here"
```

Keep these values private; do not commit real credentials or the JWT secret. For deployment, configure them in the hosting provider's secret/environment settings. Use HTTPS in production.

### Start the apps

In the terminal where you set the environment variables:

```bash
uvicorn src.api:app --port 8000
```

In that same environment (or a separate terminal with the same variables set):

```bash
streamlit run app/dashboard.py
```

### API login flow

1. Send `POST /auth/login` with JSON `{"username":"...","password":"..."}`.
2. The API returns a signed JWT bearer token that expires after 30 minutes.
3. Send `Authorization: Bearer <token>` with requests to `/predict/sensor`, `/predict/timeseries`, `/predict/image`, and `/metrics`.
4. A missing, invalid, or expired token receives HTTP `401`.

The embedded FastAPI portal displays a sign-in form and adds the token to sensor prediction requests. The Streamlit dashboard has its own sign-in gate. The public `/`, `/auth/login`, `/api`, `/regional/locations`, and API schema pages are not private data endpoints; place the service behind an HTTPS reverse proxy and restrict docs/routes at the deployment layer if those also need to be private.

## 📊 Benchmark Highlights

| Model | Modality | Task | Metric | Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | In-situ Sensors | Classification | F1-Score | **98.61%** |
| **Random Forest** | In-situ Sensors | Regression | MAE | **3.19 / 100** |
| **Support Vector Machine** | In-situ Sensors | Classification | F1-Score | **97.52%** |
| **K-Nearest Neighbors** | In-situ Sensors | Classification | F1-Score | **94.14%** |
| **PyTorch FogCNN** | Satellite / CCTV | Image Vision | Accuracy | **100.00%** |
| **PyTorch LSTM** | 24h Sequence | Time-Series | Accuracy | **72.40%** |
