"""
FastAPI Backend for Delhi NCR Fog Prediction & Monitoring System.
Serves:
- Interactive Web Portal at GET /
- REST API at /predict/sensor, /predict/timeseries, /predict/image
- Swagger Docs at /docs
"""

import os
import sys
import io
import json
from typing import Optional, List

# Ensure both project root and src directory are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [PROJECT_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image
import pandas as pd

try:
    from src.predictor import FogPredictionPipeline
except ImportError:
    from predictor import FogPredictionPipeline

app = FastAPI(
    title="Delhi NCR Fog Prediction API",
    description="Backend service powering the Delhi NCR Real-Time Fog & Smog Prediction System",
    version="1.0.0"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
pipeline = FogPredictionPipeline(MODELS_DIR)

class SensorInput(BaseModel):
    temperature_c: float = Field(default=6.8, example=6.8)
    dew_point_c: float = Field(default=6.2, example=6.2)
    relative_humidity_pct: float = Field(default=96.0, example=96.0)
    wind_speed_ms: float = Field(default=0.9, example=0.9)
    wind_direction_deg: float = Field(default=300.0, example=300.0)
    pressure_hpa: float = Field(default=1017.5, example=1017.5)
    pm25_ugm3: float = Field(default=240.0, example=240.0)
    pm10_ugm3: float = Field(default=380.0, example=380.0)
    solar_radiation_wm2: float = Field(default=0.0, example=0.0)
    model_type: Optional[str] = Field(default="Random Forest", example="Random Forest")

class TimeSeriesRecord(BaseModel):
    temperature_c: float
    dew_point_c: float
    relative_humidity_pct: float
    wind_speed_ms: float
    pressure_hpa: float
    pm25_ugm3: float
    solar_radiation_wm2: float

class TimeSeriesBatchInput(BaseModel):
    history: List[TimeSeriesRecord]

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Delhi NCR Fog Prediction System | Live Portal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; }
    .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20">
            ● LIVE SENSOR NETWORK
          </span>
          <span class="text-xs text-slate-400">Delhi Meteorological Grid</span>
        </div>
        <h1 class="text-2xl font-extrabold tracking-tight text-white mt-1">Delhi NCR Fog Prediction System</h1>
      </div>
      <div class="flex items-center gap-3">
        <a href="/docs" target="_blank" class="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition">
          Interactive Swagger Docs (/docs)
        </a>
        <a href="/api" class="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-sky-600 hover:bg-sky-500 text-white transition">
          JSON Endpoint
        </a>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
    <!-- Advisory Alert Banner -->
    <div id="alertBanner" class="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-200 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span class="text-2xl">⚠️</span>
        <div>
          <h3 id="alertTitle" class="font-bold text-amber-400">DENSE FOG ADVISORY ACTIVE</h3>
          <p id="alertDesc" class="text-xs text-amber-200/80 mt-0.5">Visibility reduced below 150m across Delhi NCR. Runway CAT-III procedures standby at IGI Airport.</p>
        </div>
      </div>
      <span class="text-xs bg-amber-500/20 text-amber-300 px-3 py-1 rounded-full font-medium">IMD Standard</span>
    </div>

    <!-- Top Metrics Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="glass-card p-5 rounded-2xl">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Quantitative Fog Index</span>
        <div class="flex items-baseline gap-2 mt-2">
          <span id="metricFogIndex" class="text-4xl font-extrabold text-sky-400">78.5</span>
          <span class="text-sm text-slate-500 font-medium">/ 100</span>
        </div>
        <p id="metricProb" class="text-xs text-sky-400/80 mt-1 font-medium">Probability: 95%</p>
      </div>

      <div class="glass-card p-5 rounded-2xl">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Est. Runway Visibility (RVR)</span>
        <div class="flex items-baseline gap-2 mt-2">
          <span id="metricVis" class="text-4xl font-extrabold text-emerald-400">120</span>
          <span class="text-sm text-slate-500 font-medium">meters</span>
        </div>
        <p id="metricStatus" class="text-xs text-amber-400 mt-1 font-medium">Condition: Foggy</p>
      </div>

      <div class="glass-card p-5 rounded-2xl">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Dew Point Depression</span>
        <div class="flex items-baseline gap-2 mt-2">
          <span id="metricDepression" class="text-4xl font-extrabold text-indigo-400">0.6</span>
          <span class="text-sm text-slate-500 font-medium">°C</span>
        </div>
        <p class="text-xs text-indigo-400/80 mt-1 font-medium">Atmospheric Condensation Active</p>
      </div>

      <div class="glass-card p-5 rounded-2xl">
        <span class="text-xs font-semibold uppercase tracking-wider text-slate-400">Aerosols (PM2.5)</span>
        <div class="flex items-baseline gap-2 mt-2">
          <span id="metricPM" class="text-4xl font-extrabold text-rose-400">240</span>
          <span class="text-sm text-slate-500 font-medium">µg/m³</span>
        </div>
        <p class="text-xs text-rose-400/80 mt-1 font-medium">Cloud Condensation Nuclei (CCN)</p>
      </div>
    </div>

    <!-- Main Content Layout -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <!-- Interactive Sensor Simulator -->
      <div class="lg:col-span-2 glass-card p-6 rounded-2xl space-y-6">
        <div class="flex justify-between items-center border-b border-slate-800 pb-4">
          <div>
            <h2 class="text-lg font-bold text-white">Live Meteorological Sensor Simulator</h2>
            <p class="text-xs text-slate-400">Adjust weather inputs in real-time to compute instant machine learning predictions</p>
          </div>
          <select id="modelSelect" class="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-sky-500">
            <option value="Random Forest">Random Forest</option>
            <option value="SVM">Support Vector Machine (SVM)</option>
            <option value="KNN">K-Nearest Neighbors (KNN)</option>
          </select>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-5 text-sm">
          <div>
            <div class="flex justify-between text-xs font-medium mb-1.5">
              <span class="text-slate-300">Air Temperature (°C)</span>
              <span id="valTemp" class="text-sky-400 font-bold">6.8 °C</span>
            </div>
            <input id="sliderTemp" type="range" min="0" max="30" step="0.1" value="6.8" class="w-full accent-sky-500">
          </div>

          <div>
            <div class="flex justify-between text-xs font-medium mb-1.5">
              <span class="text-slate-300">Dew Point Temperature (°C)</span>
              <span id="valDew" class="text-sky-400 font-bold">6.2 °C</span>
            </div>
            <input id="sliderDew" type="range" min="-5" max="25" step="0.1" value="6.2" class="w-full accent-sky-500">
          </div>

          <div>
            <div class="flex justify-between text-xs font-medium mb-1.5">
              <span class="text-slate-300">Relative Humidity (%)</span>
              <span id="valRH" class="text-sky-400 font-bold">96 %</span>
            </div>
            <input id="sliderRH" type="range" min="20" max="100" step="1" value="96" class="w-full accent-sky-500">
          </div>

          <div>
            <div class="flex justify-between text-xs font-medium mb-1.5">
              <span class="text-slate-300">Wind Speed (m/s)</span>
              <span id="valWind" class="text-sky-400 font-bold">0.9 m/s</span>
            </div>
            <input id="sliderWind" type="range" min="0.1" max="10" step="0.1" value="0.9" class="w-full accent-sky-500">
          </div>

          <div class="md:col-span-2">
            <div class="flex justify-between text-xs font-medium mb-1.5">
              <span class="text-slate-300">Particulate Matter PM2.5 (µg/m³)</span>
              <span id="valPM" class="text-rose-400 font-bold">240 µg/m³</span>
            </div>
            <input id="sliderPM" type="range" min="15" max="500" step="5" value="240" class="w-full accent-rose-500">
          </div>
        </div>

        <div class="pt-4 border-t border-slate-800 flex justify-between items-center text-xs text-slate-400">
          <span>Inference Endpoint: <code class="text-sky-400">POST /predict/sensor</code></span>
          <span id="livePing" class="text-emerald-400">● Latency: 8ms</span>
        </div>
      </div>

      <!-- Station Monitoring Grid -->
      <div class="glass-card p-6 rounded-2xl space-y-4">
        <h2 class="text-lg font-bold text-white border-b border-slate-800 pb-3">Strategic Delhi NCR Observatories</h2>
        <div class="space-y-3">
          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-semibold text-xs text-white">IGI Airport (T3 / Runway 28)</div>
              <div class="text-[11px] text-slate-400">CAT-III ILS Sensor Station</div>
            </div>
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">Fog Index: 88</span>
          </div>

          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-semibold text-xs text-white">Anand Vihar ISBT</div>
              <div class="text-[11px] text-slate-400">Trans-Yamuna Transport Hub</div>
            </div>
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">Fog Index: 84</span>
          </div>

          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-semibold text-xs text-white">Akshardham / Yamuna Belt</div>
              <div class="text-[11px] text-slate-400">High Humidity River Floodplain</div>
            </div>
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">Fog Index: 85</span>
          </div>

          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-semibold text-xs text-white">India Gate / Central Vista</div>
              <div class="text-[11px] text-slate-400">Safdarjung Base Grid</div>
            </div>
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">Fog Index: 76</span>
          </div>

          <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between">
            <div>
              <div class="font-semibold text-xs text-white">Dwarka Expressway Junction</div>
              <div class="text-[11px] text-slate-400">Western NCR Highway Corridor</div>
            </div>
            <span class="text-[11px] font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">Fog Index: 79</span>
          </div>
        </div>
      </div>
    </div>
  </main>

  <footer class="border-t border-slate-800 mt-12 py-6 text-center text-xs text-slate-500">
    Delhi NCR Fog & Smog Prediction System &bull; Powered by Random Forest, Support Vector Machines, KNN, and PyTorch Deep Learning
  </footer>

  <script>
    const sliders = ['Temp', 'Dew', 'RH', 'Wind', 'PM'];
    function updateVal(name, unit) {
      document.getElementById('val' + name).innerText = document.getElementById('slider' + name).value + ' ' + unit;
    }

    async function recompute() {
      const temp = parseFloat(document.getElementById('sliderTemp').value);
      const dew = parseFloat(document.getElementById('sliderDew').value);
      const rh = parseFloat(document.getElementById('sliderRH').value);
      const wind = parseFloat(document.getElementById('sliderWind').value);
      const pm = parseFloat(document.getElementById('sliderPM').value);
      const model = document.getElementById('modelSelect').value;

      updateVal('Temp', '°C');
      updateVal('Dew', '°C');
      updateVal('RH', '%');
      updateVal('Wind', 'm/s');
      updateVal('PM', 'µg/m³');

      const depression = (temp - dew).toFixed(1);
      document.getElementById('metricDepression').innerText = depression;
      document.getElementById('metricPM').innerText = pm;

      const t0 = performance.now();
      try {
        const res = await fetch('/predict/sensor', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            temperature_c: temp,
            dew_point_c: dew,
            relative_humidity_pct: rh,
            wind_speed_ms: wind,
            pm25_ugm3: pm,
            model_type: model
          })
        });
        const json = await res.json();
        const r = json.result;
        const latency = Math.round(performance.now() - t0);
        document.getElementById('livePing').innerText = `● Latency: ${latency}ms`;

        document.getElementById('metricFogIndex').innerText = r.fog_index;
        document.getElementById('metricProb').innerText = `Probability: ${Math.round(r.fog_probability * 100)}%`;
        document.getElementById('metricVis').innerText = Math.round(r.estimated_visibility_meters);
        document.getElementById('metricStatus').innerText = `Condition: ${r.status}`;

        if (r.fog_index >= 70) {
          document.getElementById('alertTitle').innerText = 'VERY DENSE FOG ALERT';
          document.getElementById('alertDesc').innerText = 'Runway CAT-III active. Flight delays likely across Delhi NCR.';
          document.getElementById('alertBanner').className = 'p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-200 flex items-center justify-between';
        } else if (r.fog_index >= 45) {
          document.getElementById('alertTitle').innerText = 'DENSE FOG ADVISORY ACTIVE';
          document.getElementById('alertDesc').innerText = 'Visibility reduced below 200m across Delhi NCR.';
          document.getElementById('alertBanner').className = 'p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-200 flex items-center justify-between';
        } else {
          document.getElementById('alertTitle').innerText = 'CLEAR ATMOSPHERIC VISIBILITY';
          document.getElementById('alertDesc').innerText = 'Normal visibility conditions prevailing across Delhi NCR.';
          document.getElementById('alertBanner').className = 'p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-200 flex items-center justify-between';
        }
      } catch(e) {
        console.error(e);
      }
    }

    sliders.forEach(s => document.getElementById('slider' + s).addEventListener('input', recompute));
    document.getElementById('modelSelect').addEventListener('change', recompute);
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index_html():
    return DASHBOARD_HTML

@app.get("/api")
def index_api():
    return {
        "service": "Delhi NCR Fog & Smog Prediction System",
        "region": "National Capital Region (Delhi, Noida, Gurugram, Ghaziabad)",
        "status": "operational",
        "observatories": ["Safdarjung Base", "IGI Airport Palam", "Anand Vihar Hub", "Akshardham Yamuna Corridor", "Dwarka"],
        "endpoints": [
            "/docs",
            "/predict/sensor",
            "/predict/timeseries",
            "/predict/image",
            "/metrics",
            "/regional/locations"
        ]
    }

@app.get("/regional/locations")
def regional_locations():
    return {
        "locations": [
            {"id": "igi_airport", "name": "Indira Gandhi International Airport (T3 / Runway 28)", "sensors_active": True},
            {"id": "anand_vihar", "name": "Anand Vihar ISBT & Trans-Yamuna Hub", "sensors_active": True},
            {"id": "india_gate", "name": "India Gate & Central Vista Observatory", "sensors_active": True},
            {"id": "akshardham", "name": "Akshardham & Yamuna River Floodplain Corridor", "sensors_active": True},
            {"id": "dwarka", "name": "Dwarka Expressway Junction & Sector 8", "sensors_active": True}
        ]
    }

@app.post("/predict/sensor")
def predict_sensor(data: SensorInput):
    try:
        readings = data.model_dump()
        model_choice = readings.pop("model_type", "Random Forest")
        res = pipeline.predict_from_sensors(readings, model_type=model_choice)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/timeseries")
def predict_timeseries(batch: TimeSeriesBatchInput):
    try:
        df = pd.DataFrame([rec.model_dump() for rec in batch.history])
        res = pipeline.forecast_from_history(df)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image_upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        res = pipeline.predict_from_image(image)
        return {
            "status": "success",
            "filename": file.filename,
            "result": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_benchmarks():
    metrics_path = os.path.join(MODELS_DIR, "benchmark_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"message": "Benchmark metrics operational."}
