"""
FastAPI Backend for Delhi NCR Fog Prediction & Monitoring System.
Serves REST API endpoints for:
- Sensor prediction (RF, SVM, KNN)
- Time-series sequence forecasting (LSTM)
- Image analysis (FogCNN + Dark Channel Prior)
- Multi-modal fusion & advisory
- Model benchmarking metrics
"""

import os
import io
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import pandas as pd

from src.predictor import FogPredictionPipeline

app = FastAPI(
    title="Delhi NCR Fog Prediction API",
    description="Backend service powering the Delhi NCR Real-Time Fog & Smog Prediction System",
    version="1.0.0"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
pipeline = FogPredictionPipeline(MODELS_DIR)

class SensorInput(BaseModel):
    temperature_c: float = Field(..., example=8.5, description="Ambient air temperature in Celsius")
    dew_point_c: float = Field(..., example=7.8, description="Dew point temperature in Celsius")
    relative_humidity_pct: float = Field(..., example=94.0, description="Relative humidity in percentage")
    wind_speed_ms: float = Field(..., example=1.2, description="Wind speed in meters per second")
    wind_direction_deg: float = Field(default=310.0, example=310.0, description="Wind direction in degrees")
    pressure_hpa: float = Field(default=1016.0, example=1016.0, description="Atmospheric pressure in hPa")
    pm25_ugm3: float = Field(default=180.0, example=180.0, description="PM2.5 particulate concentration")
    pm10_ugm3: float = Field(default=280.0, example=280.0, description="PM10 particulate concentration")
    solar_radiation_wm2: float = Field(default=0.0, example=0.0, description="Solar radiation in W/m²")
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

@app.get("/")
def index():
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
    return {"message": "Benchmark metrics not yet generated. Please execute training."}
