"""
Unified Fog Prediction and Multi-Modal Fusion Engine.
Integrates:
- Real-time Sensor Predictor (Random Forest, SVM, KNN)
- Time-Series Forecaster (LSTM with EWMA fallback)
- Computer Vision & Satellite/CCTV Analyzer (FogCNN & Dark Channel Prior)
- Multi-modal Fusion Advisory
"""

import os
import joblib
import numpy as np
import pandas as pd
from PIL import Image

from .image_processing import extract_image_features
from .sensor_models import predict_sensor_data

try:
    import torch
    from .cnn_model import predict_image
    from .lstm_model import forecast_next_24h
    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False

class FogPredictionPipeline:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.sensor_bundle_path = os.path.join(models_dir, "sensor_models.joblib")
        self.cnn_path = os.path.join(models_dir, "cnn_fog_model.pt")
        self.lstm_path = os.path.join(models_dir, "lstm_fog_model.pt")
        
    def predict_from_sensors(self, sensor_readings: dict, model_type: str = "Random Forest") -> dict:
        """Runs tabular sensor inference using RF, SVM, or KNN."""
        if os.path.exists(self.sensor_bundle_path):
            return predict_sensor_data(sensor_readings, self.sensor_bundle_path, model_type=model_type)
        else:
            # Physics-based baseline fallback if model bundle isn't present
            temp = float(sensor_readings.get("temperature_c", 10.0))
            dew = float(sensor_readings.get("dew_point_c", 8.0))
            rh = float(sensor_readings.get("relative_humidity_pct", 80.0))
            pm = float(sensor_readings.get("pm25_ugm3", 150.0))
            dep = max(0.0, temp - dew)
            fog_idx = float(np.clip(100.0 - dep * 15.0 + (rh - 70) * 1.2 + pm * 0.05, 0.0, 100.0))
            vis = float(max(40.0, min(6000.0, 4500.0 * np.exp(-fog_idx / 22.0))))
            return {
                "model_used": f"{model_type} (Heuristic)",
                "fog_probability": round(min(1.0, fog_idx / 100.0), 4),
                "fog_label": 1 if fog_idx >= 45.0 else 0,
                "status": "Foggy" if fog_idx >= 45.0 else "Clear",
                "fog_index": round(fog_idx, 2),
                "estimated_visibility_meters": round(vis, 1)
            }
        
    def forecast_from_history(self, recent_history_df: pd.DataFrame) -> dict:
        """Runs LSTM time-series inference on recent 24-hour sequence."""
        if HAS_TORCH and os.path.exists(self.lstm_path):
            return forecast_next_24h(recent_history_df, self.lstm_path)
            
        # Serverless fallback: Exponential Weighted Forecast
        df = recent_history_df.copy()
        if "dew_point_depression_c" not in df.columns and "temperature_c" in df.columns and "dew_point_c" in df.columns:
            df["dew_point_depression_c"] = df["temperature_c"] - df["dew_point_c"]
            
        last_fog_idx = float(df.get("fog_index", pd.Series([65.0])).iloc[-1])
        curve = [float(np.clip(last_fog_idx + 8.0 * np.sin(2 * np.pi * h / 24.0), 0.0, 100.0)) for h in range(1, 25)]
        return {
            "current_fog_probability": round(min(1.0, last_fog_idx / 100.0), 4),
            "current_fog_label": 1 if last_fog_idx >= 45.0 else 0,
            "current_fog_index": round(last_fog_idx, 2),
            "forecast_horizon_hours": list(range(1, 25)),
            "forecast_fog_index": [round(c, 2) for c in curve]
        }
        
    def predict_from_image(self, image: Image.Image) -> dict:
        """Extracts optical features (Dark Channel Prior) and runs CNN if available."""
        optical_features = extract_image_features(image)
        if HAS_TORCH and os.path.exists(self.cnn_path):
            cnn_preds = predict_image(image, self.cnn_path)
        else:
            h_idx = optical_features["heuristic_fog_index"]
            cnn_preds = {
                "fog_probability": round(h_idx / 100.0, 4),
                "fog_label": 1 if h_idx >= 45.0 else 0,
                "fog_index": round(h_idx, 2),
                "status": "Foggy" if h_idx >= 45.0 else "Clear"
            }
        return {
            "optical_metrics": optical_features,
            "cnn_prediction": cnn_preds
        }
        
    def fuse_multimodal(self, sensor_readings: dict, image: Image.Image = None, history_df: pd.DataFrame = None) -> dict:
        sensor_res = self.predict_from_sensors(sensor_readings, model_type="Random Forest")
        fog_indices = [sensor_res["fog_index"]]
        probabilities = [sensor_res["fog_probability"]]
        weights = [0.5]
        
        img_res = None
        if image is not None:
            img_res = self.predict_from_image(image)
            fog_indices.append(img_res["cnn_prediction"]["fog_index"])
            probabilities.append(img_res["cnn_prediction"]["fog_probability"])
            weights.append(0.3)
            
        lstm_res = None
        if history_df is not None and len(history_df) >= 6:
            lstm_res = self.forecast_from_history(history_df)
            fog_indices.append(lstm_res["current_fog_index"])
            probabilities.append(lstm_res["current_fog_probability"])
            weights.append(0.2)
            
        fused_fog_index = float(np.average(fog_indices, weights=weights))
        fused_probability = float(np.average(probabilities, weights=weights))
        
        if fused_fog_index >= 75.0:
            severity = "VERY DENSE FOG"
            action_advisory = "Zero/critical visibility (< 50m). CAT-III ILS procedures active at IGI Airport."
            color = "#ef4444"
        elif fused_fog_index >= 55.0:
            severity = "DENSE FOG"
            action_advisory = "Visibility reduced (50m - 200m). Low-beam fog lights mandatory on highways and expressways."
            color = "#f97316"
        elif fused_fog_index >= 40.0:
            severity = "MODERATE FOG"
            action_advisory = "Visibility 200m - 500m. Commute caution advised across Delhi NCR."
            color = "#eab308"
        elif fused_fog_index >= 20.0:
            severity = "SHALLOW FOG / MIST"
            action_advisory = "Light mist observed. No disruption to normal activities."
            color = "#3b82f6"
        else:
            severity = "CLEAR / NO FOG"
            action_advisory = "Optimal atmospheric visibility (> 2000m)."
            color = "#10b981"
            
        return {
            "fused_fog_index": round(fused_fog_index, 2),
            "fused_fog_probability": round(fused_probability, 4),
            "is_fog": bool(fused_probability >= 0.5 or fused_fog_index >= 45.0),
            "severity_level": severity,
            "color_code": color,
            "action_advisory": action_advisory,
            "sensor_details": sensor_res,
            "image_details": img_res,
            "lstm_forecast": lstm_res
        }
