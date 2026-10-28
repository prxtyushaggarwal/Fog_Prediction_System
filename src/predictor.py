"""
Unified Fog Prediction and Multi-Modal Fusion Engine.
Integrates:
- Real-time Sensor Predictor (Random Forest, SVM, KNN)
- Time-Series Forecaster (LSTM)
- Computer Vision & Satellite/CCTV Analyzer (FogCNN & Dark Channel Prior)
- Multi-modal Fusion Advisory
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image

from .image_processing import extract_image_features
from .cnn_model import predict_image
from .sensor_models import predict_sensor_data
from .lstm_model import forecast_next_24h

class FogPredictionPipeline:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.sensor_bundle_path = os.path.join(models_dir, "sensor_models.joblib")
        self.cnn_path = os.path.join(models_dir, "cnn_fog_model.pt")
        self.lstm_path = os.path.join(models_dir, "lstm_fog_model.pt")
        
    def predict_from_sensors(self, sensor_readings: dict, model_type: str = "Random Forest") -> dict:
        """Runs tabular sensor inference using RF, SVM, or KNN."""
        if not os.path.exists(self.sensor_bundle_path):
            raise FileNotFoundError(f"Sensor models not found at {self.sensor_bundle_path}. Run training first.")
        return predict_sensor_data(sensor_readings, self.sensor_bundle_path, model_type=model_type)
        
    def forecast_from_history(self, recent_history_df: pd.DataFrame) -> dict:
        """Runs LSTM time-series inference on recent 24-hour sequence."""
        if not os.path.exists(self.lstm_path):
            raise FileNotFoundError(f"LSTM model not found at {self.lstm_path}. Run training first.")
        return forecast_next_24h(recent_history_df, self.lstm_path)
        
    def predict_from_image(self, image: Image.Image) -> dict:
        """Extracts optical features (Dark Channel Prior) and runs PyTorch CNN."""
        optical_features = extract_image_features(image)
        cnn_preds = predict_image(image, self.cnn_path if os.path.exists(self.cnn_path) else None)
        return {
            "optical_metrics": optical_features,
            "cnn_prediction": cnn_preds
        }
        
    def fuse_multimodal(self, sensor_readings: dict, image: Image.Image = None, history_df: pd.DataFrame = None) -> dict:
        """
        Combines sensor readings, optical camera data, and time-series LSTM trends
        into an integrated Delhi NCR Regional Fog Advisory.
        """
        sensor_res = self.predict_from_sensors(sensor_readings, model_type="Random Forest")
        
        weights = {"sensor": 0.5, "image": 0.3, "lstm": 0.2}
        fog_indices = [sensor_res["fog_index"]]
        probabilities = [sensor_res["fog_probability"]]
        weight_sum = weights["sensor"]
        
        img_res = None
        if image is not None:
            img_res = self.predict_from_image(image)
            fog_indices.append(img_res["cnn_prediction"]["fog_index"])
            probabilities.append(img_res["cnn_prediction"]["fog_probability"])
            weight_sum += weights["image"]
            
        lstm_res = None
        if history_df is not None and len(history_df) >= 6:
            lstm_res = self.forecast_from_history(history_df)
            fog_indices.append(lstm_res["current_fog_index"])
            probabilities.append(lstm_res["current_fog_probability"])
            weight_sum += weights["lstm"]
            
        fused_fog_index = float(np.average(fog_indices, weights=[
            weights["sensor"],
            *( [weights["image"]] if image is not None else [] ),
            *( [weights["lstm"]] if lstm_res is not None else [] )
        ]))
        
        fused_probability = float(np.average(probabilities, weights=[
            weights["sensor"],
            *( [weights["image"]] if image is not None else [] ),
            *( [weights["lstm"]] if lstm_res is not None else [] )
        ]))
        
        # Severity Category based on Indian Meteorological Department (IMD) fog standards:
        # Shallow Fog (500m - 1000m), Moderate Fog (200m - 500m), Dense Fog (50m - 200m), Very Dense Fog (< 50m)
        if fused_fog_index >= 75.0:
            severity = "VERY DENSE FOG"
            action_advisory = "Zero/critical visibility (< 50m). Flight operations suspended, vehicle movement strictly restricted."
            color = "#ef4444" # red
        elif fused_fog_index >= 55.0:
            severity = "DENSE FOG"
            action_advisory = "Visibility reduced (50m - 200m). Low-beam fog lights mandatory on highways and expressways."
            color = "#f97316" # orange
        elif fused_fog_index >= 40.0:
            severity = "MODERATE FOG"
            action_advisory = "Visibility 200m - 500m. Commute caution advised across Delhi NCR."
            color = "#eab308" # yellow
        elif fused_fog_index >= 20.0:
            severity = "SHALLOW FOG / MIST"
            action_advisory = "Light mist observed. No disruption to normal activities."
            color = "#3b82f6" # blue
        else:
            severity = "CLEAR / NO FOG"
            action_advisory = "Optimal atmospheric visibility (> 2000m)."
            color = "#10b981" # green
            
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
