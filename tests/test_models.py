"""
Comprehensive Unit and Integration Tests for Fog Prediction System.
"""

import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image

from src.data_generator import generate_sensor_dataset, generate_synthetic_scene_image
from src.image_processing import extract_image_features, get_dark_channel
from src.predictor import FogPredictionPipeline

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

def test_sensor_data_generator():
    df = generate_sensor_dataset(n_samples=50)
    assert len(df) == 50
    assert "temperature_c" in df.columns
    assert "dew_point_c" in df.columns
    assert "relative_humidity_pct" in df.columns
    assert "fog_index" in df.columns
    assert "fog_label" in df.columns
    assert (df["fog_index"] >= 0).all() and (df["fog_index"] <= 100).all()

def test_image_scene_generator():
    img_cctv = generate_synthetic_scene_image("cctv", fog_level=0.5, size=(64, 64))
    assert isinstance(img_cctv, Image.Image)
    assert img_cctv.size == (64, 64)
    
    img_sat = generate_synthetic_scene_image("satellite", fog_level=0.8, size=(64, 64))
    assert isinstance(img_sat, Image.Image)

def test_dark_channel_prior():
    test_img = np.full((32, 32, 3), 120, dtype=np.uint8)
    dcp = get_dark_channel(test_img, patch_size=5)
    assert dcp.shape == (32, 32)
    assert np.allclose(dcp, 120 / 255.0, atol=0.05)

def test_image_feature_extraction():
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    feats = extract_image_features(img)
    assert "dark_channel_mean" in feats
    assert "transmission_mean" in feats
    assert "heuristic_fog_index" in feats

def test_sensor_inference():
    pipeline = FogPredictionPipeline(MODELS_DIR)
    dummy_input = {
        "temperature_c": 7.0,
        "dew_point_c": 6.8,
        "relative_humidity_pct": 98.0,
        "wind_speed_ms": 0.8,
        "wind_direction_deg": 310.0,
        "pressure_hpa": 1018.0,
        "pm25_ugm3": 250.0,
        "pm10_ugm3": 380.0,
        "solar_radiation_wm2": 0.0,
        "hour_sin": -0.86,
        "hour_cos": 0.5
    }
    
    for mod in ["Random Forest", "SVM", "KNN"]:
        res = pipeline.predict_from_sensors(dummy_input, model_type=mod)
        assert res["fog_label"] in [0, 1]
        assert 0.0 <= res["fog_index"] <= 100.0
        assert res["estimated_visibility_meters"] > 0
