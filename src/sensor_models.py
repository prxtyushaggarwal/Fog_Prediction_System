"""
Sensor Data Machine Learning Models:
Implements Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)
for both:
1. Binary Classification: Fog vs No-Fog
2. Regression: Fog Index (0 - 100) and Visibility (meters)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

FEATURE_COLS = [
    "temperature_c",
    "dew_point_c",
    "dew_point_depression_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "pressure_hpa",
    "pm25_ugm3",
    "pm10_ugm3",
    "solar_radiation_wm2",
    "hour_sin",
    "hour_cos"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical temporal features and physics-based interactions."""
    data = df.copy()
    if "timestamp" in data.columns:
        ts = pd.to_datetime(data["timestamp"])
        data["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
    else:
        # Fallback if raw feature vector without timestamp
        if "hour_sin" not in data.columns:
            data["hour_sin"] = 0.0
            data["hour_cos"] = 1.0
            
    # Calculate dew point depression if missing
    if "dew_point_depression_c" not in data.columns and "temperature_c" in data.columns and "dew_point_c" in data.columns:
        data["dew_point_depression_c"] = data["temperature_c"] - data["dew_point_c"]
        
    return data

def train_sensor_models(data_path: str, save_path: str = None) -> dict:
    """
    Trains and benchmarks Random Forest, SVM, and KNN on sensor telemetry.
    Returns model instances and comprehensive evaluation metrics.
    """
    raw_df = pd.read_csv(data_path)
    df = engineer_features(raw_df)
    
    X = df[FEATURE_COLS]
    y_cls = df["fog_label"]
    y_reg = df["fog_index"]
    
    # Train / Test split
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Classification Models ---
    rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_cls.fit(X_train, y_train_cls)
    
    svm_cls = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    svm_cls.fit(X_train_scaled, y_train_cls)
    
    knn_cls = KNeighborsClassifier(n_neighbors=7)
    knn_cls.fit(X_train_scaled, y_train_cls)
    
    # --- Regression Models ---
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_reg.fit(X_train, y_train_reg)
    
    svm_reg = SVR(kernel='rbf', C=10.0, epsilon=0.2)
    svm_reg.fit(X_train_scaled, y_train_reg)
    
    knn_reg = KNeighborsRegressor(n_neighbors=7)
    knn_reg.fit(X_train_scaled, y_reg_train := y_train_reg)
    
    # --- Evaluation ---
    metrics = {
        "classification": {},
        "regression": {}
    }
    
    # Classification evaluations
    cls_models = {
        "Random Forest": (rf_cls, X_test),
        "SVM": (svm_cls, X_test_scaled),
        "KNN": (knn_cls, X_test_scaled)
    }
    
    for name, (mod, x_eval) in cls_models.items():
        preds = mod.predict(x_eval)
        probs = mod.predict_proba(x_eval)[:, 1] if hasattr(mod, "predict_proba") else preds
        metrics["classification"][name] = {
            "accuracy": round(float(accuracy_score(y_test_cls, preds)), 4),
            "precision": round(float(precision_score(y_test_cls, preds)), 4),
            "recall": round(float(recall_score(y_test_cls, preds)), 4),
            "f1_score": round(float(f1_score(y_test_cls, preds)), 4),
            "roc_auc": round(float(roc_auc_score(y_test_cls, probs)), 4)
        }
        
    # Regression evaluations
    reg_models = {
        "Random Forest": (rf_reg, X_test),
        "SVM": (svm_reg, X_test_scaled),
        "KNN": (knn_reg, X_test_scaled)
    }
    
    for name, (mod, x_eval) in reg_models.items():
        preds = mod.predict(x_eval)
        metrics["regression"][name] = {
            "mae": round(float(mean_absolute_error(y_test_reg, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test_reg, preds))), 4),
            "r2_score": round(float(r2_score(y_test_reg, preds)), 4)
        }
        
    bundle = {
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "models": {
            "rf_cls": rf_cls,
            "svm_cls": svm_cls,
            "knn_cls": knn_cls,
            "rf_reg": rf_reg,
            "svm_reg": svm_reg,
            "knn_reg": knn_reg
        },
        "metrics": metrics
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(bundle, save_path)
        print(f"Saved sensor model bundle to {save_path}")
        
    return bundle

def predict_sensor_data(sensor_features: dict, model_bundle_path: str, model_type: str = "Random Forest") -> dict:
    """
    Predicts fog condition and fog index for real-time sensor measurements.
    """
    bundle = joblib.load(model_bundle_path)
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]
    
    # Build single-row DataFrame
    df = pd.DataFrame([sensor_features])
    df = engineer_features(df)
    
    # Ensure all feature cols exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    if model_type == "SVM":
        cls_mod = bundle["models"]["svm_cls"]
        reg_mod = bundle["models"]["svm_reg"]
        x_in = X_scaled
    elif model_type == "KNN":
        cls_mod = bundle["models"]["knn_cls"]
        reg_mod = bundle["models"]["knn_reg"]
        x_in = X_scaled
    else:  # Random Forest
        cls_mod = bundle["models"]["rf_cls"]
        reg_mod = bundle["models"]["rf_reg"]
        x_in = X
        
    prob = float(cls_mod.predict_proba(x_in)[0, 1])
    label = int(cls_mod.predict(x_in)[0])
    fog_idx = float(np.clip(reg_mod.predict(x_in)[0], 0.0, 100.0))
    
    # Visibility in meters approx
    vis_meters = float(max(40.0, min(6000.0, 4500.0 * np.exp(-fog_idx / 22.0))))
    
    return {
        "model_used": model_type,
        "fog_probability": round(prob, 4),
        "fog_label": label,
        "status": "Foggy" if label == 1 else "Clear",
        "fog_index": round(fog_idx, 2),
        "estimated_visibility_meters": round(vis_meters, 1)
    }
