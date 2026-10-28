"""
End-to-End Training Orchestrator.
Trains all components:
1. Sensor Models: Random Forest, SVM, KNN (Classification & Regression)
2. Time-Series Model: LSTM (PyTorch)
3. Computer Vision Model: FogCNN (PyTorch)
Saves models and outputs benchmarking report.
"""

import os
import sys
import json
import time

# Ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.sensor_models import train_sensor_models
from src.lstm_model import train_lstm_model
from src.cnn_model import train_cnn

def main():
    print("=" * 70)
    print("  DELHI NCR FOG PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    start_time = time.time()
    
    data_dir = os.path.join(project_dir, "data", "raw")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    sensor_csv = os.path.join(data_dir, "delhi_meteorological_sensors.csv")
    if not os.path.exists(sensor_csv):
        sensor_csv = os.path.join(data_dir, "meteorological_sensors.csv")
    image_meta_csv = os.path.join(data_dir, "image_metadata.csv")
    
    # Check if raw data exists
    if not os.path.exists(sensor_csv) or not os.path.exists(image_meta_csv):
        print("Data files not found. Generating datasets first...")
        from src.data_generator import generate_sensor_dataset, generate_image_dataset
        df_sensors = generate_sensor_dataset(n_samples=5000)
        df_sensors.to_csv(sensor_csv, index=False)
        generate_image_dataset(data_dir, n_images=120)
        
    benchmark_report = {}
    
    # Step 1: Train Classical Sensor Models (RF, SVM, KNN)
    print("\n[Stage 1/3] Training Sensor Models (Random Forest, SVM, KNN)...")
    sensor_bundle_path = os.path.join(models_dir, "sensor_models.joblib")
    sensor_bundle = train_sensor_models(sensor_csv, save_path=sensor_bundle_path)
    benchmark_report["sensor_models"] = sensor_bundle["metrics"]
    print("  -> Sensor Models Trained & Evaluated.")
    
    # Step 2: Train LSTM Time Series Model
    print("\n[Stage 2/3] Training PyTorch LSTM Sequence Model...")
    lstm_path = os.path.join(models_dir, "lstm_fog_model.pt")
    lstm_metrics = train_lstm_model(sensor_csv, save_path=lstm_path, epochs=10, batch_size=32)
    benchmark_report["lstm_model"] = {
        "val_accuracy": round(float(lstm_metrics["val_acc"]), 4),
        "val_mae": round(float(lstm_metrics["val_mae"]), 4)
    }
    print("  -> LSTM Time Series Model Trained.")
    
    # Step 3: Train FogCNN on Satellite & CCTV Images
    print("\n[Stage 3/3] Training PyTorch FogCNN for Satellite & CCTV Feeds...")
    cnn_path = os.path.join(models_dir, "cnn_fog_model.pt")
    cnn_metrics = train_cnn(image_meta_csv, data_dir, save_path=cnn_path, epochs=10, batch_size=16)
    benchmark_report["cnn_model"] = {
        "val_accuracy": round(float(cnn_metrics["val_acc"]), 4),
        "val_mae": round(float(cnn_metrics["val_mae"]), 4)
    }
    print("  -> FogCNN Model Trained.")
    
    # Save Benchmark Metrics
    metrics_path = os.path.join(models_dir, "benchmark_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(benchmark_report, f, indent=2)
        
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f" ALL MODELS SUCCESSFULLY TRAINED IN {elapsed:.1f}s!")
    print(f" Checkpoints saved to: {models_dir}")
    print(f" Benchmark metrics saved to: {metrics_path}")
    print("=" * 70)
    
    # Print neat summary table
    print("\nBENCHMARK SUMMARY:")
    print("-" * 55)
    print(f"{'Model':<18} | {'Task':<14} | {'Metric':<10} | {'Score':<8}")
    print("-" * 55)
    for model_name, m in benchmark_report["sensor_models"]["classification"].items():
        print(f"{model_name:<18} | Classification | F1-Score   | {m['f1_score']:.4f}")
    for model_name, m in benchmark_report["sensor_models"]["regression"].items():
        print(f"{model_name:<18} | Regression     | MAE        | {m['mae']:.2f}")
    print(f"{'LSTM Sequence':<18} | Time-Series    | Accuracy   | {benchmark_report['lstm_model']['val_accuracy']:.4f}")
    print(f"{'FogCNN':<18} | Satellite/CCTV | Accuracy   | {benchmark_report['cnn_model']['val_accuracy']:.4f}")
    print("-" * 55)

if __name__ == "__main__":
    main()
