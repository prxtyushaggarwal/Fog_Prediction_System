"""
PyTorch LSTM Model for Time-Series Fog Forecasting.
Handles sequential temporal dynamics of meteorological sensors to:
1. Model historical weather progression (past 24 hours).
2. Predict immediate next-step fog status & fog index.
3. Generate multi-step future fog index forecast for the next 24 hours.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLS = [
    "temperature_c",
    "dew_point_c",
    "dew_point_depression_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "pm25_ugm3",
    "solar_radiation_wm2"
]

class FogLSTM(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_COLS), hidden_dim: int = 64, num_layers: int = 2, forecast_horizon: int = 24):
        super(FogLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Immediate step classification head (Fog vs No-Fog)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Immediate step regression head (Fog Index 0-100)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Multi-step future forecast trajectory head (24-hour fog index curve)
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_horizon)
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Use final time step hidden representation
        last_hidden = lstm_out[:, -1, :]
        
        cls_logits = self.cls_head(last_hidden).squeeze(-1)
        reg_pred = self.reg_head(last_hidden).squeeze(-1)
        horizon_preds = self.forecast_head(last_hidden) # [batch_size, forecast_horizon]
        
        return cls_logits, reg_pred, horizon_preds

class TimeSeriesFogDataset(Dataset):
    def __init__(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, seq_len: int = 24, horizon: int = 24):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_samples = len(X) - seq_len - horizon + 1
        
    def __len__(self):
        return max(0, self.num_samples)
        
    def __getitem__(self, idx):
        # Sequence window
        seq_x = self.X[idx : idx + self.seq_len]
        # Target for immediate next step
        target_cls = self.y_cls[idx + self.seq_len]
        target_reg = self.y_reg[idx + self.seq_len]
        # Multi-step horizon target curve
        target_horizon = self.y_reg[idx + self.seq_len : idx + self.seq_len + self.horizon]
        
        return seq_x, target_cls, target_reg, target_horizon

def train_lstm_model(csv_path: str, save_path: str, epochs: int = 12, batch_size: int = 32, seq_len: int = 24, horizon: int = 24):
    df = pd.read_csv(csv_path)
    
    # Calculate depression if missing
    if "dew_point_depression_c" not in df.columns:
        df["dew_point_depression_c"] = df["temperature_c"] - df["dew_point_c"]
        
    X_raw = df[FEATURE_COLS].values
    y_cls_raw = df["fog_label"].values
    y_reg_raw = df["fog_index"].values
    
    # Split train/val by chronological split (essential for time-series)
    train_size = int(len(df) * 0.8)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_raw[:train_size])
    X_val_scaled = scaler.transform(X_raw[train_size:])
    
    train_dataset = TimeSeriesFogDataset(
        X_train_scaled, y_cls_raw[:train_size], y_reg_raw[:train_size],
        seq_len=seq_len, horizon=horizon
    )
    val_dataset = TimeSeriesFogDataset(
        X_val_scaled, y_cls_raw[train_size:], y_reg_raw[train_size:],
        seq_len=seq_len, horizon=horizon
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FogLSTM(input_dim=len(FEATURE_COLS), hidden_dim=64, num_layers=2, forecast_horizon=horizon).to(device)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print(f"Training FogLSTM on {device} for {epochs} epochs...")
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, b_cls, b_reg, b_horiz in train_loader:
            batch_x = batch_x.to(device)
            b_cls = b_cls.to(device)
            b_reg = b_reg.to(device)
            b_horiz = b_horiz.to(device)
            
            optimizer.zero_grad()
            cls_out, reg_out, horiz_out = model(batch_x)
            
            l_cls = bce_loss_fn(cls_out, b_cls)
            l_reg = mse_loss_fn(reg_out, b_reg) / 100.0
            l_horiz = mse_loss_fn(horiz_out, b_horiz) / 100.0
            
            loss = l_cls + 0.1 * l_reg + 0.1 * l_horiz
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            
        train_loss = total_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        mae_acc = 0.0
        
        with torch.no_grad():
            for batch_x, b_cls, b_reg, b_horiz in val_loader:
                batch_x = batch_x.to(device)
                b_cls = b_cls.to(device)
                b_reg = b_reg.to(device)
                b_horiz = b_horiz.to(device)
                
                cls_out, reg_out, horiz_out = model(batch_x)
                l_cls = bce_loss_fn(cls_out, b_cls)
                l_reg = mse_loss_fn(reg_out, b_reg) / 100.0
                l_horiz = mse_loss_fn(horiz_out, b_horiz) / 100.0
                val_loss += (l_cls + 0.1 * l_reg + 0.1 * l_horiz).item() * batch_x.size(0)
                
                prob = torch.sigmoid(cls_out)
                pred_label = (prob >= 0.5).float()
                correct += (pred_label == b_cls).sum().item()
                total += b_cls.size(0)
                mae_acc += torch.abs(reg_out - b_reg).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = correct / max(1, total)
        val_mae = mae_acc / max(1, total)
        
        if epoch % 3 == 0 or epoch == epochs:
            print(f"LSTM Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val MAE: {val_mae:.2f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            checkpoint = {
                "model_state": model.state_dict(),
                "scaler": scaler,
                "feature_cols": FEATURE_COLS,
                "seq_len": seq_len,
                "horizon": horizon,
                "metrics": {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_mae": val_mae
                }
            }
            torch.save(checkpoint, save_path)
            
    print(f"LSTM model saved successfully to {save_path}")
    return {"val_acc": val_acc, "val_mae": val_mae}

def forecast_next_24h(recent_sequence_df: pd.DataFrame, model_path: str) -> dict:
    """
    Given a DataFrame with at least 24 recent hourly sensor observations,
    forecasts immediate fog probability, fog index, and 24-hour horizon trajectory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    seq_len = checkpoint["seq_len"]
    
    df = recent_sequence_df.copy()
    if "dew_point_depression_c" not in df.columns:
        df["dew_point_depression_c"] = df["temperature_c"] - df["dew_point_c"]
        
    X_tail = df[feature_cols].tail(seq_len).values
    if len(X_tail) < seq_len:
        # Pad with repeats if not enough rows
        pad_len = seq_len - len(X_tail)
        pad = np.repeat(X_tail[:1], pad_len, axis=0)
        X_tail = np.vstack([pad, X_tail])
        
    X_scaled = scaler.transform(X_tail)
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    model = FogLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, forecast_horizon=checkpoint["horizon"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    with torch.no_grad():
        cls_logits, reg_pred, horizon_preds = model(tensor)
        prob = float(torch.sigmoid(cls_logits).item())
        fog_idx = float(torch.clamp(reg_pred, 0.0, 100.0).item())
        horizon_curve = torch.clamp(horizon_preds, 0.0, 100.0).squeeze(0).tolist()
        
    return {
        "current_fog_probability": round(prob, 4),
        "current_fog_label": 1 if prob >= 0.5 else 0,
        "current_fog_index": round(fog_idx, 2),
        "forecast_horizon_hours": [i + 1 for i in range(len(horizon_curve))],
        "forecast_fog_index": [round(val, 2) for val in horizon_curve]
    }
