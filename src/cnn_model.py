"""
PyTorch Convolutional Neural Network for Fog Detection and Index Regression.
Works on Satellite and CCTV camera feeds.
Multi-task head:
  1. Classification Head -> Fog vs No-Fog (Logits / Probability)
  2. Regression Head -> Fog Index (0 to 100 continuous score)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np

def transform_image(img: Image.Image, size=(128, 128)) -> torch.Tensor:
    """
    Standardizes image without torchvision dependency:
    Resize, normalize with standard ImageNet mean and std.
    """
    resized = img.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    arr = np.array(resized).astype(np.float32) / 255.0  # [H, W, 3]
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    # Transpose to [C, H, W]
    tensor = torch.tensor(norm.transpose(2, 0, 1), dtype=torch.float32)
    return tensor

class FogCNN(nn.Module):
    def __init__(self):
        super(FogCNN, self).__init__()
        
        # Convolutional Feature Extractor
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128 -> 64
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 64 -> 32
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32 -> 16
            
            # Block 4: 128 -> 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Dense bottleneck
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Head 1: Binary Classification (Fog vs No-Fog)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Logits for BCEWithLogitsLoss
        )
        
        # Head 2: Regression (Continuous Fog Index 0 to 100)
        self.regressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        feat = self.features(x)
        shared = self.fc_shared(feat)
        cls_logits = self.classifier(shared).squeeze(-1)
        reg_output = self.regressor(shared).squeeze(-1)
        return cls_logits, reg_output

class FogImageDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, base_dir: str, mode: str = "cctv"):
        self.meta_df = meta_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.mode = mode
        
    def __len__(self):
        return len(self.meta_df)
        
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        rel_path = row["cctv_path"] if self.mode == "cctv" else row["satellite_path"]
        img_path = os.path.join(self.base_dir, rel_path)
        
        img = Image.open(img_path).convert("RGB")
        tensor = transform_image(img, size=(128, 128))
            
        label = float(row["fog_label"])
        fog_index = float(row["fog_index"])
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(fog_index, dtype=torch.float32)

def train_cnn(meta_csv: str, base_dir: str, save_path: str, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3):
    df = pd.read_csv(meta_csv)
    # Split train/val
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # CCTV & Satellite mixed dataset
    cctv_train = FogImageDataset(train_df, base_dir, mode="cctv")
    sat_train = FogImageDataset(train_df, base_dir, mode="satellite")
    combined_train = torch.utils.data.ConcatDataset([cctv_train, sat_train])
    
    cctv_val = FogImageDataset(val_df, base_dir, mode="cctv")
    sat_val = FogImageDataset(val_df, base_dir, mode="satellite")
    combined_val = torch.utils.data.ConcatDataset([cctv_val, sat_val])
    
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FogCNN().to(device)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"Training FogCNN on {device} for {epochs} epochs...")
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for imgs, labels, indices in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            indices = indices.to(device)
            
            optimizer.zero_grad()
            cls_logits, reg_preds = model(imgs)
            
            loss_cls = bce_loss_fn(cls_logits, labels)
            loss_reg = mse_loss_fn(reg_preds, indices) / 100.0 # scale MSE
            loss = loss_cls + 0.1 * loss_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            
        train_loss = total_loss / len(combined_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        abs_err = 0.0
        
        with torch.no_grad():
            for imgs, labels, indices in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                indices = indices.to(device)
                
                cls_logits, reg_preds = model(imgs)
                loss_cls = bce_loss_fn(cls_logits, labels)
                loss_reg = mse_loss_fn(reg_preds, indices) / 100.0
                val_loss += (loss_cls + 0.1 * loss_reg).item() * imgs.size(0)
                
                probs = torch.sigmoid(cls_logits)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                abs_err += torch.abs(reg_preds - indices).sum().item()
                
        val_loss /= len(combined_val)
        val_acc = correct / total
        val_mae = abs_err / total
        
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}% - Val MAE: {val_mae:.2f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    print(f"CNN model successfully saved to {save_path}")
    return {"val_acc": val_acc, "val_mae": val_mae}

def predict_image(image: Image.Image, model_path: str = None) -> dict:
    """
    Inference helper on a single image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FogCNN().to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    tensor = transform_image(image, size=(128, 128)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cls_logits, reg_pred = model(tensor)
        prob = float(torch.sigmoid(cls_logits).item())
        fog_idx = float(torch.clamp(reg_pred, 0.0, 100.0).item())
        
    return {
        "fog_probability": round(prob, 4),
        "fog_label": 1 if prob >= 0.5 else 0,
        "fog_index": round(fog_idx, 2),
        "status": "Foggy" if prob >= 0.5 else "Clear"
    }
