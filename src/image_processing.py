"""
Image Processing Utilities for Fog Detection and Atmospheric Attenuation.
Implements:
1. Dark Channel Prior (DCP) for haze/fog estimation (He et al., IEEE TPAMI).
2. Atmospheric Light Estimation and Transmission Map.
3. Contrast & Sharpness features (RMS contrast, Laplacian variance).
"""

import numpy as np
from PIL import Image

def get_dark_channel(img_rgb: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Computes dark channel of an image:
    J^dark(x) = min_{y in Omega(x)} (min_{c in {r,g,b}} I^c(y))
    In foggy regions, the dark channel has high intensity due to atmospheric airlight.
    """
    # Min across color channels
    min_channel = np.min(img_rgb, axis=2).astype(np.float32) / 255.0
    
    # 2D min filter over patch_size x patch_size
    pad_h = patch_size // 2
    pad_w = patch_size // 2
    padded = np.pad(min_channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    dark_channel = np.zeros_like(min_channel)
    h, w = min_channel.shape
    for i in range(h):
        for j in range(w):
            dark_channel[i, j] = np.min(padded[i:i + patch_size, j:j + patch_size])
            
    return dark_channel

def estimate_atmospheric_light(img_rgb: np.ndarray, dark_channel: np.ndarray, top_pct: float = 0.001) -> np.ndarray:
    """
    Estimates global atmospheric light (A) from the top brightest pixels in the dark channel.
    """
    h, w = dark_channel.shape
    flat_dark = dark_channel.flatten()
    flat_img = img_rgb.reshape(-1, 3).astype(np.float32)
    
    num_top = max(1, int(h * w * top_pct))
    top_indices = np.argsort(flat_dark)[-num_top:]
    
    # Select pixel with highest intensity among the dark channel candidates
    brightest_candidate = flat_img[top_indices]
    intensity = np.mean(brightest_candidate, axis=1)
    best_idx = np.argmax(intensity)
    
    atm_light = brightest_candidate[best_idx] / 255.0
    # Avoid zero division
    atm_light = np.maximum(atm_light, 0.1)
    return atm_light

def estimate_transmission_map(img_rgb: np.ndarray, atm_light: np.ndarray, omega: float = 0.95, patch_size: int = 15) -> np.ndarray:
    """
    Estimates optical transmission map t(x) = 1 - omega * min_c(min_patch(I^c / A^c)).
    Low transmission = dense fog. High transmission = clear atmosphere.
    """
    norm_img = (img_rgb.astype(np.float32) / 255.0) / atm_light[None, None, :]
    dark_norm = get_dark_channel((norm_img * 255.0).astype(np.uint8), patch_size=patch_size)
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.05, 1.0)

def extract_image_features(image: Image.Image) -> dict:
    """
    Extracts photometric & fog-related handcrafted features from an image:
    - mean brightness
    - contrast (standard deviation of grayscale)
    - dark channel mean (directly correlated with fog density)
    - mean transmission (inversely correlated with fog density)
    - color saturation
    """
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    
    # Resize small for fast heuristic feature calculation
    small_img = np.array(image.resize((128, 128)).convert("RGB"))
    gray = np.mean(small_img, axis=2)
    
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Dark channel prior
    dark_channel = get_dark_channel(small_img, patch_size=9)
    dcp_mean = float(np.mean(dark_channel))
    
    atm_light = estimate_atmospheric_light(small_img, dark_channel)
    transmission = estimate_transmission_map(small_img, atm_light, patch_size=9)
    trans_mean = float(np.mean(transmission))
    
    # Fog index estimation from DCP & Transmission:
    # High dcp and low transmission => high fog index
    heuristic_fog_index = float(np.clip((dcp_mean * 75.0 + (1.0 - trans_mean) * 45.0), 0.0, 100.0))
    
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "dark_channel_mean": round(dcp_mean, 4),
        "transmission_mean": round(trans_mean, 4),
        "heuristic_fog_index": round(heuristic_fog_index, 2),
        "atmospheric_light": [round(float(v), 3) for v in atm_light]
    }
