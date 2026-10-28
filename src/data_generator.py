"""
Data generator for the Delhi NCR Fog Prediction & Monitoring System.
Generates realistic multi-modal datasets:
1. Hourly meteorological sensor time-series (Delhi Safdarjung & Palam observatories, Nov-Feb fog & smog season).
2. CCTV camera snapshots across key strategic transportation & urban nodes across Delhi NCR.
3. Satellite imagery (INSAT-3D multi-spectral channel simulation for Delhi & Indo-Gangetic fog corridor).
"""

import os
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def generate_sensor_dataset(n_samples: int = 4000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic hourly sensor data representing winter fog season in Delhi NCR.
    Physics modeled:
      - Dew Point Depression = Temp - DewPoint
      - High Relative Humidity (RH > 80%) + Low Wind Speed (< 2.0 m/s) + Extreme PM2.5/PM10 (smog/CCN) -> Dense Radiative Fog
      - Nocturnal boundary layer inversion in Delhi winter.
    """
    np.random.seed(random_seed)
    
    start_time = pd.Timestamp("2023-11-01 00:00:00")
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(n_samples)]
    
    records = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_year = ts.dayofyear
        
        # Diurnal temperature cycle in Delhi: lowest around 05:00-06:00, highest around 14:00-15:00
        diurnal_temp = -4.8 * np.cos(2 * np.pi * (hour - 5) / 24)
        seasonal_temp = 14.5 - 5.5 * np.sin(2 * np.pi * (day_of_year - 300) / 90) # coldest in late Dec/Jan (~4-9 C)
        noise_temp = np.random.normal(0, 1.1)
        temp = max(2.5, seasonal_temp + diurnal_temp + noise_temp)
        
        # Dew point is close to temperature during humid foggy nights in Delhi NCR
        dew_depression = np.random.exponential(scale=1.6 if (hour <= 8 or hour >= 21) else 4.2)
        dew_point = temp - max(0.2, dew_depression)
        
        # Relative Humidity via Magnus formula
        rh = 100 * np.exp((17.625 * dew_point) / (243.04 + dew_point)) / np.exp((17.625 * temp) / (243.04 + temp))
        rh = float(np.clip(rh + np.random.normal(0, 2.0), 30.0, 100.0))
        
        # Wind speed (m/s) - calm winds typical in Delhi winter inversions
        wind_speed = float(np.clip(
            np.random.gamma(shape=1.8, scale=0.7 if (hour <= 8 or hour >= 21) else 1.4),
            0.1, 9.0
        ))
        
        # Wind direction (degrees) - north-westerly prevailing winds
        wind_dir = float((np.random.normal(295, 30) + 360) % 360)
        
        # Atmospheric Pressure (hPa)
        pressure = float(1015.0 + 3.2 * np.cos(2 * np.pi * hour / 24) + np.random.normal(0, 0.9))
        
        # Particulate matter (Delhi winter high aerosol loading / CCN)
        pm25 = float(np.clip(np.random.lognormal(mean=5.1, sigma=0.48) * (1.35 if (hour <= 9 or hour >= 20) else 0.8), 35, 600))
        pm10 = float(pm25 * np.random.uniform(1.5, 2.2))
        
        # Solar radiation (W/m²)
        if 6 <= hour <= 18:
            solar_peak = 620 * np.sin(np.pi * (hour - 6) / 12)
            solar_radiation = float(max(0, solar_peak * (0.35 if rh > 85 else 0.92) + np.random.normal(0, 18)))
        else:
            solar_radiation = 0.0
            
        # Ground truth Fog Index (0 = clear, 100 = extreme zero-visibility dense fog)
        depression = temp - dew_point
        fog_score = (
            (100.0 - min(100.0, rh)) * -1.25 +
            max(0, 5.0 - depression) * 12.5 +
            max(0, 3.2 - wind_speed) * 8.5 +
            min(pm25, 400) * 0.08 +
            (12.0 if (hour <= 8 or hour >= 22) else 0.0)
        )
        fog_index = float(np.clip(fog_score + np.random.normal(0, 3.5), 0.0, 100.0))
        is_fog = 1 if fog_index >= 45.0 else 0
        
        # Visibility in meters (Koschmieder relationship approximation)
        visibility_m = float(np.clip(
            4500.0 * np.exp(-fog_index / 22.0) + np.random.normal(0, 35),
            40.0, 6000.0
        ))
        
        records.append({
            "timestamp": ts,
            "temperature_c": round(temp, 2),
            "dew_point_c": round(dew_point, 2),
            "dew_point_depression_c": round(depression, 2),
            "relative_humidity_pct": round(rh, 2),
            "wind_speed_ms": round(wind_speed, 2),
            "wind_direction_deg": round(wind_dir, 1),
            "pressure_hpa": round(pressure, 2),
            "pm25_ugm3": round(pm25, 2),
            "pm10_ugm3": round(pm10, 2),
            "solar_radiation_wm2": round(solar_radiation, 2),
            "fog_index": round(fog_index, 2),
            "visibility_m": round(visibility_m, 1),
            "fog_label": is_fog
        })
        
    df = pd.DataFrame(records)
    return df

def generate_synthetic_scene_image(scene_type: str, fog_level: float, size=(256, 256)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), color=(135, 206, 235) if scene_type == "cctv" else (20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    if scene_type == "cctv":
        # Sky gradient
        for y in range(h // 2):
            b_val = int(210 - y * 0.4)
            draw.line([(0, y), (w, y)], fill=(120, 175, max(50, b_val)))
        
        # Ground / Highway / Runway
        draw.rectangle([(0, h // 2), (w, h)], fill=(50, 75, 50))
        
        # Highway / Airport Runway strip
        road_pts = [(w * 0.2, h), (w * 0.45, h // 2), (w * 0.55, h // 2), (w * 0.88, h)]
        draw.polygon(road_pts, fill=(55, 55, 60))
        draw.line([(w * 0.5, h), (w * 0.5, h // 2)], fill=(240, 240, 100), width=2)
        
        # Urban skyline / Airport terminal structures
        draw.rectangle([(w * 0.1, h * 0.32), (w * 0.38, h * 0.58)], fill=(150, 155, 160))
        draw.rectangle([(w * 0.62, h * 0.28), (w * 0.92, h * 0.62)], fill=(170, 130, 110))
        for wx in range(int(w * 0.14), int(w * 0.36), 14):
            for wy in range(int(h * 0.35), int(h * 0.55), 14):
                draw.rectangle([(wx, wy), (wx + 7, wy + 7)], fill=(240, 230, 170))
                
        # Streetlights / Runway edge markers
        for lx in [w * 0.18, w * 0.82]:
            draw.line([(lx, h * 0.52), (lx, h * 0.85)], fill=(180, 180, 180), width=2)
            draw.ellipse([(lx - 3, h * 0.50), (lx + 3, h * 0.56)], fill=(255, 240, 150))
            
    else:  # Satellite Delhi NCR view (Yamuna river curve, urban sprawl)
        draw.rectangle([(0, 0), (w, h)], fill=(45, 65, 40))
        # Yamuna River curve
        river_pts = [(w * 0.5, 0), (w * 0.45, h * 0.3), (w * 0.52, h * 0.65), (w * 0.6, h)]
        for idx in range(len(river_pts) - 1):
            draw.line([river_pts[idx], river_pts[idx+1]], fill=(30, 55, 85), width=10)
        # Urban clustering (Delhi core, Noida, Gurugram)
        for _ in range(40):
            rx = np.random.randint(w * 0.2, w * 0.8)
            ry = np.random.randint(h * 0.2, h * 0.8)
            draw.rectangle([(rx, ry), (rx + 5, ry + 5)], fill=(155, 145, 135))
            
    # Atmospheric forward scattering & Koschmieder attenuation
    transmission = max(0.02, 1.0 - (fog_level ** 1.3) * 0.95)
    fog_color = (220, 225, 230)
    fog_layer = Image.new("RGB", (w, h), color=fog_color)
    alpha = 1.0 - transmission
    blended = Image.blend(img, fog_layer, alpha=alpha)
    
    if fog_level > 0.3:
        blended = blended.filter(ImageFilter.GaussianBlur(radius=fog_level * 2.2))
        
    return blended

def generate_image_dataset(output_dir: str, n_images: int = 150, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    cctv_dir = os.path.join(output_dir, "cctv")
    sat_dir = os.path.join(output_dir, "satellite")
    os.makedirs(cctv_dir, exist_ok=True)
    os.makedirs(sat_dir, exist_ok=True)
    
    locations = [
        "IGI_Airport_T3_Runway",
        "Anand_Vihar_ISBT_Hub",
        "India_Gate_Central_Vista",
        "Akshardham_Yamuna_Belt",
        "Dwarka_Expressway_Junction"
    ]
    
    records = []
    for i in range(n_images):
        fog_level = float(np.random.beta(a=0.9, b=1.1))
        fog_index = round(fog_level * 100.0, 2)
        fog_label = 1 if fog_index >= 45.0 else 0
        
        loc = np.random.choice(locations)
        cctv_filename = f"cctv_{loc}_{i:04d}_fog{int(fog_index)}.jpg"
        cctv_path = os.path.join(cctv_dir, cctv_filename)
        cctv_img = generate_synthetic_scene_image("cctv", fog_level)
        cctv_img.save(cctv_path, quality=90)
        
        sat_filename = f"sat_insat3d_delhi_{i:04d}_fog{int(fog_index)}.jpg"
        sat_path = os.path.join(sat_dir, sat_filename)
        sat_img = generate_synthetic_scene_image("satellite", fog_level)
        sat_img.save(sat_path, quality=90)
        
        records.append({
            "image_id": f"IMG_{i:04d}",
            "cctv_path": os.path.relpath(cctv_path, output_dir),
            "satellite_path": os.path.relpath(sat_path, output_dir),
            "location": loc,
            "fog_level": round(fog_level, 4),
            "fog_index": fog_index,
            "fog_label": fog_label
        })
        
    meta_df = pd.DataFrame(records)
    meta_df.to_csv(os.path.join(output_dir, "image_metadata.csv"), index=False)
    return meta_df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("[1/2] Generating Delhi NCR meteorological sensor dataset...")
    df_sensors = generate_sensor_dataset(n_samples=5000)
    sensor_path = os.path.join(raw_dir, "delhi_meteorological_sensors.csv")
    df_sensors.to_csv(sensor_path, index=False)
    print(f"Saved {len(df_sensors)} sensor records to {sensor_path}")
    
    print("[2/2] Generating Delhi NCR CCTV and Satellite imagery dataset...")
    df_images = generate_image_dataset(raw_dir, n_images=120)
    print(f"Saved {len(df_images)} image pairs to {raw_dir}")
