import pandas as pd
import numpy as np
import os

# Get project root folder (parent of src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    # If the relative path exists from current working directory, use it
    if os.path.exists(rel_path):
        return rel_path
    # Otherwise, resolve from the project root
    return os.path.join(PROJECT_ROOT, rel_path)

def clean_weather_data(filepath="data/raw/weather_daily.csv"):
    path = get_path(filepath)
    df = pd.read_csv(path, na_values=["NA", ""])
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Handle missing rainfall (2026-03-08) - interpolate
    df['rainfall_mm'] = df['rainfall_mm'].interpolate(method='linear')
    
    # 2. Handle temperature outlier (2026-03-14: 45.80°C)
    outlier_idx = df[df['temperature_c'] > 40.0].index
    for idx in outlier_idx:
        prev_val = df.loc[idx - 1, 'temperature_c']
        next_val = df.loc[idx + 1, 'temperature_c']
        df.loc[idx, 'temperature_c'] = (prev_val + next_val) / 2.0
        
    # 3. Handle missing humidity (2026-03-21) - interpolate
    df['humidity_pct'] = df['humidity_pct'].interpolate(method='linear')
    
    return df

def clean_soil_data(filepath="data/raw/soil_sensor_data.csv"):
    path = get_path(filepath)
    df = pd.read_csv(path, na_values=["NA", ""])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by zone and timestamp to ensure correct sequence for interpolation
    df = df.sort_values(by=['zone_id', 'timestamp']).reset_index(drop=True)
    
    # 1. Soil moisture: missing value on 2026-03-06 (Zone_B) and outlier (8.50%) on 2026-03-25 (Zone_B)
    df.loc[(df['zone_id'] == 'Zone_B') & (df['soil_moisture_pct'] < 10.0), 'soil_moisture_pct'] = np.nan
    df['soil_moisture_pct'] = df.groupby('zone_id')['soil_moisture_pct'].transform(lambda x: x.interpolate(method='linear'))
    
    # 2. Tank level: outlier (9900 liters) on 2026-03-14 (Zone_C)
    df.loc[df['tank_level_liters'] > 6000.0, 'tank_level_liters'] = np.nan
    df['tank_level_liters'] = df.groupby('zone_id')['tank_level_liters'].transform(lambda x: x.interpolate(method='linear'))
    
    # 3. Pump flow anomaly: 2026-03-21 (Zone_B)
    zone_b_running_flow = df[(df['zone_id'] == 'Zone_B') & (df['pump_flow_lpm'] > 5.0)]['pump_flow_lpm'].median()
    df.loc[(df['zone_id'] == 'Zone_B') & (df['sensor_status'] == 'CHECK') & (df['pump_flow_lpm'] == 0.0), 'pump_flow_lpm'] = zone_b_running_flow
    
    return df

def generate_cleaned_dataset(weather_path="data/raw/weather_daily.csv", soil_path="data/raw/soil_sensor_data.csv"):
    df_weather = clean_weather_data(weather_path)
    df_soil = clean_soil_data(soil_path)
    
    # Extract date for merging
    df_soil['date'] = df_soil['timestamp'].dt.date
    df_weather['date_only'] = df_weather['date'].dt.date
    
    # Merge on date and date_only
    df_merged = pd.merge(df_soil, df_weather, left_on='date', right_on='date_only', suffixes=('_soil', '_weather'))
    
    # Drop intermediate date columns
    df_merged = df_merged.drop(columns=['date_soil', 'date_only', 'date_weather'])
    
    cols = ['timestamp', 'zone_id', 'soil_moisture_pct', 'tank_level_liters', 'pump_flow_lpm', 
            'pump_power_watts', 'sensor_status', 'rainfall_mm', 'temperature_c', 'humidity_pct', 
            'wind_speed_mps', 'solar_index']
    df_merged = df_merged[cols]
    
    # Write to processed folder
    out_path = get_path("data/processed/cleaned_irrigation_dataset.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_merged.to_csv(out_path, index=False)
    print(f"Generated cleaned merged dataset with {len(df_merged)} rows at {out_path}.")
    return df_merged

if __name__ == "__main__":
    generate_cleaned_dataset()
