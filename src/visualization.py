import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style for premium aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Tailored color palette
COLORS = {
    'Zone_A': '#e74c3c', # Red/Tomato
    'Zone_B': '#2ecc71', # Green/Kale
    'Zone_C': '#f1c40f', # Yellow/Maize
    'Rainfall': '#3498db', # Blue
    'Temp': '#e67e22', # Orange
    'Grid': '#f0f0f0'
}

def plot_rainfall_temp(df, save_path="reports/level_4_rainfall_temp.png"):
    """Plot Rainfall and Temperature over time with dual y-axes."""
    df_unique = df[['timestamp', 'rainfall_mm', 'temperature_c']].drop_duplicates().sort_values('timestamp')
    dates = df_unique['timestamp'].dt.date
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Rainfall on left y-axis
    ax1.bar(dates, df_unique['rainfall_mm'], color=COLORS['Rainfall'], alpha=0.7, label='Rainfall (mm)', width=0.6)
    ax1.set_xlabel('Date', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Rainfall (mm)', color=COLORS['Rainfall'], fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor=COLORS['Rainfall'])
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Temperature on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(dates, df_unique['temperature_c'], color=COLORS['Temp'], marker='o', linewidth=2, label='Temperature (°C)')
    ax2.set_ylabel('Temperature (°C)', color=COLORS['Temp'], fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', labelcolor=COLORS['Temp'])
    ax2.grid(False) # Prevent overlapping grid lines
    
    plt.title('Daily Rainfall and Mean Temperature (March 2026)', fontsize=14, fontweight='bold', pad=15)
    fig.autofmt_xdate()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_soil_moisture_profiles(df, params_df, save_path="reports/level_4_soil_moisture.png"):
    """Plot soil moisture profiles over time for Zones A, B, and C with threshold lines."""
    plt.figure(figsize=(12, 6))
    
    for zone in ['Zone_A', 'Zone_B', 'Zone_C']:
        zone_df = df[df['zone_id'] == zone].sort_values('timestamp')
        dates = zone_df['timestamp'].dt.date
        
        # Plot measured soil moisture
        plt.plot(dates, zone_df['soil_moisture_pct'], marker='.', linewidth=2, color=COLORS[zone], label=f"{zone} Measured")
        
        # Get thresholds
        p = params_df[params_df['zone_id'] == zone].iloc[0]
        plt.axhline(p['min_moisture_pct'], color=COLORS[zone], linestyle='--', alpha=0.5, 
                    label=f"{zone} Min ({p['min_moisture_pct']}%)")
        plt.axhline(p['target_moisture_pct'], color=COLORS[zone], linestyle=':', alpha=0.5, 
                    label=f"{zone} Target ({p['target_moisture_pct']}%)")

    plt.title('Daily Noon Soil Moisture Profiles by Zone (March 2026)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontweight='bold', labelpad=10)
    plt.ylabel('Soil Moisture (%)', fontweight='bold', labelpad=10)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_tank_level_degradation(df, save_path="reports/level_4_tank_degradation.png"):
    """Plot water tank level degradation over time for each zone."""
    plt.figure(figsize=(12, 6))
    
    for zone in ['Zone_A', 'Zone_B', 'Zone_C']:
        zone_df = df[df['zone_id'] == zone].sort_values('timestamp')
        dates = zone_df['timestamp'].dt.date
        plt.plot(dates, zone_df['tank_level_liters'], marker='s', markersize=4, linewidth=2, color=COLORS[zone], label=zone)
        
    plt.title('Irrigation Water Tank Level Degradation (March 2026)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Date', fontweight='bold', labelpad=10)
    plt.ylabel('Tank Volume (Liters)', fontweight='bold', labelpad=10)
    plt.legend(frameon=True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pump_flow_vs_power(df, save_path="reports/level_4_pump_analysis.png"):
    """Plot pump flow rate vs pump power to show relationship and check for anomalies."""
    plt.figure(figsize=(10, 6))
    
    # Filter only when pump was active (flow > 0.1 LPM)
    active_df = df[df['pump_flow_lpm'] > 0.1]
    
    sns.scatterplot(data=active_df, x='pump_power_watts', y='pump_flow_lpm', hue='zone_id', 
                    palette=[COLORS['Zone_A'], COLORS['Zone_B'], COLORS['Zone_C']], s=80, alpha=0.8)
    
    # Draw fit line
    if len(active_df) > 1:
        coef = np.polyfit(active_df['pump_power_watts'], active_df['pump_flow_lpm'], 1)
        poly1d_fn = np.poly1d(coef)
        x_vals = np.linspace(active_df['pump_power_watts'].min() - 10, active_df['pump_power_watts'].max() + 10, 100)
        plt.plot(x_vals, poly1d_fn(x_vals), '--', color='#7f8c8d', alpha=0.8, label=f'Linear Trend')

    plt.title('Pump Flow Rate vs Power Consumption', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Pump Power (Watts)', fontweight='bold', labelpad=10)
    plt.ylabel('Pump Flow Rate (Liters/minute)', fontweight='bold', labelpad=10)
    plt.legend(frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_et_vs_solar(df, save_path="reports/level_4_et_solar.png"):
    """Plot calculated Potential Evapotranspiration vs Solar Index."""
    # Compute ET using formula
    df_unique = df[['timestamp', 'temperature_c', 'wind_speed_mps', 'solar_index', 'humidity_pct']].drop_duplicates().copy()
    
    # Empirical ET formula
    df_unique['et_calculated'] = 0.12 * df_unique['temperature_c'] + 0.35 * df_unique['wind_speed_mps'] + 2.4 * df_unique['solar_index'] - 0.025 * df_unique['humidity_pct']
    df_unique['et_calculated'] = df_unique['et_calculated'].clip(lower=0.0)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_unique['solar_index'], df_unique['et_calculated'], c=df_unique['temperature_c'], 
                          cmap='YlOrRd', s=80, edgecolors='grey', alpha=0.8)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Air Temperature (°C)', fontweight='bold', labelpad=10)
    
    plt.title('Calculated Daily Evapotranspiration vs. Solar Radiation Index', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Solar Radiation Index (Dimensionless, 0-1)', fontweight='bold', labelpad=10)
    plt.ylabel('Calculated ET (mm/day)', fontweight='bold', labelpad=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
