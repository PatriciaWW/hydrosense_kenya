import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulation import simulate_soil_moisture

def optimize_irrigation_strict_min(days, S0, R, ET, FC, S_min, drainage_coeff, I_max=10.0):
    """
    Strict Minimum Strategy:
    On each day, if the soil moisture would drop below S_min, we apply just enough
    irrigation to bring the soil moisture exactly to S_min at the end of the day.
    """
    S = S0
    I = []
    
    # We can approximate this by checking the moisture after 1 day without irrigation
    for day in range(days):
        R_val = R[day]
        ET_val = ET[day]
        
        # Estimate next day moisture with I = 0
        s_next_no_i = simulate_soil_moisture(1, S, [R_val], [0.0], [ET_val], FC, drainage_coeff, method='rk4')[0]
        
        if s_next_no_i >= S_min:
            # No irrigation needed
            irr_val = 0.0
            S_next = s_next_no_i
        else:
            # Irrigation needed to reach exactly S_min
            # Using discrete approximation: S_next = S_temp - D_t
            # Since S_min < FC, drainage will be 0.0 at S_next = S_min.
            # Thus, S_min = S + R - ET + I => I = S_min - S - R + ET
            irr_val = S_min - S - R_val + ET_val
            irr_val = max(0.0, min(I_max, irr_val))
            S_next = simulate_soil_moisture(1, S, [R_val], [irr_val], [ET_val], FC, drainage_coeff, method='rk4')[0]
            
        I.append(irr_val)
        S = S_next
        
    return np.array(I)

def optimize_irrigation_trigger_target(days, S0, R, ET, FC, S_min, S_target, drainage_coeff, I_max=10.0):
    """
    Trigger-based Target Strategy (MAD):
    If the soil moisture at the end of the day would fall below S_min, we apply
    irrigation to bring the soil moisture to S_target. Otherwise, no irrigation.
    """
    S = S0
    I = []
    
    for day in range(days):
        R_val = R[day]
        ET_val = ET[day]
        
        # Estimate next day moisture with I = 0
        s_next_no_i = simulate_soil_moisture(1, S, [R_val], [0.0], [ET_val], FC, drainage_coeff, method='rk4')[0]
        
        if s_next_no_i >= S_min:
            irr_val = 0.0
            S_next = s_next_no_i
        else:
            # We want the final moisture to be S_target.
            # If S_target > FC, drainage is active. We can use a linear approximation:
            # S_target = S_temp - D_t = S_temp - drainage_coeff * (S_temp - FC)
            # S_temp = (S_target - drainage_coeff * FC) / (1 - drainage_coeff)
            # I = S_temp - S - R + ET
            if S_target > FC:
                s_temp_target = (S_target - drainage_coeff * FC) / (1.0 - drainage_coeff)
            else:
                s_temp_target = S_target
            
            irr_val = s_temp_target - S - R_val + ET_val
            irr_val = max(0.0, min(I_max, irr_val))
            S_next = simulate_soil_moisture(1, S, [R_val], [irr_val], [ET_val], FC, drainage_coeff, method='rk4')[0]
            
        I.append(irr_val)
        S = S_next
        
    return np.array(I)
