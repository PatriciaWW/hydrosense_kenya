import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from simulation import simulate_soil_moisture

def test_simulation_zero_forcing():
    # If initial moisture is at field capacity, and all forcings are zero,
    # soil moisture should remain exactly constant at field capacity (since drainage is 0).
    days = 5
    S0 = 40.0
    R = [0.0] * days
    I = [0.0] * days
    ET = [0.0] * days
    FC = 40.0
    drainage_coeff = 0.18
    
    # Run simulation with Euler
    profile_euler = simulate_soil_moisture(days, S0, R, I, ET, FC, drainage_coeff, method='euler')
    assert np.allclose(profile_euler, S0, atol=1e-12)
    
    # Run simulation with RK4
    profile_rk4 = simulate_soil_moisture(days, S0, R, I, ET, FC, drainage_coeff, method='rk4')
    assert np.allclose(profile_rk4, S0, atol=1e-12)

def test_simulation_drainage():
    # If starting moisture is above field capacity (45% vs 40%),
    # it should drain over time towards field capacity even with zero forcing.
    days = 2
    S0 = 45.0
    R = [0.0] * days
    I = [0.0] * days
    ET = [0.0] * days
    FC = 40.0
    drainage_coeff = 0.2
    
    profile = simulate_soil_moisture(days, S0, R, I, ET, FC, drainage_coeff, method='rk4')
    
    # Moisture should decrease
    assert profile[0] < S0
    assert profile[1] < profile[0]
    # And remain above or equal to FC
    assert profile[1] >= FC
