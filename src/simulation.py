import numpy as np

def dS_dt(S, t, R_val, I_val, ET_val, FC, drainage_coeff):
    """
    Derivative of soil moisture with respect to time (in days).
    dS/dt = R(t) + I(t) - ET(t) - D(t, S)
    where D(t, S) = drainage_coeff * max(0, S - FC)
    """
    drainage = drainage_coeff * max(0.0, S - FC)
    return R_val + I_val - ET_val - drainage

def euler_step(S, t, dt, R_val, I_val, ET_val, FC, drainage_coeff):
    """Performs a single step of the Euler method."""
    deriv = dS_dt(S, t, R_val, I_val, ET_val, FC, drainage_coeff)
    return S + dt * deriv

def rk4_step(S, t, dt, R_val, I_val, ET_val, FC, drainage_coeff):
    """Performs a single step of the 4th-order Runge-Kutta method."""
    k1 = dS_dt(S, t, R_val, I_val, ET_val, FC, drainage_coeff)
    k2 = dS_dt(S + 0.5 * dt * k1, t + 0.5 * dt, R_val, I_val, ET_val, FC, drainage_coeff)
    k3 = dS_dt(S + 0.5 * dt * k2, t + 0.5 * dt, R_val, I_val, ET_val, FC, drainage_coeff)
    k4 = dS_dt(S + dt * k3, t + dt, R_val, I_val, ET_val, FC, drainage_coeff)
    return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def simulate_soil_moisture(days, S0, R, I, ET, FC, drainage_coeff, method='rk4', steps_per_day=10):
    """
    Simulates soil moisture over a sequence of days.
    R, I, ET are lists/arrays of daily values of length 'days'.
    Returns: soil moisture profile (array of daily values at the end of each day).
    """
    S = S0
    profile = []
    dt = 1.0 / steps_per_day
    
    for day in range(days):
        # Retrieve daily forcing values
        R_val = R[day]
        I_val = I[day]
        ET_val = ET[day]
        
        # Integrate over 1 day in sub-steps of dt
        for step in range(steps_per_day):
            t = day + step * dt
            if method == 'euler':
                S = euler_step(S, t, dt, R_val, I_val, ET_val, FC, drainage_coeff)
            else:
                S = rk4_step(S, t, dt, R_val, I_val, ET_val, FC, drainage_coeff)
                
        profile.append(S)
        
    return np.array(profile)

def generate_rainfall_scenarios(n_scenarios, days, prob_rain, mean_rain):
    """
    Generates stochastic rainfall scenarios using a simple two-stage model:
    1. Decide if it rains using Bernoulli trial with probability 'prob_rain'.
    2. If it rains, sample rainfall amount from an Exponential distribution with mean 'mean_rain'.
    """
    scenarios = []
    for _ in range(n_scenarios):
        scenario = []
        for _ in range(days):
            if np.random.rand() < prob_rain:
                amount = np.random.exponential(scale=mean_rain)
                scenario.append(amount)
            else:
                scenario.append(0.0)
        scenarios.append(np.array(scenario))
    return scenarios
