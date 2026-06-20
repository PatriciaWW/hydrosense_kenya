import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import trapezoidal_rule, simpson_rule

def test_trapezoidal_constant():
    # Integrate y = 2 over 3 points, dx = 1.0 (2 intervals)
    # Analytical integral: 2 * 2 = 4
    y = np.array([2.0, 2.0, 2.0])
    integral = trapezoidal_rule(y, dx=1.0)
    assert abs(integral - 4.0) < 1e-12

def test_simpson_constant():
    y = np.array([2.0, 2.0, 2.0])
    integral = simpson_rule(y, dx=1.0)
    assert abs(integral - 4.0) < 1e-12

def test_simpson_quadratic():
    # Integrate y = x^2 from x=0 to x=2, dx=1.0 (3 points: y=[0, 1, 4])
    # Analytical: [x^3/3]_0^2 = 8/3 = 2.666666...
    # Simpson's 1/3 rule integrates quadratics exactly.
    y = np.array([0.0, 1.0, 4.0])
    integral = simpson_rule(y, dx=1.0)
    assert abs(integral - 8.0/3.0) < 1e-12

def test_simpson_even_elements():
    # Integrate y = x^2 from x=0 to x=3, dx=1.0 (4 points: y=[0, 1, 4, 9])
    # Elements = 4 (even), intervals = 3 (odd)
    # Our function applies Simpson on [0, 1, 4] and Trapezoidal on [4, 9]
    # Simpson part: 8/3 = 2.6667
    # Trapezoidal part: 0.5 * (4 + 9) * 1.0 = 6.5
    # Total expected: 2.6667 + 6.5 = 9.166667
    y = np.array([0.0, 1.0, 4.0, 9.0])
    integral = simpson_rule(y, dx=1.0)
    assert abs(integral - (8.0/3.0 + 6.5)) < 1e-12
