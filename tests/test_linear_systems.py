import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import gaussian_elimination

def test_gaussian_elimination_standard():
    # 3x3 linear system
    # 2x + y - z = 8
    # -3x - y + 2z = -11
    # -2x + y + 2z = -3
    # Solution: x = 2, y = 3, z = -1
    A = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0]
    ])
    b = np.array([8.0, -11.0, -3.0])
    
    x = gaussian_elimination(A, b)
    expected = np.array([2.0, 3.0, -1.0])
    assert np.allclose(x, expected, atol=1e-12)

def test_gaussian_elimination_singular():
    # Singular system (row 2 is twice row 1)
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])
    b = np.array([5.0, 10.0])
    
    with pytest.raises(ValueError):
        gaussian_elimination(A, b)
