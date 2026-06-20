import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from numerical_methods import bisection, newton_raphson, secant

# Test function: f(x) = x^2 - 4, roots are x = 2 and x = -2
def f_test(x):
    return x**2 - 4.0

def df_test(x):
    return 2.0 * x

def test_bisection():
    root, iters, err, history = bisection(f_test, 0.0, 5.0, tol=1e-5)
    assert abs(root - 2.0) < 1e-5
    assert iters > 0
    assert len(history) == iters

def test_newton_raphson():
    root, iters, err, history = newton_raphson(f_test, df_test, 3.0, tol=1e-5)
    assert abs(root - 2.0) < 1e-5
    assert iters > 0
    assert len(history) == iters

def test_secant():
    root, iters, err, history = secant(f_test, 0.0, 3.0, tol=1e-5)
    assert abs(root - 2.0) < 1e-5
    assert iters > 0
    assert len(history) == iters

def test_bisection_invalid_bracket():
    with pytest.raises(ValueError):
        # f(0) = -4, f(1) = -3, both have negative sign
        bisection(f_test, 0.0, 1.0)
