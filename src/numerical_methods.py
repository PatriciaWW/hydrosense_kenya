import numpy as np

# ==========================================
# 1. ROOT FINDING METHODS (From Scratch)
# ==========================================

def bisection(f, a, b, tol=1e-5, max_iter=100):
    """
    Finds the root of f(x) = 0 in interval [a, b] using the Bisection method.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs.")
        
    history = []
    for iteration in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        err = abs(b - a) / 2.0
        history.append((iteration, c, fc, err))
        
        if err < tol or abs(fc) < 1e-12:
            return c, iteration, err, history
            
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
    return c, max_iter, err, history


def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    """
    Finds the root of f(x) = 0 using the Newton-Raphson method.
    """
    x = x0
    history = []
    for iteration in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-12:
            raise ZeroDivisionError("Derivative is too close to zero in Newton-Raphson.")
            
        x_new = x - fx / dfx
        err = abs(x_new - x)
        history.append((iteration, x, fx, err))
        
        if err < tol or abs(f(x_new)) < 1e-12:
            return x_new, iteration, err, history
            
        x = x_new
        
    return x, max_iter, err, history


def secant(f, x0, x1, tol=1e-5, max_iter=100):
    """
    Finds the root of f(x) = 0 using the Secant method.
    """
    history = []
    for iteration in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)
        
        if abs(fx1 - fx0) < 1e-12:
            raise ZeroDivisionError("Denominator in Secant step is too close to zero.")
            
        # Secant step formula
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        err = abs(x_new - x1)
        history.append((iteration, x1, fx1, err))
        
        if err < tol or abs(f(x_new)) < 1e-12:
            return x_new, iteration, err, history
            
        x0 = x1
        x1 = x_new
        
    return x1, max_iter, err, history


# ==========================================
# 2. NUMERICAL DIFFERENTIATION
# ==========================================

def forward_difference(f, x, h=1e-5):
    """Computes the first derivative using forward difference."""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-5):
    """Computes the first derivative using backward difference."""
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h=1e-5):
    """Computes the first derivative using central difference."""
    return (f(x + h) - f(x - h)) / (2.0 * h)

def finite_differences_array(y, dx=1.0):
    """
    Computes finite differences for a discrete array of data points.
    Returns: forward_diff, backward_diff, central_diff arrays of the same shape.
    """
    n = len(y)
    fwd = np.zeros(n)
    bwd = np.zeros(n)
    cnt = np.zeros(n)
    
    # Forward difference
    fwd[:-1] = (y[1:] - y[:-1]) / dx
    fwd[-1] = np.nan # Last point cannot be forward-differenced
    
    # Backward difference
    bwd[1:] = (y[1:] - y[:-1]) / dx
    bwd[0] = np.nan # First point cannot be backward-differenced
    
    # Central difference
    cnt[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    cnt[0] = np.nan
    cnt[-1] = np.nan
    
    return fwd, bwd, cnt


# ==========================================
# 3. NUMERICAL INTEGRATION
# ==========================================

def trapezoidal_rule(y, dx=1.0):
    """
    Integrates the array y over spacing dx using the Trapezoidal rule.
    """
    n = len(y)
    if n < 2:
        return 0.0
    return dx * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def simpson_rule(y, dx=1.0):
    """
    Integrates the array y over spacing dx using Simpson's 1/3 rule.
    If the number of elements is even (odd number of intervals),
    Simpson's rule is applied to all but the last interval,
    and the Trapezoidal rule is applied to the last interval.
    """
    n = len(y)
    if n < 2:
        return 0.0
    if n == 2:
        return trapezoidal_rule(y, dx)
        
    # Check if number of elements is odd (means number of intervals is even)
    if n % 2 == 1:
        # standard Simpson's 1/3 rule
        integral = (dx / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2]) + y[-1])
        return integral
    else:
        # Odd number of intervals: apply Simpson's rule on n-1 points, and Trapezoidal on the last interval
        simpson_part = (dx / 3.0) * (y[0] + 4.0 * np.sum(y[1:-2:2]) + 2.0 * np.sum(y[2:-3:2]) + y[-2])
        trap_part = dx * 0.5 * (y[-2] + y[-1])
        return simpson_part + trap_part


# ==========================================
# 4. LINEAR SYSTEMS SOLVER
# ==========================================

def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian elimination with partial pivoting.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Form the augmented matrix
    M = np.hstack([A, b.reshape(-1, 1)])
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Pivot search: find the row with maximum value in column i below row i
        max_row = i + np.argmax(np.abs(M[i:, i]))
        if abs(M[max_row, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
            
        # Swap current row with pivot row
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]
            
        # Eliminate entries below pivot
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]
            
    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.sum(M[i, i+1:n] * x[i+1:n])) / M[i, i]
        
    return x
