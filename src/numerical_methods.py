"""
numerical_methods.py
====================
HydroSense-Kenya | ICS 2207 Scientific Computing - Level 3

Hand-coded numerical methods engine. ALL methods implemented from scratch.
NumPy used only for array creation and basic arithmetic.
No scipy, no solver libraries.

Functions
---------
ROOT FINDING       : bisection, newton_raphson, secant
FINITE DIFFERENCES : forward_difference, backward_difference, central_difference,
                     second_derivative, finite_difference_array
NUMERICAL INTEGRATION: trapezoidal, simpsons, simpsons_three_eighths
LINEAR SYSTEMS     : gaussian_elimination, lu_decomposition, lu_solve
"""

import numpy as np


# ==========================================================================
# ROOT FINDING
# ==========================================================================

def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root of f(x)=0 on [a,b] using the bisection method.

    Repeatedly halves the bracket. Convergence is guaranteed (linear):
    error halves every iteration. Requires f(a)*f(b) < 0.

    Parameters
    ----------
    f        : callable  - function whose root we seek
    a, b     : float     - bracket endpoints; f(a)*f(b) must be < 0
    tol      : float     - stop when interval width < tol (default 1e-6)
    max_iter : int       - maximum iterations (default 100)

    Returns
    -------
    dict with keys: root, iterations, converged, history, error
    """
    if f(a) * f(b) > 0:
        raise ValueError(
            f"f(a) and f(b) must have opposite signs. "
            f"Got f({a:.4f})={f(a):.4f}, f({b:.4f})={f(b):.4f}"
        )
    history = []
    for i in range(max_iter):
        mid   = (a + b) / 2.0
        f_mid = f(mid)
        history.append(mid)
        if abs(b - a) / 2.0 < tol or f_mid == 0.0:
            return {'root': mid, 'iterations': i + 1,
                    'converged': True, 'history': history,
                    'error': abs(b - a) / 2.0}
        if f(a) * f_mid < 0:
            b = mid
        else:
            a = mid
    return {'root': (a + b) / 2.0, 'iterations': max_iter,
            'converged': False, 'history': history,
            'error': abs(b - a) / 2.0}


def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
    """
    Find root of f(x)=0 using Newton-Raphson method.

    Update rule: x_{n+1} = x_n - f(x_n) / f'(x_n)
    Convergence: quadratic (error squared each step). Requires derivative df.

    Parameters
    ----------
    f, df    : callable - function and its derivative
    x0       : float   - initial guess
    tol      : float   - stop when |x_new - x| < tol (default 1e-6)
    max_iter : int     - maximum iterations (default 50)

    Returns
    -------
    dict with keys: root, iterations, converged, history, error
    """
    x = float(x0)
    history = [x]
    for i in range(max_iter):
        fx  = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError(f"Derivative is zero at x={x:.6f}.")
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < tol:
            return {'root': x_new, 'iterations': i + 1,
                    'converged': True, 'history': history,
                    'error': abs(f(x_new))}
        x = x_new
    return {'root': x, 'iterations': max_iter,
            'converged': False, 'history': history, 'error': abs(f(x))}


def secant(f, x0, x1, tol=1e-6, max_iter=50):
    """
    Find root of f(x)=0 using the secant method.

    Approximates derivative from two previous points (no df needed):
    x_{n+1} = x_n - f(x_n)*(x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    Convergence: superlinear (~order 1.618).

    Parameters
    ----------
    f        : callable - function whose root we seek
    x0, x1  : float   - two initial guesses
    tol      : float   - stop when |x_new - x1| < tol (default 1e-6)
    max_iter : int     - maximum iterations (default 50)

    Returns
    -------
    dict with keys: root, iterations, converged, history, error
    """
    history = [x0, x1]
    for i in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-14:
            return {'root': x1, 'iterations': i + 1,
                    'converged': False, 'history': history, 'error': abs(f1)}
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            return {'root': x2, 'iterations': i + 2,
                    'converged': True, 'history': history, 'error': abs(f(x2))}
        x0, x1 = x1, x2
    return {'root': x1, 'iterations': max_iter,
            'converged': False, 'history': history, 'error': abs(f(x1))}


# ==========================================================================
# FINITE DIFFERENCES
# ==========================================================================

def forward_difference(f, x, h=1e-5):
    """
    Estimate f'(x) using forward difference: [f(x+h) - f(x)] / h
    Error order: O(h) - first-order accuracy.
    """
    return (f(x + h) - f(x)) / h


def backward_difference(f, x, h=1e-5):
    """
    Estimate f'(x) using backward difference: [f(x) - f(x-h)] / h
    Error order: O(h) - first-order accuracy.
    """
    return (f(x) - f(x - h)) / h


def central_difference(f, x, h=1e-5):
    """
    Estimate f'(x) using central difference: [f(x+h) - f(x-h)] / (2h)
    Error order: O(h^2) - more accurate than forward/backward.
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second_derivative(f, x, h=1e-5):
    """
    Estimate f''(x) using central second difference:
    [f(x+h) - 2f(x) + f(x-h)] / h^2
    Error order: O(h^2).
    """
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h ** 2)


def finite_difference_array(values, method='central'):
    """
    Apply finite differences to a discrete array (e.g. daily soil moisture).
    Returns estimated rate of change at each point (step size h = 1 day).

    Parameters
    ----------
    values : array-like - observed values
    method : str       - 'forward', 'backward', or 'central' (default)

    Returns
    -------
    np.ndarray - estimated first derivative at each point
    """
    v  = np.asarray(values, dtype=np.float64)
    n  = len(v)
    dv = np.zeros(n)
    for i in range(n):
        if i == 0:
            dv[i] = v[1] - v[0]             # forward at left endpoint
        elif i == n - 1:
            dv[i] = v[-1] - v[-2]           # backward at right endpoint
        elif method == 'forward':
            dv[i] = v[i + 1] - v[i]
        elif method == 'backward':
            dv[i] = v[i] - v[i - 1]
        else:                               # central (default, most accurate)
            dv[i] = (v[i + 1] - v[i - 1]) / 2.0
    return dv


# ==========================================================================
# NUMERICAL INTEGRATION
# ==========================================================================

def trapezoidal(y, x=None, dx=1.0):
    """
    Estimate integral using the composite trapezoidal rule.
    Area = sum of trapezoids: (h/2) * (y_i + y_{i+1})
    Error order: O(h^2) overall.

    Parameters
    ----------
    y  : array-like - function values
    x  : array-like - x positions (optional; uses dx if None)
    dx : float      - uniform step size (default 1.0)

    Returns
    -------
    float - estimated integral
    """
    y = np.asarray(y, dtype=np.float64)
    if x is not None:
        h_vals = np.diff(np.asarray(x, dtype=np.float64))
    else:
        h_vals = np.full(len(y) - 1, dx)
    total = 0.0
    for i in range(len(y) - 1):
        total += (h_vals[i] / 2.0) * (y[i] + y[i + 1])
    return total


def simpsons(y, x=None, dx=1.0):
    """
    Estimate integral using composite Simpson's 1/3 rule.
    Fits a parabola through every 3 points: (h/3)*(y0 + 4*y1 + y2)
    Error order: O(h^4) - much more accurate than trapezoidal.

    Parameters
    ----------
    y  : array-like - function values (odd length preferred)
    x  : array-like - x positions (optional)
    dx : float      - uniform step size (default 1.0)

    Returns
    -------
    float - estimated integral
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if x is not None:
        h = (np.asarray(x, dtype=np.float64)[-1] - np.asarray(x, dtype=np.float64)[0]) / (n - 1)
    else:
        h = dx
    if n < 3:
        return trapezoidal(y, x, dx)
    total = 0.0
    i = 0
    while i <= n - 3:
        total += (h / 3.0) * (y[i] + 4.0 * y[i + 1] + y[i + 2])
        i += 2
    if n % 2 == 0:          # even number of points: handle last interval
        total += (h / 2.0) * (y[-2] + y[-1])
    return total


def simpsons_three_eighths(y, dx=1.0):
    """
    Estimate integral using Simpson's 3/8 rule.
    Fits a cubic through every 4 points: (3h/8)*(y0+3*y1+3*y2+y3)
    Requires number of intervals divisible by 3.

    Parameters
    ----------
    y  : array-like - function values
    dx : float      - uniform step size (default 1.0)

    Returns
    -------
    float - estimated integral
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 4:
        return trapezoidal(y, dx=dx)
    total = 0.0
    i = 0
    while i <= n - 4:
        total += (3.0 * dx / 8.0) * (y[i] + 3.0*y[i+1] + 3.0*y[i+2] + y[i+3])
        i += 3
    remainder = (n - 1) % 3
    if remainder == 1:
        total += (dx / 2.0) * (y[-2] + y[-1])
    elif remainder == 2:
        total += (dx / 3.0) * (y[-3] + 4.0*y[-2] + y[-1])
    return total


# ==========================================================================
# LINEAR SYSTEMS
# ==========================================================================

def gaussian_elimination(A, b):
    """
    Solve Ax=b using Gaussian elimination with partial pivoting.

    Transforms the augmented matrix [A|b] to upper-triangular form,
    then solves by back-substitution. Partial pivoting (swapping rows
    to maximise the pivot element) improves numerical stability.

    Parameters
    ----------
    A : array-like (n x n) - coefficient matrix
    b : array-like (n,)    - right-hand side vector

    Returns
    -------
    dict with keys: solution, converged, residual ||Ax-b||

    HydroSense use: solve 3-zone water allocation system Ax = b
    """
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    n = len(b)
    aug = np.hstack([A, b.reshape(-1, 1)])

    for col in range(n):
        max_row = col + int(np.argmax(np.abs(aug[col:, col])))
        aug[[col, max_row]] = aug[[max_row, col]]
        if abs(aug[col, col]) < 1e-12:
            raise ValueError(f"Matrix is singular at column {col}.")
        for row in range(col + 1, n):
            factor = aug[row, col] / aug[col, col]
            aug[row, col:] -= factor * aug[col, col:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i, i]

    residual = float(np.linalg.norm(A @ x - b))
    return {'solution': x, 'converged': residual < 1e-6, 'residual': residual}


def lu_decomposition(A):
    """
    Decompose A = L*U using Doolittle's algorithm.

    L is lower triangular with 1s on diagonal.
    U is upper triangular.
    Decompose once, then solve for multiple right-hand sides cheaply.

    Parameters
    ----------
    A : array-like (n x n) - square matrix

    Returns
    -------
    dict with keys: L, U
    """
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        L[i, i] = 1.0
        for k in range(i + 1, n):
            if abs(U[i, i]) < 1e-12:
                raise ValueError(f"Zero pivot at ({i},{i}). Matrix may be singular.")
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    return {'L': L, 'U': U}


def lu_solve(L, U, b):
    """
    Solve Ax=b given LU decomposition A = L*U.

    Step 1 - Forward substitution:  solve L*y = b  for y
    Step 2 - Back substitution:     solve U*x = y  for x

    Parameters
    ----------
    L, U : np.ndarray (n x n) - from lu_decomposition()
    b    : array-like (n,)    - right-hand side vector

    Returns
    -------
    np.ndarray - solution vector x
    """
    b = np.array(b, dtype=np.float64)
    n = len(b)

    # Forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return x