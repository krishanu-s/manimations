import numpy as np


# TODO Make the interfaces clearer from the outside.
# TODO Add docstrings
# TODO Make the 2d version non-recursive to improve algorithm speed.

def interpolate_vals(vals: np.ndarray, t_min: float, dt: float, t: float) -> float:
    """Given an input t, and a sequence of function output values
    f(0), f(dt), f(2*dt), ..., f(N*dt), linearly interpolates between the given values
    to estimate f(t)."""
    assert len(vals.shape) == 1
    t -= t_min
    if t >= (vals.shape[0] - 1) * dt:
        return vals[-1]
    else:
        k = int(t // dt)
        alpha = t / dt - k
        return (1 - alpha) * vals[k] + alpha * vals[k + 1]
    
def interpolate_vals_2d(
    vals: np.ndarray,
    x_min: float, dx: float, x: float,
    t_min: float, dt: float, t: float
):
    # Assumes x-dimension is first, t-dimension is second
    assert len(vals.shape) == 2
    x -= x_min
    if x >= (vals.shape[0] - 1) * dx:
        v = vals[-1]
    else:
        k = int(x // dx)
        alpha = x / dx - k
        v = (1 - alpha) * vals[k] + alpha * vals[k + 1]
    return interpolate_vals(v, t_min, dt, t)