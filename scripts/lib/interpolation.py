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

def interpolate_vals_3d(
    vals: np.ndarray,
    xmin: float, dx: float, x: float,
    ymin: float, dy: float, y: float,
    zmin: float, dz: float, z: float,
):
    assert len(vals.shape) == 3
    x -= xmin
    y -= ymin
    z -= zmin

    kx = int(x // dx)
    ky = int(y // dy)
    kz = int(z // dz)

    a_x = x / dx - kx
    a_y = y / dy - ky
    a_z = z / dz - kz

    if a_x == 0:
        if a_y == 0:
            if a_z == 0:
                return vals[kx, ky, kz]
            else:
                return (1 - a_z) * vals[kx, ky, kz] + a_z * vals[kx, ky, kz + 1]
        else:
            if a_z == 0:
                return (1 - a_y) * vals[kx, ky, kz] + a_y * vals[kx, ky + 1, kz]
            else:
                return (1 - a_y) * (1 - a_z) * vals[kx, ky, kz] + (1 - a_y) * a_z * vals[kx, ky, kz + 1] + a_y * (1 - a_z) * vals[kx, ky + 1, kz] + a_y * a_z * vals[kx, ky + 1, kz + 1]
    else:
        if a_y == 0:
            if a_z == 0:
                return (1 - a_x) * vals[kx, ky, kz] + a_x * vals[kx + 1, ky, kz]
            else:
                return (1 - a_x) * (1 - a_z) * vals[kx, ky, kz] + (1 - a_x) * a_z * vals[kx, ky, kz + 1] + a_x * (1 - a_z) * vals[kx + 1, ky, kz] + a_x * a_z * vals[kx + 1, ky, kz + 1]
        else:
            if a_z == 0:
                return (1 - a_x) * (1 - a_y) * vals[kx, ky, kz] + (1 - a_x) * a_y * vals[kx, ky + 1, kz] + a_x * (1 - a_y) * vals[kx + 1, ky, kz] + a_x * a_y * vals[kx + 1, ky + 1, kz]
            else:
                return (1 - a_x) * (1 - a_y) * (1 - a_z) * vals[kx, ky, kz] + (1 - a_x) * a_y * (1 - a_z) * vals[kx, ky + 1, kz] + a_x * (1 - a_y) * (1 - a_z) * vals[kx + 1, ky, kz] + a_x * a_y * (1 - a_z) * vals[kx + 1, ky + 1, kz] + (1 - a_x) * (1 - a_y) * a_z * vals[kx, ky, kz + 1] + (1 - a_x) * a_y * a_z * vals[kx, ky + 1, kz + 1] + a_x * (1 - a_y) * a_z * vals[kx + 1, ky, kz + 1] + a_x * a_y * a_z * vals[kx + 1, ky + 1, kz + 1]
