from typing import Callable
from .point import Point2D
import numpy as np

# TODO Vectorize this to numpy arrays.
class RungeKutta2:
    """Method for forward-stepping a differential equation of the form (d/dt)^2(x) = f(t, x)"""
    def __init__(self, t: float, val: Point2D, f: Callable[[float, float], float]):
        self.t = t
        self.val = val
        self.f = f

    def step(self, dt: float):
        """Step forward in time by dt"""
        k1 = Point2D(self.val.y, self.f(self.t, self.val.x))
        k2 = Point2D(
            self.val.y + k1.x * dt / 2,
            self.f(self.t + dt / 2, self.val.x + k1.x * dt/2)
        )
        k3 = Point2D(
            self.val.y + k2.x * dt / 2,
            self.f(self.t + dt / 2, self.val.x + k2.x * dt/2)
        )
        k4 = Point2D(
            self.val.y + k3.x * dt,
            self.f(self.t + dt, self.val.x + k3.x * dt)
        )
        self.t += dt
        self.val.translate_x((k1.x + 2 * k2.x + 2 * k3.x + k4.x) * dt/6)
        self.val.translate_y((k1.y + 2 * k2.y + 2 * k3.y + k4.y) * dt/6)


class AutonomousDiffEqSolver:
    """Generic interface for numerical solvers for autonomous differential equations."""
    @property
    def get_x(self):
        raise NotImplementedError
    
    def step(self, dt: float):
        """Step forward in time by dt."""
        raise NotImplementedError

class AutonomousFirstOrderDiffEqSolver(AutonomousDiffEqSolver):
    t: float
    val: np.ndarray
    """Numerically solves a differential equation of the form x'(t) = f(x(t)), where x(t)
    can be a scalar-valued or vector-valued function.
    The underlying solver is Runge-Kutta-4."""
    def __init__(
        self,
        t0: float,  # Initial time
        x0: float | np.ndarray,  # Initial value
        f: Callable[[np.ndarray], np.ndarray]  # Numpy-aware function which maintains array shape
    ):
        # Assert the same shape
        self.t = t0
        if isinstance(x0, float):
            self.val = np.array(x0)
        else:
            self.val = x0
        self.f = f

    @property
    def get_x(self) -> np.ndarray:
        return self.val

    def step(self, dt: float):
        """Step forward in time by dt"""
        k1 = self.f(self.val)
        k2 = self.f(self.val + k1 * dt/2)
        k3 = self.f(self.val + k2 * dt/2)
        k4 = self.f(self.val + k3 * dt)
        self.t += dt
        self.val += (k1 + k2 * 2 + k3 * 2 + k4) * (dt/6)

class AutonomousSecondOrderDiffEqSolver(AutonomousDiffEqSolver):
    t: float
    val: np.ndarray
    """Numerically solves a differential equation of the form x''(t) = f(x(t), x'(t)), where x(t)
    can be a scalar-valued or vector-valued function.
    The underlying solver is Runge-Kutta-4."""
    def __init__(
        self,
        t0: float,  # Initial time
        x0: float | np.ndarray,  # Initial value
        v0: float | np.ndarray,  # Initial derivative
        f: Callable[[np.ndarray], np.ndarray]  # Numpy-aware function
    ):
        # Assert the same shape
        assert type(x0) == type(v0)
        if isinstance(x0, np.ndarray):
            assert x0.shape == v0.shape
        self.t = t0
        self.val = np.stack((x0, v0), axis=0) # Shape (2, N)
        self.f = f

    @property
    def get_x(self) -> np.ndarray:
        return self.val[0]

    def step(self, dt: float):
        """Step forward in time by dt"""
        k1 = np.stack((self.val[1], self.f(self.val)), axis=0)
        p1 = self.val + k1 * (dt / 2)
        k2 = np.stack((p1[1],self.f(p1)), axis=0)
        p2 = self.val + k2 * (dt / 2)
        k3 = np.stack((p2[1], self.f(p2)), axis=0)
        p3 = self.val + k3 * dt
        k4 = np.stack((p3[1], self.f(p3)), axis=0)

        self.t += dt
        self.val += (k1 + k2 * 2 + k3 * 2 + k4) * (dt/6)
