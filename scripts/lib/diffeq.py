from typing import Callable
from .point import Point2D

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

# TODO Vectorize this method to numpy arrays
class RungeKuttaAutonomous:
    """Method for forward-stepping a differential equation of the form x''(t) = f(x(t), x'(t))."""
    def __init__(self, t0: float,  val: Point2D,  f: Callable[[Point2D], float]
        ):
        self.t = t0
        self.val = val
        self.f = f

    def step(self, dt: float):
        """Step forward in time by dt"""
        k1 = Point2D(self.val.y, self.f(self.val))
        p1 = self.val + k1 * (dt / 2)
        k2 = Point2D(p1.y,self.f(p1))
        p2 = self.val + k2 * (dt / 2)
        k3 = Point2D(p2.y, self.f(p2))
        p3 = self.val + k3 * dt
        k4 = Point2D(p3.y, self.f(p3))

        self.t += dt
        self.val += (k1 + k2 * 2 + k3 * 2 + k4) * (dt/6)