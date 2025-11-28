"""Animation of the heat equation

The Laplacian operator.

Random walks"""

import numpy as np
import manim as m
from lib import (
    AutonomousFirstOrderDiffEqSolver,
    AutonomousSecondOrderDiffEqSolver,
    ParametrizedHomotopy,
    interpolate_vals,
    interpolate_vals_2d
    )

class OneDimFunction:
    """Real array-valued function f(x) defined on an interval [x_min, x_max], represented by its
    values at discrete points spaced apart by x_step."""
    x_min: float
    x_max: float
    dx: float # x_max - x_min = N * dx
    left_bdy: np.ndarray # Shape (...), represents the boundary value at x_min
    right_bdy: np.ndarray # Shape (...), represents the boundary value at x_max
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        num_steps: int,
        x_min_val: float | None = None,
        x_max_val: float | None = None
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / num_steps
        if x_min_val is None:
            self.x_min_val = 0
        else:
            self.x_min_val = x_min_val

        if x_max_val is None:
            self.x_max_val = 0
        else:
            self.x_max_val = x_max_val
        
    def laplacian(self, vals: np.ndarray) -> np.ndarray:
        """Returns the Laplacian (d/dx)^2 f(x), using (f(x_{n-1}) - 2f(x_n) + f_(x_{n+1})) / (Δx)^2."""
        # TODO Issue: this results in numerical instability as dx becomes small.
        right_bdy = self.x_max_val * np.expand_dims(np.ones_like(vals[0]), 0)
        left_bdy = self.x_min_val * np.expand_dims(np.ones_like(vals[0]), 0)
        result = np.concatenate((vals[1:], right_bdy), axis=0)
        result += np.concatenate((left_bdy, vals[:-1]), axis=0)
        result -= 2 * vals
        result /= self.dx ** 2
        return result

class OneDimHeatEquationScene(m.Scene):
    """(d/dt) f(x, t) = C * (d/dx)^2 f(x, t)"""
    def construct(self):
        x_min, x_max = 0, 1
        num_steps = 10
        dx = (x_max - x_min) / num_steps
        init_vals = np.array([
            np.sin(x * np.pi)
            for x in np.linspace(x_min + dx, x_max - dx, num_steps - 1)
            ])
        fn = OneDimFunction(x_min, x_max, num_steps, 0, 0)
        c = 1.0
        solver = AutonomousFirstOrderDiffEqSolver(t0=0, x0=init_vals.copy(), f=lambda arr: c * fn.laplacian(arr))

        result = [init_vals.copy()]
        print(init_vals)

        dt = 1e-3
        for i in range(1000):
            solver.step(dt)
            result.append(solver.val.copy())
            print(solver.val)
        
        # x dimension is first, t dimension is second
        result = np.stack(result, axis=-1)

        # Make animation
        l = 3.0 # Scale of width
        a = 3.0 # Scale of height
        curve = m.ParametricFunction(
            function=lambda x: (l * x, a * interpolate_vals(init_vals, x_min, dx, x), 0),
            t_range=(x_min, x_max, 0.1)
        )
        self.add(curve)

        homotopy = ParametrizedHomotopy(
            homotopy=lambda x, t: (l * x, a * interpolate_vals_2d(result, x_min, dx, x, 0, dt, t), 0),
            mobject=curve,
            rate_func=m.linear
        )
        self.play(homotopy)

class OneDimWaveEquationScene(m.Scene):
    def construct(self):
        """(d/dt)^2 f(x, t) = C * (d/dx)^2 f(x, t)"""
        x_min, x_max = 0, 1
        num_steps = 10
        dx = (x_max - x_min) / num_steps
        init_vals = np.array([
            np.sin(x * np.pi)
            for x in np.linspace(x_min + dx, x_max - dx, num_steps - 1)
            ])
        fn = OneDimFunction(x_min, x_max, num_steps, 0, 0)
        c = 3.0
        solver = AutonomousSecondOrderDiffEqSolver(
            t0=0,
            x0=init_vals.copy(),
            v0=np.zeros_like(init_vals),
            f=lambda arr: c * fn.laplacian(arr[0])
            )

        result = [init_vals.copy()]
        print(init_vals)

        dt = 1e-3
        for i in range(1000):
            solver.step(dt)
            result.append(solver.val[0].copy())
            print(solver.val[0])
        
        # x dimension is first, t dimension is second
        result = np.stack(result, axis=-1)

        # Make animation
        L = 3.0 # Scale of width
        A = 3.0 # Scale of height
        curve = m.ParametricFunction(
            function=lambda x: (L * x, A * interpolate_vals(init_vals, x_min, dx, x), 0),
            t_range=(x_min, x_max, 0.1)
        )
        self.add(curve)

        homotopy = ParametrizedHomotopy(
            homotopy=lambda x, t: (L * x, A * interpolate_vals_2d(result, x_min, dx, x, 0, dt, t), 0),
            mobject=curve,
            rate_func=m.linear
        )
        self.play(homotopy)