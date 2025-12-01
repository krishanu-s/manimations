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

class Function:
    """Interface for manipulating scalar-valued functions defined on a compact region of
    space - usually rectilinear."""
    @classmethod
    def laplacian(cls, vals: np.ndarray) -> np.ndarray:
        """Calculates the spacelike Laplacian of the function. This requires some constraint
        on the function values on the boundary of the region."""
        raise NotImplementedError

class OneDimFunctionWithBoundary(Function):
    """Real array-valued function f(x) defined on an interval [xmin, xmax], represented by its
    values at discrete points xmin + dx, xmin + 2 * dx, ..., xmin + (N-1) * dx, where
    dx = (xmax - xmin) / N.

    For the purpose of time-evolution (e.g. via a differential equation), it is assumed that 
    f(xmin) and f(xmax) are fixed boundary values"""
    xmin: float
    xmax: float
    num_steps: int # Number of subdivisions of the interval [xmin, xmax]
    dx: float # Step size between sample points
    ymin: float # Represents the boundary value at xmin
    ymax: float # Represents the boundary value at xmax
    
    def __init__(
        self,
        xmin: float = 0,
        xmax: float = 1,
        num_steps: int = 10,
        ymin: float = 0,
        ymax: float = 0,
        init_vals: np.ndarray | None = None
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.num_steps = num_steps
        self.dx = (xmax - xmin) / num_steps
        self.ymin = ymin
        self.ymax = ymax

        # If no initialization values are given, set the function to be linear.
        if init_vals is None:
            dy = (ymax - ymin) / num_steps
            self._set_vals(np.linspace(ymin + dy, ymax - dy, num_steps - 1))
        else:
            self._set_vals(init_vals)

        self._set_normalization_kernel()

    def _set_normalization_kernel(self):
        """Set th"""
        pass

    def _set_vals(self, vals: np.ndarray):
        """Sets the internal function values to the given one."""
        assert vals.shape[0] == self.num_steps - 1
        self.vals = vals

    def _get_vals(self):
        """Outputs the function values from xmin to xmax."""
        return np.concatenate((
            self.ymin * np.expand_dims(np.ones_like(self.vals[0]), 0),
            self.vals,
            self.ymax * np.expand_dims(np.ones_like(self.vals[0]), 0),
        ), axis=0)
    

    def laplacian(self, vals: np.ndarray | None = None) -> np.ndarray:
        """
        Returns the Laplacian (d/dx)^2 f(x), using the numerical approximation
        (f(x_{n-1}) - 2f(x_n) + f_(x_{n+1})) / (Δx)^2. If no function is given,
        uses
        """
        # TODO This results in numerical instability as dx becomes small.
        # You can view this Laplacian approximation as an integral against [1, -2, 1] / (Δx)^2.
        # Consider calculating by integrating against a different kernel, e.g. a normal distribution
        # of some width σ. This would be done as follows:
        # - Define self.kernel as yet another function, centered around x = 0. This will be a
        #   linear combination of normal distributions.
        # - Define self.convolution_with_kernel as 
        if vals is None:
            vals = self.vals
        else:
            assert vals.shape[0] == self.num_steps - 1
        
        result = np.concatenate((vals[1:], self.ymax * np.expand_dims(np.ones_like(vals[0]), 0)), axis=0)
        result += np.concatenate((self.ymin * np.expand_dims(np.ones_like(vals[0]), 0), vals[:-1]), axis=0)
        result -= 2 * vals
        result /= self.dx ** 2
        return result


class OneDimHeatEquationScene(m.Scene):
    """(d/dt) f(x, t) = C * (d/dx)^2 f(x, t)"""
    def _set_params(self):
        # Set parameters for the domain of the function
        self.xmin, self.xmax = 0, 1
        self.num_steps = 10

        # Set parameters for forward time-evolution
        self.run_time = 5.0
        self.num_iters = 1000

        
        self.dt = self.run_time / self.num_iters
        self.dx = (self.xmax - self.xmin) / self.num_steps
        assert self.dt < self.dx ** 2, "Must set a smaller time-step value for numerical stability."

    def construct(self):
        self._set_params()
        
        # Function values at dx, 2 * dx, 3 * dx, ..., (N-1) * dx
        init_vals = np.sin(np.pi * np.linspace(self.xmin, self.xmax, self.num_steps + 1))
        fn = OneDimFunctionWithBoundary(self.xmin, self.xmax, self.num_steps, 0, 0, init_vals=init_vals[1:-1])
        c = 1.0
        solver = AutonomousFirstOrderDiffEqSolver(
            t0=0,
            x0=init_vals[1:-1].copy(),
            f=lambda arr: c * fn.laplacian(arr))

        # Solve and record results
        result = [fn._get_vals()]
        for _ in range(self.num_iters):
            solver.step(self.dt)
            fn._set_vals(solver.val)
            result.append(fn._get_vals())
        
        # x dimension is first, t dimension is second
        result = np.stack(result, axis=-1)

        # Make animation
        l = 3.0 # Scale of width
        a = 3.0 # Scale of height
        curve = m.ParametricFunction(
            function=lambda x: (l * x, a * interpolate_vals(result[0], self.xmin, self.dx, x), 0),
            t_range=(self.xmin, self.xmax, self.dx)
        )
        self.add(curve)

        homotopy = ParametrizedHomotopy(
            homotopy=lambda x, t: (l * x, a * interpolate_vals_2d(result, self.xmin, self.dx, x, 0, 1 / self.num_iters, t), 0),
            mobject=curve,
            rate_func=m.linear,
            run_time=self.run_time
        )
        self.play(homotopy)

class OneDimWaveEquationScene(m.Scene):
    def _set_params(self):
        # Set parameters for the domain of the function
        self.xmin, self.xmax = 0, 1
        self.num_steps = 10

        # Set parameters for forward time-evolution
        self.run_time = 5.0
        self.num_iters = 1000

        
        self.dt = self.run_time / self.num_iters
        self.dx = (self.xmax - self.xmin) / self.num_steps
        assert self.dt < self.dx ** 2, "Must set a smaller time-step value for numerical stability."

    def construct(self):
        """(d/dt)^2 f(x, t) = C * (d/dx)^2 f(x, t)"""
        # Set discretization of interval
        xmin, xmax = 0, 1
        num_steps = 40
        dx = (xmax - xmin) / num_steps

        # Function values at dx, 2 * dx, 3 * dx, ..., (N-1) * dx
        fourier_coefficients = [1]
        init_vals = sum(
            coeff * np.sin(i * np.pi * np.linspace(xmin, xmax, num_steps + 1))
            for i, coeff in enumerate(fourier_coefficients)
            )

        # Initialize the function and solver
        fn = OneDimFunctionWithBoundary(xmin, xmax, num_steps, 0, 0, init_vals=init_vals[1:-1])
        c = 1.0
        solver = AutonomousSecondOrderDiffEqSolver(
            t0=0,
            x0=init_vals[1:-1].copy(),
            v0=np.zeros_like(init_vals[1:-1]),
            f=lambda arr: c * fn.laplacian(arr[0]))

        
        # Set parameters for forward time-evolution
        run_time = 5.0
        num_iters = 1000
        dt = run_time / num_iters
        assert dt < dx ** 2, "Must set a smaller time-step value for numerical stability."

        # Solve and record results
        result = [fn._get_vals()]
        for _ in range(num_iters):
            solver.step(dt)
            fn._set_vals(solver.val[0])
            result.append(fn._get_vals())
        
        # x dimension is first, t dimension is second
        result = np.stack(result, axis=-1)

        # Make animation
        l = 5.0 # Scale of width
        a = 1.0 # Scale of height
        curve = m.ParametricFunction(
            function=lambda x: (l * x, a * interpolate_vals(result[0], xmin, dx, x), 0),
            t_range=(xmin, xmax, dx)
        )
        self.add(curve)

        homotopy = ParametrizedHomotopy(
            homotopy=lambda x, t: (l * x, a * interpolate_vals_2d(result, xmin, dx, x, 0, 1 / num_iters, t), 0),
            mobject=curve,
            rate_func=m.linear,
            run_time = run_time,
        )
        self.play(homotopy)