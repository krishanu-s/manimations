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

# TODO Figure out how to forward-solve the wave equation in (1+1)-D and (2+1)-D
# with arbitrary boundary conditions. Numerical stability will be a major hurdle here.

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

class TwoDimFunctionWithBoundary(Function):
    """
    Real array-valued function f(x, y) defined on a subset of a rectangle in the plane.
    Includes boundary conditions on the perimeter of the rectangle.
    """
    pass


class WaveEquation1D(m.Scene):
    """
    Dumping ground for simulating and animating the time-evolution of a function f(x, t)
    according to the differential equation (d/dt)^2 f(x, t) = (d/dx)^2 f(x, t), subject
    to some initial conditions.

    The core of the simulation consists of a few components:

    (*) At any time, we represent (f(X, t), d_x f(X, t)) at a discrete mesh X for a single time-value t.
    This representation evolves according to the differential equation above.

    (*) How one calculates the Laplacian (d/dx)^2 f(x, t)
    
    There are various versions of this simulation to solve. If one thinks of this problem
    as solving for f(x, t) on the interior of a region bounded by a 1-D manifold in the plane,
    then the various versions of this problem correspond to different manifold shapes.

    (|_|) t >= 0 and xmin <= x <= xmax, where f(xmin, t), f(xmax, t), and f(x, 0) are given functions.
        Represents a string controlled at two ends.

    (|_) t >= 0 and x >= xmin, where f(xmin, t) and f(x, 0) are given functions.
        Represents a string controlled at one end and unbounded at the other
    
    TODO Once these are solved, we will move onto 2D animations.
    """

    def single_constrained(self):
        xmin = 0.0
        xmax = 5.0 # Maximum sample value for f
        xdiff = xmax - xmin

        # Number of sample points for f
        Nx = 100
        dx = xdiff / Nx

        # Temporal resolution on f
        Nt = 10000
        total_t = 15.0
        dt = total_t / Nt

        # Initial string value and its time derivative
        def f0(x: float):
            return np.array([
                np.exp(-x) * np.cos(np.pi * x),
                0
                ])

        # Left boundary value and its time derivative
        def fmin(t):
            return np.array([
                [np.cos(4 * t)],
                [-4 * np.sin(4 * t)]
            ])
        
        # Vals is of shape (N + 1,), and represents f(X, t)
        # Output is of shape (N ,)
        def laplacian(vals: np.ndarray):
            l = (vals[:-2] + vals[2:] - 2 * vals[1:-1])
            # TODO This is our estimation of the laplacian value at the endpoint,
            # but there might be a better way
            bdy = np.array([2 * l[-1] - l[-2]]) # Estimate laplacian value at the endpoint
            return np.concatenate((l, bdy), axis=0) / (dx ** 2)
        
        # Vals_and_df is of shape (2, N + 1), and represents (f(X, t), d_x * f(X, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1, 1:], laplacian(vals_and_df[0])), axis=0)

            # Step forward to f(x, t + dt/2) via the derivative
            # Step forward to d_x * f(x, t + dt/2) via the Laplacian
            # Append on boundary values
            p1 = np.concatenate((
                fmin(t + dt/2),
                vals_and_df[:, 1:] + (dt / 2) * k1,
            ), axis=-1)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1, 1:], laplacian(p1[0])), axis=0)
            p2 = np.concatenate((
                fmin(t + dt/2),
                vals_and_df[:, 1:] + (dt / 2) * k2,
            ), axis=-1)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1, 1:], laplacian(p2[0])), axis=0)
            p3 = np.concatenate((
                fmin(t + dt),
                vals_and_df[:, 1:] + dt * k3,
            ), axis=-1)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1, 1:], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = np.concatenate((
                fmin(t + dt),
                vals_and_df[:, 1:] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4),
            ), axis=-1)
            return new_vals_and_df
        
        # Set initial values
        vals = np.stack([f0(x) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated
        # to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            # print(vals)
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        # Make animation
        l = 2.0 # Scale of width
        a = 0.5 # Scale of height
        offset = np.array([-5.0, 0.0, 0.0])
        curve = m.ParametricFunction(

            function=lambda x: offset + np.array([l * x, a * interpolate_vals(result[0], xmin, dx, x), 0]),
            t_range=(xmin, xmax, dx)
        )
        self.add(curve)

        homotopy = ParametrizedHomotopy(
            homotopy=lambda x, t: offset + np.array([l * x, a * interpolate_vals_2d(result, xmin, dx, x, 0, 1 / Nt, t), 0]),
            mobject=curve,
            rate_func=m.linear,
            run_time = total_t,
        )
        # TODO Animate movement of bob as well
        # TODO Make this controlled by a ValueTracker
        self.play(homotopy)

    def double_constrained(self):
        """String constrained/controlled at two endpoints."""
        xmin = 0.0
        xmax = 1.0
        xdiff = xmax - xmin # This quantity occurs enough that we give it a name

        # Number of sample points for f
        Nx = 10
        dx = xdiff / Nx

        # Temporal resolution on f
        Nt = 1000
        total_t = 2.0
        dt = total_t / Nt

        # Initial string value and its time derivative
        w = 3 # Integer representing the number of half-wavelengths of the initial wave
        def f0(x: float):
            return np.array([
                np.sin(np.pi * w * (x - xmin) / xdiff),
                0
                ])

        # Left boundary value and its time derivative
        def fmin(t):
            return np.array([
                [0],
                [0]
            ])
        
        # Right boundary value and its time derivative
        def fmax(t):
            return np.array([
                [0],
                [0]
            ])

        
        # Vals is of shape (N + 1,), and represents f(X, t).
        # Output is of shape (N - 1,)
        def laplacian(vals: np.ndarray):
            return (vals[:-2] + vals[2:] - 2 * vals[1:-1]) / (dx ** 2)

        # Vals_and_df is of shape (2, N + 1), and represents (f(X, t), d_x * f(X, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1, 1:-1], laplacian(vals_and_df[0])), axis=0)

            # Step forward to f(x, t + dt/2) via the derivative
            # Step forward to d_x * f(x, t + dt/2) via the Laplacian
            # Append on boundary values
            p1 = np.concatenate((
                fmin(t + dt/2),
                vals_and_df[:, 1:-1] + (dt / 2) * k1,
                fmax(t + dt/2),
            ), axis=-1)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1, 1:-1], laplacian(p1[0])), axis=0)
            p2 = np.concatenate((
                fmin(t + dt/2),
                vals_and_df[:, 1:-1] + (dt / 2) * k2,
                fmax(t + dt/2),
            ), axis=-1)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1, 1:-1], laplacian(p2[0])), axis=0)
            p3 = np.concatenate((
                fmin(t + dt),
                vals_and_df[:, 1:-1] + dt * k3,
                fmax(t + dt),
            ), axis=-1)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1, 1:-1], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = np.concatenate((
                fmin(t + dt),
                vals_and_df[:, 1:-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4),
                fmax(t + dt),
            ), axis=-1)
            return new_vals_and_df


        # Set initial values
        vals = np.stack([f0(x) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated
        # to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            # print(vals)
            result.append(vals[0].copy())
        
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
            homotopy=lambda x, t: (l * x, a * interpolate_vals_2d(result, xmin, dx, x, 0, 1 / Nt, t), 0),
            mobject=curve,
            rate_func=m.linear,
            run_time = total_t,
        )
        self.play(homotopy)

    def construct(self):
        # self.double_constrained()
        self.single_constrained()

        # # Define f(0, t), f_t(0, t)
        # def left_bdy(t: float):
        #     return np.array([np.sin(t), np.cos(t)])
        
        # # Set initial values
        # num_steps = 10
        # xmin = 0
        # xmax = 1
        # dx = (xmax - xmin) / num_steps
        # init_vals = np.concatenate(
        #     (np.expand_dims(left_bdy(0), -1), np.zeros((2, num_steps))),
        #     axis=-1)

        # def laplacian(arr: np.ndarray):
        #     result = (arr[2:] + arr[:-2] - 2 * arr[1:-1]) / (dx ** 2)
        #     return np.concatenate((np.array([0]), result, np.array([result[-1]])), axis=0)
        
        # solver = AutonomousSecondOrderDiffEqSolver(
        #     t0=0,
        #     x0=init_vals[0],
        #     v0=init_vals[1],
        #     f=lambda arr: laplacian(arr[0]))
        
        # result = [init_vals[0].copy()]
        # run_time = 5.0
        # num_iters = 5000
        # dt = run_time / num_iters
        # assert dt < dx ** 2, "Must set a smaller time-step value for numerical stability."
        # for _ in range(num_iters):
        #     solver.step(dt)
        #     solver.val[:, 0] = left_bdy(solver.t)
        #     result.append(solver.val[0].copy())
            
        # result = np.stack(result, axis=-1)

        # # Make animation
        # l = 5.0 # Scale of width
        # a = 1.0 # Scale of height
        # curve = m.ParametricFunction(
        #     function=lambda x: (l * x, a * interpolate_vals(result[0], xmin, dx, x), 0),
        #     t_range=(xmin, xmax, dx)
        # )
        # self.add(curve)

        # homotopy = ParametrizedHomotopy(
        #     homotopy=lambda x, t: (l * x, a * interpolate_vals_2d(result, xmin, dx, x, 0, 1 / num_iters, t), 0),
        #     mobject=curve,
        #     rate_func=m.linear,
        #     run_time = run_time,
        # )
        # self.play(homotopy)


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