"""Animation of the heat equation

The Laplacian operator.

Random walks"""

import math
import numpy as np
import manim as m
from lib import (
    AutonomousFirstOrderDiffEqSolver,
    AutonomousSecondOrderDiffEqSolver,
    ParametrizedHomotopy,
    interpolate_vals,
    interpolate_vals_2d,
    interpolate_vals_3d
    )

# TODO Figure out how to forward-solve the wave equation in (1+1)-D and (2+1)-D
# with arbitrary boundary conditions. Numerical stability will be a major hurdle here.

class DotCloudScene(m.Scene):
    def construct(self):
        density = 30
        dots = m.DotCloud(radius=2.0, density=density)
        self.add(dots)

        time = m.ValueTracker(0)
        dots.add_updater(lambda mobj: mobj.become(
            m.DotCloud(radius=2.0 * (1 + time.get_value() / 5), density=density)
        ))

        self.play(time.animate.increment_value(5), run_time=5.0, rate_func=m.linear)

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


class WaveEquation2D(m.ThreeDScene):
    """
    Dumping ground for simulating and animating the time-evolution of a function f(x, y, t)
    according to the differential equation (d/dt)^2 f = ((d/dx)^2 + (d/dy)^2) f, subject
    to some boundary conditions.

    The key ideas are as follows:

    (1) The function's values (and their time-derivatives) are stored over some rectilinear region of the plane [xmin, xmax] \times [ymin, ymax] as a finite mesh. That is, the state is an array of shape (2, Nx + 1, Ny + 1). The rectilinear region covers the function's domain, and the value on points of the mesh which lie outside of the function domain are assigned a value of 0.
    (2) The time-evolution at each mesh point proceeds according to several possible rules:
       (a) If the point lies in the domain and its four neighbors also lie in the domain, compute its Laplacian by finite-difference and use this to do time evolution.
       (b) If the point lies outside the domain, no time-evolution occurs.
       (c) If the point lies within the domain but some of its neighbors lie outside the domain, then we have pre-stored the locations of the boundary points nearest to it along the horizontal and vertical grid lines. We compute its Laplacian by a suitable modification of finite-difference in such a way that the zero-th and first-order components in the Taylor series cancel, leaving a result of the form (for the x-direction, for example) d_x^2 f(x, y) + O(dx). Handling these points will be the tricky part of the algorithm.
       (d) (TODO) If the mesh point is in the near-vicinity of a point source, ...

    Points of type (c) are pre-computed, their corresponding boundary (x, y) points and distances pre-computed, and the required coefficients as well. Handle the computing of d_x^2 and of d_y^2 completely independently, for simplicity.

    The different forms of this problem amount to different shapes of the boundary. We tackle
    several cases in increasing order of complexity:

    (1) t >= 0, xmin <= x <= xmax, and ymin <= y <= ymax, where f(x, y, 0), f(xmin, y, t), f(xmax, y, t), f(x, ymin, t), and f(x, ymax, t) are given functions. Represents a sheet with rectangular boundary. This is completely analogous to the 1D case.

    (2) Replace the rectilinear constraints in (1) by an arbitrary curve parametrized in some fashion, but where the function is still defined on the (compact) interior of the curve. This involves some innovation in how one stores the function values, as well as how one computes the Laplacian, given the uneven distances from the mesh points to the boundary. Key examples to use: circles and ellipses.

    (3) Extend to handle non-compact domains.

    (4) Add point sources to these cases.
    """
    def make_animation(self, result: np.ndarray, speed: float = 5.0):
        # TODO This is the time-limiting part of animation. Find a way to make it more efficient.
        # E.g., maybe animate in 2D with shading density.
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax
        Nx, Ny = self.Nx, self.Ny
        total_t, Nt = self.total_t, self.Nt

        dx = (xmax - xmin) / Nx
        dy = (ymax - ymin) / Ny
        dt = total_t / Nt

        # TODO It's ideal if we can first interpolate the results down to the intended frame-rate / time values.
        # Another way is to use a 2D gradient field, rather than a m.Surface

        # Add the initial surface
        # TODO defining a function which calls directly from the stored arrays.
        #      The rendering is the main limiting factor.
        surface = m.Surface(
            lambda x, y: np.array([x, y, interpolate_vals_2d(result[:, :, 0], xmin, dx, x, ymin, dy, y)]),
            resolution=(Nx, Ny),
            u_range=[xmin, xmax],
            v_range=[ymin, ymax])
        self.add(surface)

        # Animate its time evolution
        time = m.ValueTracker(0)
        surface.add_updater(lambda mobj: mobj.become(
            m.Surface(
                lambda x, y: np.array([x, y, 1.0 * interpolate_vals_3d(
                    result,
                    xmin, dx, x,
                    ymin, dy, y,
                    0, dt, time.get_value()
                    )]),
                resolution=(Nx, Ny),
                u_range=[xmin, xmax],
                v_range=[ymin, ymax]
                )
        ))
        self.play(time.animate.set_value(total_t), run_time=(1/speed) * total_t, rate_func=m.linear)
        

    def circular_compact(self):
        """Animates a wave with a source at t = 0 and a spatial boundary x^2 + y^2 = 1."""

        xmin, xmax = -1.2, 1.2
        ymin, ymax = -1.2, 1.2
        xdiff = xmax - xmin
        ydiff = ymax - ymin

        # Spatial resolution for f
        Nx = 50
        dx = xdiff / Nx
        Ny = 50
        dy = ydiff / Ny

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx, self.Ny = Nx, Ny

        # Function value on boundary
        def f_bdy(p: np.ndarray, t: float):
            assert p.shape[-1] == 2
            return np.zeros(p.shape[:-1])

        # Calculate mask for points near to the boundary
        x_pos_interior_mask = np.zeros((Nx - 1, Ny + 1))
        x_pos_near_bdy_mask = np.zeros((Nx - 1, Ny + 1, 3))
        x_neg_interior_mask = np.zeros((Nx - 1, Ny + 1))
        x_neg_near_bdy_mask = np.zeros((Nx - 1, Ny + 1, 3))

        y_pos_interior_mask = np.zeros((Nx + 1, Ny - 1))
        y_pos_near_bdy_mask = np.zeros((Nx + 1, Ny - 1, 3))
        y_neg_interior_mask = np.zeros((Nx + 1, Ny - 1))
        y_neg_near_bdy_mask = np.zeros((Nx + 1, Ny - 1, 3))

        exterior_mask = np.zeros((Nx + 1, Ny + 1)) # Used for setting f values

        for i, x in enumerate(np.linspace(xmin, xmax, Nx + 1)):
            for j, y in enumerate(np.linspace(ymin, ymax, Ny + 1)):
                # Outside domain, mask value is irrelevant
                if x**2 + y**2 >= 1:
                    exterior_mask[i, j] = 1
                    continue
                
                # Otherwise, calculate whether (x + dx)^2 + y^2 > 1, (x - dx)^2 + y^2 > 1
                if i > 0 and i < Nx:
                    if (x + dx) ** 2 + y ** 2 >= 1:
                        # find the number 0 < a < 1 such that (x + a * dx)^2 + y^2 == 1
                        # then would calculate Laplacian using (f(x + a*dx, y) - f(x, y)) / a
                        a = (math.sqrt(1 - y ** 2) - x) / dx
                        x_pos_near_bdy_mask[i - 1, j] = np.array([a, math.sqrt(1 - y ** 2), y])
                    else:
                        x_pos_interior_mask[i - 1, j] = 1

                    if (x - dx) ** 2 + y ** 2 >= 1:
                        b = (math.sqrt(1 - y ** 2) + x) / dx
                        x_neg_near_bdy_mask[i - 1, j] = np.array([b, -math.sqrt(1 - y ** 2), y])
                    else:
                        x_neg_interior_mask[i - 1, j] = 1
                
                if j > 0 and j < Ny:
                    if x ** 2 + (y + dy) ** 2 >= 1:
                        a = (math.sqrt(1 - x ** 2) - y) / dy
                        y_pos_near_bdy_mask[i, j - 1] = np.array([a, x, math.sqrt(1 - x ** 2)])
                    else:
                        y_pos_interior_mask[i, j - 1] = 1

                    if x ** 2 + (y - dy) ** 2 >= 1:
                        b = (math.sqrt(1 - x ** 2) + y) / dy
                        y_neg_near_bdy_mask[i, j - 1] = np.array([b, x, -math.sqrt(1 - x ** 2)])
                    else:
                        y_neg_interior_mask[i, j - 1] = 1
        
        # Input shape (Nx + 1, Ny + 1), output shape (Nx - 1, Ny + 1)
        def d_x_pos(vals: np.ndarray, t: float):
            a = x_pos_interior_mask * (vals[2:, :] - vals[1:-1, :])
            b = x_pos_near_bdy_mask[:, :, 0] * (f_bdy(x_pos_near_bdy_mask[:, :, 1:], t) - vals[1:-1, :])
            return a + b
        
        def d_x_neg(vals: np.ndarray, t: float):
            a = x_neg_interior_mask * (vals[:-2, :] - vals[1:-1, :])
            b = x_neg_near_bdy_mask[:, :, 0] * (f_bdy(x_neg_near_bdy_mask[:, :, 1:], t) - vals[1:-1, :])
            return a + b
        
        def d_y_pos(vals: np.ndarray, t: float):
            a = y_pos_interior_mask * (vals[:, 2:] - vals[:, 1:-1])
            b = y_pos_near_bdy_mask[:, :, 0] * (f_bdy(y_pos_near_bdy_mask[:, :, 1:], t) - vals[:, 1:-1])
            return a + b
        
        def d_y_neg(vals: np.ndarray, t: float):
            a = y_neg_interior_mask * (vals[:, :-2] - vals[:, 1:-1])
            b = y_neg_near_bdy_mask[:, :, 0] * (f_bdy(y_neg_near_bdy_mask[:, :, 1:], t) - vals[:, 1:-1])
            return a + b
        
        # Input shape (Nx + 1, Ny + 1), output shape (Nx - 1, Ny + 1)
        def l_x(vals: np.ndarray, t: float):
            return d_x_pos(vals, t) + d_x_neg(vals, t)
        
        # Input shape (Nx + 1, Ny + 1), output shape (Nx + 1, Ny - 1)
        def l_y(vals: np.ndarray, t: float):
            return d_y_pos(vals, t) + d_y_neg(vals, t)
        
        # Input shape (Nx + 1, Ny + 1), output shape (Nx + 1, Ny + 1)
        def laplacian(vals: np.ndarray, t: float):
            lx = np.concatenate((np.zeros((1, Ny + 1)), l_x(vals, t), np.zeros((1, Ny + 1))), axis=0)
            ly = np.concatenate((np.zeros((Nx + 1, 1)), l_y(vals, t), np.zeros((Nx + 1, 1))), axis=1)
            return lx + ly
        
        # TODO This function is designed to set the values at the point sources and at any boundaries which have not already been handled.
        def add_boundary_values(vals: np.ndarray, t: float):
            return vals
        
        # Temporal resolution on f
        Nt = 10000
        total_t = 30.0
        self.total_t, self.Nt = total_t, Nt
        dt = total_t / Nt

        # Initial value and its time derivative. Input is shape (..., 2), output is shape (2, ...)
        k = 1
        def f0(p: np.ndarray):
            r = np.linalg.norm(p, axis=-1)
            mask = (r <= 1)
            vals = mask * np.exp(-r) * np.cos(r * np.pi * (k + 0.5))
            return np.stack((vals, np.zeros_like(vals)), axis=0)
        
        # Vals_and_df is of shape (2, Nx + 1, Ny + 1), and represents (f(X, Y, t), d_t * f(X, Y, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1], laplacian(vals_and_df[0], t)), axis=0)
            p1 = add_boundary_values(vals_and_df + (dt / 2) * k1, t + dt/2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1], laplacian(p1[0], t + dt / 2)), axis=0)
            p2 = add_boundary_values(vals_and_df + (dt / 2) * k2, t + dt/2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1], laplacian(p2[0], t + dt / 2)), axis=0)
            p3 = add_boundary_values(vals_and_df + dt * k3, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1], laplacian(p3[0], t + dt)), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)
            return new_vals_and_df
        
        # Set initial values
        inputs = np.stack([
            np.stack([
                np.array([x, y])
                for y in np.linspace(ymin, ymax, Ny + 1)], axis=0)
            for x in np.linspace(xmin, xmax, Nx + 1)], axis=0)
        vals = f0(inputs)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        self.make_animation(result)
    
    def unbounded_dipole(self):
        """Animates a wave propagating on an unconstrained plane from a pair of opposing sources at (0, d/2) and (0, -d/2) with respective strengths +a and -a and where the second source is delayed from the first by T = d / c (where c=1)."""
        d = 0.5
        a = 4.0 / d

        xmin, xmax = -5.0, 5.0
        ymin, ymax = -5.0, 5.0
        xdiff = xmax - xmin
        ydiff = ymax - ymin

        # Spatial resolution for f
        Nx = 40
        assert Nx % 2 == 0
        dx = xdiff / Nx
        Ny = 40
        assert Ny % 2 == 0
        dy = ydiff / Ny

        # Dipole check condition
        ind = (d / 2) / (ydiff / Ny)
        assert math.isclose(ind, int(ind))
        ind = int(ind)

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx, self.Ny = Nx, Ny

        # Temporal resolution on f
        Nt = 10000
        total_t = 15.0
        dt = total_t / Nt

        self.total_t, self.Nt = total_t, Nt

        # Initial value and its time derivative
        def f0(x: float, y: float):
            return np.array([
                0,
                0
                ])
        
        # Value and its time-derivative at the two point sources
        w = 3.0
        def fpos(t: float):
            return a * np.array([
                np.sin(w * t),
                w * np.cos(w * t)
            ])
        def fneg(t: float):
            return -a * np.array([
                np.sin(w * (t - d)),
                w * np.cos(w * (t - d))
            ])
        
        # TODO This function is designed to set the values at the point sources and at any boundaries.
        def add_boundary_values(vals: np.ndarray, t: float):
            vals[:, Nx // 2, ind + Ny // 2] = fpos(t)
            vals[:, Nx // 2, -ind + Ny // 2] = fneg(t)
            return vals
        
        # Vals is of shape (Nx + 1, Ny + 1) and represents f(X, Y, t).
        # Outputs is of shape (Nx + 1, Ny + 1).
        # TODO This function is written to handle an open boundary at both xmin and xmax.
        def l_x(vals: np.ndarray):
            l = (vals[:-2, :] + vals[2:, :] - 2 * vals[1:-1, :])
            bdy_min = np.array([2 * l[0] - l[1]])
            bdy_max = np.array([2 * l[-1] - l[-2]])
            return np.concatenate((bdy_min, l, bdy_max), axis=0) / (dx ** 2)


        # TODO This function is written to handle an open boundary at both ymin and ymax.
        def l_y(vals: np.ndarray):
            l = (vals[:, :-2] + vals[:, 2:] - 2 * vals[:, 1:-1])
            bdy_min = np.expand_dims(2 * l[:, 0] - l[:, 1], axis=1)
            bdy_max = np.expand_dims(2 * l[:, -1] - l[:, -2], axis=1)
            return np.concatenate((bdy_min, l, bdy_max), axis=1) / (dy ** 2)
        
        def laplacian(vals: np.ndarray):
            return l_x(vals) + l_y(vals)

        # Vals_and_df is of shape (2, N + 1, N + 1), and represents (f(X, Y, t), d_t * f(X, Y, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1], laplacian(vals_and_df[0])), axis=0)
            p1 = add_boundary_values(vals_and_df + (dt / 2) * k1, t + dt / 2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1], laplacian(p1[0])), axis=0)
            p2 = add_boundary_values(vals_and_df + (dt / 2) * k2, t + dt / 2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1], laplacian(p2[0])), axis=0)
            p3 = add_boundary_values(vals_and_df + dt * k3, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)
            return new_vals_and_df

        # Set initial values
        vals = np.stack(
            [
                np.stack([f0(x, y) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
                for y in np.linspace(ymin, ymax, Ny + 1)
            ], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        self.make_animation(result)


    def unbounded_point_source(self):
        """Animates a wave propagating on an unconstrained plane from a single source at x = y = 0."""
        
        xmin, xmax = -5.0, 5.0
        ymin, ymax = -5.0, 5.0
        xdiff = xmax - xmin
        ydiff = ymax - ymin

        # Spatial resolution for f
        Nx = 40
        assert Nx % 2 == 0
        dx = xdiff / Nx
        Ny = 40
        assert Ny % 2 == 0
        dy = ydiff / Ny

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx, self.Ny = Nx, Ny

        # Temporal resolution on f
        Nt = 10000
        total_t = 15.0
        dt = total_t / Nt

        self.total_t, self.Nt = total_t, Nt

        # Initial value and its time derivative
        def f0(x: float, y: float):
            return np.array([
                0,
                0
                ])
        
        # Value and its time-derivative at the point source
        # TODO When handling multiple point sources, we will define one of these functions for every point source
        w = 3.0
        def fpoint(t: float):
            return np.array([
                np.sin(w * t),
                w * np.cos(w * t)
            ])
        
        # TODO This function is designed to set the values at the point sources and at any boundaries.
        def add_boundary_values(vals: np.ndarray, t: float):
            vals[:, Nx // 2, Ny // 2] = fpoint(t)
            return vals
        
        # Vals is of shape (Nx + 1, Ny + 1) and represents f(X, Y, t).
        # Outputs is of shape (Nx + 1, Ny + 1).
        # TODO This function is written to handle an open boundary at both xmin and xmax.
        def l_x(vals: np.ndarray):
            l = (vals[:-2, :] + vals[2:, :] - 2 * vals[1:-1, :])
            bdy_min = np.array([2 * l[0] - l[1]])
            bdy_max = np.array([2 * l[-1] - l[-2]])
            return np.concatenate((bdy_min, l, bdy_max), axis=0) / (dx ** 2)


        # TODO This function is written to handle an open boundary at both ymin and ymax.
        def l_y(vals: np.ndarray):
            l = (vals[:, :-2] + vals[:, 2:] - 2 * vals[:, 1:-1])
            bdy_min = np.expand_dims(2 * l[:, 0] - l[:, 1], axis=1)
            bdy_max = np.expand_dims(2 * l[:, -1] - l[:, -2], axis=1)
            return np.concatenate((bdy_min, l, bdy_max), axis=1) / (dy ** 2)
        
        def laplacian(vals: np.ndarray):
            return l_x(vals) + l_y(vals)

        # Vals_and_df is of shape (2, N + 1, N + 1), and represents (f(X, Y, t), d_t * f(X, Y, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1], laplacian(vals_and_df[0])), axis=0)
            p1 = add_boundary_values(vals_and_df + (dt / 2) * k1, t + dt / 2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1], laplacian(p1[0])), axis=0)
            p2 = add_boundary_values(vals_and_df + (dt / 2) * k2, t + dt / 2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1], laplacian(p2[0])), axis=0)
            p3 = add_boundary_values(vals_and_df + dt * k3, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)
            return new_vals_and_df

        # Set initial values
        vals = np.stack(
            [
                np.stack([f0(x, y) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
                for y in np.linspace(ymin, ymax, Ny + 1)
            ], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        self.make_animation(result)

    def rectilinear_compact(self):
        # Bounds
        xmin, xmax = 0.0, 1.0
        ymin, ymax = 0.0, 1.0
        xdiff = xmax - xmin
        ydiff = ymax - ymin

        # Number of sample points in each dimension
        Nx = 20
        Ny = 20
        dx = xdiff / Nx
        dy = ydiff / Ny

        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.Nx, self.Ny = Nx, Ny

        # Temporal resolution on f
        Nt = 1000
        total_t = 3.0
        dt = total_t / Nt

        self.total_t, self.Nt = total_t, Nt

        # Initial sheet value and its time derivative
        def f0(x: float, y: float):
            return np.array([np.sin(3 * np.pi * x) * np.sin(3 * np.pi * y), 0])

        # Spatial boundaries: values and time derivatives. Used for stacking onto the laplacian-advanced function values on the domain interior
        def f_xmin(t: float):
            return np.zeros((2, 1, Ny - 1))
        def f_xmax(t: float):
            # Shape (2, 1, Ny - 1)
            return np.zeros((2, 1, Ny - 1))
        def f_ymin(t: float):
            # Shape (2, Nx + 1, 1)
            return np.zeros((2, Nx + 1, 1))
        def f_ymax(t: float):
            # Shape (2, Nx + 1, 1)
            return np.zeros((2, Nx + 1, 1))

        def add_boundary_values(vals: np.ndarray, t: float):
            return np.concatenate((
                f_ymin(t),
                np.concatenate((f_xmin(t), vals, f_xmax(t),), axis=-2),
                f_ymax(t),),
                axis=-1)

        
        # Vals is of shape (N + 1, N + 1) and represents f(X, Y, t).
        # Outputs is of shape (N - 1, N - 1)
        def laplacian(vals: np.ndarray):
            l_x = (vals[:-2, 1:-1] + vals[2:, 1:-1] - 2 * vals[1:-1, 1:-1]) / (dx ** 2)
            l_y = (vals[1:-1, :-2] + vals[1:-1, 2:] - 2 * vals[1:-1, 1:-1]) / (dy ** 2)
            return l_x + l_y
        
        # Vals_and_df is of shape (2, N + 1, N + 1), and represents (f(X, Y, t), d_t * f(X, Y, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1, 1:-1, 1:-1], laplacian(vals_and_df[0])), axis=0)
            p1 = add_boundary_values(vals_and_df[:, 1:-1, 1:-1] + (dt / 2) * k1, t + dt/2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1, 1:-1, 1:-1], laplacian(p1[0])), axis=0)
            p2 = add_boundary_values(vals_and_df[:, 1:-1, 1:-1] + (dt / 2) * k2, t + dt/2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1, 1:-1, 1:-1], laplacian(p2[0])), axis=0)
            p3 = add_boundary_values(vals_and_df[:, 1:-1, 1:-1] + dt * k3, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1, 1:-1, 1:-1], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df[:, 1:-1, 1:-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)
            return new_vals_and_df

        # Set initial values
        vals = np.stack(
            [
                np.stack([f0(x, y) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
                for y in np.linspace(ymin, ymax, Ny + 1)
            ], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        self.make_animation(result)

    def construct(self):
        # self.set_camera_orientation(phi=45 * m.DEGREES, theta=-30 * m.DEGREES, zoom=1.)
        self.set_camera_orientation(phi=0, theta=0, zoom=0.75)
        # self.circular_compact()

        # self.set_camera_orientation(phi=45 * m.DEGREES, theta=-30 * m.DEGREES, zoom=0.5)
        # self.unbounded_point_source()
        self.unbounded_dipole()
        # self.rectilinear_compact()



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

    (1) t >= 0 and xmin <= x <= xmax, where f(xmin, t), f(xmax, t), and f(x, 0) are given functions.
        Represents a string controlled at two ends.

    (2) t >= 0 and x >= xmin, where f(xmin, t) and f(x, 0) are given functions.
        Represents a string controlled at one end and unbounded at the other
    
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
        
        def add_boundary_values(vals: np.ndarray, t: float):
            return np.concatenate((fmin(t), vals), axis=-1)
        
        # Vals_and_df is of shape (2, N + 1), and represents (f(X, t), d_t * f(X, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1, 1:], laplacian(vals_and_df[0])), axis=0)

            # Step forward to f(x, t + dt/2) via the derivative
            # Step forward to d_x * f(x, t + dt/2) via the Laplacian
            # Append on boundary values
            p1 = add_boundary_values(vals_and_df[:, 1:] + (dt / 2) * k1, t + dt / 2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1, 1:], laplacian(p1[0])), axis=0)
            p2 = add_boundary_values(vals_and_df[:, 1:] + (dt / 2) * k2, t + dt / 2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1, 1:], laplacian(p2[0])), axis=0)
            p3 = add_boundary_values(vals_and_df[:, 1:] + dt * k2, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1, 1:], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df[:, 1:] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)
            return new_vals_and_df
        
        # Set initial values
        vals = np.stack([f0(x) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        # Make animation
        self.make_animation(result, l=2.0, a=0.5)
    
    def double_constrained(self):
        """String constrained/controlled at two endpoints."""
        # Set all params
        self.xmin, self.xmax = 0.0, 1.0
        self.Nx = 50
        self.Nt = 1000
        self.total_t = 5.0

        # Spatial bounds for f
        xmin, xmax = self.xmin, self.xmax
        xdiff = xmax - xmin

        # Number of sample points for f
        Nx = self.Nx
        dx = xdiff / Nx

        # Temporal resolution on f
        Nt = self.Nt
        total_t = self.total_t
        dt = total_t / Nt

        # Initial string value and its time derivative
        def f0(x: float):
            fourier_coeffs = [1.0, -0.5]
            return sum(np.array([
                coeff * np.sin(np.pi * (w + 1) * (x - xmin) / xdiff),
                0
                ]) for w, coeff in enumerate(fourier_coeffs))

        # Left boundary value and its time derivative
        def fmin(t):
            return np.array([
                [0],
                [0]
            ])
        
        # Right boundary value and its time derivative
        def fmax(t):
            return np.array([[0], [0]])


        # Vals is of shape (N + 1,), and represents f(X, t).
        # Output is of shape (N - 1,)
        def laplacian(vals: np.ndarray):
            return (vals[:-2] + vals[2:] - 2 * vals[1:-1]) / (dx ** 2)

        def add_boundary_values(vals: np.ndarray, t: float):
            return np.concatenate((fmin(t), vals, fmax(t)), axis=-1)

        # Vals_and_df is of shape (2, N + 1), and represents (f(X, t), d_t * f(X, t))
        def step(vals_and_df: np.ndarray, t: float, dt: float):
            # TODO Refactor this
            ## Iteration 1 of RK

            # Derivative of vector
            k1 = np.stack((vals_and_df[1, 1:-1], laplacian(vals_and_df[0])), axis=0)

            # Step forward to f(x, t + dt/2) via the derivative
            # Step forward to d_x * f(x, t + dt/2) via the Laplacian
            # Append on boundary values
            p1 = add_boundary_values(vals_and_df[:, 1:-1] + (dt / 2) * k1, t + dt / 2)

            ## Iteration 2 of RK
            k2 = np.stack((p1[1, 1:-1], laplacian(p1[0])), axis=0)
            p2 = add_boundary_values(p1[:, 1:-1] + (dt / 2) * k2, t + dt / 2)

            ## Iteration 3 of RK
            k3 = np.stack((p2[1, 1:-1], laplacian(p2[0])), axis=0)
            p3 = add_boundary_values(p2[:, 1:-1] + dt * k2, t + dt)

            ## Iteration 4 of RK
            k4 = np.stack((p3[1, 1:-1], laplacian(p3[0])), axis=0)
            
            ## Calculate new vals
            new_vals_and_df = add_boundary_values(vals_and_df[:, 1:-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4), t + dt)

            return new_vals_and_df


        # Set initial values
        vals = np.stack([f0(x) for x in np.linspace(xmin, xmax, Nx + 1)], axis=-1)
        result = [vals[0].copy()]

        # Iterate to get subsequent values
        # TODO Could do this when building the animation, so that it can be calculated to frames.
        t = 0
        for _ in range(Nt):
            vals = step(vals, t, dt)
            t += dt
            result.append(vals[0].copy())
        
        result = np.stack(result, axis=-1)

        # Make animation
        self.make_animation(result, l=5.0, a=1.0)

    def make_animation(
        self,
        result: np.ndarray,
        l: float = 5.0, # Scale of width
        a: float = 1.0, # Scale of amplitude
        ):
        """Animates the vibrating string."""
        # TODO could make this controlled by a ValueTracker.
        xmin, xmax = self.xmin, self.xmax
        Nx = self.Nx
        dx = (xmax - xmin) / Nx

        total_t = self.total_t
        Nt = self.Nt
        # Make animation
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
        self.double_constrained()
        # self.single_constrained()


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