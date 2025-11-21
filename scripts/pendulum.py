# Visualizations associated to a pendulum

import math
from typing import Callable, Any
import numpy as np
from scipy.special import ellipj
import manim as m
from lib import Point2D

"""
(Oscillating pendulum comes into view)
I'm guessing (something about old grandfather clocks)

There's a "fact" many of us learned in high school about pendulums which states that
the period of oscillation is directly proportional to the square root of the length.

(animation)
For example, multiply the length by four, and the total swing time doubles.
Cut it in half, and the swing time divides by the square root of two.

Interestingly this rule-of-thumb says nothing about the swing angle.

(animation)

If you start two identical pendulums off at different initial angles, they'll oscillate
at the same frequency. Well ... (let animation go) ... not exactly. It turns out that the
oscillation period is longer when the initial angle is longer. (Draw a curve showing the
oscillation period as a function of the initial angle. Then show the pendulum next to
a spring and the difference in the differential equations.)

(Aside on timekeeping and calibration. How this was problematic for long-term accuracy
of pendulum clocks, and also for transporting clocks. Mention figures on the daily error
of clocks.)

(Pendulums and precision timekeeping. Harmonic oscillators.)

(Also, who even wanted to keep accurate time anyways? Talk about navigation. Astronomy.
British crown and the longitude problem. In service of colonization and trade. Research this.)

(Focus on Christiaan Huygens and the challenge of the accurate pendulum clock. Draw the
idea: if the pendulum is ideally constricted along the sides, it moves faster towards
the edges and its period thus becomes independent of the initial angle. "Isochronous")

(Now go into mathematics to show that this is a cycloid.)
"""

# Given a function H: [0, 1] x [0, T] -> R^2, animate it as a curve homotopy.
class ParametrizedMobject(m.Mobject):
    """Mobject whose points are indexed by values in [0, 1]."""
    def apply_function(self, func):
        # Apply the function
        pass

class ParametrizedHomotopy(m.Animation):
    """
    A function H: [0, 1] x [0, T] -> R^2.

    At the time this object is created, an entire family of 

    When the second coordinate is restricted to 0, this parametrizes a curve, i.e.
    a m.ParametricFunction object. The m.ParametricFunction object has "init_points"
    which then should be moved according to the homotopy."""
    # TODO
    # First, modify the computation of bezier handles in
    # 
    # `utils.bezier.py:get_smooth_open_cubic_bezier_handle_points()'
    #
    # to be done for an entire *family* of curves with a correspondence between
    # their anchor points.
    #
    # A family of curves (defined by their anchor points) is to be generated
    # at the time the ParametrizedHomotopy object is created, and then all
    # of the handles are computed in a vectorized fashion. These anchors and
    # handles are stored as an attribute of the homotopy. Two possible ways to go about this:
    # 
    # - Call up manim._config.config['frame_rate'] to determine the *exact* timestamps
    #   at which anchors and handles will be needed for the rendering at hand.
    # - Generate a fine enough time-mesh of N curves (i.e. (N, N_pts) anchors), then use these
    #   to calculate the corresponding (N, 2 * N_pts) handles in a vectorized fashion.
    #   Then equip the function "interpolate_submobject(alpha)" and "function_at_time_t"
    #   to linearly interpolate between the nearest anchors/handles, merely calling on the
    #   data that's already stored within the ParametrizedHomotopy.
    # 
    #   If you want to go one step further, the time-evolution itself can be smoothed
    #   in a cubic-bezier way, though this is probably unnecessary.
    def __init__(
        self,
        homotopy: Callable[[float, float], tuple[float, float, float]],
        mobject: ParametrizedMobject,
        run_time: float = 3,
        apply_function_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.homotopy = homotopy
        self.apply_function_kwargs = (
            apply_function_kwargs if apply_function_kwargs is not None else {}
        )
        super().__init__(mobject, run_time=run_time, **kwargs)

    def function_at_time_t(self, t: float) -> tuple[float, float, float]:
        return lambda p: self.homotopy(*p, t)

    def interpolate_submobject(
        self,
        submobject: ParametrizedMobject,
        starting_submobject: ParametrizedMobject,
        alpha: float,
    ) -> None:
        submobject.points = starting_submobject.points
        # For this type of submobject, the points are stored as floats in [0, 1]
        submobject.apply_function(
            self.function_at_time_t(alpha), **self.apply_function_kwargs
        )

# TODO Move these into their own file called "diffeq.py" and standardize the interfaces.
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

# TODO Vectorize this method.
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

# TODO Move these into their own file called "oscillators.py" and standardize the interfaces.
class Pendulum:
    """
    Modeling various kinds of pendulums. The path of the pendulum bob can be parametrized
    as (x(θ), y(θ)) with θ in [-π/2, π/2] where θ is the angle of the path, with θ = 0 being
    the lowest point and θ = \pm π/2 being the endpoints where the path is vertical.
    """
    l: float # 1/ω^2
    length: float # For drawing
    attach_point: np.ndarray[float]

    def __init__(self, l: float, **kwargs):
        self.l = l
        self.set_initial_conditions(0, 0)

        # Coordinates for drawing
        self.attach_point = kwargs.get("attach_point", np.array([0.0, 0.0, 0.0]))
        self.length = kwargs.get("length", 3.0)
    
    def solve(self, theta0: float, omega0: float, dt: float = 0.01, num_steps: int = 100) -> list[float]:
        """Solves the differential equation with given initial conditions θ(0) and θ'(0),
        using an analytic solver if available and otherwise deferring to Runge-Kutta.
        Outputs a list of values θ(t) for t = 0, dt, 2 * dt, ..., N * dt."""

        if hasattr(self, 'analytic_solution_angle') and callable(self.analytic_solution_angle):
            sol = self.analytic_solution_angle(theta0, omega0)
            return [sol(k * dt) for k in range(0, num_steps + 1)]
        else:
            self.set_initial_conditions(theta0, omega0)
            vals = [self.solver.val.to_array()]
            for _ in range(num_steps):
                self.step(dt, 10)
                vals.append(self.solver.val.to_array()[0])
            return vals

    def set_initial_conditions(self, theta0: float, omega0: float):
        """Sets the initial conditions for the pendulum in terms of angular position
         and velocity, and initializes a solver."""
        self.solver = RungeKuttaAutonomous(
            t0=0,
            val=Point2D(theta0, omega0),
            f=self.diff_eq
            )
        
    def diff_eq(self, Point2D) -> float:
        """Differential equation for the second derivative of the angular position
        as a function of t and x. Implemented in the sub-class."""
        raise NotImplementedError
    
    def angle_to_arc_length(self, theta: float) -> float:
        """Converts an angular displacement in the range [-π/2, π/2] to an arc-length displacement."""
        raise NotImplementedError
    
    def arc_length_to_angle(self, x: float) -> float:
        """Converts an arc-length displacement to an angular displacement in the range [-π/2, π/2]."""
        raise NotImplementedError
    
    def kinetic_energy(self, v: float) -> float:
        """Kinetic energy of the pendulum. Used for conservation of energy.
        Implemented in the sub-class."""
        raise NotImplementedError
    
    def potential_energy(self, x: float) -> float:
        """Kinetic energy of the pendulum. Used for conservation of energy.
        Implemented in the sub-class."""
        raise NotImplementedError
    
    def step(self, dt: float, num_iterations: float = 10):
        """Advance the simulation by time dt."""

        # Total energy before update
        # total_energy = self.kinetic_energy(self.v) + self.potential_energy(self.x)

        for _ in range(num_iterations):
            self.solver.step(dt / num_iterations)
        
        # TODO Project the current state (x, v) onto the manifold
        # 0.5 * (l * v) ** 2 + l * (1 - cos(x))
        # TODO Make sure to deal with the cases where x is close to 0 or v is close to 0
        # diff = total_energy - self.kinetic_energy(self.solver.val.y)
        # self.solver.val.x = math.acos(1 - diff / self.l)

class SimplePendulum(Pendulum):
    """Models a simple pendulum whose bob moves along a circular arc."""

    def diff_eq(self, val: Point2D) -> float:
        """Since x'' = -g * sin(θ) and x = L * θ, we get θ'' = (-g/L) * sin(θ)."""
        return -math.sin(val.x) / self.l
    
    def kinetic_energy(self, v: float):
        return 0.5 * (self.l * v) ** 2
    
    def potential_energy(self, x: float):
        return self.l * (1 - math.cos(x))
    
    # def analytic_solution_arc_length(self, x0: float, v0: float) -> Callable[[float], float]:
    #     """Given (x(0), x'(0)), returns an explicit function x(t) describing the motion of the
    #     pendulum, where x(t) is the arc position."""
    #     theta0 = self.arc_length_to_angle(x0)
    #     omega0 = self.arc_length_to_angle(v0)
    #     f = self.analytic_solution_angle(theta0, omega0)
    #     return lambda t: self.angle_to_arc_length(f(t))
    
    # def analytic_solution_angle(self, theta0: float, omega0: float) -> Callable[[float], float]:
    #     """Given (θ(0), θ'(0)), returns an explicit function θ(t) describing the motion of the
    #     pendulum, where θ(t) is the angle."""
    #     # TODO Write the terms as an infinite series. Then based on the given value of t_max,
    #     # decide how many terms of the series to include.
    #     # TODO Express in terms of the jacobi elliptic functions
    #     raise NotImplementedError
    
    def period(self, theta: float, eps: float = 1e-4) -> float:
        """
        Given the maximal angle of swing, returns the length of a single period of oscillation.
        Calculates to precision epsilon.
        Ref: https://math.stackexchange.com/questions/2257095/exact-period-of-simple-pendulum
        """
        # k is between -1/sqrt(2) and 1/sqrt(2)
        k = np.sin((theta + math.pi/2)/2)
        total = 0
        summand = 1
        i = 1
        # TODO Replace this with a "while" condition on the size of the summand.
        while summand > eps:
            total += summand
            summand *= (k * (2*i-1)/(2*i)) ** 2
            i += 1
        return 2 * math.pi * self.l * total

    def angle_to_arc_length(self, theta: float) -> float:
        return self.l * theta
    
    def arc_length_to_angle(self, x: float) -> float:
        return x / self.l

    def param_string(self, a: float, theta: float) -> np.ndarray[float]:
        """Returns 3D coordinates for the points along the pendulum, where theta is the angular
        position and a is the proportion of length along the pendulum."""
        return self.attach_point + a * self.length * np.array([np.sin(theta), - np.cos(theta), 0.])

class IsochronousPendulum(Pendulum):
    """
    Models an isochronous pendulum whose bob moves along a cycloidal arc.
    
    The position of the pendulum is given by x(θ) = (L/4) * (2θ + sin(2θ))
    and y(θ) = (L/4) * (1 - cos(2θ)), where L is the length.

    PROOF OF ISOCHRONICITY:
    The parametric equation for the path satisfies the following two relationships:
    
    - y'(θ)/x'(θ) = tan(θ), i.e. the slope to the horizontal is θ.
    - √((x')^2 + (y')^2) = L * cos(θ), i.e. the arc length increases at this rate.

    Let's say that the linear velocity of the bob with respect to time t is given by
    v(t), and that the bob is at position (x(θ), y(θ)) for some time-dependent θ(t).
    By the first equation, the linear acceleration of the pendulum bob is g * sin(θ(t)),
    i.e.
    
    (1) v' = - g * sin(θ).

    We can express the angular velocity as 
    
    (2) θ' = v / (L * cos(θ))
    
    Thus, v'' = - g * cos(θ) * θ' = - (g/L) * v. It follows that v
    (and also the position of the bob) satisfies Hooke's Law and always oscillates with
    frequency ω = √(g/L). The general solution is v(t) = Asin(ωt) + Bcos(ωt)
    where A^2 + B^2 <= g/L.

    If the pendulum string has length L, equation (2) becomes θ' = v / (L * cos(θ)),
    yielding v'' = - (g/L) * v or x'' = (-g/L) * x

    MODELING THE OSCILLATION:
    We can either explicitly express the angle θ(t) as arcsin(v'(t)/g) for a valid solution
    v(t), or we can integrate the following differential equation for θ:

    θ'' = tan(θ) * (-g/L + (θ')^2)
    """
    # TODO Add a method of forward-solving in terms of linear position on the arc, rather
    # than angle. This is because the differential equation for the angle has numerical
    # instability.

    def parametrization(self, theta: float) -> np.ndarray[float]:
        """Expresses the position (x(θ), y(θ)) of the pendulum bob as a function
        of the slope angle. Used for drawing."""
        if theta == 0:
            return np.array([0, -self.length, 0])
        else:
            return self.parametrization(0) + (self.length / 4) * np.array([
                2 * theta + np.sin(2 * theta),
                1 - np.cos(2 * theta),
                0
                ])
    
    def kinetic_energy(self, v: float):
        # TODO low priority
        raise NotImplementedError
    
    def potential_energy(self, x: float):
        # TODO low priority
        raise NotImplementedError
    
    def angle_to_arc_length(self, theta: float, omega: float | None = None) -> float | tuple[float, float]:
        """Converts an angular displacement (and velocity) in the range [-π/2, π/2] to an arc-length displacement (and velocity)."""
        x = self.l * np.sin(theta)
        if omega is None:
            return x
        else:
            return x, self.l * np.cos(theta) * omega
    
    def arc_length_to_angle(self, x: float, v: float | None = None) -> float | tuple[float, float]:
        """Converts an arc-length displacement (and velocity) to an angular displacement (and velocity) in the range [-π/2, π/2]."""
        assert np.abs(x) <= self.l, "Exceeded the maximum allowed arc length."
        theta = np.arcsin(x / self.l)
        if v is None:
            return theta
        else:
            return theta, v / np.sqrt(self.l ** 2 - self.x ** 2)
    
    def analytic_solution_arc_length(self, x0: float, v0: float) -> Callable[[float], float]:
        """Given (x(0), x'(0)), returns an explicit function x(t) describing the motion of the
        pendulum, where x(t) is the arc position."""
        # x(t) = Acos(ωt) + Bsin(ωt)
        w = 1 / math.sqrt(self.l)
        return lambda t: x0 * np.cos(w * t) + (v0 / w) * np.sin(w * t)
    
    def analytic_solution_angle(self, theta0: float, omega0: float) -> Callable[[float], float]:
        """Given (θ(0), θ'(0)), returns an explicit function θ(t) describing the motion of the
        pendulum, where θ(t) is the angle."""
        x0, v0 = self.angle_to_arc_length(theta0, omega0)
        f = self.analytic_solution_arc_length(x0, v0)
        return lambda t: self.arc_length_to_angle(f(t), None)
    
    def diff_eq(self, val: Point2D):
        """Differential equation for the angular position θ, expressed as θ'' = f(θ, θ')"""
        return np.tan(val.x) * (- 1/self.l + val.y ** 2)
    
    def step(self, dt: float):
        if self.solver.val.x > np.pi/2 - dt:
            raise InterruptedError("Not advancing simulation for numerical stability reasons.")
        super().step(dt)
    
    def wrapped_point(self, a: float) -> np.ndarray[float]:
        """Defines the position of the point on the pendulum string at ratio a when it
        is wrapped against the obstruction."""
        # Angle according to parametrization which yields arc length alpha * L
        alpha = np.arccos(1 - a)
        result = (self.length / 4) * np.array([
            2 * alpha - np.sin(2 * alpha),
            np.cos(2 * alpha) - 1,
            0])
        return result
    
    def make_blocks(self) -> tuple[m.ArcPolygon]:
        block_right = m.Polygon(
            *[self.wrapped_point(a) for a in np.linspace(0, 1, 20)],
            (self.length / 4) * np.array([np.pi, 0, 0]),
            stroke_width=1.0,
            fill_opacity=0.3,
            fill_color=m.BLUE
            )
        block_left = m.Polygon(
            *[np.array([-p[0], p[1], p[1]]) for p in block_right.points],
            stroke_width=1.0,
            fill_opacity=0.3,
            fill_color=m.BLUE
            )
        return block_right,block_left
    
    def param_string(self, a: float, theta: float) -> np.ndarray[float]:
        """Returns 3D coordinates for the points along the pendulum, where theta is the angular
        position and a is the proportion of length along the pendulum."""
        # TODO Fix this for the case where theta is negative.
        if theta == 0:
            return np.array([0., -self.length * a, 0.])
        
        sign = theta / np.abs(theta)

        # Proportion of arc length which is wrapped
        d = np.abs(1 - np.cos(theta))

        if a <= d:
            # Wrapped point at ratio a
            alpha = np.arccos(1 - a)
            return (self.length / 4) * np.array([
                sign * (2 * alpha - np.sin(2 * alpha)),
                np.cos(2 * alpha) - 1,
                0
                ])
        else:
            # Go to the wrapped point at d
            wrapped_point = (self.length / 4) * np.array([
                2 * theta - np.sin(2 * theta),
                np.cos(2 * theta) - 1,
                0
                ])
            
            # Extend out the remaining distance
            return wrapped_point + (a - d) * self.length * np.array([np.sin(theta), - np.cos(theta), 0.])

class Spring:
    """An oscillating spring."""
    pass

class PendulumScene(m.Scene):
    def set_params(self):
        self.pendulum_type = Pendulum
        self.run_time = 3.0
        self.num_steps = 100
        raise NotImplementedError
    
    def pendulum_kwargs(self):
        raise NotImplementedError

    def construct(self):
        self.set_params()

        # Integrate the differential equation to solve for reference points
        dt = self.run_time / self.num_steps
        pendulum: Pendulum = self.pendulum_type(l=0.03, length=4.0)

        if isinstance(pendulum, IsochronousPendulum):
            self.add(*pendulum.make_blocks())

        
        def make_homotopy(theta0, omega0, bob_color=m.BLUE) -> m.Homotopy:
            vals = pendulum.solve(theta0, omega0, dt, self.num_steps)

            def angular_position(t: float) -> float:
                """Given a time t, finds the best-possible approximation to the angular
                position at time t by interpolating between the two nearest points."""
                if t == self.run_time:
                    return vals[-1]
                else:
                    k = int(t // dt)
                    alpha = t / dt - k
                    return (1 - alpha) * vals[k] + alpha * vals[k + 1]


            # Add the pendulum string to the scene.
            curve = m.ParametricFunction(
                function=lambda a: pendulum.param_string(a, theta0),
                t_range=(0, 1, 0.1),
                stroke_width = 1.5,
                # stroke_color = [1., 1., 1.],
                # TODO: This is not efficient time-wise. If the rendering time
                # becomes cumbersome, this could be fixed by modifying the computation
                # of bezier handles in
                # 
                # `utils.bezier.py:get_smooth_closed_cubic_bezier_handle_points()'
                #
                # to be done for an entire *family* of curves with a correspondence between
                # their anchor points. This calculation will have to be bundled into
                # a custom m.Animation-type class we define, called e.g. ParametrizedHomotopy.
                # 
                # The family of curves (defined by their anchor points) is generated
                # at the time the ParametrizedHomotopy object is created,
                # and then all of the handles are computed in a vectorized fashion. These
                # anchors and handles are stored as an attribute of the homotopy. Two possible ways:
                # - Call up manim._config.config['frame_rate'] to determine the exact timestamps
                #   at which anchors and handles will be needed for the animation at hand.
                # - Generate a fine enough time-mesh for the anchors, then use this to calculate
                #   all of the handles in a vectorized fashion. The function "interpolate_submobject(alpha)"
                # 
                # This attribute is then directly used in the "interpolate_submobject" method.
                #
                # A third (perhaps most elegant) possible way: just as the coordinates of the anchor points
                # P_0(t), P_1(t), ... are functions of t, the coordinates of the handles
                # B_1(t), B_2(t), B_3(t), B_4(t), ... are linear combinations of the
                # anchor point functions and can be stored. This is a direct operation on
                # the homotopy function, and can be a subroutine of m.ParametrizedHomotopy.
                make_smooth_after_applying_functions=True
            )
            self.add(curve)

            # Add the bob
            bob = m.Dot(
                point=pendulum.param_string(1, theta0),
                radius=0.08,
                fill_opacity=0.5,
                fill_color=bob_color
                )
            bob.add_updater(lambda z: z.move_to(curve.points[-1]))
            self.add(bob)

            # Define the homotopy describing the swing of the pendulum
            def htpy(x, y, z, t) -> np.ndarray[float]:
                # If the input point is not on the curve, then it's a handle
                # of a Bezier curve and is being re-computed anyways.
                try:
                    a = curve.proportion_from_point(np.array([x, y, z]))
                    theta = angular_position(t)
                    return pendulum.param_string(a, theta)
                except:
                    return np.array([0., 0., 0.])
            
            homotopy = m.Homotopy(htpy, curve, run_time=self.run_time, rate_func=m.linear)
            return homotopy
        
        homotopies = [
            make_homotopy(theta0, 0., bob_color)
            for theta0, bob_color in zip(*self.pendulum_kwargs())]
        self.play(*homotopies)

class SimplePendulumScene(PendulumScene):
    def set_params(self):
        self.pendulum_type = SimplePendulum
        self.run_time = 3.0
        self.num_steps = 500
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, 1.0, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors

class IsochronousPendulumScene(PendulumScene):
    def set_params(self):
        self.pendulum_type = IsochronousPendulum
        self.run_time = 3.0
        self.num_steps = 100
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, np.pi/2, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors
