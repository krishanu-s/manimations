# Visualizations associated to a pendulum

import math
from typing import Callable, Any
import numpy as np
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
    """Modeling various kinds of pendulums."""
    l: float # 1/ω
    length: float # For drawing
    attach_point: np.ndarray[float]

    def __init__(self, l: float, **kwargs):
        self.l = l
        self.set_initial_conditions(0, 0)

        # Coordinates for drawing
        self.attach_point = kwargs.get("attach_point", np.array([0.0, 0.0, 0.0]))
        self.length = kwargs.get("length", 3.0)
    
    @property
    def x(self):
        """Angular position."""
        return self.solver.val.x
    @property
    def v(self):
        """Angular velocity."""
        return self.solver.val.y

    def set_initial_conditions(self, x0: float, v0: float):
        """Sets the initial conditions for the pendulum and initializes a solver."""
        # TODO Make this a RungeKuttaAutonomous
        self.solver = RungeKutta2(t=0, val=Point2D(x0, v0), f=self.diff_eq)
        
    def diff_eq(self, t: float, x: float) -> float:
        """Differential equation for the second derivative of the angular position
        as a function of t and x. Implemented in the sub-class."""
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
    def diff_eq(self, t, x):
        # TODO Make this a function of x, x'
        return -math.sin(x) / self.l
    def kinetic_energy(self, v: float):
        return 0.5 * (self.l * v) ** 2
    def potential_energy(self, x: float):
        return self.l * (1 - math.cos(x))

    def draw_string(self, a: float, theta: float) -> np.ndarray[float]:
        """Returns 3D coordinates for the points along the pendulum, where theta is the angular
        position and a is the proportion of length along the pendulum."""
        return self.attach_point + a * self.length * np.array([np.sin(theta), - np.cos(theta), 0.])

class IsochronousPendulum(Pendulum):
    """Models an isochronous pendulum whose bob moves along a cycloidal arc.
    
    The path of the pendulum bob can be parametrized as (x(θ), y(θ)) with
    θ in [-π/2, π/2] where θ is the angle of the path, with θ = 0 being
    the lowest point and θ = \pm π/2 being the endpoints where the path
    is vertical, by the equations x(θ) = 2θ + sin(2θ) and y(θ) = 1 - cos(2θ).
    This is assuming the pendulum string has length 4.

    PROOF OF ISOCHRONICITY:
    The parametric equation for the path satisfies the following two relationships:
    
    - y'(θ)/x'(θ) = tan(θ), i.e. the slope to the horizontal is θ.
    - √((x')^2 + (y')^2) = 4 * cos(θ), i.e. the arc length increases at this rate.

    Let's say that the linear velocity of the bob with respect to time t is given by
    v(t), and that the bob is at position (x(θ), y(θ)) for some time-dependent θ(t).
    By the first equation, the linear acceleration of the pendulum bob is g * sin(θ(t)),
    i.e.
    
    (1) v' = - g * sin(θ).

    We can express the angular velocity as 
    
    (2) θ' = v / (4 * cos(θ))
    
    Thus, v'' = - g * cos(θ) * θ' = - (g/4) * v. It follows that v
    (and also the position of the bob) satisfies Hooke's Law and always oscillates with
    frequency ω = (√g)/2. The general solution is v(t) = Asin(ωt) + Bcos(ωt)
    where A^2 + B^2 <= 4g.

    If the pendulum string has length L, equation (2) becomes θ' = v / (L * cos(θ)),
    yielding v'' = - (g/L) * v.

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
    
    def set_initial_conditions(self, x0: float, v0: float):
        """Sets the initial conditions for the pendulum in terms of angular position
         and velocity, and initializes a solver."""
        # if x0.abs() > np.pi/2 - 0.1:
        #     print("Cannot set an initial angle too close to pi/2.")
        #     return
            
        self.solver = RungeKuttaAutonomous(
            t0=0,
            val=Point2D(x0, v0),
            f=self.diff_eq
            )
    
    def diff_eq(self, val: Point2D):
        """Differential equation for the angular position θ, expressed as θ'' = f(θ, θ')"""
        return np.tan(val.x) * (- 1/self.l + val.y ** 2)
    
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
    
    def draw_string(self, a: float, theta: float) -> np.ndarray[float]:
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


class PendulumScene(m.Scene):
    @property
    def pendulum_type(self) -> Pendulum:
        raise NotImplementedError
    
    @property
    def run_time(self) -> float:
        raise NotImplementedError
    
    def pendulum_kwargs(self):
        raise NotImplementedError

    def construct(self):

        # Integrate the differential equation to solve for reference points
        num_steps = 1000
        dt = self.run_time / num_steps
        pendulum: Pendulum = self.pendulum_type(l=0.03, length=4.0)

        if isinstance(pendulum, IsochronousPendulum):
            self.add(*pendulum.make_blocks())

        
        # Set the initial conditions and forward-solve
        def get_vals(x0: float, v0: float) -> list[np.ndarray[float]]:
            """Sets the initial conditions into the pendulum and """
            pendulum.set_initial_conditions(x0=x0, v0=v0)
            # TODO alternative: analytically solve and retrieve the values.
            vals = [pendulum.solver.val.to_array()]
            for _ in range(num_steps):
                pendulum.step(dt, 10)
                vals.append(pendulum.solver.val.to_array())
            return vals
        
        def make_homotopy(x0, v0, bob_color=m.BLUE) -> m.Homotopy:
            vals = get_vals(x0=x0, v0=v0)

            def angular_position(t: float) -> float:
                """Given a time t, finds the best-possible approximation to the angular
                position at time t by interpolating between the two nearest points."""
                if t == self.run_time:
                    return vals[-1][0]
                else:
                    k = int(t // dt)
                    alpha = t / dt - k
                    return (1 - alpha) * vals[k][0] + alpha * vals[k + 1][0]


            # Add the pendulum string to the scene.
            curve = m.ParametricFunction(
                function=lambda a: pendulum.draw_string(a, x0),
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
                make_smooth_after_applying_functions=True
            )
            self.add(curve)

            # Add the bob
            bob = m.Dot(
                point=pendulum.draw_string(1, x0),
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
                    return pendulum.draw_string(a, theta)
                except:
                    return np.array([0., 0., 0.])
            
            homotopy = m.Homotopy(htpy, curve, run_time=self.run_time, rate_func=m.linear)
            return homotopy
        
        homotopies = [
            make_homotopy(x0, 0., bob_color)
            for x0, bob_color in zip(*self.pendulum_kwargs())]
        self.play(*homotopies)

class SimplePendulumScene(PendulumScene):
    @property
    def run_time(self) -> float:
        return 10.0
    @property
    def pendulum_type(self):
        return SimplePendulum
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, 1.0, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors

class IsochronousPendulumScene(PendulumScene):
    @property
    def run_time(self) -> float:
        return 10.0
    @property
    def pendulum_type(self):
        return IsochronousPendulum
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, 1.0, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors
