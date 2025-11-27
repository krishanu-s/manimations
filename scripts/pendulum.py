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
the period of oscillation is directly proportional to the square root of the length,
and is roughly independent of the initial swing angle.

For example, multiply the length by four, and the total swing time doubles.
Cut it in half, and the swing time divides by the square root of two.

Now, this independence of the swing angle starts to break down for larger swing angles.
The oscillation period is a bit longer when the swing angle is larger.

(animation)

(Draw a curve showing the oscillation period as a function of the initial angle.)

The dependence between the two isn't a very simple one, either. Deriving this curve requires
using infinite series, and the value is closely related to so-called elliptic integrals.

In any case, what it means is that a simple pendulum clock will become inaccurate by the
end of the day, especially if you move it around.

HISTORY OF TIMEKEEPING

Sources:
- https://museum.seiko.co.jp/en/knowledge/inventors_01/
- Wikipedia...

But who needs such an accurate timekeeping device, anyways?

(History of pendulum timekeeping. 
- Galileo studying pendulums in early 1600s, and noticed they are roughly isochronous.
  Measurement error led him to think so, but Mersenne and others discovered they are not
- Huygens proposed pendulum, and patented it in 1658.
- Hooke in 1658.
- Huygens in 1673 mathematical treatise on timekeeping and oscillation. Precursor to modern
  variational calculus. Introduced cycloidal pendulum. While the mathematics was sound,
  the engineering made it useless.
- Hooke and the introduction of springs to regulate clocks
- Huygens in 1675 translated this design into the spiral spring and balance wheel
  pocket watch

Relate to the problem of calculating longitude. Calculating the longitude of one's present
location is equivalent to finding the time difference to another point of known longitude.
So, calibrate a clock at the time you depart your origin, and thereafter you can calculate
longitude to the level of accuracy of your clock.
1 hour of time difference equals 15 degrees. 1 minute is a quarter degree, or 15 nautical miles.

Any time you reach a new location whose longitude and latitude you know, you can
recalibration based on knowing the time of sunrise (based on calendar)


This was an observation pointed out in the 1500s by a Dutch mapmaker, Gemma Frisius. Coincidentally
this lands around the beginning of the colonial eras of the Dutch and British.

In fact, this was considered such an important problem that the British government formed
a Board of Longitude in 1714, offering large prizes to any inventor who could produce an accurate
method to measure longitude at sea, and materially supporting certain individuals
working on a solution.

Source: https://en.wikipedia.org/wiki/History_of_longitude#Government_initiatives
(Even before this, there had been other such instances. Starting with Philip II of Spain in 1567
who offered a prize for solving the longitude problem. Many observatories established to
solve this very problem. Method of lunar distances.)

(State power guiding mathematical inquiry in the pursuit of conquest.)

The Empire's colonization efforts would be helped significantly by an accurate timekeeping
device, because it would allow them to both create accurate maps through surveying and also
place oneself on a map through celestial navigation. The required accuracy to estimate longitude
to within one nautical mile, would be a few seconds' precision. Needed to maintain this on a
moving ship. Eventually after centuries, culminated with John Harrison in 1770s who built
an accurate marine chronometer.


(Then show the pendulum next to a spring and the difference in the differential equations.)

(Aside on timekeeping and calibration. How this was problematic for long-term accuracy
of pendulum clocks, and also for transporting clocks. Mention figures on the daily error
of clocks.)

(Pendulums and precision timekeeping. Harmonic oscillators.)

(Longitude problem.)

(Also, who even wanted to keep accurate time anyways? Talk about navigation. Astronomy.
British crown and the longitude problem. In service of colonization and trade. Research this.)

(Focus on Christiaan Huygens and the challenge of the accurate pendulum clock. Draw the
idea: if the pendulum is ideally constricted along the sides, it moves faster towards
the edges and its period thus becomes independent of the initial angle. "Isochronous")

(Now go into mathematics to show that this is a cycloid.)

(Eventually, technology moved onto spring-based watches for accuracy and gravity-independence.
What we got from this era is a nice piece of mathematics -- and an important part of its story
is knowing how it arose.)
"""

from lib import ParametrizedHomotopy, RungeKuttaAutonomous


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

        try:
            sol = self.analytic_solution_angle(theta0, omega0)
            return [sol(k * dt) for k in range(0, num_steps + 1)]
        except:
            self.set_initial_conditions(theta0, omega0)
            vals = [self.solver.val.to_array()[0]]
            for _ in range(num_steps):
                self.step(dt, 10)
                vals.append(self.solver.val.to_array()[0])
            return vals
        
    def analytic_solution_angle(self, theta0: float, omega0: float) -> Callable[[float], float]:
        """Given (θ(0), θ'(0)), returns an explicit function θ(t) describing the motion of the
        pendulum, where θ(t) is the angle."""
        raise NotImplementedError

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
    
    def analytic_solution_arc_length(self, x0: float, v0: float) -> Callable[[float], float]:
        """Given (x(0), x'(0)), returns an explicit function x(t) describing the motion of the
        pendulum, where x(t) is the arc position."""
        theta0 = self.arc_length_to_angle(x0)
        omega0 = self.arc_length_to_angle(v0)
        f = self.analytic_solution_angle(theta0, omega0)
        return lambda t: self.angle_to_arc_length(f(t))
    
    def analytic_solution_angle(self, theta0: float, omega0: float) -> Callable[[float], float]:
        """Given (θ(0), θ'(0)), returns an explicit function θ(t) describing the motion of the
        pendulum, where θ(t) is the angle.
        Ref: https://en.wikipedia.org/wiki/Jacobi_elliptic_functions"""
        if omega0 == 0:
            k = np.sin(theta0 / 2)
            def soln(t):
                _, cn, dn, _ = ellipj(math.sqrt(1 / self.l) * t, k**2)
                cd = cn / dn
                return 2 * np.arcsin(k * cd)
            return soln
        else:
            raise NotImplementedError("Analytic solution not computed for nonzero initial velocity")
    
    def period(self, theta: float, eps: float = 1e-4) -> float:
        """
        Given the maximal angle of swing, returns the length of a single period of oscillation.
        Calculates to precision epsilon.
        Ref: https://math.stackexchange.com/questions/2257095/exact-period-of-simple-pendulum
        """
        # k is between -1/sqrt(2) and 1/sqrt(2)
        k = np.sin((theta + math.pi/2)/2)
        summand = 1
        total = summand
        i = 0
        while summand > eps:
            i += 1
            summand *= (k * (2*i-1)/(2*i)) ** 2
            total += summand
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
    
    (*) y'(θ)/x'(θ) = tan(θ), i.e. the slope to the horizontal is θ. This is essentially
        a "tautological" consequence of any path parametrization.
    (**) √((x'(θ))^2 + (y'(θ))^2) = L * cos(θ), i.e. the arc length increases at this rate.
        This is a property that is needed to make the differential equation for v(t)
        into that of a harmonic oscillator.

    In fact, these two properties are enough to retrieve the parametrizations equations, as

    √((x'(θ))^2 + (y'(θ))^2) = x'(θ) * √(1 + tan^2(θ)) = x'(θ) / cos(θ)
    => x'(θ) = L * cos^2(θ) = (L/2) * (1 + cos(2θ))
    which implies
    => x(θ) = (L/2) * (θ + sin(2θ)/2) + x(0)
    and also implies
    => y'(θ) = L * cos(θ) * sin(θ) = (L/2) * sin(2θ)
    => y(θ) = (L/2) * (cos(0)/2 - cos(2θ)/2) + y(0)


    Let's say that the linear velocity of the bob with respect to time t is given by
    v(t), and that the bob is at position (x(θ), y(θ)) for some time-dependent θ(t).
    By the first equation, the linear acceleration of the pendulum bob is g * sin(θ(t)),
    i.e.
    
    (1) v'(t) = - g * sin(θ(t)).

    We can express the angular velocity as 
    
    (2) θ'(t) = v(t) / (L * cos(θ(t)))
    
    Thus, v''(t) = - g * cos(θ(t)) * θ'(t) = - (g/L) * v(t). It follows that v(t)
    (and also the position of the bob) satisfies Hooke's Law and always oscillates with
    frequency ω = √(g/L). The general solution is v(t) = Asin(ωt) + Bcos(ωt)
    where A^2 + B^2 <= g/L.

    If the pendulum string has length L, equation (2) becomes θ' = v / (L * cos(θ)),
    yielding v'' = - (g/L) * v or x'' = (-g/L) * x

    MODELING THE OSCILLATION:
    We can either explicitly express the angle θ(t) as arcsin(v'(t)/g) for a valid solution
    v(t), or we can integrate the following differential equation for θ:

    θ'' = tan(θ) * (-g/L + (θ')^2)

    OBTAINING THE PARAMETRIZATION FROM ISOCHRONICITY:
    Suppose we have a curve we know is isochronous. How would we deduce it is a cycloid?
    Equations (*) and (**) combined are enough to solve for x(θ) and y(θ), and thus retrieve
    the cycloid equation. So it's enough to deduce (*) and (**). (*) is tautological, and
    implies that √((x')^2 + (y')^2) = x' / cos(θ). From here, we perform the same substitutions
    as above to get v'' = v * g * (cos^2(θ) / x'(θ)). The last part in parentheses has no
    time-dependence
    """

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
    
    def step(self, dt: float, num_iterations: float = 10):
        if self.solver.val.x > np.pi/2 - dt:
            raise InterruptedError("Not advancing simulation for numerical stability reasons.")
        super().step(dt, num_iterations)
    
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
        self.l = 0.01
        self.length = 4.0
        raise NotImplementedError
    
    def pendulum_kwargs(self):
        raise NotImplementedError

    def construct(self):
        self.set_params()

        # Integrate the differential equation to solve for reference points
        dt = self.run_time / self.num_steps
        pendulum: Pendulum = self.pendulum_type(l=self.l, length=self.length)

        if isinstance(pendulum, IsochronousPendulum):
            self.add(*pendulum.make_blocks())

        def interpolate_vals(vals: list[float], t: float):
            """Given an input t, and a sequence of function output values
            f(0), f(dt), f(2*dt), ..., linearly interpolates between the given values
            to estimate f(t)."""
            if t > (len(vals) - 1) * dt:
                return vals[-1]
            else:
                k = int(t // dt)
                alpha = t / dt - k
                return (1 - alpha) * vals[k] + alpha * vals[k + 1]
        
        def make_homotopy(theta0, omega0, bob_color=m.BLUE) -> m.Homotopy:
            vals = pendulum.solve(theta0, omega0, dt, self.num_steps)
            angular_position = lambda t: interpolate_vals(vals, t)

            # Add the pendulum string to the scene.
            curve = m.ParametricFunction(
                function=lambda a: pendulum.param_string(a, theta0),
                t_range=(0, 1, 0.1),
                stroke_width = 1.5,
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
            homotopy = ParametrizedHomotopy(
                homotopy=lambda alpha, t: pendulum.param_string(alpha, angular_position(t * self.run_time)),
                mobject=curve, run_time=self.run_time, rate_func=m.linear)
            return homotopy
        
        homotopies = [
            make_homotopy(theta0, 0., bob_color)
            for theta0, bob_color in zip(*self.pendulum_kwargs())]
        self.play(*homotopies)

class SimplePendulumScene(PendulumScene):
    def set_params(self):
        self.pendulum_type = SimplePendulum
        self.run_time = 10.0
        self.num_steps = 200
        self.l = 0.1
        self.length = 4.0
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, 0.5, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors

class IsochronousPendulumScene(PendulumScene):
    def set_params(self):
        self.pendulum_type = IsochronousPendulum
        self.run_time = 10.0
        self.num_steps = 200
        self.l = 0.1
        self.length = 4.0
    
    def pendulum_kwargs(self):
        initial_angles = np.linspace(0.1, np.pi/4, 5)
        colors = [m.BLUE, m.RED, m.ORANGE, m.GREEN, m.YELLOW]
        return initial_angles, colors

