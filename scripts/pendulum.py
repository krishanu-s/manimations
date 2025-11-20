# Visualizations associated to a pendulum

import math
from typing import Callable
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
class ParametrizedHomotopy(m.Animation):
    """
    A function H: [0, 1] x [0, T] -> R^2.

    When the second coordinate is restricted to 0, this parametrizes a curve, i.e.
    a m.ParametricFunction object. The m.ParametricFunction object has "init_points"
    which then should be moved according to the homotopy."""
    pass

class PendulumScene(m.Scene):
    def construct(self):
        # Set the initial parameters of the scene
        run_time = 3.0
        x0 = 0.5
        v0 = 0.0

        # Integrate the differential equation to solve for reference points
        num_steps = 50
        dt = run_time / num_steps
        pendulum = IsochronousPendulum(l=0.02)
        pendulum.set_initial_conditions(x0=x0, v0=v0)
        vals = [pendulum.solver.val.to_array()]
        for _ in range(num_steps):
            pendulum.step(dt)
            vals.append(pendulum.solver.val.to_array())
        assert len(vals) == num_steps + 1

        for v in vals:
            print(v)
        raise InterruptedError

        def angular_position(t: float) -> float:
            """Given a time t, finds the best-possible approximation to the angular
            position at time t by interpolating between the two nearest points."""
            if t == run_time:
                return vals[-1][0]
            else:
                k = int(t // dt)
                alpha = t / dt - k
                return (1 - alpha) * vals[k][0] + alpha * vals[k + 1][0]



        # Add the pendulum string to the scene.
        curve = m.ParametricFunction(
            function=lambda a: pendulum.draw_string(a, x0),
            t_range=(0, 1, 0.1),
            make_smooth_after_applying_functions=True
        )
        self.add(curve)


        def htpy(x, y, z, t) -> np.ndarray[float]:
            # If the input point is not on the curve, then it's a handle
            # of a Bezier curve and is being re-computed anyways.
            try:
                a = curve.proportion_from_point(np.array([x, y, z]))
                theta = angular_position(t)
                return pendulum.draw_string(a, theta)
            except:
                return np.array([0., 0., 0.])
            

        homotopy = m.Homotopy(htpy, curve, run_time=run_time, rate_func=m.linear)

        self.play(homotopy)


        # # Draw a parametrized curve with n + 1 points, and n cubic Bezier curves
        # # connecting the consecutive pairs
        # n = 10
        # x_range = [0, 1, 1/n]

        # def curve_fun(x):
        #     return np.array([np.cos(x), np.sin(x), 0])
        
        # curve = m.ParametricFunction(
        #     function=curve_fun,
        #     t_range=x_range,
        #     # This kwarg ensures that the handles of the Bezier curve
        #     # are manually recomputed after every transformation of the Mobject.
        #     # TODO: This is not efficient time-wise. If the rendering time
        #     # becomes cumbersome, this could be fixed by modifying the computation
        #     # of bezier handles in
        #     # 
        #     # `utils.bezier.py:get_smooth_closed_cubic_bezier_handle_points()'
        #     #
        #     # to be done for an entire *family* of curves with a correspondence between
        #     # their anchor points. This calculation will have to be bundled into
        #     # a custom m.Animation-type class we define, called e.g. ParametrizedHomotopy.
        #     # 
        #     # The family of curves (defined by their anchor points) is generated
        #     # at the time the ParametrizedHomotopy object is created, and then
        #     # all of the handles are computed in a vectorized fashion. These
        #     # anchors and handles are stored as an attribute of the homotopy.
        #     # 
        #     # This attribute is then directly used in the "interpolate_submobject" method.
        #     make_smooth_after_applying_functions=True
        #     )
        # # Consists of a single family member, whose points are those of the curve.

        # self.add(curve)

        # # Apply a homotopy to the curve
        # def curve_homotopy(x, t):
        #     # TODO Find the nearest two t-values in the computed trajectory
        #     # and linearly interpolate between them. to get the theta value.
        #     return np.array([(1+t) * np.cos(x), (1+t) * np.sin(x), 0])
        # def htpy(x, y, z, t):
        #     # If the input point is not on the curve, then it's a handle
        #     # of a Bezier curve and is being re-computed anyways.
        #     try:
        #         a = curve.proportion_from_point(np.array([x, y, z]))
        #         return curve_homotopy(a, t)
        #     except:
        #         return (0, 0, 0)

        # homotopy = m.Homotopy(htpy, curve, run_time=1.0, rate_func=m.smooth)

        # self.play(homotopy)

if __name__ == "__main__":
    """
    This code products a parametrized curve with points P_0, P_1, ..., P_n,
    with P_k = f(k / n). Between each pair (P_{k-1}, P_k), a cubic Bezier curve
    is drawn by trisecting the interval ((k-1)/n, k/n).

    Hence, curve.points consists of 4n points.
    """
    n = 100
    x_range = [0, 1, 1/n]
    def curve_fun(x):
        return np.array([np.cos(x), np.sin(x), 0])

    curve = m.ParametricFunction(function=curve_fun, t_range=x_range)
    # print("t values:", curve.scaling.function(np.arange(0, 1, curve.t_step)))
    # print("t step:", curve.t_step)
    # print("Points array shape:", curve.points.shape)
    # print("Points:", curve.points)
    # t_values = [math.acos(p[0]) for p in curve.points]
    # for t in t_values:
    #     print(t)

    """Next, we continuously deform the curve by moving all intermediate points."""
    def curve_homotopy(x, t):
        return np.array([(1+t) * np.cos(x), (1+t) * np.sin(x), 0])
    
    def htpy(x, y, z, t):
        a = curve.proportion_from_point(np.array([x, y, z]))
        return curve_homotopy(a, t)
    
    m.Homotopy(htpy, curve)


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
    
    def step(self, dt: float):
        """Advance the simulation by time dt."""
        # Number of iterations of Runge-Kutta method to fit
        num_iterations = 10

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
    """
    Models an isochronous pendulum whose bob moves along a cycloidal arc.
    
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

    θ'' = tan(θ) * (-g/L + θ')
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
        raise NotImplementedError
    def potential_energy(self, x: float):
        raise NotImplementedError
    
    def set_initial_conditions(self, x0: float, v0: float):
        """Sets the initial conditions for the pendulum in terms of angular position
         and velocity, and initializes a solver."""
        self.solver = RungeKuttaAutonomous(
            t0=0,
            val=Point2D(x0, v0),
            f=self.diff_eq
            )
    
    def diff_eq(self, val: Point2D):
        """Differential equation for the angular position θ, expressed as θ'' = f(θ, θ')"""
        return np.tan(val.x) * (val.y - 1/self.l)
    
    def wrapped_point(self, alpha: float) -> np.ndarray[float]:
        """Defines the position of the point on the pendulum string at ratio alpha when it
        is wrapped against the obstruction."""
        return np.array([self.length / 4, 0, 0])
    
    def draw_string(self, alpha: float, theta: float) -> np.ndarray[float]:
        """Returns 3D coordinates for the points along the pendulum, where theta is the angular
        position and a is the proportion of length along the pendulum."""
        d = 1 - np.sin(theta) / theta
        if alpha <= d:
            return self.wrapped_point(alpha)
        else:
            return self.wrapped_point(d) + (alpha - d) * self.length * np.array([np.sin(theta), - np.cos(theta), 0.])
