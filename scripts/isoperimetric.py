from __future__ import annotations

import cmath
import math
from collections import deque
from typing import Annotated, Dict, Iterable, Literal, Tuple, Union

import manimlib as m
import numpy as np
from manimlib import *

FloatArray = np.ndarray[int, np.dtype[np.float64]]
Vect2 = Annotated[FloatArray, Literal[2]]
Vect3 = Annotated[FloatArray, Literal[3]]
Vect4 = Annotated[FloatArray, Literal[4]]


def get_angle(v: FloatArray) -> float:
    return (math.atan2(v[1], v[0]) - PI / 2) % TAU


def polar_vec(theta: float) -> Vect3:
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def rot90(v: Vect3) -> Vect3:
    """Rotates the vector counterclockwise by 90 degrees"""
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ v


def normalize_vect3(v: FloatArray) -> FloatArray:
    """Normalizes a vector"""
    return v / np.linalg.norm(v)


def clamp(t: float, tmin: float, tmax: float) -> float:
    return max(min(t, tmax), tmin)


class Fourier:
    cos_coeffs: list[float]
    sin_coeffs: list[float]
    degree: int

    def __init__(self, cos_coeffs, sin_coeffs, degree):
        self.cos_coeffs = cos_coeffs
        self.sin_coeffs = sin_coeffs
        self.degree = degree

        if len(self.cos_coeffs) < degree:
            self.cos_coeffs += [0.0] * (degree - len(self.cos_coeffs))

        if len(self.sin_coeffs) < degree:
            self.sin_coeffs += [0.0] * (degree - len(self.sin_coeffs))

    @classmethod
    def interp_from_values(cls, values: FloatArray) -> Fourier:
        """Interpolates from a given set of values, outputting a new Fourier object"""
        n = len(values)
        assert n % 2 == 0
        d = n // 2
        cos_coeffs = [np.sum(values) / n]
        sin_coeffs = [0]
        for d in range(1, d):
            cos_vector = np.array([math.cos(TAU * d * i / n) for i in range(n)])
            sin_vector = np.array([math.sin(TAU * d * i / n) for i in range(n)])
            cos_coeffs.append(values.dot(cos_vector) * 2 / n)
            sin_coeffs.append(values.dot(sin_vector) * 2 / n)

        cos_coeffs.append(sum((-1) ** i * values[i] for i in range(n)) / n)
        sin_coeffs.append(0)
        return Fourier(cos_coeffs, sin_coeffs, d)

    def compute_val(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x(t) value at parameter t in [0, 1]"""
        cos_term = sum(cf * np.cos(TAU * d * t) for d, cf in enumerate(self.cos_coeffs))
        sin_term = sum(cf * np.sin(TAU * d * t) for d, cf in enumerate(self.sin_coeffs))
        return cos_term + sin_term

    def compute_dval(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x'(t) value at parameter t in [0, 1]"""
        cos_term = sum(
            -cf * (TAU * d) * np.sin(TAU * d * t)
            for d, cf in enumerate(self.cos_coeffs)
        )
        sin_term = sum(
            cf * (TAU * d) * np.cos(TAU * d * t) for d, cf in enumerate(self.sin_coeffs)
        )
        return cos_term + sin_term

    def compute_d2val(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x''(t) value at parameter t in [0, 1]"""
        cos_term = sum(
            -cf * ((TAU * d) ** 2) * np.cos(TAU * d * t)
            for d, cf in enumerate(self.cos_coeffs)
        )
        sin_term = sum(
            -cf * ((TAU * d) ** 2) * np.sin(TAU * d * t)
            for d, cf in enumerate(self.sin_coeffs)
        )
        return cos_term + sin_term

    def scale(self, c: float):
        self.cos_coeffs = [cf * c for cf in self.cos_coeffs]
        self.sin_coeffs = [cf * c for cf in self.sin_coeffs]
        return self


class ClosedCurve(VMobject):
    """A closed curve defined by a function (x(t), y(t)): [0, 1] -> R^2,
    represented by 2d coefficients, i.e.
    cos(0t), cos(2pi*t), cos(4pi*t), ..., cos(2(d-1)pi*t), cos(2dpi*t),
    sin(2pi*t), sin(4pi*t), ..., sin(2(d-1)pi*t).
    Also represented by 2d anchor points, to which the Fourier coefficients are fitted.
    The Fourier representation is considered the source of ground truth for computations."""

    degree: int
    fourier_x: Fourier
    fourier_y: Fourier
    num_segments: int

    @classmethod
    def interpolate_from_points(cls, *anchors) -> ClosedCurve:
        """Interpolates the fourier series from a sequence of points, and then makes the curve accordingly."""
        n = len(anchors)
        assert n % 2 == 0

        curve = ClosedCurve()
        curve.degree = n // 2
        curve.fourier_x = Fourier.interp_from_values(np.array([a[0] for a in anchors]))
        curve.fourier_y = Fourier.interp_from_values(np.array([a[1] for a in anchors]))
        curve.num_segments = n

        # by default, draw a number of Bezier segments equal to the number of anchors
        a = list(anchors) + [anchors[0]]
        h1, h2 = get_smooth_cubic_bezier_handle_points(a)
        for i in range(len(anchors)):
            curve.add_cubic_bezier_curve(a[i], h1[i], h2[i], a[i + 1])

        return curve

    @classmethod
    def make_from_fourier(cls, fourier_x: Fourier, fourier_y: Fourier, n: int):
        assert n % 2 == 0

        curve = ClosedCurve()
        curve.degree = n // 2
        curve.fourier_x = fourier_x
        curve.fourier_y = fourier_y
        curve.num_segments = n

        anchors = [curve.compute_xyz(i / n) for i in range(n)]
        a = list(anchors) + [anchors[0]]
        h1, h2 = get_smooth_cubic_bezier_handle_points(a)
        for i in range(len(anchors)):
            curve.add_cubic_bezier_curve(a[i], h1[i], h2[i], a[i + 1])
        return curve

    def color_by_curvature(self, color_scaling=15.0, num_segments: int = 500) -> VGroup:
        """Makes a color-gradient colored curve where the segments are
        colored according to curvature."""
        anchors = [self.compute_xyz(i / num_segments) for i in range(num_segments)]
        a = list(anchors) + [anchors[0]]
        h1, h2 = get_smooth_cubic_bezier_handle_points(a)
        curve_pieces = VGroup(
            VMobject().add_cubic_bezier_curve(a[i], h1[i], h2[i], a[i + 1])
            for i in range(num_segments)
        )

        # White is curvature 1.
        # Q: Is a log scale better? Or something else to spread out the curvature values?
        cg = color_gradient([RED, WHITE, BLUE], 100)
        curvatures = [self.get_curvature(i / num_segments) for i in range(num_segments)]
        colors = list(
            map(
                lambda c: cg[int(clamp((c - 1) * color_scaling + 50, 0, 99))],
                curvatures,
            )
        )
        for i in range(num_segments):
            curve_pieces[i].set_color(colors[i])

        return curve_pieces

    def scale(self, c: float):
        """Scales the curve points by c"""
        self.fourier_x.scale(c)
        self.fourier_y.scale(c)
        return self

    def compute_area(self, num_steps: int = 100) -> float:
        """Computes the total area by finite-difference approximating
        the integral of (x * dy - y * dx) / 2.
        TODO Vectorize the computations."""
        t_vals = np.array([i / num_steps for i in range(num_steps)])
        x_vals = self.compute_x(t_vals)
        y_vals = self.compute_y(t_vals)
        dx_vals = self.compute_dx(t_vals)
        dy_vals = self.compute_dy(t_vals)
        return np.sum(x_vals * dy_vals - y_vals * dx_vals) / (2 * num_steps)

    def compute_arc_length(self, num_steps: int = 100) -> float:
        """Computes the total arc-length by finite-difference approximating
        the integral of ((x')^2 + (y')^2)^{1/2}."""
        t_vals = np.array([i / num_steps for i in range(num_steps)])
        dx_vals = self.compute_dx(t_vals)
        dy_vals = self.compute_dy(t_vals)
        l_vals = np.pow(np.pow(dx_vals, 2) + np.pow(dy_vals, 2), 0.5)
        return np.sum(l_vals) / num_steps

    def make_bezier_segments(self, num_segments):
        """Resets the drawn version of the curve to higher resolution"""
        # Clear points
        self.set_points(np.empty((0, 3)))

        # Make a new set of anchors at the desired resolution
        anchors = [self.compute_xyz(i / num_segments) for i in range(num_segments)]

        # Add the Bezier curves
        a = list(anchors) + [anchors[0]]
        h1, h2 = get_smooth_cubic_bezier_handle_points(a)
        for i in range(len(anchors)):
            self.add_cubic_bezier_curve(a[i], h1[i], h2[i], a[i + 1])

        self.num_segments = num_segments

        return self

    def make_tangent_circle_at(
        self,
        t: float,
    ) -> Circle:
        """Outputs a tangent circle at the given t value"""
        rc = 1 / self.get_curvature(t)
        return Circle(
            arc_center=self.compute_xyz(t) + self.get_normal_vec(t) * rc,
            radius=abs(rc),
        ).set_style(stroke_opacity=0.6)

    def make_tangent_radius_at(self, t: float) -> DashedLine:
        """Outputs the radius to the tangent circle at the given t value"""
        return DashedLine(
            self.compute_xyz(t),
            self.compute_xyz(t) + self.get_normal_vec(t) * 1 / self.get_curvature(t),
        ).set_style(stroke_opacity=0.6)

    def make_moving_tangent_circle_at(self, t: ValueTracker) -> Circle:
        """Outputs a tangent circle at the given t value"""
        rc = 1 / self.get_curvature(t.get_value())
        circle = Circle(
            arc_center=self.compute_xyz(t.get_value())
            + self.get_normal_vec(t.get_value()) * rc,
            radius=abs(rc),
        )
        circle.add_updater(
            lambda mobj: mobj.become(
                Circle(
                    arc_center=self.compute_xyz(t.get_value())
                    + self.get_normal_vec(t.get_value())
                    / self.get_curvature(t.get_value()),
                    radius=1 / abs(self.get_curvature(t.get_value())),
                )
            )
        )
        return circle

    def get_normal_vec(self, t: float) -> Vect3:
        """Constructs the normal vector, pointing inwards."""
        return rot90(self.get_tangent_vec(t))

    def get_tangent_vec(self, t: float) -> Vect3:
        """Constructs the tangent vector."""
        return normalize_vect3(np.array([self.compute_dx(t), self.compute_dy(t), 0.0]))

    def compute_xyz(self, t: float) -> Vect3:
        """Outputs the point at the given t value"""
        return np.array(
            [self.fourier_x.compute_val(t), self.fourier_y.compute_val(t), 0.0]
        )

    def compute_x(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x(t) value at parameter t in [0, 1]"""
        return self.fourier_x.compute_val(t)

    def compute_y(self, t: float | FloatArray) -> float | FloatArray:
        """Compute y(t) value at parameter t in [0, 1]"""
        return self.fourier_y.compute_val(t)

    def compute_dx(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x'(t) value at parameter t in [0, 1]"""
        return self.fourier_x.compute_dval(t)

    def compute_dy(self, t: float | FloatArray) -> float | FloatArray:
        """Compute y'(t) value at parameter t in [0, 1]"""
        return self.fourier_y.compute_dval(t)

    def compute_d2x(self, t: float | FloatArray) -> float | FloatArray:
        """Compute x''(t) value at parameter t in [0, 1]"""
        return self.fourier_x.compute_d2val(t)

    def compute_d2y(self, t: float) -> float:
        """Compute y''(t) value at parameter t in [0, 1]"""
        return self.fourier_y.compute_d2val(t)

    def get_curvature(self, t: float) -> float:
        """Computes the curvature, using the Fourier interpolation."""
        # Use the Fourier coefficients to compute the curvature, using the formula
        # C = x'(s)y''(s) - x''(s)y'(s)
        # where s parametrizes arc-length changing at rate 1. Equivalently, parametrizing by t,
        # C = (x'(t)y''(t) - x''(t)y'(t)) / ((x')^2 + (y')^2)^{3/2}
        xprime = self.compute_dx(t)
        yprime = self.compute_dy(t)
        return (xprime * self.compute_d2y(t) - self.compute_d2x(t) * yprime) / math.pow(
            xprime**2 + yprime**2, 1.5
        )

    def get_curvatures(self) -> float:
        """Computes the curvature at the i-th anchor, using the Fourier interpolation."""
        if not self.fourier_x_cos:
            self.do_fourier_interp()

        return np.array(
            [
                self.get_curvature(i / self.num_segments)
                for i in range(self.num_segments)
            ]
        )

    def deform_along(self, h_values: list[float]) -> ClosedCurve:
        """Deforms the curve along the given list of h-values.
        Then does a step of spacing out the anchors to ensure they
        don't drift too close together."""
        new_anchors = [
            self.compute_xyz(i / self.num_segments)
            - h_values[i] * self.get_normal_vec(i / self.num_segments)
            for i in range(self.num_segments)
        ]
        new_curve = ClosedCurve.interp_from_values(*new_anchors)
        return new_curve


class ClosedCurveScene(Scene):
    """Base class for scenes involving a closed curve"""

    degree: int = 10
    num_pts: int = 200

    # Construction of a curve straight from Fourier decomposition of the x and y coordinates
    def make_curve(
        self, degree: int | None = None, perturbations: list[float] | None = None
    ):
        # Degree of the Fourier interpolation, also half the number of points taken
        if degree is None:
            degree = self.degree

        # Perturbations to Fourier modes of radius values
        if perturbations is None:
            perturbations = [0.20, -0.15, 0.15, -0.12, -0.02, 0.04]

        n = 2 * degree
        angles = np.linspace(0, TAU * (1 - 1 / n), n)
        radial_vecs = list(map(lambda t: np.array([np.cos(t), np.sin(t), 0]), angles))

        # Define the curve in polar coordinates, where the radius is deformed from a constant function
        # by various cosine/sine curves, scaled by Gaussians.
        radii = np.array([1.0] * n)
        for j, val in enumerate(perturbations):
            radii += val * np.array([np.cos(TAU * i * j / n) for i in range(n)])
            radii += val * np.array([np.sin(TAU * i * j / n) for i in range(n)])
        curve = ClosedCurve.interpolate_from_points(
            *[radii[i] * radial_vecs[i] for i in range(n)]
        )

        # Up the resolution
        curve.make_bezier_segments(self.num_pts)

        return curve

    def construct(self):
        self.embed()


class DefineCurvature(ClosedCurveScene):
    """Link the two definitions of curvature, i.e.
    - (geometric) osculating circle
    - (analytic) derivative of tangent angle
    """

    def construct(self):
        tmin, tmax = 0.01, 0.98
        n = 2 * self.degree
        curve = self.make_curve()
        self.frame.set_height(6.0).move_to(np.array([2, 0, 0]))
        self.play(ShowCreation(curve), run_time=1.0)
        self.wait()

        tangent_tracker = ValueTracker(0.00)

        # Show the curvature at several points, using a circle
        # circle = curve.make_moving_tangent_circle_at(tangent_tracker)
        circle = curve.make_tangent_circle_at(0.0)
        rad = curve.make_tangent_radius_at(0.0)
        pt = Dot(curve.compute_xyz(0), radius=0.05).set_color(BLUE)
        pt.add_updater(
            lambda mobj: mobj.move_to(curve.compute_xyz(tangent_tracker.get_value()))
        )
        self.play(
            ShowCreation(circle), ShowCreation(rad), ShowCreation(pt), run_time=1.0
        )

        self.wait()
        rc_formula = Tex("r_C").set_height(0.15).next_to(rad, UP, 0.1)
        self.play(FadeIn(rc_formula))
        self.play(Indicate(rc_formula), Indicate(rad))
        self.wait()
        self.play(FadeOut(rc_formula))
        self.wait()
        for v in (0.3, 0.55, 0.72, 0.8):
            self.play(
                Transform(circle, curve.make_tangent_circle_at(v)),
                Transform(rad, curve.make_tangent_radius_at(v)),
                tangent_tracker.animate.set_value(v),
                run_time=1.0,
            )
            self.wait()
        circle.add_updater(
            lambda mobj: mobj.become(
                curve.make_tangent_circle_at(tangent_tracker.get_value())
            )
        )
        self.play(
            Transform(circle, curve.make_moving_tangent_circle_at(tangent_tracker))
        )
        rad.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                curve.compute_xyz(tangent_tracker.get_value()),
                curve.compute_xyz(tangent_tracker.get_value())
                + curve.get_normal_vec(tangent_tracker.get_value())
                * 1
                / curve.get_curvature(tangent_tracker.get_value()),
            )
        )
        self.play(tangent_tracker.animate.set_value(tmin), run_time=4.0)
        self.play(FadeOut(circle), FadeOut(rad))

        # Color code according to the curvature

        ## Analytic depiction of curvature
        # Add a graph on the right which shows the tangent vector angle as a function of distance along the curve
        axes = Axes(
            (0, 1),
            (-0.5, TAU + 0.5),
            width=3.0,
            height=3.0,
            y_axis_config={"include_ticks": False},
        ).next_to(curve, RIGHT, 1.5)
        self.play(ShowCreation(axes))

        ax_labels = [
            Tex("\\theta").set_height(0.3).next_to(axes, LEFT, 0.2),
            Text("Distance (t)").set_height(0.3).next_to(axes, DOWN, 0.2),
        ]
        self.play(*[FadeIn(axl) for axl in ax_labels])
        self.wait()

        def tangent_angle_fn(t_val: float):
            return get_angle(curve.get_tangent_vec(t_val))

        def tangent_angle_fn_derivative(t_val: float, delta: float = 1e-3):
            return (
                tangent_angle_fn(t_val + delta) - tangent_angle_fn(t_val - delta)
            ) / (2 * delta)

        graph_of_tangent_angle = ParametricCurve(
            lambda t: axes.c2p(t, tangent_angle_fn(t)),
            (tmin, tmax, 0.01),
        ).set_color(GREEN)

        self.play(ShowCreation(graph_of_tangent_angle))

        # Depict a tangent vector moving along the curve, and a corresponding point

        self.add(tangent_tracker)
        tangent_arr = Arrow(
            curve.compute_xyz(0),
            curve.compute_xyz(0) + curve.get_tangent_vec(0) * 0.5,
            buff=0,
        ).set_color(BLUE)
        tangent_arr.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                curve.compute_xyz(tangent_tracker.get_value()),
                curve.compute_xyz(tangent_tracker.get_value())
                + 0.5 * curve.get_tangent_vec(tangent_tracker.get_value()),
            )
        )
        tangent_angle = VGroup()
        tangent_angle.add(
            Line(
                curve.compute_xyz(0),
                curve.compute_xyz(0) + 0.5 * np.array([0.0, 1.0, 0.0]),
            )
            .set_style(stroke_width=2.0)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    curve.compute_xyz(tangent_tracker.get_value()),
                    curve.compute_xyz(tangent_tracker.get_value())
                    + 0.5 * np.array([0.0, 1.0, 0.0]),
                )
            )
        )
        tangent_angle.add(
            Arc(
                start_angle=PI / 2,
                angle=tangent_angle_fn(0),
                radius=0.2,
                arc_center=curve.compute_xyz(0),
            )
            .set_style(stroke_width=2.0)
            .add_updater(
                lambda mobj: mobj.become(
                    Arc(
                        start_angle=PI / 2,
                        angle=tangent_angle_fn(tangent_tracker.get_value()),
                        radius=0.2,
                        arc_center=curve.compute_xyz(tangent_tracker.get_value()),
                    ).set_style(stroke_width=2.0)
                )
            )
        )
        tangent_graph_pt = Dot(
            axes.c2p(0, tangent_angle_fn(0)),
            radius=0.05,
        ).set_color(BLUE)
        tangent_graph_pt.add_updater(
            lambda mobj: mobj.move_to(
                axes.c2p(
                    tangent_tracker.get_value() % 1.0,
                    tangent_angle_fn(tangent_tracker.get_value() % 1.0),
                )
            )
        )
        self.add(tangent_tracker)
        self.play(FadeIn(tangent_arr), FadeIn(tangent_angle), FadeIn(tangent_graph_pt))

        def slope_line_updater(mobj):
            tval = tangent_tracker.get_value() % 1.0
            fval = tangent_angle_fn(tval)
            pval = axes.c2p(tval, fval)
            m = tangent_angle_fn_derivative(tval)
            dval = normalize(
                axes.c2p(1 / np.sqrt(1 + m**2), m / np.sqrt(1 + m**2)) - axes.c2p(0, 0)
            )
            mobj.put_start_and_end_on(
                pval - 0.3 * dval,
                pval + 0.3 * dval,
            )

        slope_line = (
            Line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
            .set_color(RED)
            .add_updater(slope_line_updater)
        )

        self.play(ShowCreation(slope_line))
        self.wait()

        formula = VGroup(Tex("\\mathrm{curvature} = \\frac{1}{r_C} = \\theta'(t) = "))
        formula.add(
            DecimalNumber()
            .add_updater(
                lambda mobj: mobj.set_value(
                    curve.get_curvature(tangent_tracker.get_value())
                )
            )
            .next_to(formula[0])
        )
        formula.set_width(4.0).next_to(curve, DOWN, 0.5)
        self.play(FadeIn(formula), FadeIn(circle), FadeIn(rad))
        self.wait()
        self.play(
            tangent_tracker.animate.set_value(tmax), run_time=8.0, rate_func=linear
        )


class ArcDeformation(ClosedCurveScene):
    """Show deformation of a small arc of constant curvature"""

    def construct(self):
        curve = self.make_curve()
        self.add(curve)
        self.wait()

        # Zoom in on a specific location
        t_val = 0.35
        rad = 0.02
        self.play(
            self.frame.animate.set_height(0.5).move_to(curve.compute_xyz(t_val)),
            run_time=2.0,
        )
        self.wait()

        circle = curve.make_tangent_circle_at(t_val).set_style(stroke_opacity=0.5)

        # Highlight the arc in consideration
        c = ParametricCurve(
            lambda t: curve.compute_xyz(t), (t_val - rad, t_val + rad, 0.002)
        )
        deformed_c = ParametricCurve(
            lambda t: curve.compute_xyz(t), (t_val - rad, t_val + rad, 0.002)
        ).set_style(stroke_color=GREEN)

        # Deform it
        h = 0.05
        def_arrows = [
            Arrow(
                curve.compute_xyz(t),
                curve.compute_xyz(t) - h * curve.get_normal_vec(t),
                buff=0,
            )
            for t in np.linspace(t_val - rad, t_val + rad, 10)
        ]
        area = Polygon(
            *[curve.compute_xyz(t) for t in np.linspace(0.33, 0.37, 50)],
            *[
                curve.compute_xyz(t) - h * curve.get_normal_vec(t)
                for t in np.linspace(0.37, 0.33, 50)
            ],
        ).set_style(stroke_opacity=0.0, fill_opacity=0.5, fill_color=BLUE)

        dots = VGroup(
            Dot(curve.compute_xyz(t_val - rad), radius=0.005).set_color(GREEN),
            Dot(curve.compute_xyz(t_val + rad), radius=0.005).set_color(GREEN),
        )
        self.play(ShowCreation(deformed_c), ShowCreation(dots))
        self.wait()

        self.add(c)
        self.play(
            *[FadeIn(a) for a in def_arrows],
        )

        self.play(
            Transform(
                deformed_c,
                ParametricCurve(
                    lambda t: curve.compute_xyz(t) - h * curve.get_normal_vec(t),
                    (t_val - rad, t_val + rad, 0.002),
                ).set_style(stroke_color=GREEN),
            ),
        )
        self.wait()
        self.play(
            FadeIn(area),
        )
        self.wait()

        # Draw radial lines to do calculation
        radial_lines = [
            DashedLine(
                curve.compute_xyz(t),
                curve.compute_xyz(t) + 1.0 * curve.get_normal_vec(t),
                dash_length=0.03,
            )
            for t in (t_val - rad, t_val + rad)
        ]
        r_label = (
            Tex("r")
            .set_width(0.02)
            .set_color(radial_lines[1].color)
            .next_to(radial_lines[1], RIGHT, 0.01)
        )
        self.play(*[ShowCreation(rl) for rl in radial_lines], FadeIn(circle))
        self.wait()

        # Add symbols/labels
        arrow_label = (
            Tex("h")
            .set_width(0.02)
            .set_color(def_arrows[0].color)
            .next_to(def_arrows[0], RIGHT, 0.01)
        )
        c_label = Tex("s").set_width(0.02).next_to(curve.compute_xyz(t_val), DR, 0.05)
        deformed_c_label = (
            Tex("s \\cdot \\frac{r+h}{r}")
            .set_height(0.06)
            .set_color(deformed_c.color)
            .next_to(
                curve.compute_xyz(t_val) - h * curve.get_normal_vec(t_val), UL, 0.05
            )
        )
        area_label = (
            Tex("\\frac{s}{2r}((r+h)^2 - r^2)")
            .set_height(0.03)
            .set_color(BLUE)
            .next_to(area, LEFT, -0.05)
        )
        # area_label_arrow =
        self.play(Indicate(c, scale_factor=1.1), FadeIn(c_label))
        self.wait()
        self.play(Indicate(deformed_c, scale_factor=1.1), FadeIn(deformed_c_label))
        self.wait()
        self.play(
            *[Indicate(a, scale_factor=1.1) for a in def_arrows], FadeIn(arrow_label)
        )
        self.wait()
        self.play(Indicate(area, scale_factor=1.02), FadeIn(area_label))

        self.embed()


class GlobalDeformation(ClosedCurveScene):
    """Show a global function h(t) graphed, and the corresponding global
    deformation"""

    def construct(self):
        self.frame.move_to(np.array([0.0, 0.0, 0.0])).set_height(4.0)

        deg = 10
        n = 2 * deg
        curve = self.make_curve(
            deg, [0.25 * np.random.randn() for _ in range(4)]
        ).set_style(stroke_width=2.0)

        # Compute the deformation
        ones_vector = np.ones(n)
        curvatures = np.array([curve.get_curvature(i / n) for i in range(n)])
        projected = curvatures - ones_vector * (ones_vector.dot(curvatures)) / n
        h_values = -0.1 * projected

        # Make the new curve
        new_curve = ClosedCurve.interpolate_from_points(
            *[
                curve.compute_xyz(i / n) - h_values[i] * curve.get_normal_vec(i / n)
                for i in range(n)
            ]
        ).make_bezier_segments(self.num_pts)
        new_curve.set_style(stroke_width=2.0, stroke_color=YELLOW)

        self.add(curve, new_curve)

        # Make arrows
        def make_arrow(t: float):
            start = curve.compute_xyz(t)
            end = new_curve.compute_xyz(t)
            color = BLUE if np.linalg.norm(end) > np.linalg.norm(start) else RED
            return Arrow(
                start,
                end,
                buff=0,
            ).set_color(color)

        arrows = list(map(make_arrow, np.linspace(0, 1 - 1 / 80, 80)))
        self.add(*arrows)


class IsoperimetricFlow(ClosedCurveScene):
    """Showing deformation of a curve a la the isoperimetric inequality, i.e. calculus of variations"""

    def construct(self):
        tmin, tmax = 0.03, 0.97
        deg = 10
        n = 2 * deg
        curve = self.make_curve(deg)

        axes = Axes(
            (0, 1),
            (-0.5, TAU + 0.5),
            width=3.0,
            height=3.0,
            y_axis_config={"include_ticks": False},
        ).next_to(curve, RIGHT, 1.5)

        ax_labels = [
            Tex("\\theta").set_height(0.3).next_to(axes, LEFT, 0.2),
            Text("Distance (t)").set_height(0.3).next_to(axes, DOWN, 0.2),
        ]
        self.add(curve, axes, *ax_labels)
        self.frame.set_height(8.0).move_to(np.array([2, 0, 0]))

        graph_of_tangent_angle = ParametricCurve(
            lambda t: axes.c2p(t, get_angle(curve.get_tangent_vec(t))),
            (tmin, tmax, 0.002),
        ).set_color(GREEN)
        self.add(graph_of_tangent_angle)

        l0 = curve.compute_arc_length()
        # Make a number which tracks area and arc length
        a_text = VGroup()
        a_text.add(Text("A ="))
        a_text.add(DecimalNumber(curve.compute_area()).next_to(a_text[0], RIGHT, 0.3))
        a_text.next_to(curve, DOWN, 1.0)
        l_text = VGroup()
        l_text.add(Text("L ="))
        l_text.add(
            DecimalNumber(curve.compute_arc_length()).next_to(l_text[0], RIGHT, 0.3)
        )
        l_text.next_to(a_text, DOWN, 0.5)
        self.add(a_text, l_text)

        # Define perturbation function at each point in the way which
        # keeps the length fixed and maximally increases the area
        # - Length change = <h_values, curvatures>
        # - Area change   = <h_values, 1>
        # - Normalized: <h_values, h_values> = 1
        # We project 1-vector onto the orthogonal complement of curvatures, then use that.
        ones_vector = np.ones(n)
        arrow_length = 0.1

        # Deformation step.
        for j in range(50):
            print(f"Iteration {j}")
            curvatures = np.array([curve.get_curvature(i / n) for i in range(n)])

            projected = curvatures - ones_vector * (ones_vector.dot(curvatures)) / n
            # h_values = arrow_length * normalize_vect3(projected)
            h_values = -0.2 * arrow_length * projected

            # Define the new curve
            new_curve = ClosedCurve.interpolate_from_points(
                *[
                    curve.compute_xyz(i / n) - h_values[i] * curve.get_normal_vec(i / n)
                    for i in range(n)
                ]
            ).make_bezier_segments(self.num_pts)

            # Calculate the length so the curve can be rescaled
            l = new_curve.compute_arc_length()
            new_curve.scale(l0 / l)

            # Increase the resolution
            new_curve.make_bezier_segments(self.num_pts)

            # Make arrows which represent normal deformation
            arrows = list(
                map(
                    lambda tup: Arrow(
                        curve.compute_xyz(tup[0] / n),
                        curve.compute_xyz(tup[0] / n)
                        - tup[1] * curve.get_normal_vec(tup[0] / n),
                        buff=0,
                    ).set_color(BLUE),
                    zip(range(n), 5 * h_values),
                )
            )
            self.add(*arrows)
            # self.play(*[FadeIn(a) for a in arrows], run_time=0.5)

            self.play(
                Transform(curve, new_curve),
                Transform(
                    graph_of_tangent_angle,
                    ParametricCurve(
                        lambda t: axes.c2p(t, get_angle(new_curve.get_tangent_vec(t))),
                        (tmin, tmax, 0.002),
                    ).set_color(GREEN),
                ),
                *[FadeOut(a) for a in arrows],
                rate_func=linear,
                run_time=0.2,
            )
            a_text[-1].set_value(new_curve.compute_area())
            l_text[-1].set_value(new_curve.compute_arc_length())
            self.remove(curve)
            self.add(new_curve)
            curve = new_curve
