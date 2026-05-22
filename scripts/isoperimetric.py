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

    def compute_val(self, t: float) -> float:
        """Compute x(t) value at parameter t in [0, 1]"""
        cos_term = sum(
            cf * math.cos(TAU * d * t) for d, cf in enumerate(self.cos_coeffs)
        )
        sin_term = sum(
            cf * math.sin(TAU * d * t) for d, cf in enumerate(self.sin_coeffs)
        )
        return cos_term + sin_term

    def compute_dval(self, t: float) -> float:
        """Compute x'(t) value at parameter t in [0, 1]"""
        cos_term = sum(
            -cf * (TAU * d) * math.sin(TAU * d * t)
            for d, cf in enumerate(self.cos_coeffs)
        )
        sin_term = sum(
            cf * (TAU * d) * math.cos(TAU * d * t)
            for d, cf in enumerate(self.sin_coeffs)
        )
        return cos_term + sin_term

    def compute_d2val(self, t: float) -> float:
        """Compute x''(t) value at parameter t in [0, 1]"""
        cos_term = sum(
            -cf * ((TAU * d) ** 2) * math.cos(TAU * d * t)
            for d, cf in enumerate(self.cos_coeffs)
        )
        sin_term = sum(
            -cf * ((TAU * d) ** 2) * math.sin(TAU * d * t)
            for d, cf in enumerate(self.sin_coeffs)
        )
        return cos_term + sin_term


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
        print(a)
        return curve

    def compute_xyz(self, t: float) -> Vect3:
        """Outputs the point at the given t value"""
        return np.array(
            [self.fourier_x.compute_val(t), self.fourier_y.compute_val(t), 0.0]
        )

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

    def make_tangent_circle_at(self, t: float) -> Circle:
        """Outputs a tangent circle at the given t value"""
        rc = 1 / abs(self.get_curvature(t))
        return Circle(
            arc_center=self.compute_xyz(t) + self.get_normal_vec(t) * rc,
            radius=rc,
        )

    def make_moving_tangent_circle_at(self, t: ValueTracker) -> Circle:
        """Outputs a tangent circle at the given t value"""
        rc = 1 / abs(self.get_curvature(t.get_value()))
        circle = Circle(
            arc_center=self.compute_xyz(t.get_value())
            + self.get_normal_vec(t.get_value()) * rc,
            radius=rc,
        )
        circle.add_updater(
            lambda mobj: mobj.become(
                Circle(
                    arc_center=self.compute_xyz(t.get_value())
                    + self.get_normal_vec(t.get_value())
                    / abs(self.get_curvature(t.get_value())),
                    radius=1 / abs(self.get_curvature(t.get_value())),
                )
            )
        )
        return circle

    def get_normal_vec(self, t: float) -> Vect3:
        """Constructs the normal vector at parameter t according to the Fourier expansion."""
        return rot90(self.get_tangent_vec(t))

    def get_tangent_vec(self, t: float) -> Vect3:
        """Constructs the tangent vector at parameter t according to the Fourier expansion."""
        return normalize_vect3(np.array([self.compute_dx(t), self.compute_dy(t), 0.0]))

    def compute_dx(self, t: float) -> float:
        """Compute x'(t) value at parameter t in [0, 1]"""
        return self.fourier_x.compute_dval(t)

    def compute_dy(self, t: float) -> float:
        """Compute y'(t) value at parameter t in [0, 1]"""
        return self.fourier_y.compute_dval(t)

    def compute_d2x(self, t: float) -> float:
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

    degree: int = 5
    num_pts: int = 200

    # Construction of a curve straight from Fourier decomposition of the x and y coordinates
    def make_curve(self):
        n = 2 * self.degree
        angles = np.linspace(0, TAU * (1 - 1 / n), n)
        radial_vecs = list(map(lambda t: np.array([np.cos(t), np.sin(t), 0]), angles))

        # Define the curve in polar coordinates, where the radius is deformed from a constant function
        # by various cosine/sine curves, scaled by Gaussians.
        radii = np.array([1.0] * n)
        mean_perturbation = 0.25
        for j in range(5):
            radii += (
                mean_perturbation
                * np.random.randn()
                * np.array([np.cos(TAU * i * j / n) for i in range(n)])
            )
            radii += (
                mean_perturbation
                * np.random.randn()
                * np.array([np.sin(TAU * i * j / n) for i in range(n)])
            )
        curve = ClosedCurve.interpolate_from_points(
            *[radii[i] * radial_vecs[i] for i in range(n)]
        )

        # Up the resolution
        curve.make_bezier_segments(self.num_pts)

        return curve


class DefineCurvature(ClosedCurveScene):
    """Depict the two definitions of curvature, i.e.
    - (geometric) osculating circle
    - (analytic) derivative of tangent angle
    """

    def construct(self):
        n = 2 * self.degree
        curve = self.make_curve()
        self.play(ShowCreation(curve))

        tangent_tracker = ValueTracker(0.01)

        # Show the curvature at several points, using a circle
        # circle = curve.make_moving_tangent_circle_at(tangent_tracker)
        circle = curve.make_tangent_circle_at(0.0)
        pt = Dot(curve.compute_xyz(0), radius=0.1).set_color(BLUE)
        pt.add_updater(
            lambda mobj: mobj.move_to(curve.compute_xyz(tangent_tracker.get_value()))
        )
        self.play(ShowCreation(circle), ShowCreation(pt))
        # self.play(tangent_tracker.animate.set_value(0.2))
        # self.play(tangent_tracker.animate.set_value(0.6))
        # self.play(tangent_tracker.animate.set_value(0.8))
        for v in (0.3, 0.55, 0.8):
            self.play(
                Transform(circle, curve.make_tangent_circle_at(v)),
                tangent_tracker.animate.set_value(v),
            )
        self.play(FadeOut(circle))
        tangent_tracker.set_value(0.01)

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

        def tangent_angle_fn(t_val: float):
            return get_angle(curve.get_tangent_vec(t_val))

        graph_of_tangent_angle = ParametricCurve(
            lambda t: axes.c2p(t, tangent_angle_fn(t)),
            (0.01, 0.99, 0.01),
        ).set_color(RED)

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
            radius=0.1,
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

        self.embed()


# Showing deformation of a curve a la the isoperimetric inequality, i.e. calculus of variations
class Isoperimetric(ClosedCurveScene):
    def construct(self):
        n = self.degree
        curve = self.make_curve()
        self.add(curve)
        # Define perturbation function at each point in the way which
        # keeps the length fixed and maximally increases the area
        # - Length change = <h_values, curvatures>
        # - Area change   = <h_values, 1>
        # - Normalized: <h_values, h_values> = 1
        # We project 1-vector onto the orthogonal complement of curvatures, then use that.
        ones_vector = np.ones(n)
        arrow_length = 0.5

        # Deformation step: put into ClosedCurve
        for j in range(10):
            print(f"Iteration {j}")
            curvatures = np.array([curve.get_curvature(i / n) for i in range(n)])

            projected = (
                ones_vector
                - curvatures
                * (ones_vector.dot(curvatures))
                / np.linalg.norm(curvatures) ** 2
            )
            # h_values = arrow_length * normalize_vect3(projected)
            h_values = 0.2 * arrow_length * projected

            # Make arrows which represent normal deformation
            arrows = list(
                map(
                    lambda tup: Arrow(
                        curve.compute_xyz(tup[0] / num_tex_symbols),
                        curve.compute_xyz(tup[0] / n)
                        - tup[1] * curve.get_normal_vec(tup[0] / n),
                        buff=0,
                    ).set_color(BLUE),
                    zip(range(n), 5 * h_values),
                )
            )
            self.play(*[FadeIn(a) for a in arrows], run_time=1.0)

            # Define the new curve
            new_curve = ClosedCurve.interpolate_from_points(
                *[
                    curve.compute_xyz(i / n) - h_values[i] * curve.get_normal_vec(i / n)
                    for i in range(n)
                ]
            )
            self.play(
                Transform(curve, new_curve),
                *[FadeOut(a) for a in arrows],
                rate_func=linear,
                run_time=0.5,
            )
            self.remove(curve)
            self.add(new_curve)
            curve = new_curve

        self.embed()
