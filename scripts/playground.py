# Dumping ground for new ideas: particularly the most challenging animations which will make their
# way into other files.

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
            arc_center=self.compute_xyz(t) + self.get_normal_vec(t) * rc,
            radius=rc,
        )
        circle.add_updater(
            lambda mobj: mobj.become(
                Circle(
                    arc_center=self.compute_xyz(t)
                    + self.get_normal_vec(t) / abs(self.get_curvature(t.get_value())),
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

    # def make_arrow(self, i: int, h: float) -> Arrow:
    #     """Constructs an arrow normal to the curve of the given length"""
    #     return Arrow(
    #         self.get_anchor(i),
    #         self.get_anchor(i) - h * self.get_normal_vec(i / self.num_segments),
    #         buff=0,
    #     ).set_color(BLUE)

    # def _forward_curvature(self, i: int) -> float:
    #     """Picks the three points of the Bezier curve segment going forward, and
    #     uses these to compute curvature."""
    #     a0 = self.data["point"][6 * i]
    #     h = self.data["point"][6 * i + 1]
    #     a1 = self.data["point"][6 * i + 2]
    #     return 0.5 * np.linalg.norm(a0 - 2 * h + a1) / np.linalg.norm(h - a0) ** 2

    # def _backward_curvature(self, i: int) -> float:
    #     """Picks the three points of the Bezier curve segment going backward, and
    #     uses these to compute curvature."""
    #     if i == 0:
    #         a0 = self.data["point"][-1]
    #         h = self.data["point"][-2]
    #         a1 = self.data["point"][-3]
    #     else:
    #         a0 = self.data["point"][6 * i - 2]
    #         h = self.data["point"][6 * i - 3]
    #         a1 = self.data["point"][6 * i - 4]
    #     return -0.5 * np.linalg.norm(a0 - 2 * h + a1) / np.linalg.norm(h - a0) ** 2

    def get_curvature(self, t: float) -> float:
        """Computes the curvature, using the Fourier interpolation."""
        # Use the Fourier coefficients to compute the curvature, using the formula
        # C = x'(s)y''(s) - x''(s)y'(s)
        # where s parametrizes arc-length changing at rate 1. Equivalently, parametrizing by t,
        # C = (x'(t)y''(t) - x''(t)y'(t)) / ((x')^2 + (y')^2)^{3/2}
        # TODO Not sure if this is correct? It relies on the accuracy of the Fourier decomposition.
        xprime = self.compute_dx(t)
        yprime = self.compute_dy(t)
        return (xprime * self.compute_d2y(t) - self.compute_d2x(t) * yprime) / math.pow(
            xprime**2 + yprime**2, 1.5
        )

        # return 0.5 * (self._forward_curvature(i) + self._backward_curvature(i))

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
            # self.get_anchor(i)
            self.compute_xyz(i / self.num_segments)
            - h_values[i] * self.get_normal_vec(i / self.num_segments)
            for i in range(self.num_segments)
        ]
        new_curve = make_closed_curve_from_points(*new_anchors)
        return new_curve

    def space_out(self) -> ClosedCurve:
        """Produces a new closed curve whose anchors lie on this one, but which
        are equally spaced (by arc length) along the curve."""
        # Do a quick-and-dirty estimate of the arc lengths
        arc_lengths = []
        anchors_and_handles = []
        for i in range(self.num_segments):
            anchors_and_handles.extend(
                [
                    (
                        self.data["point"][6 * i],
                        self.data["point"][6 * i + 1],
                        self.data["point"][6 * i + 2],
                    ),
                    (
                        self.data["point"][6 * i + 2],
                        self.data["point"][6 * i + 3],
                        self.data["point"][6 * i + 4],
                    ),
                ]
            )
            arc_lengths.extend(
                [
                    qbez_arc_length(
                        self.data["point"][6 * i],
                        self.data["point"][6 * i + 1],
                        self.data["point"][6 * i + 2],
                    ),
                    qbez_arc_length(
                        self.data["point"][6 * i + 2],
                        self.data["point"][6 * i + 3],
                        self.data["point"][6 * i + 4],
                    ),
                ]
            )

        # Get the total arc length and spacing between new anchors
        total_length = sum(arc_lengths)
        normalized_arc_lengths = [l / total_length for l in arc_lengths]

        # Construct the new anchors
        new_anchors = []
        length, ind = 0.0, 0
        for i in range(self.num_segments):
            desired_length = i / self.num_segments
            while length + normalized_arc_lengths[ind] <= desired_length:
                length += normalized_arc_lengths[ind]
                ind += 1
            new_anchors.append(
                qbez_arc_length_interp(
                    *anchors_and_handles[ind],
                    alpha=(desired_length - length) / normalized_arc_lengths[ind],
                )
            )

        new_curve = make_closed_curve_from_points(*new_anchors)
        return new_curve


def qbez_arc_length_interp(
    a0: Vect3, h: Vect3, a1: Vect3, alpha: float, num_segments: int = 10
):
    """Outputs the point a fraction alpha of the way along the quadratic bezier curve"""
    curve_p = lambda t: (1 - t) ** 2 * a0 + 2 * t * (1 - t) * h + t**2 * a1
    total_length = qbez_arc_length(a0, h, a1, num_segments)
    desired_length = alpha * total_length
    length, ind = 0.0, 0
    next_segment_length = np.linalg.norm(
        curve_p((ind + 1) / num_segments) - curve_p(ind / num_segments)
    )
    while length + next_segment_length <= desired_length:
        length += next_segment_length
        ind += 1
        next_segment_length = np.linalg.norm(
            curve_p((ind + 1) / num_segments) - curve_p(ind / num_segments)
        )
    ratio = (desired_length - length) / next_segment_length
    return curve_p((ind + ratio) / num_segments)


def qbez_arc_length(a0: Vect3, h: Vect3, a1: Vect3, num_segments: int = 10):
    """Estimates the arc length of a quadratic Bezier segment by approximating linearly."""
    curve_p = lambda t: (1 - t) ** 2 * a0 + 2 * t * (1 - t) * h + t**2 * a1
    return sum(
        np.linalg.norm(curve_p((i + 1) / num_segments) - curve_p(i / num_segments))
        for i in range(num_segments)
    )
    pass


def make_closed_curve_from_points(*anchors) -> VMobject:
    """Makes a smooth Bezier curve from a sequence of points, where the first
    point is repeated as the last point"""
    return make_curve_from_points(*anchors, anchors[0])


def make_curve_from_points(*anchors) -> ClosedCurve:
    """Makes a smooth Bezier curve from a sequence of points."""
    h1, h2 = get_smooth_cubic_bezier_handle_points(anchors)
    c = ClosedCurve()
    for i in range(len(anchors) - 1):
        c.add_cubic_bezier_curve(anchors[i], h1[i], h2[i], anchors[i + 1])
    return c


class Curvature(Scene):
    """Depict the two definitions of curvature, i.e.
    - (geometric) osculating circle
    - (analytic) derivative of tangent angle
    """

    degree: int = 5
    num_pts: int = 50

    def construct(self):
        n = 2 * self.degree
        angles = np.linspace(0, TAU * (1 - 1 / n), n)
        radial_vecs = list(map(lambda t: np.array([np.cos(t), np.sin(t), 0]), angles))

        # Define the curve in polar coordinates, where the radius is deformed from a constant function
        # by various cosine/sine curves, scaled by Gaussians.
        radii = np.array([1.0] * n)
        mean_perturbation = 0.2
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
        curve.make_bezier_segments(200)
        self.play(ShowCreation(curve))

        # Show the curvature at several points

        # Color code according to the curvature

        # Add a graph on the right which shows the tangent vector angle as a function of distance along the curve
        axes = Axes(
            (0, 1),
            (-0.5, TAU + 0.5),
            width=3.0,
            height=3.0,
            y_axis_config={"include_ticks": False},
        ).next_to(curve, RIGHT, 1.5)
        self.play(ShowCreation(axes))

        def get_angle(v) -> float:
            return (math.atan2(v[1], v[0]) - PI / 2) % TAU

        def polar_vec(theta: float) -> Vect3:
            return np.array([np.cos(theta), np.sin(theta), 0.0])

        def tangent_angle_fn(t_val: float):
            return get_angle(curve.get_tangent_vec(t_val))

        graph_of_tangent_angle = ParametricCurve(
            lambda t: axes.c2p(t, tangent_angle_fn(t)),
            (0.01, 0.99, 0.01),
        ).set_color(RED)

        self.play(ShowCreation(graph_of_tangent_angle))

        # Depict a tangent vector moving along the curve, and a corresponding point

        tangent_tracker = ValueTracker(0.01)
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
class Isoperimetric(Scene):
    # First, define closed curves in the plane, parametrized as f: [0, 2pi] -> R^2
    # according to a Fourier decomposition
    def construct(self):
        # Define a curve.
        # TODO Find a smoother way of defining a curve -- maybe by Fourier coefficients?
        num_pts = 10
        angles = np.linspace(0, TAU * (1 - 1 / num_pts), num_pts)
        radial_vecs = list(map(lambda t: np.array([np.cos(t), np.sin(t), 0]), angles))
        radii = [1 + 0.1 * np.random.randn() for _ in range(num_pts)]
        curve = ClosedCurve.interpolate_from_points(
            *[radii[i] * radial_vecs[i] for i in range(num_pts)]
        )
        self.add(curve)

        # Up the resolution
        curve.make_bezier_segments(100)

        # Show curvature
        circle = curve.make_tangent_circle_at(0.0)
        self.add(ShowCreation(circle))
        self.play(Transform(circle, curve.make_tangent_circle_at(0.4)))
        self.play(Transform(circle, curve.make_tangent_circle_at(0.7)))
        self.play(Transform(circle, curve.make_tangent_circle_at(0.0)))
        self.play(FadeOut(circle))

        # Define perturbation function at each point in the way which
        # keeps the length fixed and maximally increases the area
        # - Length change = <h_values, curvatures>
        # - Area change   = <h_values, 1>
        # - Normalized: <h_values, h_values> = 1
        # We project 1-vector onto the orthogonal complement of curvatures, then use that.
        ones_vector = np.ones(num_pts)
        arrow_length = 0.5

        for j in range(10):
            print(f"Iteration {j}")
            curve.do_fourier_interp()
            curvatures = curve.get_curvatures()

            projected = (
                ones_vector
                - curvatures
                * (ones_vector.dot(curvatures))
                / np.linalg.norm(curvatures) ** 2
            )
            # h_values = arrow_length * normalize_vect3(projected)
            h_values = 0.2 * arrow_length * projected

            arrows = VGroup(
                *list(
                    map(
                        lambda tup: curve.make_arrow(*tup),
                        zip(range(num_pts), 5 * h_values),
                    )
                )
            )
            self.play(ShowCreation(arrows), run_time=0.1)

            new_curve = make_closed_curve_from_points(
                *[
                    curve.compute_xyz(i / num_pts)
                    - h_values[i] * curve.get_normal_vec(i / num_pts)
                    for i in range(num_pts)
                ]
            )
            self.play(
                Transform(curve, new_curve),
                # Transform(anchors, new_anchors),
                # Transform(intermediate_anchors, new_intermediate_anchors),
                FadeOut(arrows),
                rate_func=linear,
                run_time=0.5,
            )

        self.embed()


# # ── Poincaré disk helpers ──────────────────────────────────────────────────

# def mobius_add(a: complex, b: complex) -> complex:
#     """Möbius addition (hyperbolic translation) in the Poincaré disk."""
#     return (a + b) / (1 + np.conj(a) * b)


# def disk_distance_euclidean(h: float) -> float:
#     """Convert hyperbolic distance h to Euclidean distance from origin."""
#     return float(np.tanh(h / 2))


# def reflect_across_diameter(z: complex, angle: float) -> complex:
#     """Reflect z across the diameter at given angle from positive x-axis."""
#     return np.conj(z * np.exp(-1j * angle)) * np.exp(1j * angle)


# def reflect_across_geodesic(z: complex, center: complex, radius: float) -> complex:
#     """Reflect z across a geodesic given by a circle (center, radius)
#     that is orthogonal to the unit circle."""
#     return center + (radius ** 2) / np.conj(z - center)

# # ── Fundamental triangle ───────────────────────────────────────────────────

# class FundamentalTriangle:
#     """Coxeter triangle with angles (π/p, π/2, π/q) for p=6, q=4."""

#     def __init__(self, p: int = 6, q: int = 4):
#         self.p = p
#         self.q = q
#         self.alpha = np.pi / p   # π/6
#         self.gamma = np.pi / q   # π/4
#         self._compute_vertices()

#     def _compute_vertices(self):
#         """Compute (A, B, C) in the Poincaré disk.

#         A is at origin (angle π/6).  B is on the positive real axis (right
#         angle).  C is in the upper half-plane at angle α from the x-axis
#         (angle π/4).
#         """
#         α, γ = self.alpha, self.gamma

#         # Hypotenuse AC
#         h_AC = float(np.arccosh(1.0 / (np.tan(α) * np.tan(γ))))

#         # Leg AB: cosh(AB) = cos(α) / sin(γ)
#         h_AB = float(np.arccosh(np.cos(α) / np.sin(γ)))

#         self.A = complex(0.0, 0.0)
#         d_AB = disk_distance_euclidean(h_AB)
#         self.B = complex(d_AB, 0.0)
#         d_AC = disk_distance_euclidean(h_AC)
#         self.C = d_AC * np.exp(1j * α)


# # ── Reflection generators ──────────────────────────────────────────────────

# class ReflectionGenerators:
#     """Three hyperbolic reflections R1, R2, R3 for the (p,q,2) Coxeter group.

#     R1: across side BC  (opposite vertex A)
#     R2: across side CA  (opposite vertex B)
#     R3: across side AB  (opposite vertex C)
#     """

#     def __init__(self, p: int = 6, q: int = 4):
#         self.tri = FundamentalTriangle(p, q)
#         A, B, C = self.tri.A, self.tri.B, self.tri.C

#         # R2: diameter at angle α (π/6)
#         self.theta_R2 = self.tri.alpha

#         # R3: diameter along positive real axis (angle 0)
#         self.theta_R3 = 0.0

#         # R1: geodesic through B and C (circle orthogonal to unit disk)
#         Bx, By = B.real, B.imag
#         Cx, Cy = C.real, C.imag

#         # Solve for circle center c0 = (x, y) orthogonal to unit circle
#         # that passes through B and C:
#         #   Re(c̄₀ B) = (|B|² + 1)/2
#         #   Re(c̄₀ C) = (|C|² + 1)/2
#         rhs_B = (Bx**2 + By**2 + 1) / 2
#         rhs_C = (Cx**2 + Cy**2 + 1) / 2
#         det = Bx * Cy - By * Cx

#         if abs(det) < 1e-14:
#             self.R1_is_diameter = True
#             self.theta_R1 = float(np.angle(C))
#         else:
#             self.R1_is_diameter = False
#             x = (rhs_B * Cy - By * rhs_C) / det
#             y = (Bx * rhs_C - rhs_B * Cx) / det
#             self.R1_center = complex(float(x), float(y))
#             self.R1_radius = float(np.sqrt(abs(self.R1_center) ** 2 - 1))

#     def apply_R1(self, z: complex) -> complex:
#         if self.R1_is_diameter:
#             return reflect_across_diameter(z, self.theta_R1)
#         return reflect_across_geodesic(z, self.R1_center, self.R1_radius)

#     def apply_R2(self, z: complex) -> complex:
#         return reflect_across_diameter(z, self.theta_R2)

#     def apply_R3(self, z: complex) -> complex:
#         return reflect_across_diameter(z, self.theta_R3)

#     def apply(self, index: int, z: complex) -> complex:
#         return [self.apply_R1, self.apply_R2, self.apply_R3][index - 1](z)


# # ── Word enumeration (tile centers) ────────────────────────────────────────

# def generate_tile_centers(gens: ReflectionGenerators,
#                            max_depth: int) -> List[complex]:
#     """BFS over the Coxeter group word graph, yielding unique tile centers."""
#     seed = complex(0.0, 0.0)
#     visited = {seed}
#     centers = [seed]
#     distance = {seed: 0}
#     queue = deque([(seed, 0)])  # (center, last_generator_index)

#     while queue:
#         z, last_gen = queue.popleft()
#         d = distance.get(z, 0)
#         if d >= max_depth:
#             continue
#         for g in (1, 2, 3):
#             if g == last_gen:
#                 continue  # R_g² = I
#             z_new = gens.apply(g, z)
#             if z_new not in visited:
#                 visited.add(z_new)
#                 distance[z_new] = d + 1
#                 centers.append(z_new)
#                 queue.append((z_new, g))

#     return centers


# # ── Hexagon construction ───────────────────────────────────────────────────

# def hexagon_vertices(center: complex, radius: float,
#                       n: int = 6) -> List[complex]:
#     """Vertices of a regular hyperbolic n-gon centered at `center`."""
#     angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
#     verts_origin = [radius * np.exp(1j * a) for a in angles]
#     if abs(center) < 1e-15:
#         return verts_origin
#     return [mobius_add(center, v) for v in verts_origin]


# # ── Main API ───────────────────────────────────────────────────────────────

# def build_642_tiling(max_depth: int = 4) -> Tuple[np.ndarray, np.ndarray]:
#     """Generate (6,4,2) tiling.

#     Returns:
#         centers: (N, 2) array of hexagon centers
#         vertices: (N, 6, 2) array of hexagon vertices
#     """
#     gens = ReflectionGenerators(p=6, q=4)

#     # Vertex distance: cosh(h) = cos(π/6) / sin(π/4)
#     h_v = float(np.arccosh(np.cos(np.pi / 6) / np.sin(np.pi / 4)))
#     vdist = disk_distance_euclidean(h_v)

#     centers = generate_tile_centers(gens, max_depth)

#     all_centers = []
#     all_vertices = []
#     for c in centers:
#         verts = hexagon_vertices(complex(c.real, c.imag), vdist)
#         all_centers.append((c.real, c.imag))
#         all_vertices.append([(v.real, v.imag) for v in verts])

#     return np.array(all_centers), np.array(all_vertices)


def wp(z: complex | np.ndarray, tau: complex, n_pts: int = 10) -> np.ndarray:
    """Approximation to Weierstrass P-function. Automatically vectorized over z
    and casts to the input type."""
    result = np.pow(z, -2)
    for x in range(-n_pts, n_pts):
        for y in range(-n_pts, n_pts):
            if (x == 0) and (y == 0):
                continue

            p = x + y * tau
            result += np.pow(z + p, -2) - 1 / p**2

    if isinstance(z, complex):
        return complex(result)
    else:
        return result


def wp_prime(z: complex | np.ndarray, tau: complex, n_pts: int = 10) -> np.ndarray:
    """Approximation to derivative of Weierstrass P-functionAutomatically vectorized over z
    and casts to the input type."""
    result = -2 * sum(
        np.pow(z + x + y * tau, -3)
        for x in range(-n_pts, n_pts)
        for y in range(-n_pts, n_pts)
    )

    if isinstance(z, complex):
        return complex(result)
    else:
        return result


class Laplacian(m.Scene):
    """Visualizations of the Laplacian of a manifold equipped with a Riemannian metric.
    Begin with the simplest case, one-dimensional R. Wave and heat equation.
    Next, move up to two-dimensional R^2. Again, wave and heat equation.
    "Harmonics"
    Then functions on S^1.
    Spherical harmonics: functions on S^2."""

    def construct(self):
        pass


class EllipticCurve(m.Scene):
    """Parametrize the points of an elliptic curve y^2 = x^3 + ax + b"""

    def construct(self):
        # Construct a two-dimensional version of the curve
        t_axes = m.ThreeDAxes((-4, 4), (-4, 4), (-4, 4)).set_stroke(width=3.0)

        # Orient the frame so that x and y-axes are visible
        self.frame.reorient(0, 0, 0)

        a = m.ValueTracker(-1.0)
        b = m.ValueTracker(0.0)

        # Construct the curve
        curve_1 = m.ParametricCurve(
            lambda x: t_axes.c2p(
                x, np.sqrt(complex(x**3 + x * a.get_value() + b.get_value())).real, 0
            ),
            (-4, 4, 0.01),
        ).set_stroke(width=2.0, color=m.BLUE)

        curve_2 = m.ParametricCurve(
            lambda x: t_axes.c2p(
                x, -np.sqrt(complex(x**3 + x * a.get_value() + b.get_value())).real, 0
            ),
            (-4, 4, 0.01),
        ).set_stroke(width=2.0, color=m.BLUE)

        self.play(
            m.ShowCreation(t_axes), m.ShowCreation(curve_1), m.ShowCreation(curve_2)
        )

        curve_1.add_updater(
            lambda mobj: mobj.become(
                m.ParametricCurve(
                    lambda x: t_axes.c2p(
                        x,
                        np.sqrt(complex(x**3 + x * a.get_value() + b.get_value())).real,
                        0,
                    ),
                    (-4, 4, 0.1),
                ).set_stroke(width=2.0, color=m.BLUE)
            )
        )

        curve_2.add_updater(
            lambda mobj: mobj.become(
                m.ParametricCurve(
                    lambda x: t_axes.c2p(
                        x,
                        -np.sqrt(
                            complex(x**3 + x * a.get_value() + b.get_value())
                        ).real,
                        0,
                    ),
                    (-4, 4, 0.1),
                ).set_stroke(width=2.0, color=m.BLUE)
            )
        )

        self.embed()


class EllipticFunctionToCurve(m.Scene):
    """Visualizing how the Weierstrass P-function parametrizes the points on an elliptic curve."""

    def construct(self):
        # Value trackers which determine z value
        z_real = m.ValueTracker(0.5)
        z_imag = m.ValueTracker(0.5)

        # Value trackers which determine tau value
        t_real = m.ValueTracker(0.0)
        t_imag = m.ValueTracker(1.0)

        x_max = 5
        z_plane = m.ComplexPlane((-2, 2), (-2, 2)).move_to((0, 0, 0))

        z_pt = m.GlowDot()
        z_pt.add_updater(
            lambda mobj: mobj.move_to(
                z_plane.n2p(complex(real.get_value(), imag.get_value()))
            )
        )

        x_plane = m.ComplexPlane((-x_max, x_max), (-x_max, x_max)).move_to((0, 15, 0))
        y_plane = m.ComplexPlane((-x_max, x_max), (-x_max, x_max)).move_to((0, 0, 0))

        # Updater for wp value
        def x_updater(mobj):
            z = complex(z_real.get_value(), z_imag.get_value())
            t = complex(t_real.get_value(), t_imag.get_value())
            val = wp(z, t)
            mobj.move_to(x_plane.n2p(val))

        # Updater for wp derivative value
        def y_updater(mobj):
            z = complex(z_real.get_value(), z_imag.get_value())
            t = complex(t_real.get_value(), t_imag.get_value())
            val = wp_prime(z, t)
            mobj.move_to(y_plane.n2p(val))

        x_pt = m.GlowDot()
        x_pt.add_updater(x_updater)

        y_pt = m.GlowDot()
        y_pt.add_updater(y_updater)

        self.embed()


class JInvariant(m.Scene):
    """Visualizing the J-invariant of a lattice L, i.e. the unique modular function (up to invertible
    rational map) which maps its fundamental domain (H mod L) to the Riemann sphere"""

    def construct(self):
        pass


class FunctionSpaceOnS1(m.Scene):
    """Depict the space of functions on S^1 (or any other compact manifold with metric), which has an inner
    product given by multiplying pointwise and then integration. Under this inner product,
    - the linear map given by multiplying pointwise by a function is symmetric/hermitian;
    - the linear map given by differentiation (needs to be properly defined in higher dimensions) is symplectic."""


# Convenience functions for the Poincare disk
class PoincareDisk:
    z_values: list[complex] = []
    dots: list[Dot] = []
    alpha: ComplexValueTracker

    """Poincare disk of radius 1"""

    def __init__(self):
        self.z_values = []
        self.dots = []
        self.alpha = ComplexValueTracker(0.0)

    def set_plane(self, plane: ComplexPlane):
        self.plane = plane

    def add_points(self, *vals):
        self.z_values.extend(vals)


def _make_hyperbolic_line(
    endpoints: tuple[complex, complex], plane: ComplexPlane
) -> ArcBetweenPoints:
    """Given two unit complex numbers z1 and z2, constructs the arc connecting them in the Poincare disk."""
    z1, z2 = endpoints
    assert np.isclose(abs(z1), 1.0)
    assert np.isclose(abs(z2), 1.0)

    p1 = cmath.phase(z1)
    p2 = cmath.phase(z2)
    while p2 < p1:
        p2 += TAU

    # Ensure z2 is counterclockwise of z1 by less than PI
    if p2 - p1 > PI:
        return _make_hyperbolic_line((z2, z1), plane)

    return ArcBetweenPoints(
        start=plane.n2p(z2), end=plane.n2p(z1), angle=PI - (p2 - p1)
    ).set_color(RED)


class UpperHalfPlane(Mobject):
    pass


class ConformalMaps(Scene):
    """Experimenting with visualizations of various conformal maps. For example:
    - Parametrization of maps D_1 -> D_1, by mapping orthogonal circles (parametrized by points outside the disk?)
    - Parametrization of maps H -> H
    """

    def construct(self):
        plane = ComplexPlane((-3, 3, 1.0), (-3, 3, 1.0))
        # self.add(plane)

        # Draw the disk
        disk = ParametricCurve(
            lambda t: plane.n2p(np.exp(complex(0, TAU * t))), (0, 1, 0.01)
        )
        self.play(ShowCreation(disk))

        # The function [[-1, a], [-a*, 1]] interchanges 0 and a

        a = ComplexValueTracker(0.0)

        def transform(z: complex) -> complex:
            # The function [[1, a], [a*, 1]] sends 0 -> a and -a -> 0.
            return (z + a.get_value()) / (a.get_value().conjugate() * z + 1)

        def get_size(z: complex) -> float:
            return 1 - abs(z) ** 2

        # Draw some reference points, whose values and size track according to the value of a
        def make_dot(z: complex, plane: ComplexPlane = plane):
            dot = Dot(plane.n2p(z), radius=0.05).set_color(BLUE)
            dot.add_updater(
                lambda mobj: mobj.move_to(plane.n2p(transform(z))).set_width(
                    0.1 * get_size(transform(z))
                )
            )
            return dot

        # TODO Make a version which creates a hyperbolic line, with an updater function.
        # In essence, it is a circular arc passing through two points on the boundary, and where the number
        # of degrees of the arc is supplementary to the angular distance between those two points.
        def _make_hyperbolic_line(endpoints: tuple[complex, complex]):
            """Given two unit complex numbers z1 and z2, constructs the arc connecting them"""
            z1, z2 = endpoints
            assert np.isclose(abs(z1), 1.0)
            assert np.isclose(abs(z2), 1.0)

            p1 = cmath.phase(z1)
            p2 = cmath.phase(z2)
            while p2 < p1:
                p2 += TAU

            # Ensure z2 is counterclockwise of z1 by less than PI
            if p2 - p1 > PI:
                return _make_hyperbolic_line((z2, z1))

            return ArcBetweenPoints(
                start=plane.n2p(z2), end=plane.n2p(z1), angle=PI - (p2 - p1)
            ).set_color(RED)

        def make_hyperbolic_line(endpoints):
            arc = _make_hyperbolic_line(endpoints, plane)
            arc.add_updater(
                lambda mobj: mobj.become(
                    _make_hyperbolic_line(
                        (transform(endpoints[0]), transform(endpoints[1])), plane
                    )
                )
            )
            return arc

        z_values = [0.0, 0.5, -0.5, complex(0, 0.5), complex(0, -0.5)]
        # points = map(make_dot, z_values)

        # arcs = map(make_hyperbolic_line, [
        #     (1.0, complex(0, 1)),
        #     (-1.0, complex(0, 1)),
        #     (1.0, complex(0, -1)),
        #     (-1.0, complex(0, -1)),
        # ])

        # TODO Will need to refine this some to make something pretty.

        # pts = build_642_tiling(4)
        # cx_dots = map(make_dot, [complex(*y) for y in pts[0]])
        # self.play(*[FadeIn(p) for p in cx_dots])

        # self.play(*[FadeIn(p) for p in points])
        # self.play(*[FadeIn(a) for a in arcs])

        self.embed()

        # Draw a hyperbolic line which can vary

        # Use this to draw gridlines

        # The function [[-1, a], [-a*, 1]] interchanges 0 and a
        # The function [[1, a], [a*, 1]] sends 0 -> a and -a -> 0. Compose this with a rotation

        # Specify the function as a fractional linear transformation
        # [[a, b], [b*, a*]] where |a|^2 - |b|^2 = 1, then draw a path in SU(1, 1) from I_2 to it


class InvLaplace(Scene):
    """Playing around to try to get an animation for the prime number theorem"""

    def construct(self):
        # Plane where s lives
        xmin = -4
        xmax = 4
        ymin = -4
        ymax = 4
        s_plane = ComplexPlane(x_range=(xmin, xmax, 1.0), y_range=(ymin, ymax, 1.0))

        self.play(ShowCreation(s_plane))

        # s varies from a - i∞ to a + i∞
        a = 1.2
        s = ComplexValueTracker(a)
        s_line = DashedLine(
            s_plane.n2p(complex(a, xmin)), s_plane.n2p(complex(a, xmax))
        )
        s_line_label = Tex("\\mathrm{Re}(s) = a", font_size=36).next_to(
            s_line, DOWN, 0.5
        )
        s_dot = GlowDot(s_plane.n2p(s.get_value())).add_updater(
            lambda mobj: mobj.move_to(s_plane.n2p(s.get_value()))
        )

        self.play(ShowCreation(s_line), FadeIn(s_line_label), ShowCreation(s_dot))

        # Draw in all the lines, for every s value
        laplace_rays = [
            Line(s_plane.n2p(0), s_plane.n2p(3 * complex(a, im_part))).set_style(
                stroke_width=1.0, stroke_opacity=0.5
            )
            for im_part in np.linspace(xmin, xmax, 10)
        ]
        self.play(*[ShowCreation(l) for l in laplace_rays])

        # t values from 0 to ∞, and controls

        # Plane where the exponentials live
        # e^{-st} = e^{-(a+bi)t} = e^{-at}e^{-ibt}. Thus, magnitude depends only on t, not position of s
        exp_plane = (
            ComplexPlane(x_range=(-4, 4, 1.0), y_range=(-4, 4, 1.0))
            .set_width(8.0)
            .next_to(s_plane, RIGHT, 2.0)
        )
        self.play(ShowCreation(exp_plane))

        # # Images of the laplace rays
        # laplace_spirals = [
        #     ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * complex(a, im_part))), t_range=(0, 3, 0.01)).set_style(stroke_width=1.0)
        #     for im_part in np.linspace(xmin, xmax, 10)
        # ]

        # # Map via (s, t) -> exp(-st)
        # exp_s_curve = ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * s.get_value())), t_range=(0, 2, 0.01)).set_style(stroke_width=1.0)
        # exp_s_curve.add_updater(lambda mobj: mobj.become(ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * s.get_value())), t_range=(0, 2, 0.01)).set_style(stroke_width=1.0)))

        # # Fix a value t0
        # t0 = ValueTracker(1.0)

        # # Make Laplace spiral
        # spiral = ParametricCurve(lambda t: exp_plane.n2p(np.exp((t0.get_value()-t) * s.get_value() - t0.get_value() * s.get_value().real)), t_range=(0, 6, 0.01)).set_style(stroke_width=1.0).set_color(RED)
        # spiral.add_updater(lambda mobj: mobj.become(
        #     ParametricCurve(
        #         lambda t: exp_plane.n2p(np.exp((t0.get_value()-t) * s.get_value() - t0.get_value() * s.get_value().real)), t_range=(0, 6, 0.01)
        #     ).set_style(stroke_width=1.0).set_color(RED)
        # ))
        # self.play(ShowCreation(spiral))

        t0 = 0.5
        laplace_spirals = [
            ParametricCurve(
                lambda t: exp_plane.n2p(
                    np.exp(-t * complex(a, im_part) + t0 * complex(0, im_part))
                ),
                t_range=(0, 6, 0.01),
            )
            .set_style(stroke_width=1.0)
            .set_color(RED)
            for im_part in np.linspace(-10, 10, 20)
        ]

        self.embed()


def vector_to_function_graph(v: np.ndarray):
    """Converts an array of shape (N,) to a bar-graph function on the interval [0, N]."""
    pass


class VectorTracker(Mobject):
    """Stores a vector with real entries, where the individual entries act as value trackers."""

    value_type: type = np.float64

    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.vector = np.zeros((dim,), dtype=self.value_type)
        super().__init__(**kwargs)

    def init_uniforms(self) -> None:
        super().init_uniforms()
        self.uniforms["vector"] = self.vector.copy()

    def get_vector(self) -> np.ndarray:
        return self.uniforms["vector"]

    def set_vector(self, vector: np.ndarray):
        self.uniforms["vector"] = vector
        return self

    def get_value(self, i: int):
        return self.uniforms["vector"][i]

    def set_value(self, i: int, val: float):
        self.uniforms["vector"][i] = val
        return self

    def slide_value(self, i: int, val: float | complex):
        """Continuously changes a single entry of the matrix. This is the result which *should* be produced by
        self.animate.set_value(i, j, val), but is not because that method doesn't recompute the eigenvals/vecs
        at all intermediate steps."""
        self.clear_updaters()
        init_val = self.get_value(i)
        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_value(i, init_val + (val - init_val) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def slide_vector(self, vec: np.ndarray):
        """Continuously vary the entire vector in a linear fashion towards the given target vector."""
        self.clear_updaters()
        init_vec = self.get_vector()

        tracker = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_vector(
                init_vec + (vec - init_vec) * tracker.get_value()
            )
        )
        return tracker.animate.set_value(1.0)

    def rotate_vector(self, vec: np.ndarray):
        """Assumes the current, and final vectors are both normalized.
        Continuously vary the vector along a great circle towards the given target vector."""
        self.clear_updaters()
        vec /= np.linalg.norm(vec)
        init_vec = self.get_vector()
        init_vec *= 1 / np.linalg.norm(init_vec)

        # rotation_axis = np.cross(init_vec, vec)
        # rotation_axis /= np.linalg.norm(rotation_axis)
        angle = math.acos(init_vec.dot(vec))

        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_vector(
                init_vec
                + (vec - init_vec)
                * (
                    0.5
                    * np.sin(angle * v.get_value())
                    / (np.sin(0.5 * angle) * np.cos(angle * (v.get_value() - 0.5)))
                )
            )
        )
        return v.animate.set_value(1.0)


class MarkovChain(m.Scene):
    def construct(self):
        # Define the matrix for the Markov chain
        transition_matrix = np.array(
            [[0, 0.5, 0.4], [0.3, 0, 0.6], [0.7, 0.5, 0]], dtype=float
        )

        # Set an initial value
        v = VectorTracker(3).set_vector(np.array([1.0, 0.0, 0.0]))

        ## Make a graph representing this chain

        graph = m.VGroup()

        # Vertices
        radius = 0.4
        vxs = [
            Circle(arc_center=(-3, 3, 0), radius=radius).set_color(WHITE),
            Circle(arc_center=(-2, 4.5, 0), radius=radius).set_color(WHITE),
            Circle(arc_center=(-1, 3, 0), radius=radius).set_color(WHITE),
        ]

        def cis(theta):
            return np.array([np.cos(theta), np.sin(theta), 0])

        # Arrows
        arrow_opacity = 0.5
        transition_font_size = 16
        a_10 = (
            ArcBetweenPoints(
                start=vxs[1].get_center() + radius * cis((240 - 15) * DEGREES),
                end=vxs[0].get_center() + radius * cis((60 + 15) * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_10 = DecimalNumber(
            number=transition_matrix[0, 1], font_size=transition_font_size
        ).next_to(a_10.get_center(), cis(150 * DEGREES), 0.2)
        a_01 = (
            ArcBetweenPoints(
                start=vxs[0].get_center() + radius * cis((60 - 15) * DEGREES),
                end=vxs[1].get_center() + radius * cis((240 + 15) * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_01 = DecimalNumber(
            number=transition_matrix[1, 0], font_size=transition_font_size
        ).next_to(a_01.get_center(), cis(330 * DEGREES), -0.1)

        a_21 = (
            ArcBetweenPoints(
                start=vxs[2].get_center() + radius * cis((120 - 15) * DEGREES),
                end=vxs[1].get_center() + radius * cis((300 + 15) * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_21 = DecimalNumber(
            number=transition_matrix[1, 2], font_size=transition_font_size
        ).next_to(a_21.get_center(), cis(30 * DEGREES), 0.2)
        a_12 = (
            ArcBetweenPoints(
                start=vxs[1].get_center() + radius * cis((300 - 15) * DEGREES),
                end=vxs[2].get_center() + radius * cis((120 + 15) * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_12 = DecimalNumber(
            number=transition_matrix[2, 1], font_size=transition_font_size
        ).next_to(a_12.get_center(), cis(210 * DEGREES), -0.1)

        a_02 = (
            ArcBetweenPoints(
                start=vxs[0].get_center() + radius * cis((360 - 15) * DEGREES),
                end=vxs[2].get_center() + radius * cis((180 + 15) * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_02 = DecimalNumber(
            number=transition_matrix[2, 0], font_size=transition_font_size
        ).next_to(a_02.get_center(), cis(270 * DEGREES), 0.2)
        a_20 = (
            ArcBetweenPoints(
                start=vxs[2].get_center() + radius * cis((180 - 15) * DEGREES),
                end=vxs[0].get_center() + radius * cis(15 * DEGREES),
                angle=30 * DEGREES,
            )
            .add_tip(width=0.1, length=0.2)
            .set_style(stroke_opacity=arrow_opacity)
        )
        m_20 = DecimalNumber(
            number=transition_matrix[0, 2], font_size=transition_font_size
        ).next_to(a_20.get_center(), cis(90 * DEGREES), -0.1)

        arrows = [a_10, a_01, a_21, a_12, a_02, a_20]
        arrow_labels = [m_10, m_01, m_21, m_12, m_02, m_20]

        graph.add(*vxs, *arrows, *arrow_labels)
        graph.fix_in_frame()
        graph.set_width(4.0)
        graph.move_to((-4, 2, 0))
        self.play(FadeIn(graph))

        vx_values = []
        font_size = 30
        vx_values.append(
            DecimalNumber(font_size=font_size)
            .move_to(vxs[0].get_center())
            .add_updater(lambda mobj: mobj.set_value(v.get_value(0)))
        )
        vx_values.append(
            DecimalNumber(font_size=font_size)
            .move_to(vxs[1].get_center())
            .add_updater(lambda mobj: mobj.set_value(v.get_value(1)))
        )
        vx_values.append(
            DecimalNumber(font_size=font_size)
            .move_to(vxs[2].get_center())
            .add_updater(lambda mobj: mobj.set_value(v.get_value(2)))
        )
        self.add(*vx_values)
        # self.play(*[FadeIn(mobj) for mobj in vx_values])
        graph.add(*vx_values)
        graph.fix_in_frame()

        # arrows = [
        #     for i in range(3) for j in range(3)
        # ]

        # a_01 = ArcBetweenPoints(
        #     start=v1.get_center() - np.array([0.4, 0, 0]),
        #     end=v0.get_center() + np.array([0, 0.4, 0]),
        #     angle=120 * DEGREES,
        # )
        # a_01.add_tip(at_start=True)

        ## Draw the bar graph representation
        function_graph = m.VGroup()
        axes = Axes(
            x_range=(0, 3),
            y_range=(0, 1.5),
            y_axis_config={
                "include_tip": True,
            },
            x_axis_config={
                "include_tip": False,
            },
            # x_axis_config={
            #     "unit_size": 2,
            # },
        )
        function_graph.add(axes)
        function_graph.fix_in_frame()
        function_graph.set_width(4.0)
        function_graph.next_to(graph, RIGHT, 3.0)
        ax_w = (axes.c2p(1, 0, 0) - axes.c2p(0, 0, 0))[0]
        ax_h = (axes.c2p(0, 1, 0) - axes.c2p(0, 0, 0))[1]
        min_h = 1e-3
        function_graph.add(
            Rectangle(width=ax_w, height=1.0)
            .set_style(fill_opacity=1.0)
            .add_updater(
                lambda mobj: mobj.set_height(
                    (v.get_value(0) + min_h) * ax_h, stretch=True
                ).move_to(axes.c2p(0.5 + 0, v.get_value(0) * 0.5, 0))
            )
        )
        function_graph.add(
            Rectangle(width=ax_w, height=1.0)
            .set_style(fill_opacity=1.0)
            .add_updater(
                lambda mobj: mobj.set_height(
                    (v.get_value(1) + min_h) * ax_h, stretch=True
                ).move_to(axes.c2p(1.5, v.get_value(1) * 0.5, 0))
            )
        )
        function_graph.add(
            Rectangle(width=ax_w, height=1.0)
            .set_style(fill_opacity=1.0)
            .add_updater(
                lambda mobj: mobj.set_height(
                    (v.get_value(2) + min_h) * ax_h, stretch=True
                ).move_to(axes.c2p(2.5, v.get_value(2) * 0.5, 0))
            )
        )

        function_graph.fix_in_frame()
        self.play(FadeIn(function_graph))

        ## Make the 3D representation
        three_d_rep = VGroup()

        # Axes and background domain
        t_axes = ThreeDAxes((-0.5, 1.5), (-0.5, 1.5), (-0.5, 1.5)).set_width(4.0)
        triangle_domain = Polygon(
            t_axes.c2p(1, 0, 0), t_axes.c2p(0, 1, 0), t_axes.c2p(0, 0, 1)
        ).set_style(fill_opacity=0.3)

        # Point which is evolving over time
        three_d_pt = Dot(t_axes.c2p(*v.get_vector()), radius=0.05).set_color(RED)
        three_d_pt.add_updater(lambda mobj: mobj.move_to(t_axes.c2p(*v.get_vector())))

        # Evolution of the domain itself
        c0 = VectorTracker(3).set_vector(np.array([1, 0, 0], dtype=float))
        c1 = VectorTracker(3).set_vector(np.array([0, 1, 0], dtype=float))
        c2 = VectorTracker(3).set_vector(np.array([0, 0, 1], dtype=float))
        evolving_domain = (
            Polygon(
                t_axes.c2p(*c0.get_vector()),
                t_axes.c2p(*c1.get_vector()),
                t_axes.c2p(*c2.get_vector()),
            )
            .set_style(fill_opacity=0.5)
            .set_color(BLUE)
        )
        evolving_domain.add_updater(
            lambda mobj: mobj.become(
                Polygon(
                    t_axes.c2p(*c0.get_vector()),
                    t_axes.c2p(*c1.get_vector()),
                    t_axes.c2p(*c2.get_vector()),
                )
                .set_style(fill_opacity=0.5)
                .set_color(BLUE)
            )
        )

        three_d_rep.add(t_axes, triangle_domain, three_d_pt, evolving_domain)

        evolving_domain.add_updater(
            lambda mobj: mobj.become(
                Polygon(
                    t_axes.c2p(*c0.get_vector()),
                    t_axes.c2p(*c1.get_vector()),
                    t_axes.c2p(*c2.get_vector()),
                )
                .set_style(fill_opacity=0.5)
                .set_color(BLUE)
            )
        )
        three_d_rep.move_to((0, -2, 0))

        self.play(FadeIn(three_d_rep))

        ## Add an iteration indicator
        it_text = Text("Iteration:").fix_in_frame()
        it_text.next_to(graph, DOWN, 1)
        it_num = Integer(0).next_to(it_text, RIGHT, 0.2)
        self.add(it_text, it_num)

        for i in range(10):
            it_num.set_value(i + 1)
            self.play(
                v.animate.set_vector(np.matmul(transition_matrix, v.get_vector())),
                c0.animate.set_vector(np.matmul(transition_matrix, c0.get_vector())),
                c1.animate.set_vector(np.matmul(transition_matrix, c1.get_vector())),
                c2.animate.set_vector(np.matmul(transition_matrix, c2.get_vector())),
                run_time=0.5,
            )

        self.embed()


class StirlingsApproximation(Scene):
    def construct(self):
        # Flash up Stirling's formula at the top
        stirlings_formula = (
            Tex("n! \\approx \\left(\\frac{n}{e}\\right)^n\\sqrt{2\pi n}", font_size=36)
            .move_to((0, 3, 0))
            .fix_in_frame()
        )
        self.play(ShowCreation(stirlings_formula))
        self.wait()
        log_stirlings_formula = (
            Tex(
                "\\sum\limits_{k=1}^{n}\\log(k) \\approx \\left( n - \\frac{1}{2} \\right)\\log(n) - n + \\frac{1}{2}\\log(2\\pi)",
                font_size=36,
            )
            .move_to((0, 1, 0))
            .fix_in_frame()
        )
        self.play(ShowCreation(log_stirlings_formula))
        self.wait()
        self.play(
            FadeOut(stirlings_formula), log_stirlings_formula.animate.move_to((0, 3, 0))
        )
        n = 4

        self.frame.move_to(np.array([n / 2, -n / 2, 0]))
        self.frame.set_height(3 * n)

        # Use a diagram of moving rectangles to show the grid sum
        blue_rectangles = VGroup()
        for i in range(1, n + 1):
            blue_rectangles.add(
                VGroup(
                    *[
                        Rectangle(width=1, height=1 / i)
                        .move_to((0, j / i, 0), DL)
                        .set_fill(BLUE, 0.4)
                        .set_style(stroke_width=2.0)
                        for j in range(i)
                    ]
                )
            )
        for i in range(n):
            blue_rectangles[i].move_to((i, 0, 0), DL)

        # Fade in the rectangles and sum value
        self.play(ShowCreation(blue_rectangles))
        blue_sum_tex = Tex(f"{n}").next_to(blue_rectangles, UP, 0.5).set_color(BLUE)
        self.play(FadeIn(blue_sum_tex))

        # Create the grid sum
        VERTICAL_DISP = 1.5
        for j in range(1, n):
            self.play(
                *[
                    blue_rectangles[i][j].animate.move_to(
                        (i, -j * VERTICAL_DISP, 0), DL
                    )
                    for i in range(j, n)
                ]
            )

        # Add in red rectangles to complement
        red_rectangles = VGroup()
        for i in range(1, n):
            red_rectangles.add(
                VGroup(
                    *[
                        Rectangle(width=1, height=1 / (j + 1))
                        .move_to((j, -i * VERTICAL_DISP, 0), DL)
                        .set_fill(RED, 0.4)
                        .set_style(stroke_width=2.0)
                        for j in range(i)
                    ]
                )
            )
        self.play(ShowCreation(red_rectangles))

        # Add in sum values for red rectangles
        red_sum_labels = VGroup()
        for i in range(1, n):
            red_sum_labels.add(
                Tex(f"H_{i}").next_to(red_rectangles[i - 1], LEFT, 0.5).set_color(RED)
            )
        self.play(FadeIn(red_sum_labels))

        # Add in sum values for rows
        arrows = []
        row_sum_labels = []
        for i in range(n):
            arrows.append(
                Arrow(
                    (n + 0.5, 0.5 - i * VERTICAL_DISP, 0),
                    (n + 1.5, 0.5 - i * VERTICAL_DISP, 0),
                    buff=0.0,
                )
            )
            row_sum_labels.append(Tex(f"H_{n}").next_to(arrows[-1], RIGHT, 0.5))

        self.play(
            *[ShowCreation(a) for a in arrows], *[FadeIn(r) for r in row_sum_labels]
        )

        # Write the main formula
        harmonic_sum_formula = Tex("n + H_1 + H_2 + \\ldots + H_{n-1} = nH_n").move_to(
            (n / 2 + 1, -n * VERTICAL_DISP, 0)
        )

        self.play(FadeIn(harmonic_sum_formula))

        # Zoom out
        self.play(
            self.frame.animate.set_height(4 * n),
        )
        self.play(
            self.frame.animate.move_to((-n, -n / 2, 0)),
        )

        #### Make the integral diagram to relate H_n and the logarithm

        xmin = -1
        xmax = 6
        ymin = -1
        ymax = 4
        ax = Axes((xmin, xmax), (ymin, ymax)).move_to((-3 * n, -n / 2, 0))
        graph = ParametricCurve(
            lambda t: ax.c2p(t, 1 / t, 0), (1 / ymax, xmax, 0.05)
        ).set_style(stroke_width=1.0)
        graph_label = Tex("y=\\frac{1}{x}").next_to(graph, UR, -1.5)

        self.play(ShowCreation(ax), ShowCreation(graph), FadeIn(graph_label))

        int_a = 1
        int_b = xmax - 1
        quad = (
            ParametricCurve(lambda t: ax.c2p(t, 1 / t, 0), (1, int_b, 0.05))
            .add_line_to(ax.c2p(int_b, 0))
            .add_line_to(ax.c2p(1, 0))
            .add_line_to(ax.c2p(1, 1))
            .set_stroke(width=0)
            .set_fill(GREEN, 0.4)
        )
        quad_a_label = Tex("1").move_to(ax.c2p(int_a, -0.5, 0))
        quad_b_label = Tex("n").move_to(ax.c2p(int_b, -0.5, 0))
        quad_tex = Tex("\\log(n)").set_color(GREEN).next_to(quad, UP, 0.4)
        self.play(
            FadeIn(quad),
            ShowCreation(quad_a_label),
            ShowCreation(quad_b_label),
            FadeIn(quad_tex),
        )

        fn_dots = [Dot(ax.c2p(i, 1 / i, 0), radius=0.05) for i in range(1, int_b + 1)]

        # Trapezoid approximation
        quad_traps = [
            Polygon(
                ax.c2p(i, 0),
                ax.c2p(i + 1, 0),
                ax.c2p(i + 1, 1 / (i + 1)),
                ax.c2p(i, 1 / i),
            )
            .set_fill(RED, 0.2)
            .set_style(stroke_width=1.0)
            for i in range(1, int_b)
        ]
        self.play(
            *[ShowCreation(t) for t in quad_traps], *[ShowCreation(d) for d in fn_dots]
        )

        # Turn trapezoids into rectangles
        quad_rects = [
            Rectangle(
                width=(ax.x_axis.n2p(1 / 2) - ax.x_axis.n2p(0))[0],
                height=(ax.y_axis.n2p(1) - ax.y_axis.n2p(0))[1],
            )
            .move_to(ax.c2p(1, 0), DL)
            .set_fill(RED, 0.2)
            .set_style(stroke_width=1.0)
        ]
        for i in range(2, int_b):
            quad_rects.append(
                Rectangle(
                    width=(ax.x_axis.n2p(1) - ax.x_axis.n2p(0))[0],
                    height=(ax.y_axis.n2p(1 / i) - ax.y_axis.n2p(0))[1],
                )
                .move_to(ax.c2p(i - 0.5, 0), DL)
                .set_fill(RED, 0.2)
                .set_style(stroke_width=1.0)
            )
        quad_rects.append(
            Rectangle(
                width=(ax.x_axis.n2p(1 / 2) - ax.x_axis.n2p(0))[0],
                height=(ax.y_axis.n2p(1 / int_b) - ax.y_axis.n2p(0))[1],
            )
            .move_to(ax.c2p(int_b - 0.5, 0), DL)
            .set_fill(RED, 0.2)
            .set_style(stroke_width=1.0)
        )

        self.play(*[FadeOut(t) for t in quad_traps], *[FadeIn(r) for r in quad_rects])

        # Add additional portions of rectangles to make it a harmonic number
        end_rects = [
            Rectangle(width=0.5, height=1.0)
            .move_to(ax.c2p(1, 0), DR)
            .set_fill(YELLOW, 0.4)
            .set_style(stroke_width=1.0),
            Rectangle(width=0.5, height=1 / int_b)
            .move_to(ax.c2p(int_b, 0), DL)
            .set_fill(YELLOW, 0.4)
            .set_style(stroke_width=1.0),
        ]
        self.play(*[ShowCreation(r) for r in end_rects])

        log_formula = Tex(
            "\\log(n) \\approx H_n - \\frac{1}{2} - \\frac{1}{2n}"
        ).next_to(ax, DOWN, 0.5)
        self.play(FadeIn(log_formula))

        # Draw the connector
        connector = CurvedDoubleArrow(
            end_point=harmonic_sum_formula.get_bounding_box_point(LEFT)
            - np.array([1.0, 0, 0]),
            start_point=log_formula.get_bounding_box_point(RIGHT)
            + np.array([1.0, 0, 0]),
            angle=2 * DEGREES,
        ).set_style(stroke_width=2.0)
        self.play(ShowCreation(connector))

        # self.embed()
