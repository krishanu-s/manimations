# Dumping ground for new ideas: particularly the most challenging animations which will make their
# way into other files.

import math
from typing import Callable

import manimlib as m
import numpy as np
from manimlib import *


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
        s_line = DashedLine(s_plane.n2p(complex(a, xmin)), s_plane.n2p(complex(a, xmax)))
        s_line_label = Tex("\\mathrm{Re}(s) = a", font_size=36).next_to(s_line, DOWN, 0.5)
        s_dot = GlowDot(s_plane.n2p(s.get_value())).add_updater(lambda mobj: mobj.move_to(s_plane.n2p(s.get_value())))

        self.play(ShowCreation(s_line), FadeIn(s_line_label), ShowCreation(s_dot))

        # Draw in all the lines, for every s value
        laplace_rays = [
            Line(s_plane.n2p(0), s_plane.n2p(3 * complex(a, im_part))).set_style(stroke_width=1.0, stroke_opacity=0.5)
            for im_part in np.linspace(xmin, xmax, 10)
        ]
        self.play(*[ShowCreation(l) for l in laplace_rays])

        # t values from 0 to ∞

        # Plane where the exponentials live
        exp_plane = ComplexPlane(x_range=(-2, 2, 1.0), y_range=(-2, 2, 1.0)).set_width(8.0).next_to(s_plane, RIGHT, 2.0)
        self.play(ShowCreation(exp_plane))

        # Images of the laplace rays
        laplace_spirals = [
            ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * complex(a, im_part))), t_range=(0, 3, 0.01)).set_style(stroke_width=1.0)
            for im_part in np.linspace(xmin, xmax, 10)
        ]

        # Map via (s, t) -> exp(-st)
        exp_s_curve = ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * s.get_value())), t_range=(0, 2, 0.01)).set_style(stroke_width=1.0)
        exp_s_curve.add_updater(lambda mobj: mobj.become(ParametricCurve(lambda t: exp_plane.n2p(np.exp(-t * s.get_value())), t_range=(0, 2, 0.01)).set_style(stroke_width=1.0)))
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
