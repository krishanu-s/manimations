"""Testing ground"""

import math

# from lib.conic import *
from functools import partial

import manimlib as m
import numpy as np
from manim import *
from pyweierstrass import weierstrass as ws

# def stereographic_projection(conic: ConicSection, basepoint: list[float], t: float,) -> np.ndarray:
#     """
#     Defines a map from P^1 to a given conic. That is, parametrizes the points of said
#     conic section as a pair of rational functions t -> (x(t), y(t)).

#     Outputs 3D array, where the third coordinate is zero."""
#     # TODO
#     pass


# def elliptic_parametrization(elliptic_curve, tau: complex, z: complex) -> tuple[complex]:
#     """
#     Defines a map from the torus C / <1, T> to the elliptic curve y^2 = 4(x-r1)(x-r2)(x-r3) defined by
#     z -> (p(z), p'(z)),  where p denotes the Weierstrass p-function of the lattice defined by T, and
#     r1 = p(1/2), r2 = p(T/2), r3 = p((1+T)/2),
#     """
#     return (ws.wp(z, [1/2, tau/2]), ws.wpprime(z, [1/2, tau/2]))


class DefinePythagoreanTriples(m.Scene):
    """Define Pythagorean triples."""

    def construct(self):
        """Draw a right triangle with sidelengths a, b, and c. Then write in the Pythagorean theorem"""
        self.clear()

        # Draw a Pythagorean triangle
        a = m.ValueTracker(1.0)
        b = m.ValueTracker(1.0)

        def update_triangle(mobj: m.Polygon):
            mobj.set_points(
                [
                    np.array([0, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([1, 0, 0]),
                ]
            )

        triangle = m.Polygon(
            np.array([0, 0, 0]),
            np.array([0, a.get_value(), 0]),
            np.array([b.get_value(), 0, 0]),
        ).set_color(m.WHITE)
        # triangle.add_updater(lambda z: z.become(Polygon(
        #     np.array([0, 0, 0]),
        #     np.array([0, a.get_value(), 0]),
        #     np.array([b.get_value(), 0, 0]),
        #     ).set_color(WHITE)))

        # triangle.add_updater(lambda z: z.set_points(
        #     [np.array([0, 0, 0]),
        #     np.array([0, a.get_value(), 0]),
        #     np.array([b.get_value(), 0, 0]),]
        #     ).set_color(WHITE)))

        right_angle = m.Elbow(0.1)

        label_a = m.DecimalNumber(1, font_size=24).move_to(np.array([-0.3, 0.5, 0.0]))
        label_a.add_updater(lambda z: z.set_value(a.get_value()))
        label_a.add_updater(
            lambda z: z.move_to(np.array([-0.3, a.get_value() / 2, 0.0]))
        )

        label_b = m.DecimalNumber(1, font_size=24).move_to(np.array([0.5, -0.3, 0.0]))
        label_b.add_updater(lambda z: z.set_value(b.get_value()))
        label_b.add_updater(
            lambda z: z.move_to(np.array([b.get_value() / 2, -0.3, 0.0]))
        )

        label_c = m.DecimalNumber(math.sqrt(2), font_size=24).move_to(
            np.array([0.7, 0.7, 0.0])
        )
        label_c.add_updater(
            lambda z: z.set_value(math.sqrt(a.get_value() ** 2 + b.get_value() ** 2))
        )
        label_c.add_updater(
            lambda z: z.move_to(
                np.array([0.2 + b.get_value() / 2, 0.2 + a.get_value() / 2, 0.0])
            )
        )

        self.play(
            m.ShowCreation(triangle, rate_func=m.linear),
            m.ShowCreation(right_angle, rate_func=m.linear),
            # m.FadeIn(label_a, label_b, label_c),
        )
        self.play(m.Write(m.Tex("a^2 + b^2 = c^2").move_to(np.array([1.0, -1.0, 0.0]))))

        self.embed()

        # Change values a, b along some path, ending up at (3, 4)

        self.play(a.animate.set_value(2.0), run_time=1.5)
        self.play(b.animate.set_value(3.0), run_time=1.5)


class StereographicProjection(m.Scene):
    """Demonstrates stereographic projection in 2D for a circle"""

    def construct(self):
        # Parameter which corresponds to the y-coordinate of the point on the y-axis.
        t = m.ValueTracker(-1.5)
        pt = np.array([1, 0, 0])  # Basepoint for stereographic projection

        self.add(m.Axes([-2.5, 2.5], [-2.5, 2.5]))

        curve = m.Circle(1)
        self.play(m.ShowCreation(curve))

        basepoint = m.Dot(pt)
        self.play(m.FadeIn(basepoint))

        length = 8
        theta = math.atan2(t.get_value(), 1)
        slope_vec = np.array([np.cos(theta), np.sin(theta), 0])
        line = m.Line(pt + (length / 2) * slope_vec, pt - (length / 2) * slope_vec)

        ## TODO Switch to grow from midpoint
        self.play(m.ShowCreation(line))

        self.embed()


# class SingularCubic(Scene):
#     """A short scene showing how one can parametrize a singular cubic curve"""

# TODO Some scenes showing some number theory results
# TODO some geometry depicting elliptic curves


# Scene describing how to parametrize Pythagorean triples, and integer solutions to any other homogeneous
# quadratic equation, by parametrizing rational points on a conic section.
class PythagoreanTriples(Scene):
    def scene_0(self):
        self.clear()
        pass

    def scene_1(self):
        """Draw a right triangle with sidelengths a, b, and c. Then write in the Pythagorean theorem"""
        self.clear()

        # Draw a Pythagorean triangle
        a = ValueTracker(1.0)
        b = ValueTracker(1.0)

        def update_triangle(mobj: Polygon):
            mobj.set_points(
                [
                    np.array([0, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([1, 0, 0]),
                ]
            )

        triangle = Polygon(
            np.array([0, 0, 0]),
            np.array([0, a.get_value(), 0]),
            np.array([b.get_value(), 0, 0]),
        ).set_color(WHITE)
        triangle.add_updater(
            lambda z: z.become(
                Polygon(
                    np.array([0, 0, 0]),
                    np.array([0, a.get_value(), 0]),
                    np.array([b.get_value(), 0, 0]),
                ).set_color(WHITE)
            )
        )

        right_angle = Elbow(0.1)

        label_a = DecimalNumber(1, font_size=24).move_to([-0.3, 0.5, 0])
        label_a.add_updater(lambda z: z.set_value(a.get_value()))
        label_a.add_updater(lambda z: z.move_to([-0.3, a.get_value() / 2, 0]))

        label_b = DecimalNumber(1, font_size=24).move_to([0.5, -0.3, 0])
        label_b.add_updater(lambda z: z.set_value(b.get_value()))
        label_b.add_updater(lambda z: z.move_to([b.get_value() / 2, -0.3, 0]))

        label_c = DecimalNumber(math.sqrt(2), font_size=24).move_to([0.7, 0.7, 0])
        label_c.add_updater(
            lambda z: z.set_value(math.sqrt(a.get_value() ** 2 + b.get_value() ** 2))
        )
        label_c.add_updater(
            lambda z: z.move_to([0.2 + b.get_value() / 2, 0.2 + a.get_value() / 2, 0])
        )

        self.play(
            Create(triangle, rate_func=linear),
            Create(right_angle, rate_func=linear),
            FadeIn(label_a, label_b, label_c),
        )
        self.play(Write(MathTex("a^2 + b^2 = c^2").move_to([1, -1, 0])))

        # Change values a, b along some path, ending up at (3, 4)

        self.play(a.animate.set_value(2.0), run_time=1.5)
        self.play(b.animate.set_value(3.0), run_time=1.5)

    def scene_3(self):
        self.clear()
        pass

    def scene_4(self):
        self.clear()
        pass

    def scene_5(self):
        self.clear()

        # # Parameter for the slope of the stereographic projection line
        # m = manim.ValueTracker()
        # m.set_value(-2.0)

        # # Cartesian form of the conic section
        # cart_eq = CartesianConicEquation(1, 0, 1, 0, 0, -1)
        # polar_eq = cart_eq.to_polar()

        # curve = manim.ParametricFunction(function=lambda t: polar_eq.param(t), t_range=(0, manim.TAU,))

        # axes = manim.Axes((-3, 3), (-3, 3))
        # self.play(manim.FadeIn(axes, curve))

        # # Add a temporary value tracker to change the equation
        # c_xy = manim.ValueTracker()
        # def update_curve(c: manim.ParametricFunction) -> manim.ParametricFunction:
        #     cart_eq.c_xy = c_xy.get_value()
        #     polar_eq = cart_eq.to_polar()
        #     return manim.ParametricFunction(function=lambda t: polar_eq.param(t), t_range=(0, manim.TAU,))
        # curve.add_updater(update_curve)
        # self.play(c_xy.animate.set_value(1.0))
        # curve.remove_updater(update_curve)

    def scene_6(self):
        self.clear()
        pass

    def construct(self):
        # self.scene_1()
        self.scene_2()
        # self.test_scene()
