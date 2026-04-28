"""Animating the core concepts in complex analysis"""

import math

import manimlib as m
import numpy as np
from manimlib import *
from scipy.special import gamma


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


# Weierstrass p function, vectorized


# Gamma function, vectorized
def gamma_func(arr: np.ndarray):
    return np.apply_along_axis(gamma, axis=-1, arr=arr)


# Riemann zeta function, vectorized TODO


class TestingComplexFunctions(m.Scene):
    """Testing ground for animating complex-valued functions"""

    def _configure_scene(self):
        # Set background color
        # self.camera.background_rgba = m.color_to_rgba("#FFFFFF", 1.0)
        pass

    def construct(self):
        self._configure_scene()

        # Set up axes
        self.frame.reorient(0, 0, 0)
        xmin = -2.0
        xmax = 2.0
        ymin = -2.0
        ymax = 2.0
        axes = m.Axes(x_range=(xmin, xmax), y_range=(ymin, ymax))
        bounding_box = m.Rectangle(xmax - xmin, ymax - ymin)
        bounding_box.move_to((0, 0, 0))
        self.play(m.ShowCreation(axes), m.ShowCreation(bounding_box))

        # Radius and center of domain as a disk
        r = m.ValueTracker(5.0)
        cx = m.ValueTracker(0.0)
        cy = m.ValueTracker(0.0)

        # (x, y) position of pole
        px = m.ValueTracker(1.0)
        py = m.ValueTracker(0.0)

        # Define a background function
        nx, ny = 101, 101

        # A simple rational function, vectorized
        def cx_fn(z):
            # return np.exp(np.pow(z, -1))
            # return z
            return np.pow(z - complex(1, 0), -1)

        heatmap = m.PlaneHeatMap((xmin, xmax), (ymin, ymax), (nx, ny))
        heatmap.init_heatmap(m.HeatMapType.COMPLEX)
        heatmap.set_background_opacity(0.3)
        # Basic function
        heatmap.set_f(lambda z: np.pow(z - complex(px.get_value(), py.get_value()), -1))

        # WP function
        # heatmap.set_f(lambda z: p_func(z, complex(1 / 4, np.sqrt(3) / 4)))

        heatmap.add_updater(
            lambda mobj: mobj.set_domain(
                lambda z: (z.real - cx.get_value()) ** 2
                + (z.imag - cy.get_value()) ** 2
                < r.get_value() ** 2
            )
        )

        self.embed()


class WhatIsAComplexDerivative(m.Scene):
    """Explaining the definition of an analytic/holomorphic function

    1. The complex derivative, explained visually in two ways
        (a) Deforming an image
        (b) Heatmap
    2. Some examples via heatmap:
    - f(x) = x + 1
    - f(x) = Cx
    - f(x) = x^3
    - f(x) = e^x
    - f(x) = 1/x
    - (Any others from book)
    """

    def construct(self):
        pass


class GoursatCauchyTheorem(Scene):
    """
    Proof of Goursat's theorem -- if f: U -> C is complex-differentiable on all of U, then the integral
    around any closed contour in U is equal to zero. We prove it for triangles, and then triangulate."""

    def construct(self):
        # Fill opacity for integrated regions
        INT_FILL_OPACITY = 0.2

        # Set the initial vertices
        vertices = [
            np.array([-2.5, -1, 0]),
            np.array([-0.5, 2, 0]),
            np.array([3.0, -1, 0]),
        ]

        def make_triangle_vertices(*d_values):
            """Each sequence of integers (d_1, d_2, ..., d_n) in {0, 1, 2, 3}^n
            specifies one of the subdivided triangles in Goursat's theorem.
            Returns the vertices of that triangle, in the same planar
            orientation as the original."""
            # Start with the initial triangle vertices
            vxs = [vertices[0], vertices[1], vertices[2]]

            # 0 = scale down by 1/2 factor towards v0
            # 1 = scale down by 1/2 factor towards v1
            # 2 = scale down by 1/2 factor towards v2
            # 3 = scale down by -1/2 factor towards centroid
            for d in d_values:
                if d == 0:
                    vxs[1] = 0.5 * (vxs[1] + vxs[0])
                    vxs[2] = 0.5 * (vxs[2] + vxs[0])

                elif d == 1:
                    vxs[0] = 0.5 * (vxs[0] + vxs[1])
                    vxs[2] = 0.5 * (vxs[2] + vxs[1])

                elif d == 2:
                    vxs[0] = 0.5 * (vxs[0] + vxs[2])
                    vxs[1] = 0.5 * (vxs[1] + vxs[2])

                else:
                    centroid = (vxs[0] + vxs[1] + vxs[2]) / 3
                    vxs[0] = 0.5 * (3 * centroid - vxs[0])
                    vxs[1] = 0.5 * (3 * centroid - vxs[1])
                    vxs[2] = 0.5 * (3 * centroid - vxs[2])

            return vxs

        def make_arrows(triangle_vxs: list[np.ndarray], thickness: float = 3.0):
            """Makes three arrows going around the triangle, indicating a contour integral"""
            centroid = sum(triangle_vxs) / 3
            return [
                Arrow(
                    triangle_vxs[i] * 0.85 + centroid * 0.15,
                    triangle_vxs[(i + 1) % 3] * 0.85 + centroid * 0.15,
                    thickness=thickness,
                    buff=0.0,
                ).set_color(BLUE)
                for i in range(3)
            ]

        # Hard-code the sequence of triangles
        d_seq = (0, 2, 3, 1, 2, 0, 3)

        # Calculate the final convergent point
        z = sum(make_triangle_vertices(*d_seq)) / 3
        z_pt = Dot(z, radius=0.05).set_color(RED)

        # Make the full scale version of the triangle and contour
        a0 = VGroup(*make_arrows(vertices))
        t0 = Polygon(*vertices).set_style(
            fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
        )

        # Put the first iteration

        self.play(ShowCreation(a0), ShowCreation(t0))

        # Make the second iteration
        t1 = VGroup()
        a1 = VGroup()
        for d in range(4):
            triangle_vxs = make_triangle_vertices(d)
            a1.add(VGroup(*make_arrows(triangle_vxs, thickness=1.2)))
            t1.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
                )
            )

        # Transform
        # TODO A better option is to draw this triangle to the right
        self.play(FadeOut(t0), FadeOut(a0), FadeIn(t1), FadeIn(a1))

        # Make only one of the triangles solid-filled
        self.play(
            *[
                t1[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[0]
            ]
        )
        self.play(*[FadeOut(a1[d]) for d in range(4) if d != d_seq[0]])

        # Make the third iteration
        t2 = VGroup()
        a2 = VGroup()
        for d1 in range(4):
            triangle_vxs = make_triangle_vertices(d_seq[0], d1)
            a2.add(VGroup(*make_arrows(triangle_vxs, thickness=0.5)))
            t2.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
                )
            )

        # Transform
        self.play(FadeOut(t1[d_seq[0]]), FadeOut(a1[d_seq[0]]), FadeIn(t2), FadeIn(a2))

        # Make only one of the triangles solid-filled
        self.play(
            *[
                t2[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[1]
            ]
        )
        self.play(*[FadeOut(a2[d]) for d in range(4) if d != d_seq[1]])

        # On and on for 3, ...
        t3 = VGroup()
        a3 = VGroup()
        for d2 in range(4):
            triangle_vxs = make_triangle_vertices(d_seq[0], d_seq[1], d2)
            a3.add(VGroup(*make_arrows(triangle_vxs, thickness=0.5)))
            t3.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
                )
            )

        self.play(
            FadeOut(t2[d_seq[1]]),
            FadeOut(a2[d_seq[1]]),
            FadeIn(t3),
            FadeIn(a3),
            run_time=0.5,
        )

        self.play(
            *[
                t3[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[2]
            ]
        )
        self.play(*[FadeOut(a3[d]) for d in range(4) if d != d_seq[2]], run_time=0.5)

        # ... 4, ...
        t4 = VGroup()
        a4 = VGroup()
        for d3 in range(4):
            triangle_vxs = make_triangle_vertices(d_seq[0], d_seq[1], d_seq[2], d3)
            a4.add(VGroup(*make_arrows(triangle_vxs, thickness=0.5)))
            t4.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
                )
            )

        self.play(
            FadeOut(t3[d_seq[2]]),
            FadeOut(a3[d_seq[2]]),
            FadeIn(t4),
            FadeIn(a4),
            run_time=0.5,
        )

        self.play(
            *[
                t4[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[3]
            ]
        )
        self.play(*[FadeOut(a4[d]) for d in range(4) if d != d_seq[3]], run_time=0.5)

        # For the rest of the way, just remove the arrows and put in the triangle bisections
        # TODO

        # Then show formulas. Key point:
        # - The function breaks into a constant part + linear part + remainder
        # - Integral of the constant part vanishes
        # - Integral of the linear part vanishes
        # - The remainder has the form (psi(z) * (z-z0)) where psi(z) -> 0 as z -> z0.
        # Length of contour halves at each point
        # Distance from centroid z halves at each point
        # Therefore, the nontrivial portion of the function int (psi(z) * (z-z0)) scales as 1/4 times
        # the supremum of psi(z) around the triangle, which itself goes to 0.
        self.embed()


class InfinitelyDifferentiable(m.Scene):
    """
    The complex derivative is itself a function.
    The complex derivative itself has a complex derivative.
    (Intuitively, why? Because there are power series representations. This, in turn, follows from the Cauchy integral
    formula. And that, in turn, is derived from Goursat's theorem.)
    Then by induction, a function with complex derivative is infinitely differentiable
    """

    def construct(self):
        pass


class PowerSeries(m.Scene):
    """
    Power series representation in a disk contained in the domain
    """

    def construct(self):
        pass


class ContourIntegrals(m.Scene):
    """
    Integral around a closed contour where interior is holomorphic, is zero.
    Depict proof of this for triangles (Goursat's theorem) and then triangulate an arbitrary contour.
    """

    def construct(self):
        pass


class ResidueCalculus(m.Scene):
    """Explain the behavior of singularities.

    1. Three types of singularities: removable, pole, and essential
    2. Cauchy integral formula, and more general form
    3. Meromorphic functions as maps to CP^1 (and, as domain as well).

    """

    def construct(self):
        pass


class AnalyticContinuation(m.ThreeDScene):
    """Explaining how analytic continuation works.

    1. Depict what a removable singularity is using the function f(x) = 1 + x + x^2 +... on R.
    2. Show how the same function can be continued beyond (-1, 1) in many different ways, and
       only one lines up with the power series
    3. Lift this same function up to the complex plane and show analytic continuation
    4. (...)
    5. Depict for gamma function
    6. Depict for zeta function

    """

    def construct(self):
        def fn(t):
            return [t, 0, 1 / (1 - t)]

        self.frame.reorient(0, 90, 0)
        axes = m.ThreeDAxes(x_range=(-3, 3), y_range=(-3, 3), z_range=(-3, 3))
        self.play(m.ShowCreation(axes))

        # Set the camera so the x-axis and z-axis are visible
        r = m.ValueTracker(0.05)
        curve = m.ParametricCurve(
            lambda t: axes.c2p(*fn(t)),
            t_range=[-1 + r.get_value(), 1 - r.get_value(), r.get_value()],
            color=m.BLUE,
        )
        self.play(m.ShowCreation(curve))
        # Do all the business about extending the function beyond (-1, 1)
        # self.play(
        #     m.ShowCreation(
        #         m.Circle(arc_center=fn(-1), radius=r.get_value()),
        #     )
        # )

        # Move to 3D
        self.play(self.frame.animate.reorient(0, 0, 0))

        # Convert to a complex function
        xmin = -3.0
        xmax = 3.0
        ymin = -3.0
        ymax = 3.0
        heatmap = m.PlaneHeatMap((xmin, xmax), (ymin, ymax), (201, 201))

        nx, ny = heatmap.resolution
        re, im = np.meshgrid(np.linspace(ymin, ymax, ny), np.linspace(xmin, xmax, nx))
        cx_array = np.stack((np.ravel(re), np.ravel(im)), axis=-1)
        domain_pts = heatmap.data["point"].copy()

        self.embed()


class GammaFunction(m.Scene):
    """Construction of the Gamma function.
    - Coin flipping question, about variance.
    - Introduce the formula for the binomial coefficient nCk, then put up Stirling's approximation.
    - A combinatorial approach to Stirling's approximation
        - Visualizing the basic approximation to H_n, the n-th harmonic number
        - Use this to get an approximation for sum(log n)
    - Complex analysis to introduce the Gamma function
        - Polynomials have a finite number of zeros ; how would we get an infinite number of zeros all on one side?
        - Factorization of polynomials ; factorization of sin(πz)/π

    """

    def construct(self):
        pass


class HarmonicFunctions(m.Scene):
    """
    Using complex-analytic functions to relate solutions to the heat/wave equation in different domains.
    """

    def construct(self):
        pass
