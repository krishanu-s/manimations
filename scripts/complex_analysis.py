"""Animating the core concepts in complex analysis"""

import math

import manimlib as m
import numpy as np
from pyweierstrass import weierstrass as ws
from scipy.special import gamma


# Weierstrass p function, vectorized
def p_func(arr: np.ndarray, tau: complex = 0.0 + 0.5j):
    if len(arr.shape) == 0:
        return complex(ws.wp(arr, [0.5, tau]))

    return np.stack(tuple(p_func(z, tau) for z in arr), axis=0)


# Gamma function, vectorized
def gamma_func(arr: np.ndarray):
    return np.apply_along_axis(gamma, axis=-1, arr=arr)


# Riemann zeta function, vectorized


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


class InfinitelyDifferentiable(m.Scene):
    """
    The complex derivative is itself a function.
    The complex derivative itself has a complex derivative.
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
