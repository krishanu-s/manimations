"""Animating the core concepts in complex analysis"""

from __future__ import annotations

import cmath
import math
from typing import Annotated, Dict, Iterable, Literal, Tuple, Union

import manimlib as m
import numpy as np
from manimlib import *
from scipy.special import gamma

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


class Curve(VMobject):
    """A curve composed of a sequence of bezier curves."""

    num_segments: int

    @classmethod
    def make_from_anchors(cls, *anchors) -> Curve:
        """Makes a curve passing through the given anchors."""
        n = len(anchors) - 1
        curve = Curve()
        curve.num_segments = n

        h1, h2 = get_smooth_cubic_bezier_handle_points(anchors)
        for i in range(n):
            curve.add_cubic_bezier_curve(anchors[i], h1[i], h2[i], anchors[i + 1])
        return curve

    @property
    def anchors(self) -> list[Vect3]:
        """Retrieves the anchors."""
        return list(self.data["point"][::6]) + [self.data["point"][-1]]

    def transform(self, fn: Callable[[Vect3], Vect3]) -> Curve:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve."""
        return Curve.make_from_anchors(*map(fn, self.anchors))


class CurveGrid(VGroup):
    """A grid of curves passing through an array of points, intersecting to form an m-by-n grid.
    The m vertical (x) curves each are defined by N anchors, where N - 1 is a multiple of n - 1.
    The n horizontal (y)  curves are each defined by M anchors, where M - 1 is a multiple of m - 1."""

    anchors_x: FloatArray  # Array of shape (m, N, 3).
    anchors_y: FloatArray  # Array of shape (n, M, 3).
    num_curves_x: int  # m
    num_curves_y: int  # n
    num_anchors_x: int  # M
    num_anchors_y: int  # N

    @classmethod
    def make_from_anchors(cls, anchors_x: FloatArray, anchors_y: FloatArray):
        """Initializes the grid from the set of anchors for each crossing set."""
        grid = CurveGrid()
        grid.anchors_x = anchors_x
        grid.anchors_y = anchors_y
        grid.num_curves_x = anchors_x.shape[0]
        grid.num_anchors_y = anchors_x.shape[1]

        grid.num_curves_y = anchors_y.shape[0]
        grid.num_anchors_x = anchors_y.shape[1]

        for i in range(grid.num_curves_x):
            grid.add(Curve.make_from_anchors(*list(anchors_x[i, :, :])))

        for i in range(grid.num_curves_y):
            grid.add(Curve.make_from_anchors(*list(anchors_y[i, :, :])))

        return grid

    @classmethod
    def make_linear(
        cls,
        xlims: tuple[float, float],
        ylims: tuple[float, float],
        num_curves_x: int,
        num_anchors_x: int,
        num_curves_y: int,
        num_anchors_y: int,
    ):
        xmin, xmax = xlims
        ymin, ymax = ylims
        anchors_x = np.array(
            [
                [[x, y, 0.0] for y in np.linspace(ymin, ymax, num_anchors_y)]
                for x in np.linspace(xmin, xmax, num_curves_x)
            ]
        )
        anchors_y = np.array(
            [
                [[x, y, 0.0] for x in np.linspace(xmin, xmax, num_anchors_x)]
                for y in np.linspace(ymin, ymax, num_curves_y)
            ]
        )
        return CurveGrid.make_from_anchors(anchors_x, anchors_y)

    def get_curve_x(self, i: int):
        return self[i]

    def get_curve_y(self, i: int):
        return self[self.num_curves_x + i]

    def get_anchors_x(self):
        return self.anchors_x

    def get_anchors_y(self):
        return self.anchors_y

    def transform(self, fn: Callable[[Vect3], Vect3]) -> CurveGrid:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve.
        Assumes the function isn't vectorized."""
        anchors_x = self.get_anchors_x()
        anchors_y = self.get_anchors_y()
        mapped_anchors_x = np.stack(
            [
                np.stack(
                    [fn(anchors_x[i, j]) for j in range(self.num_anchors_y)], axis=0
                )
                for i in range(self.num_curves_x)
            ],
            axis=0,
        )

        mapped_anchors_y = np.stack(
            [
                np.stack(
                    [fn(anchors_y[i, j]) for j in range(self.num_anchors_x)], axis=0
                )
                for i in range(self.num_curves_y)
            ],
            axis=0,
        )
        return CurveGrid.make_from_anchors(mapped_anchors_x, mapped_anchors_y)

    def transform_vectorized(self, fn: Callable[[FloatArray], FloatArray]) -> CurveGrid:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve.
        Assumes the function is vectorized."""
        return CurveGrid.make_from_anchors(
            fn(self.get_anchors_x()), fn(self.get_anchors_y())
        )


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


def pdisk_aut(a: complex, z: complex):
    """Consider the automorphism of the Poincare disk interchanging 0 and a.
    Applies this function to z."""
    return (a - z) / (1 - z * a.conjugate())


def vect3_to_cx(v: Vect3):
    return complex(v[0], v[1])


def cx_to_vect3(z: complex):
    return np.array([z.real, z.imag, 0.0])


# Riemann zeta function, vectorized TODO
class RiemannMapping(Scene):
    """Some animations related to the Riemann mapping theorem."""

    def do_fn(
        self, grid: CurveGrid, fn: Callable[[complex], complex], run_time: float = 1.0
    ) -> CurveGrid:
        """Deforms the grid to its image under an arbitrary function."""
        tgrid = grid.transform(lambda v: cx_to_vect3(fn(vect3_to_cx(v))))
        self.play(grid.animate.become(tgrid), run_time=run_time)
        self.remove(grid)
        grid = tgrid
        self.add(grid)
        return grid

    def do_pdisk_aut(
        self, grid: CurveGrid, a: complex, run_time: float = 1.0
    ) -> CurveGrid:
        """Deforms the grid to its image under the automorphism represented by the complex number a."""
        return self.do_fn(grid, lambda z: pdisk_aut(a, z), run_time=run_time)

    def do_sqrt(self, grid: CurveGrid, run_time: float = 1.0) -> CurveGrid:
        """Deforms the grid to its image under the square root function.

        WARNING: Assumes that the grid doesn't cross the negative real line. This isn't checked."""
        return self.do_fn(grid, cmath.sqrt, run_time=run_time)

    def do_riemann_mapping_expansion(
        self, grid: CurveGrid, a: complex, do_steps: bool = True, run_time: float = 3.0
    ) -> CurveGrid:
        """Given a domain in the Poincare disk which contains 0, and a point a
        which lies outside of it, performs a series of animations to deform the
        grid to be larger while still lying within the Poincare disk."""
        if do_steps:
            grid = self.do_pdisk_aut(grid, a, run_time=run_time / 3)
            grid = self.do_sqrt(grid, run_time=run_time / 3)
            grid = self.do_pdisk_aut(grid, cmath.sqrt(a), run_time=run_time / 3)
        else:
            grid = self.do_fn(
                grid,
                lambda z: pdisk_aut(cmath.sqrt(a), cmath.sqrt(pdisk_aut(a, z))),
                run_time=run_time,
            )
        return grid

    def construct(self):
        circle = ParametricCurve(polar_vec, (0, TAU, TAU / 100))
        self.add(circle)

        # Make a square grid
        grid = CurveGrid.make_linear((-0.2, 0.2), (-0.2, 0.2), 9, 49, 9, 49)
        self.add(grid)

        # Expand it
        tgrid = grid.transform(lambda v: v * 2.5 * np.sqrt(2))
        self.play(grid.animate.become(tgrid))
        self.remove(grid)
        grid = tgrid
        self.add(grid)

        ## Automorphism associated to a point not inside the domain.
        grid = self.do_riemann_mapping_expansion(grid, complex(0.8, 0))

        # Do iterative expansions
        a = complex(0.9, 0)
        for i in range(10):
            grid = self.do_riemann_mapping_expansion(grid, a, False, 0.2)
            print(f"Iteration {i}")

        grid = self.do_fn(grid, lambda z: z * complex(0, 1), 0.2)

        for i in range(10):
            grid = self.do_riemann_mapping_expansion(grid, a, False, 0.2)
            print(f"Iteration {i}")

        grid = self.do_fn(grid, lambda z: z * complex(0, 1), 0.2)

        for i in range(10):
            grid = self.do_riemann_mapping_expansion(grid, a, False, 0.2)
            print(f"Iteration {i}")

        grid = self.do_fn(grid, lambda z: z * complex(0, 1), 0.2)

        for i in range(10):
            grid = self.do_riemann_mapping_expansion(grid, a, False, 0.2)
            print(f"Iteration {i}")

        # TODO Find a method to locate points in the Poincare disk which lie outside of the grid.
        # One possibility is to follow any curve a bit beyond the last point, without going outside of the disk.

        self.embed()


class LocalGridScene(Scene):
    """Depicts two coordinate planes related by a complex-analytic function, and the effect
    of this function on a local coordinate patch around a single point.

    The hope is to use this to show Goursat's theorem:
    - first showing that all contour integrals around a point are zero for a linear function
    - then showing that for a function whose derivative is zero at a point, the integral of a small contour
    of radius r around that point goes as smaller than p(r)/r^2 where p(r) -> 0 as r -> 0
    - thus deduce that if the function is holomorphic at a point, then the integral of a small contour
    of radius r around that point goes as smaller than p(r)/r^2 where p(r) -> 0 as r -> 0
    - then using that there's some point inside the contour such that small contours around that point
    majorize the big one
    """

    def construct(self):
        self.embed()
        # Make input axis
        axes_in = Axes((-5, 5), (-5, 5))
        self.play(ShowCreation(axes_in))

        # Point around which grid will be defined
        pt_x = ValueTracker(2.0)
        pt_y = ValueTracker(2.0)

        # Parameters for local grid of point
        rad = ValueTracker(0.7)
        nx, ny = 6, 6

        # Make local grid
        def make_hline(y: float):
            l = Line(ORIGIN, ORIGIN).set_style(
                stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE
            )
            l.add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    axes_in.c2p(
                        pt_x.get_value() - rad.get_value(),
                        pt_y.get_value() + y * rad.get_value(),
                    ),
                    axes_in.c2p(
                        pt_x.get_value() + rad.get_value(),
                        pt_y.get_value() + y * rad.get_value(),
                    ),
                )
            )
            return l

        def make_vline(x: float):
            l = Line(ORIGIN, ORIGIN).set_style(
                stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE
            )
            l.add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    axes_in.c2p(
                        pt_x.get_value() + x * rad.get_value(),
                        pt_y.get_value() - rad.get_value(),
                    ),
                    axes_in.c2p(
                        pt_x.get_value() + x * rad.get_value(),
                        pt_y.get_value() + rad.get_value(),
                    ),
                )
            )
            return l

        def make_local_grid(num_x, num_y):
            hlines = map(make_hline, np.linspace(-1, 1, num_y + 1))
            vlines = map(make_vline, np.linspace(-1, 1, num_x + 1))
            return VGroup(*hlines, *vlines)

        local_grid = make_local_grid(nx, ny)
        self.play(ShowCreation(local_grid))

        # Make output axes
        axes_out = Axes((-5, 5), (-5, 5)).next_to(axes_in, RIGHT, 4.0)
        self.play(ShowCreation(axes_out))

        # Define the mapping function
        def cx_fn(x: float, y: float) -> tuple[float, float]:
            z = complex(x, y)
            fz = z * z
            return fz.real, fz.imag

        # Make image of the local grid
        # TODO Make these updaters more efficient.
        # TODO Make modification of the function possible.
        def make_hline_img(y: float, fn: Callable[[float, float], tuple[float, float]]):
            l = ParametricCurve(lambda t: ORIGIN, (-1, 1, 0.05)).set_style(
                stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE
            )

            l.add_updater(
                lambda mobj: mobj.become(
                    ParametricCurve(
                        lambda t: axes_out.c2p(
                            *fn(
                                pt_x.get_value() + t * rad.get_value(),
                                pt_y.get_value() + y * rad.get_value(),
                            )
                        ),
                        (-1, 1, 0.05),
                    ).set_style(stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE)
                )
            )
            return l

        def make_vline_img(x: float, fn: Callable[[float, float], tuple[float, float]]):
            l = ParametricCurve(lambda t: ORIGIN, (-1, 1, 0.05)).set_style(
                stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE
            )
            l.add_updater(
                lambda mobj: mobj.become(
                    ParametricCurve(
                        lambda t: axes_out.c2p(
                            *fn(
                                pt_x.get_value() + x * rad.get_value(),
                                pt_y.get_value() + t * rad.get_value(),
                            )
                        ),
                        (-1, 1, 0.05),
                    ).set_style(stroke_width=1.0, stroke_opacity=0.6, stroke_color=BLUE)
                )
            )
            return l

        def make_local_grid_img(
            num_x, num_y, fn: Callable[[float, float], tuple[float, float]] = cx_fn
        ):
            hlines_img = map(
                lambda y: make_hline_img(y, fn), np.linspace(-1, 1, num_y + 1)
            )
            vlines_img = map(
                lambda x: make_vline_img(x, fn), np.linspace(-1, 1, num_x + 1)
            )
            return VGroup(*hlines_img, *vlines_img)

        local_grid_img = make_local_grid_img(nx, ny)

        self.play(ShowCreation(local_grid_img))

        self.embed()


class TestingComplexFunctions(m.Scene):
    """Testing ground for animating complex-valued functions"""

    def _configure_scene(self):
        # Set background color
        # self.camera.background_rgba = m.color_to_rgba("#FFFFFF", 1.0)
        pass

    def make_frames(self):
        # Axes size
        xmin = -5.0
        xmax = 5.0
        ymin = -5.0
        ymax = 5.0

        # First complex function display
        self.axes = m.Axes(x_range=(xmin, xmax), y_range=(ymin, ymax))
        bbox = m.Rectangle(xmax - xmin, ymax - ymin)
        bbox.move_to((0, 0, 0))
        self.cx_frame = VGroup(self.axes, bbox)
        self.play(m.ShowCreation(self.cx_frame))

        # Second complex function display
        self.cx_frame_2 = self.cx_frame.copy().next_to(self.cx_frame, RIGHT, 2.0)
        self.axes_2 = self.cx_frame_2[0]
        self.play(m.ShowCreation(self.cx_frame_2))

    def construct(self):
        self._configure_scene()
        self.frame.reorient(0, 0, 0)

        self.make_frames()
        self.embed()
        # self.animate_heatmaps()

    def animate_heatmaps(self):
        """Do heatmap animations for sin and gamma functions"""
        # Function resolution
        nx, ny = 101, 101
        cx_frame = self.cx_frame
        cx_frame_2 = self.cx_frame_2
        axes = self.axes
        xmin, xmax, _ = self.axes.x_axis.x_range
        ymin, ymax, _ = self.axes.y_axis.x_range

        # Populating the first one
        # Radius and center of domain as a disk
        r = m.ValueTracker(5.0)
        cx = m.ValueTracker(0.0)
        cy = m.ValueTracker(0.0)

        # (x, y) position of pole
        px = m.ValueTracker(1.0)
        py = m.ValueTracker(0.0)

        # Initialize heatmaps
        heatmap = PlaneHeatMap((xmin, xmax), (ymin, ymax), (nx, ny)).move_to(
            axes.get_center()
        )
        heatmap.init_heatmap(m.HeatMapType.COMPLEX)
        heatmap.set_background_opacity(0.3)

        heatmap_2 = PlaneHeatMap((xmin, xmax), (ymin, ymax), (nx, ny)).move_to(
            cx_frame_2[1].get_center()
        )
        heatmap_2.init_heatmap(m.HeatMapType.COMPLEX)
        heatmap_2.set_background_opacity(0.3)

        heatmap.set_f(lambda z: np.sin(PI * z) / PI)
        heatmap_2.set_f(lambda z: z)

        # sin(pi * z) / pi on the left, z on the right
        self.play(ShowCreation(heatmap), ShowCreation(heatmap_2))
        sin_tex = Tex(
            "\\frac{\\sin(\\pi z)}{\\pi} = z \\prod\\limits_{n=1}^{\infty}\\left(1 - \\frac{z^2}{n^2}\\right)"
        ).next_to(cx_frame, UP, 1.0)
        self.play(ShowCreation(sin_tex))

        # Show successive Hadamard approximations
        for n in range(1, 7):
            self.play(
                heatmap_2.animate.set_f(
                    lambda z: reduce(
                        (lambda x, y: x * y),
                        [z, *[1 - np.pow(z, 2) / (i**2) for i in range(1, n)]],
                    )
                )
            )

        # G(z) function
        gamma_tex = Tex(
            "\\Gamma(z) = \\frac{e^{\\gamma z}}{z}\\prod\\limits_{n=1}^{\infty} e^{\\frac{z}{n}}\\left(1 + \\frac{z}{n}\\right)^{-1}"
        ).next_to(cx_frame, UP, 1.0)
        self.play(heatmap.animate.set_f(gamma_func), Transform(sin_tex, gamma_tex))
        g = sum(1 / n for n in range(1, 5000)) - np.log(5000)
        for n in range(1, 7):
            self.play(
                heatmap_2.animate.set_f(
                    lambda z: reduce(
                        (lambda x, y: x * y),
                        [
                            np.pow(z, -1),
                            np.exp(-g * z),
                            *[
                                np.exp(z / i) * np.pow(1 + z / i, -1)
                                for i in range(1, n)
                            ],
                        ],
                    )
                )
            )

        # WP function
        # heatmap.set_f(lambda z: p_func(z, complex(1 / 4, np.sqrt(3) / 4)))

        # heatmap.add_updater(
        #     lambda mobj: mobj.set_domain(
        #         lambda z: (z.real - cx.get_value()) ** 2
        #         + (z.imag - cy.get_value()) ** 2
        #         < r.get_value() ** 2
        #     )
        # )

        self.embed()


## CH 0 portions

## CH 1 portions
# - Define a complex derivative formally.
# - Conformality intuition.
# - Examples, and non-examples.
# - Complex derivative as itself another *function*


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


## CH 2 portions

## CH 2 portions
# - Complex integration along a path. Path-dependence from a to b is a new phenomenon.
# - If a function f is the complex derivative of some other function, then its integral is path-independent. Equivalently,
# integral around contours is zero.
# - Goal is to show that if a function f itself has a complex derivative, then integral around contours is zero.
# This is Goursat's theorem.


class WhatIsComplexIntegration(Scene):
    """Let's say that you have a complex-valued function defined on the plane -- meaning that it takes a complex
    value at every point."""

    def construct(self):
        pass


class GoursatTheorem(Scene):
    """
    Proof of Goursat's theorem -- if f: U -> C is complex-differentiable on all of U, then the integral
    around any closed contour in U is equal to zero. We prove it for triangles, and then for an arbitrary
    contour we triangulate."""

    # Inset for arrows going around the interior of a contour
    CONTOUR_ARROW_GAP = 0.15

    # Fill opacity for integrated regions
    INT_FILL_OPACITY = 0.2

    def make_arrows(self, vxs: list[np.ndarray], thickness: float = 3.0):
        """Makes arrows going around the polygon with the given vertices, indicating a contour integral.
        Sets the arrows slightly inside the contour."""
        n = len(vxs)
        centroid = sum(vxs) / n
        c = self.CONTOUR_ARROW_GAP
        return [
            Arrow(
                vxs[i] * (1 - c) + centroid * c,
                vxs[(i + 1) % n] * (1 - c) + centroid * c,
                thickness=thickness,
                buff=0.0,
            ).set_color(BLUE)
            for i in range(n)
        ]

    def construct(self):
        INT_FILL_OPACITY = self.INT_FILL_OPACITY

        # Write the theorem to be proven
        thm_statement = Tex("\\int_C f(z)dz = 0").move_to((-3, 3, 0)).fix_in_frame()

        # Set the initial vertices for the triangle
        vertices = [
            np.array([-2.5, -1, 0]),
            np.array([-0.5, 2, 0]),
            np.array([3.0, -1, 0]),
        ]
        n_vertices = len(vertices)

        def make_polygon_vertices(*d_values):
            """Each sequence of integers (d_1, d_2, ..., d_n) in {0, 1, 2, 3}^n
            specifies one of the subdivided triangles in Goursat's theorem.
            Returns the vertices of that triangle, in the same planar
            orientation as the original."""
            # Start with the initial triangle vertices
            vxs = vertices.copy()

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

        # Hard-code the sequence of subcontours used
        d_seq = (0, 2, 3, 1, 2, 0, 3)

        # Make the full scale version of the triangle and contour
        a0 = VGroup(*self.make_arrows(vertices))
        t0 = Polygon(*vertices).set_style(
            fill_opacity=INT_FILL_OPACITY, fill_color=BLUE
        )
        t0_label = Tex("C").next_to(t0, RIGHT, 0.2)
        self.play(ShowCreation(a0), ShowCreation(t0), ShowCreation(t0_label))

        # Make the first iteration
        t1 = VGroup()
        a1 = VGroup()
        for d in range(4):
            triangle_vxs = make_polygon_vertices(d)
            a1.add(VGroup(*self.make_arrows(triangle_vxs, thickness=1.2)))
            t1.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE, stroke_width=4.0
                )
            )

        # Transform
        # TODO A better option is to draw this triangle to the right
        self.play(FadeOut(t0), FadeOut(a0), FadeIn(t1), FadeIn(a1))

        # Label the one with the largest integral
        t1_label = Tex("T^{(1)}").move_to(t1[d_seq[0]].get_center())

        # Make only one of the triangles solid-filled
        self.play(
            *[
                t1[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[0]
            ],
            FadeIn(t1_label),
        )
        self.play(*[FadeOut(a1[d]) for d in range(4) if d != d_seq[0]])

        # Make the second iteration
        t2 = VGroup()
        a2 = VGroup()
        for d1 in range(4):
            triangle_vxs = make_polygon_vertices(d_seq[0], d1)
            a2.add(VGroup(*self.make_arrows(triangle_vxs, thickness=0.5)))
            t2.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE, stroke_width=1.5
                )
            )

        # Transform
        self.play(FadeOut(t1_label))
        self.play(FadeOut(t1[d_seq[0]]), FadeOut(a1[d_seq[0]]), FadeIn(t2), FadeIn(a2))

        # Label the one with the largest integral
        t2_label = Tex("T^{(2)}", font_size=18).move_to(t2[d_seq[1]].get_center())

        # Make only one of the triangles solid-filled
        self.play(
            *[
                t2[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[1]
            ],
            FadeIn(t2_label),
        )
        self.play(*[FadeOut(a2[d]) for d in range(4) if d != d_seq[1]])

        # On and on for 3, ...
        self.play(FadeOut(t2_label))
        t3 = VGroup()
        a3 = VGroup()
        for d2 in range(4):
            triangle_vxs = make_polygon_vertices(d_seq[0], d_seq[1], d2)
            a3.add(VGroup(*self.make_arrows(triangle_vxs, thickness=0.5)))
            t3.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE, stroke_width=1.5
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
            triangle_vxs = make_polygon_vertices(d_seq[0], d_seq[1], d_seq[2], d3)
            a4.add(VGroup(*self.make_arrows(triangle_vxs, thickness=0.5)))
            t4.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE, stroke_width=1.5
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

        # ... 5, ...
        t5 = VGroup()
        a5 = VGroup()
        for d4 in range(4):
            triangle_vxs = make_polygon_vertices(
                d_seq[0], d_seq[1], d_seq[2], d_seq[3], d4
            )
            a5.add(VGroup(*self.make_arrows(triangle_vxs, thickness=0.5)))
            t5.add(
                Polygon(*triangle_vxs).set_style(
                    fill_opacity=INT_FILL_OPACITY, fill_color=BLUE, stroke_width=1.5
                )
            )
        self.play(
            FadeOut(t4[d_seq[3]]),
            FadeOut(a4[d_seq[3]]),
            FadeIn(t5),
            FadeIn(a5),
            run_time=0.5,
        )
        self.play(
            *[
                t5[d].animate.set_style(fill_opacity=0.0)
                for d in range(4)
                if d != d_seq[4]
            ]
        )
        self.play(*[FadeOut(a5[d]) for d in range(4) if d != d_seq[4]], run_time=0.5)

        # Then show formulas. Key point:
        # - The function breaks into a constant part + linear part + remainder
        # - Integral of the constant part vanishes
        # - Integral of the linear part vanishes
        # - The remainder has the form (psi(z) * (z-z0)) where psi(z) -> 0 as z -> z0.
        # Length of contour halves at each point
        # Distance from centroid z halves at each point
        # Therefore, the nontrivial portion of the function int (psi(z) * (z-z0)) scales as 1/4 times
        # the supremum of psi(z) around the triangle, which itself goes to 0.

        # Mark the point of convergence
        z = sum(make_polygon_vertices(*d_seq)) / n_vertices
        z_pt = Dot(z, radius=0.05).set_color(RED)
        z_label = Tex("z_0", font_size=30).next_to(z_pt, DOWN, 0.2).set_color(RED)
        self.play(FadeIn(z_pt), FadeIn(z_label))

        ## Go through all of the triangles in order again, and morph the integral value
        integral_value = (
            Tex("\\abs{\\int_{C} f(z)dz}").move_to((0, -2, 0)).fix_in_frame()
        )
        bound = Tex("\\le 4\\abs{\\int_{T^{(1)}} f(z)dz}").next_to(
            integral_value, RIGHT, 0.15
        )
        bound_2 = Tex("\\le 4^2\\abs{\\int_{T^{(2)}} f(z)dz}").next_to(
            integral_value, RIGHT, 0.15
        )
        bound_3 = Tex("\\le 4^3\\abs{\\int_{T^{(3)}} f(z)dz}").next_to(
            integral_value, RIGHT, 0.15
        )
        bound_n = Tex("\\le 4^n\\abs{\\int_{T^{(n)}} f(z)dz}").next_to(
            integral_value, RIGHT, 0.15
        )

        # Write down the integral
        self.play(ShowCreation(integral_value))

        # Unwind all the way back up
        self.play(FadeOut(t5), FadeOut(a5), FadeIn(t4[d_seq[3]]), run_time=0.3)
        self.play(FadeOut(t4), FadeIn(t3[d_seq[2]]), run_time=0.3)
        self.play(FadeOut(t3), FadeIn(t2[d_seq[1]]), run_time=0.3)
        self.play(FadeOut(t2), FadeIn(t1[d_seq[0]]), run_time=0.3)
        self.play(FadeOut(t1), FadeIn(t0), run_time=0.3)
        self.embed()

        # Write down the bounds, and restrict to the triangles in order

        self.play(ShowCreation(bound), FadeOut(t0), FadeIn(t1), FadeIn(t1_label))

        self.play(
            Transform(bound, bound_2),
            FadeOut(t1[d_seq[0]]),
            FadeIn(t2),
            FadeIn(t2_label),
            FadeOut(t1_label),
        )

        self.play(
            Transform(bound, bound_3),
            FadeOut(t2[d_seq[1]]),
            FadeIn(t3),
            FadeOut(t2_label),
        )

        self.play(Transform(bound, bound_n), FadeOut(t3[d_seq[2]]), FadeIn(t4))
        self.play(FadeOut(t4[d_seq[3]]), FadeIn(t5))

        #

        self.embed()


## CH 3 portions
# - Integral of 1/z
# - Convergence of power series in a disk.
# - Cauchy residue formula.
# - Analytic continuation


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


## CH 4
# - Types of singularities, and rates of growth, with examples
# - Key example of 1/z, and residue calculus
# - The Riemann sphere
# - Branch cuts, inverse trig, logarithms, ...


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


## CH 5
# - The solution to the Basel problem which involves an infinite factorization of sin(z)
# - Show successive polynomial approximations of sin(z) with more and more roots
# - Do the same for the Gamma function (Introduce this function via Fourier analysis?)

## CH 6
# -


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
