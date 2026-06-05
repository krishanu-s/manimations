"""Animations associated to the Riemann mapping theorem, which says that any simply-connected,
open domain Ω in the complex plane is conformally equivalent to the Poincare unit disk D.

Suppose that you have two different surfaces. Is there a continuous mapping between points
on the first surface and points on the second surface, such that local angles between
lines are preserved?

A simple case of this is to just imagine the two surfaces are domains in the plane, each
bounded by some continuous curve. The curve could have kinks and angles (which would create
a polygon), and the region it defines could be unbounded. In this case, it turns out that
it's always possible as long as (1) neither region is the entire plane, and (2) neither
region encloses a hole. This is called the {\\bf Riemann Mapping Theorem.}

A mapping which locally preserves angles between lines is called "conformal", because in
some sense it implies that the shape of one surface conforms to the other. The term originates
with mapmakers in the 1500s which were looking for flat map projections of the Earth which
preserve angles between directions of travel (but not distances).

Conformal maps are the geometry behind the field of complex analysis. In fact, the definition
of what it means for a function on complex numbers to have a "complex derivative" is \\emph{equivalent}
to being conformal as a mapping between regions of the complex plane.

To me, the Riemann is an unintuitive result about geometry. Conformal equivalence means that
there is a continuous map which preserves angles between curves. I thought this means
you can't turn a corner into a straight line without breaking conformality. And yet you can!
The reason is that in order to make a map conformal, you only have to convert corners into
straight lines at the *boundary* of the domain, and it's possible to make the derivative
go to zero (resp. infinity) for mapping an angle < PI (resp. > PI) to a straight line).

It also provides a method to transform images on one surface into images on another. An extension of this
theorem says (...)

*** Implications for elliptic PDEs, i.e. the heat equation ***

It also has implications for the solution to the Dirichlet problem on Ω, namely finding
the steady-state of the heat equation (i.e. the long-term heat distribution in the domain)
associated to initial boundary conditions. You just map to the unit disk and solve the
problem there, where it's explicitly solvable (because the eigenbasis is known, it is
a product of Bessel functions in the radius with Fourier functions i.e. sin/cos in the angle).

A little bit harder is solving the time-dependent heat equation -- i.e. not just
finding the long-term heat distribution, but calculating how it evolves through time.
This is more difficult because the mapping Ω -> D is not *isometric*, i.e. it doesn't
preserve distances. The effect is that we have a varying diffusion coefficient on D,
namely the heat diffuses at a different rate at each point of D. Solving *this* equation
can only be done approximately. But thankfully, it can be done, and here's the reasoning.

    - Suppose that the heat diffusion rate at each point x in D is given by p(x). Explicitly,
    p(x) = |f'(x)|^2, where f is the Riemann mapping.
    - In the ordinary heat equation, p(x) = 1 everywhere, and we have the Bessel-Fourier eigenbasis.
    This eigenbasis is quantized (because the disk is bounded), i.e. it's enumerated by integer
    pairs (m, k).
    - When p(x) is nonconstant, time-evolution mixes the Bessel-Fourier eigenbasis, according to
    an infinite-dimensional matrix M with rows/columns parametrized by (m, k).
    - However, since p(x) is *smooth*, its own Bessel-Fourier decomposition is concentrated
    in low modes. The implication of this is that the entry M_{m, k, m', k'} is exponentially
    small in |m-m'| + |k-k'|.
    - Thus, we can proceed with two approximations:
        - Only consider Bessel-Fourier modes with small m and k, effectively making M finite-dimensional.
        - Ignore entries of M for which |m-m'| + |k-k'| is large, effectively making M low-rank.
        - Now, evolving the system by time t consists of multiplying by exp(Mt), which is doable.
    - The number of modes we have to consider will depend logarithmically (I think?) on the length
    of simulation time. It also depends on how smooth p(x) is, which in turn depends on how nice
    the map f: Ω -> D is. The upshot is that this approximation method behaves poorly at corners, and
    is suited for domains with a smooth boundary.

This time-dependent case can also be used for parabolic PDEs (e.g. the wave equation), where again we
focus on a finite number of modes.

*** Proof of the theorem itself, by construction ***

The theorem itself proceeds by starting from one such injective mapping, and then iteratively
"growing" it to cover all the points in the unit disk. Some analytic properties of the
function space imply that this process converges to an actual function, and rigidity of
maps from the disk to itself shows the uniqueness of the limit.

The starting point is known as Schwarz's lemma. Explicitly, it says that any function f: D -> D
such that f(0) = 0, has
- f'(0) <= 1, and |f(z)| <= |z| for all z in the disk, and
- equality holds iff f is a rotation
The second part is the rigidity result. It says that despite the fact that there's an
infinite-dimensional space of functions from D to D, the functions which maximize the
derivative at 0 is one-dimensional.

The animation depicts how the theorem construction goes, by "expanding" a starting map.
"""

from __future__ import annotations

import cmath
import math
from typing import Annotated, Dict, Iterable, Literal, Tuple, Union

import numpy as np
from manimlib import *
from scipy.special import gamma, jn_zeros, jv

FloatArray = np.ndarray[int, np.dtype[np.float64]]
Vect2 = Annotated[FloatArray, Literal[2]]
Vect3 = Annotated[FloatArray, Literal[3]]
Vect4 = Annotated[FloatArray, Literal[4]]

# Utilities


def get_angle(v: FloatArray) -> float:
    return (math.atan2(v[1], v[0]) - PI / 2) % TAU


def polar_vec(theta: float) -> Vect3:
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def rot90(v: Vect3) -> Vect3:
    """Rotates the vector counterclockwise by 90 degrees"""
    return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ v


def rot(v: Vect3, theta: float) -> Vect3:
    """Rotates the vector counterclockwise by theta"""
    return (
        np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        @ v
    )


def normalize_vect3(v: FloatArray) -> FloatArray:
    """Normalizes a vector"""
    return v / np.linalg.norm(v)


def clamp(t: float, tmin: float, tmax: float) -> float:
    return max(min(t, tmax), tmin)


def vect3_to_cx(v: Vect3):
    """Converts a vector to a complex number."""
    return complex(v[0], v[1])


def cx_to_vect3(z: complex):
    """Converts a complex number to a vector."""
    return np.array([z.real, z.imag, 0.0])


# Hyperbolic geometry
def pdisk_aut(a: complex, z: complex):
    """Consider the automorphism of the Poincare disk interchanging 0 and a.
    Applies this function to z."""
    return (a - z) / (1 - z * a.conjugate())


class Disk(Surface):
    def __init__(
        self,
        u_range: Tuple[float, float] = (0.0, 1.0),  # Radius
        v_range: Tuple[float, float] = (0.0, TAU),  # Theta
        resolution: Tuple[int, int] = (51, 101),
        radius: float = 1.0,
        true_normals: bool = True,
        clockwise=False,
        **kwargs,
    ):
        self.radius = radius
        self.clockwise = clockwise
        super().__init__(
            u_range=u_range, v_range=v_range, resolution=resolution, **kwargs
        )
        # Add bespoke normal specification to avoid issue at poles
        if true_normals:
            self.data["d_normal_point"] = self.data["point"] * (
                (radius + self.normal_nudge) / radius
            )

    def uv_func(self, u: float, v: float) -> np.ndarray:
        sign = -1 if self.clockwise else +1
        return self.radius * np.array(
            [
                u * math.cos(sign * v),
                u * math.sin(sign * v),
                0.0,
            ]
        )


class DiskHeatMap(HeatMapMixin, Disk):
    pass


# Scene showing the evolution of a Fourier-Bessel function on the
class Foo(Scene):
    def construct(self):
        frame = self.frame
        frame.reorient(120, 55, 0)

        # Get zeros of the Bessel functions for the purpose of defining Fourier-Bessel harmonics
        max_m = 10
        max_k = 10
        bessel_zeros = np.stack([jn_zeros(m, max_k) for m in range(max_m)], axis=0)

        # Definition of Fourier-Bessel harmonics which are eigenfunctions of the Laplace operator on the disk
        def phi(m, k):
            j_mk = bessel_zeros[np.abs(m), k]  # Bessel zero
            # First coordinate is r, second coordinate is theta
            return lambda polar_coords: jv(np.abs(m), j_mk * polar_coords[:, 0]) * np.exp(
                complex(0, 1) * m * polar_coords[:, 1]
            )

        # Define a linear combination of said harmonics.
        def fn(
            polar_coords: FloatArray, # Array of [r, theta] pairs
            coeffs: FloatArray, # Fourier-Bessel coefficients
            t: float # Time
        ):
            result = np.zeros(polar_coords.shape[:-1], dtype=np.complex128)
            for m in range(-max_m + 1, max_m):
                for k in range(max_k):
                    c = coeffs[m + max_m - 1, k]
                    if c != 0:
                        j_mk = bessel_zeros[np.abs(m), k]  # Bessel zero
                        l_mk = -j_mk ** 2 # Eigenvalue
                        result += np.exp(l_mk * t) * jv(np.abs(m), j_mk * polar_coords[:, 0]) * np.exp(
                            complex(0, 1) * m * polar_coords[:, 1]
                        )
            return result


        coeffs = np.zeros((2 * max_m - 1, max_k))

        # Given a function v(theta) on the boundary of the disk (i.e. the circle), uses the Poisson kernel to produce
        # a steady-state solution v(r, theta) which is harmonic
        def solve_dirichlet(v0: Callable[[float], float], fourier_deg: int = 10):
            # Decompose v(-) into a Fourier basis
            num_pts = 2 * fourier_deg + 1
            t_values = np.linspace(0, TAU * (1 - 1/num_pts), num_pts)
            v_array = np.array([v0(t) for t in t_values])
            fourier_coeffs = []
            for m in range(-fourier_deg, fourier_deg + 1):
                fourier_coeffs.append(v_array.dot(np.array([np.exp(complex(0, m*t)) for t in t_values])) / num_pts)

            # Rescale the m-th Fourier by r^|m|, and reconstitute
            return lambda polar_coords: sum([
                fourier_coeffs[m + fourier_deg] * np.exp(complex(0, m) * polar_coords[:, 1]) * np.pow(polar_coords[:, 0], abs(m))
                for m in range(-fourier_deg, fourier_deg + 1)
            ])

        # TODO Inner product of two functions of (r, theta).

        # Decomposes a function into the Fourier-Bessel basis, outputting coefficients.
        def decompose_into_fourier_bessel(v, max_m: int = 10, max_k: int = 10, num_r: int = 20, num_t: int = 20):
            # Inner product of phi_{m, k} with itself is J_{m+1}(j_mk)^2 / 2.
            # So we take the dot product with all of the Fourier-Bessel functions, times 1/TAU * 2/J_{m+1}(j_mk)^2
            r_values = np.linspace(0, 1, num_r)
            t_values = np.linspace(0, TAU, num_t)
            polar_coords = np.array([[r, t] for r in r_values for t in t_values])
            v_array = v(polar_coords)
            coeffs = np.zeros((2 * max_m - 1, max_k))
            for m in range(-max_m + 1, max_m):
                for k in range(max_k):
                    fb_array = phi(m, k)(polar_coords)

                    # Include a factor of r in integration
                    for ir, r in enumerate(r_values):
                        fb_array[num_t * ir:num_t * (ir+1)] *= r

                    # Calculate integral, divided by TAU
                    coeff = v_array.dot(fb_array) /  (num_t  * num_r)

                    # Scale factor
                    j_mk = bessel_zeros[np.abs(m), k]
                    coeff *= 2 / math.pow(jv(abs(m + 1), j_mk), 2)

                    # Add to coefficients matrix
                    coeffs[m + max_m - 1, k] = coeff

            return coeffs



        # Plot a heatmap of the function onto the disk
        disk = DiskHeatMap()
        disk.init_heatmap(HeatMapType.REAL)
        self.add(disk)
        t = ValueTracker()

        # Given an initial condition, find the steady state
        v0 = lambda t: np.cos(2*t)
        ev = solve_dirichlet(v0)

        # Find the decomposition of ev into the Fourier-Bessel basis.
        ev_coeffs = decompose_into_fourier_bessel(ev)

        # Set an updater for the heatmap which evolves through time

        # Do time evolution

        # Plot a 3D evolving version
        # disk = Disk()
        # self.add(disk)

        self.embed()


#
class Curve(VMobject):
    """A curve composed of a sequence of bezier curves."""

    # TODO Move this into a utility file.

    num_segments: int

    @classmethod
    def make_from_anchors(cls, *anchors, **style_kwargs) -> Curve:
        """Makes a curve passing through the given anchors."""
        n = len(anchors) - 1
        curve = Curve()
        curve.set_style(**style_kwargs)
        curve.num_segments = n

        h1, h2 = get_smooth_cubic_bezier_handle_points(anchors)
        for i in range(n):
            curve.add_cubic_bezier_curve(anchors[i], h1[i], h2[i], anchors[i + 1])
        return curve

    @property
    def anchors(self) -> list[Vect3]:
        """Retrieves the anchors."""
        return list(self.data["point"][::6]) + [self.data["point"][-1]]

    def transform(self, fn: Callable[[Vect3], Vect3], **style_kwargs) -> Curve:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve."""
        return Curve.make_from_anchors(*map(fn, self.anchors), **style_kwargs)


# Generic grid of curves (which are meant to be orthogonal at the intersections)
class CurveGridGeneric(VGroup):
    """A grid of curves tiling the interior of a planar domain"""

    # Three subfamilies:
    # - Boundary curves
    # - X-grid curves (vertical)
    # - Y-grid curves (horizontal)
    anchors_x: list[FloatArray]  # Array of shape (m, N, 3).
    anchors_y: list[FloatArray]  # Array of shape (n, M, 3).
    num_curves_x: int  # m
    num_curves_y: int  # n

    @classmethod
    def make_from_boundary(
        cls,
        bdy_curves: list[Curve],
        num_curves_x,
        num_curves_y,
        num_anchors_x,
        num_anchors_y,
    ):
        """
        Used at the outset to initialize a grid for a generic convex region.
        Assumes the sequence of boundary curves goes in order and is oriented,
        i.e. each one ends where the previous one starts.
        """
        grid = CurveGridGeneric()
        grid.num_curves_x = num_curves_x
        grid.num_curves_y = num_curves_y

        # (1) Find xmin, xmax, ymin, ymax
        xmin = min([a[0] for curve in bdy_curves for a in curve.anchors])
        xmax = max([a[0] for curve in bdy_curves for a in curve.anchors])
        ymin = min([a[1] for curve in bdy_curves for a in curve.anchors])
        ymax = max([a[1] for curve in bdy_curves for a in curve.anchors])

        # (2) Set x-values of vertical lines, and y-values of horizontal lines
        x_values = np.linspace(xmin, xmax, num_curves_x + 2)[1:-1]
        y_values = np.linspace(ymin, ymax, num_curves_y + 2)[1:-1]

        # (3) For each x-value, find the min/max y-values along the boundary curve. Same for the y-values
        y_min_values = []
        y_max_values = []
        x_min_values = []
        x_max_values = []

        anchors = []
        for curve in bdy_curves:
            anchors.extend(curve.anchors[:-1])

        # First, cyclically permute the anchors until the first one in the list has minimal x-coordinate
        min_index = min(range(len(anchors)), key=lambda i: anchors[i][0])
        anchors = anchors[min_index:] + anchors[:min_index]

        # Proceed through, marking the y-coordinates every time we cross a grid line, in order.
        # Grid lines are numbered 1 through num_curves_x.
        x_crossings = [[] for _ in range(len(x_values))]
        x_delta = (xmax - xmin) / (num_curves_x + 1)

        for i in range(len(anchors)):
            a, b = anchors[i], anchors[(i + 1) % len(anchors)]
            if a[0] > b[0]:
                a, b = b, a
            a_x = ((a[0] - xmin) / x_delta) - 1
            b_x = ((b[0] - xmin) / x_delta) - 1
            for j in range(
                max(math.ceil(a_x), 0), min(math.floor(b_x) + 1, num_curves_x)
            ):
                x_crossings[j].append(
                    a[1] + (x_values[j] - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
                )

        # Use the result to construct the min and max values
        grid.anchors_x = []
        for i, li in enumerate(x_crossings):
            while len(li) > 0:
                grid.anchors_x.append(
                    np.stack(
                        [
                            np.array([x_values[i], y, 0])
                            for y in np.linspace(li[0], li[-1], num_anchors_x)
                        ],
                        axis=0,
                    )
                )
                li = li[1:-1]

        # Next, cyclically permute the anchors until the first one in the list has minimal y-coordinate
        min_index = min(range(len(anchors)), key=lambda i: anchors[i][1])
        anchors = anchors[min_index:] + anchors[:min_index]

        # Proceed through, marking the y-coordinates every time we cross a grid line, in order.
        # Grid lines are numbered 1 through num_curves_x.
        y_crossings = [[] for _ in range(len(y_values))]
        y_delta = (ymax - ymin) / (num_curves_y + 1)

        for i in range(len(anchors)):
            a, b = anchors[i], anchors[(i + 1) % len(anchors)]
            if a[1] > b[1]:
                a, b = b, a
            a_y = ((a[1] - ymin) / y_delta) - 1
            b_y = ((b[1] - ymin) / y_delta) - 1
            for j in range(
                max(math.ceil(a_y), 0), min(math.floor(b_y) + 1, num_curves_y)
            ):
                y_crossings[j].append(
                    a[0] + (y_values[j] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                )

        # Use the result to construct the min and max values
        grid.anchors_y = []
        for i, li in enumerate(y_crossings):
            while len(li) > 0:
                grid.anchors_y.append(
                    np.stack(
                        [
                            np.array([x, y_values[i], 0])
                            for x in np.linspace(li[0], li[-1], num_anchors_y)
                        ],
                        axis=0,
                    )
                )
                li = li[1:-1]

        # (4) Make the grid lines.
        x_curves = [
            Curve.make_from_anchors(
                *a,
                stroke_width=1.0,
                stroke_opacity=0.5,
                stroke_color=BLUE,
            )
            for a in grid.anchors_x
        ]
        y_curves = [
            Curve.make_from_anchors(
                *a,
                stroke_width=1.0,
                stroke_opacity=0.5,
                stroke_color=BLUE,
            )
            for a in grid.anchors_y
        ]

        grid.add(VGroup(*bdy_curves))
        grid.add(VGroup(*x_curves))
        grid.add(VGroup(*y_curves))
        return grid

    @classmethod
    def make_from_boundary_and_anchors(
        cls,
        bdy_curves: list[Curve],
        anchors_x: list[FloatArray],
        anchors_y: list[FloatArray],
    ):
        """Here, we are given the anchors defining each grid curve explicitly.
        This is used when transforming another such grid."""
        grid = cls()
        grid.add(VGroup(*bdy_curves))
        grid.make_grid_curves(anchors_x, anchors_y)
        return grid

    def make_grid_curves(
        self,
        anchors_x: FloatArray | list[FloatArray],
        anchors_y: FloatArray | list[FloatArray],
        grid_stroke_opacity=0.5,
        grid_stroke_width=1.0,
    ):
        self.anchors_x = anchors_x
        self.anchors_y = anchors_y
        self.num_curves_x = len(anchors_x)
        self.num_curves_y = len(anchors_y)

        # X-curves
        self.add(
            VGroup(
                *[
                    Curve.make_from_anchors(
                        *a,
                        stroke_width=grid_stroke_width,
                        stroke_opacity=grid_stroke_opacity,
                        stroke_color=BLUE,
                    )
                    for a in anchors_x
                ]
            )
        )

        # Y-curves
        self.add(
            VGroup(
                *[
                    Curve.make_from_anchors(
                        *a,
                        stroke_width=grid_stroke_width,
                        stroke_opacity=grid_stroke_opacity,
                        stroke_color=BLUE,
                    )
                    for a in anchors_y
                ]
            )
        )

        return self

    def become(self, grid: CurveGridGeneric, match_updaters=False):
        """Becomes another grid, including updating all anchors"""
        super().become(grid, match_updaters=match_updaters)
        self.anchors_x = grid.anchors_x
        self.anchors_y = grid.anchors_y

    def get_curve_x(self, i: int):
        return self[1][i]

    def get_curve_y(self, i: int):
        return self[2][i]

    def get_anchors_x(self):
        return self.anchors_x

    def get_anchors_y(self):
        return self.anchors_y

    def get_boundary_points(self) -> list[Vect3]:
        """Gets anchors along the bounding curves. Assumes the region is non-self-intersecting"""
        result = []
        for curve in self[0]:
            result.extend(curve.anchors)
        return result

    def get_bbox(self) -> tuple[float, float, float, float]:
        """Returns (xmin, xmax, ymin, ymax)"""
        bdy_pts = self.get_boundary_points()
        return (
            min(v[0] for v in bdy_pts),
            max(v[0] for v in bdy_pts),
            min(v[1] for v in bdy_pts),
            max(v[1] for v in bdy_pts),
        )

    def get_phase_bounds(self) -> tuple[float, float]:
        """Gets theta_min < theta_max so that all points in the grid have phase lying within
        these two angles. Assumes the grid is simply-connected and doesn't encircle 0."""
        bdy_pts = self.get_boundary_points()
        num_pts = len(bdy_pts)

        # Get the phases, sorted
        as_complex = [vect3_to_cx(p) for p in bdy_pts]
        phases = list(map(cmath.phase, [vect3_to_cx(p) for p in bdy_pts]))
        phases = sorted(phases)

        # Find the largest difference -- this is the jump from theta_max to theta_min
        diffs = [(phases[(i + 1) % num_pts] - phases[i]) % TAU for i in range(num_pts)]
        ind = np.argmax(diffs)
        theta_max = phases[ind]
        theta_min = phases[(ind + 1) % num_pts]

        # Shift so that -pi < theta_min < pi and theta_min < theta_max
        if theta_max - theta_min < 0:
            theta_max += TAU
        if theta_min >= PI:
            theta_min -= TAU
            theta_max -= TAU

        return theta_min, theta_max

    def get_sqrt_fn(self) -> Callable[[complex], complex]:
        theta_min, theta_max = self.get_phase_bounds()
        phi = (theta_min + theta_max) / 2
        return lambda z: cmath.sqrt(z * cmath.exp(complex(0, -phi))) * cmath.exp(
            complex(0, phi / 2)
        )

    def transform(self, fn: Callable[[Vect3], Vect3]) -> CurveGridGeneric:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve.
        Assumes the function isn't vectorized."""
        anchors_x = self.get_anchors_x()
        anchors_y = self.get_anchors_y()
        mapped_bdy_curves = [c.transform(fn, stroke_color=BLUE) for c in self[0]]
        mapped_anchors_x = [np.stack([fn(a) for a in anc], axis=0) for anc in anchors_x]

        mapped_anchors_y = [np.stack([fn(a) for a in anc], axis=0) for anc in anchors_y]
        return CurveGridGeneric.make_from_boundary_and_anchors(
            mapped_bdy_curves, mapped_anchors_x, mapped_anchors_y
        )


# Grid of curves obtained by transforming a rectilinear one
class CurveGrid(CurveGridGeneric):
    """Special case of CurveGridGeneric, where the array of intersection points has shape
    m-by-n, i.e. the grid is a transformation of a rectilinear one.
    The m vertical (x) curves each are defined by N anchors.
    The n horizontal (y)  curves are each defined by M anchors."""

    @classmethod
    def make_from_anchors(cls, anchors_x: FloatArray, anchors_y: FloatArray):
        """Initializes the grid from two arrays of anchors -- one defining the vertical lines, and one
        defining the horizontal lines. Is the same as the image of a rectilinear grid under
        a conformal map."""
        grid = cls()

        # Boundary curves
        boundary_stroke_opacity = 1.0
        boundary_stroke_width = 3.0
        grid.add(
            VGroup(
                *[
                    Curve.make_from_anchors(
                        *list(a[i, :, :]),
                        stroke_opacity=boundary_stroke_opacity,
                        stroke_width=boundary_stroke_width,
                        stroke_color=BLUE,
                    )
                    for a in (anchors_x, anchors_y)
                    for i in (0, -1)
                ]
            )
        )

        grid.make_grid_curves(anchors_x, anchors_y)

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
        """Makes a rectilinear grid bounded by a rectangular region. The most specialized type of grid."""
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

    def transform(self, fn: Callable[[Vect3], Vect3]) -> CurveGrid:
        """Transforms the anchors according to the given function, and uses the images to draw a new curve.
        Assumes the function isn't vectorized."""
        anchors_x = self.get_anchors_x()
        anchors_y = self.get_anchors_y()
        mapped_anchors_x = np.stack(
            [
                np.stack([fn(a) for a in anchors_x[i]], axis=0)
                for i in range(self.num_curves_x)
            ],
            axis=0,
        )

        mapped_anchors_y = np.stack(
            [
                np.stack([fn(a) for a in anchors_y[i]], axis=0)
                for i in range(self.num_curves_y)
            ],
            axis=0,
        )
        return CurveGrid.make_from_anchors(mapped_anchors_x, mapped_anchors_y)


class RiemannMapping(Scene):
    """Some animations related to the Riemann mapping theorem."""

    # Helper functions
    def map_domain_to_pdisk(self, bdy_curves: list[Curve], num_iterations: int = 0):
        """
        Given a curve or collection of curves bounding a region, does the following:
            (1) Computes a grid for the region
            (2) Maps conformally to the disk -- i.e. computes the image
            (3) Runs the algorithm in Stein-Shakarchi
        """
        domain_grid = CurveGridGeneric.make_from_boundary(bdy_curves, 15, 15, 50, 50)

        # Translates and scales so that it's within a bounding square centered at 0 with sidelengths sqrt(2)
        xmin, xmax, ymin, ymax = domain_grid.get_bbox()
        xavg = (xmin + xmax) / 2
        yavg = (ymin + ymax) / 2
        marked_point = np.array([xavg, yavg, 0.0])

        scale_x = math.sqrt(2) / (xmax - xmin)
        scale_y = math.sqrt(2) / (ymax - ymin)
        tgrid = domain_grid.transform(
            lambda v: min(scale_x, scale_y) * np.array([v[0] - xavg, v[1] - yavg, 0.0])
        )

        return domain_grid, tgrid

    def compute_riemann_expansion(self, grid: CurveGridGeneric) -> CurveGridGeneric:
        """Does the raw computation of the Riemann expansion"""
        # Get the boundary point which is furthest from the origin
        best_p = min(
            [p for p in grid.get_boundary_points() if np.linalg.norm(p) < 1.0],
            key=np.linalg.norm,
            default=None,
        )

        # Scale it slightly out towards norm 1 partway
        a = vect3_to_cx(best_p) / math.pow(np.linalg.norm(best_p), 0.2)

        # Do mapping:
        # - First calculate alpha value
        # - Then compute required correctional rotations
        # - Then compute the composite effect of the mappings
        tgrid = grid.transform(lambda v: cx_to_vect3(pdisk_aut(a, vect3_to_cx(v))))
        sqrt_fn = tgrid.get_sqrt_fn()
        pa = cmath.phase(a)
        sa = sqrt_fn(a)

        # If a has negative phase, but the middle of the phase bounds is positive
        # (and they're near each other, e.g. on opposide sides of +pi / -pi)
        # then an additional correctional rotation by pi will be needed
        cx_rot = cmath.exp(complex(0, pa) / 2)
        theta_min, theta_max = tgrid.get_phase_bounds()
        phi = (theta_min + theta_max) / 2
        if abs(phi - pa) > PI:
            cx_rot *= -1

        tgrid = grid.transform(
            lambda v: cx_to_vect3(
                pdisk_aut(sa, sqrt_fn(pdisk_aut(a, vect3_to_cx(v)))) * cx_rot
            )
        )

        return tgrid

    def anim_fn(
        self,
        grid: CurveGrid,
        fn: Callable[[complex], complex],
        other_animations: [Animation] = [],
        **anim_kwargs,
    ) -> CurveGrid:
        """Deforms the grid to its image under an arbitrary function."""
        # TODO Find some way to modify the grid in-place rather than adding/removing.
        tgrid = grid.transform(lambda v: cx_to_vect3(fn(vect3_to_cx(v))))
        self.play(grid.animate.become(tgrid), *other_animations, **anim_kwargs)
        self.remove(grid)
        grid = tgrid
        self.add(grid)
        return grid

    def anim_pdisk_aut(self, grid: CurveGrid, a: complex, **anim_kwargs) -> CurveGrid:
        """Deforms the grid to its image under the automorphism represented by the complex number a."""
        return self.anim_fn(grid, lambda z: pdisk_aut(a, z), **anim_kwargs)

    def anim_sqrt(self, grid: CurveGrid, **anim_kwargs) -> CurveGrid:
        """Deforms the grid to its image under the square root function.

        WARNING: Assumes that the grid doesn't cross the negative real line. This isn't checked."""
        return self.anim_fn(grid, cmath.sqrt, **anim_kwargs)

    def anim_riemann_mapping_expansion(
        self,
        grid: CurveGrid,
        a: complex,
        anim_steps: bool = True,
        show_dots: bool = True,
        run_time: float = 3.0,
        wait_time_between_steps=0.5,
        **anim_kwargs,
    ) -> CurveGrid:
        """
        Primary workhorse step of the algorithm.

        Given a domain in the Poincare disk which contains 0, and a point a
        which lies outside of it, performs a series of animations to deform the
        grid to be larger while still lying within the Poincare disk, as follows:
            (i) Applies the unitary transformation associated to a which interchanges
            0 and a and has (negative) real derivative at 0.
            (ii) Applies a square-root map on the image.
            (iii) Applies the unitary transformation associated to sqrt(a) which interchanges
            0 and sqrt(a) and has (negative) real derivative at 0.
            (iv) Applies a rotation by phase(a)/2 to get back to the original orientation.
        """
        # Do the initial transformation to calculate the angles for the square root
        tgrid = grid.transform(lambda v: cx_to_vect3(pdisk_aut(a, vect3_to_cx(v))))
        sqrt_fn = tgrid.get_sqrt_fn()
        pa = cmath.phase(a)
        sa = sqrt_fn(a)

        # Correctional rotation at the end
        cx_rot = cmath.exp(complex(0, pa) / 2)

        # If a has negative phase, but the middle of the phase bounds is positive
        # (and they're near each other, e.g. on opposide sides of +pi / -pi)
        # then an additional correctional rotation by pi will be needed
        theta_min, theta_max = tgrid.get_phase_bounds()
        phi = (theta_min + theta_max) / 2
        if abs(phi - pa) > PI:
            cx_rot *= -1

        # Setup if we're showing the dots
        if show_dots:
            zero_dot = Dot((0, 0, 0), radius=0.02).set_color(WHITE)
            a_dot = VGroup()
            a_dot.add(Dot((a.real, a.imag, 0), radius=0.02).set_color(RED))
            a_dot.add(
                Tex("\\alpha", font_size=12).set_color(RED).next_to(a_dot[0], UR, 0.03)
            )
            sa_dot = VGroup()
            sa_dot.add(Dot((sa.real, sa.imag, 0), radius=0.02).set_color(GREEN))
            sa_dot.add(
                Tex("\\sqrt{\\alpha}", font_size=12)
                .set_color(GREEN)
                .next_to(sa_dot[0], UR, 0.03)
            )

        # Do animation
        if show_dots & anim_steps:
            self.play(FadeIn(zero_dot), FadeIn(a_dot), run_time=run_time / 10)
            grid = self.anim_pdisk_aut(
                grid,
                a,
                run_time=run_time / 4,
                other_animations=[zero_dot.animate.move_to((a.real, a.imag, 0))],
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            self.play(FadeIn(sa_dot), run_time=run_time / 10)
            grid = self.anim_fn(
                grid,
                sqrt_fn,
                run_time=run_time / 4,
                other_animations=[zero_dot.animate.move_to((sa.real, sa.imag, 0))],
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            grid = self.anim_pdisk_aut(
                grid,
                sa,
                run_time=run_time / 4,
                other_animations=[
                    zero_dot.animate.move_to((0, 0, 0)),
                ],
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            grid = self.anim_fn(
                grid,
                lambda z: z * cx_rot,
                run_time=run_time / 4,
                other_animations=[
                    FadeOut(a_dot),
                    FadeOut(sa_dot),
                    FadeOut(zero_dot),
                ],
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
        elif (show_dots != True) & anim_steps:
            grid = self.anim_pdisk_aut(grid, a, run_time=run_time / 4, **anim_kwargs)
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            grid = self.anim_fn(grid, sqrt_fn, run_time=run_time / 4, **anim_kwargs)
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            grid = self.anim_pdisk_aut(grid, sa, run_time=run_time / 4, **anim_kwargs)
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)

            grid = self.anim_fn(
                grid, lambda z: z * cx_rot, run_time=run_time / 4, **anim_kwargs
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
        elif show_dots & (anim_steps != True):
            self.play(
                FadeIn(zero_dot), FadeIn(a_dot), FadeIn(sa_dot), run_time=run_time / 3
            )
            grid = self.anim_fn(
                grid,
                lambda z: pdisk_aut(sa, sqrt_fn(pdisk_aut(a, z))) * cx_rot,
                run_time=run_time,
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
            self.play(
                FadeIn(zero_dot), FadeOut(a_dot), FadeOut(sa_dot), run_time=run_time / 3
            )
        else:
            grid = self.anim_fn(
                grid,
                lambda z: pdisk_aut(sa, sqrt_fn(pdisk_aut(a, z))) * cx_rot,
                run_time=run_time,
                **anim_kwargs,
            )
            if wait_time_between_steps != 0:
                self.wait(duration=wait_time_between_steps)
        return grid

    def construct(self):
        """Animates the iterative algorithm given in Stein-Shakarchi 8.3.3 for
        holomorphically mapping an arbitrary simply-connected domain in C onto
        the unit disk, by maximizing the derivative of the function at the
        preimage of 0."""
        # TODO Generalize this an arbitrary region, by
        # (1) generalizing CurveGrid to include boundary curves and internal curves
        # (2) writing a method in CurveGrid which makes gridlines to an arbitrary
        # boundary curve.

        self.frame.set_height(4.0)

        other_curve = Curve.make_from_anchors(
            *[
                np.array(
                    [
                        0.5 * np.cos(t) + 0.15 * np.sin(2 * t),
                        0.5 * np.sin(t) - 0.1 * np.cos(3 * t),
                        0,
                    ]
                )
                for t in np.linspace(0, TAU, 100)
            ]
        ).set_color(BLUE)
        other_grid = CurveGridGeneric.make_from_boundary([other_curve], 15, 15, 30, 30)

        # Poincare disk
        circle = ParametricCurve(polar_vec, (0, TAU, TAU / 100))
        self.add(circle)

        self.embed()

        # Tracker for the derivative at 0
        deriv_at_zero = VGroup()
        deriv_at_zero.add(Tex("f'(0)="))
        deriv_at_zero.add(
            DecimalNumber(num_decimal_places=5).next_to(deriv_at_zero[0], RIGHT, 0.2)
        )
        deriv_at_zero.set_height(0.3).next_to(circle, DOWN, 0.5)
        self.add(deriv_at_zero)

        # Make a square grid
        grid = CurveGrid.make_linear((-0.2, 0.2), (-0.2, 0.2), 21, 101, 21, 101)
        self.add(grid)
        df_tracker = ValueTracker(0.2 * np.sqrt(2))
        deriv_at_zero[1].add_updater(
            lambda mobj: mobj.set_value(df_tracker.get_value())
        )

        # Expand it
        tgrid = grid.transform(lambda v: v * 2.5 * np.sqrt(2))
        self.play(
            grid.animate.become(tgrid),
            df_tracker.animate.set_value(1.0),
            rate_func=linear,
        )
        self.remove(grid)
        grid = tgrid
        self.add(grid)

        ## Automorphism associated to a point not inside the domain.
        a = complex(0, 0.8)
        grid = self.anim_riemann_mapping_expansion(
            grid,
            a,
            anim_steps=True,
            show_dots=True,
            run_time=3.0,
            wait_time_between_steps=1.0,
        )
        df_tracker.set_value(
            df_tracker.get_value() * (1 + abs(a)) / (2 * abs(cmath.sqrt(a)))
        )

        # Do iterative expansions to points on the boundary. First with steps shown,
        # then sped along
        for i in range(2):
            # Get the boundary point which is furthest from the origin
            best_p = min(
                [p for p in grid.get_boundary_points() if np.linalg.norm(p) < 1.0],
                key=np.linalg.norm,
                default=None,
            )

            # Scale it slightly out towards norm 1 partway
            a = vect3_to_cx(best_p) / math.pow(np.linalg.norm(best_p), 0.2)

            # Do mapping
            grid = self.anim_riemann_mapping_expansion(
                grid,
                a,
                anim_steps=True,
                show_dots=True,
                run_time=3.0,
                wait_time_between_steps=0.5,
            )

            df_tracker.set_value(
                df_tracker.get_value() * (1 + abs(a)) / (2 * abs(cmath.sqrt(a)))
            )

        for i in range(200):
            # Get the boundary point which is furthest from the origin
            best_p = min(
                [p for p in grid.get_boundary_points() if np.linalg.norm(p) < 1.0],
                key=np.linalg.norm,
                default=None,
            )

            # Scale it slightly out towards norm 1 partway
            a = vect3_to_cx(best_p) / math.pow(np.linalg.norm(best_p), 0.2)

            # Do mapping
            # TODO If a is in the lower-right
            grid = self.anim_riemann_mapping_expansion(
                grid,
                a,
                anim_steps=False,
                show_dots=(i < 20),
                run_time=0.05 * 50 / (50 + i),
                rate_func=linear,
                wait_time_between_steps=0.0,
            )

            df_tracker.set_value(
                df_tracker.get_value() * (1 + abs(a)) / (2 * abs(cmath.sqrt(a)))
            )
