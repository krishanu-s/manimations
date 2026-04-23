# Dumping ground for new ideas: particularly the most challenging animations which will make their
# way into other files.

import math
from typing import Callable

import manimlib as m
import numpy as np

from .linear_algebra.utils import (
    MatrixTracker,
    RadialFn,
    SymmetricMatrixTracker,
    VectorTracker,
    make_random_symmetric_matrix,
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

    pass
