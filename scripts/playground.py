# Dumping ground for new ideas: particularly the most challenging animations which will make their
# way into other files.

import math
from typing import Callable

import manimlib as m
import numpy as np


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


class VectorTracker(m.Mobject):
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
        v = m.ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_value(i, init_val + (val - init_val) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def slide_vector(self, vec: np.ndarray):
        """Continuously vary the entire vector in a linear fashion towards the given target vector."""
        self.clear_updaters()
        init_vec = self.get_vector()

        tracker = m.ValueTracker(0)
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

        v = m.ValueTracker(0)
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


class MatrixTracker(m.Mobject):
    """Stores a finite-dimensional square matrix with complex entries, where the individual entries act
    as value trackers. Contains convenience functions for computing, e.g., eigenvalues and eigenvectors,
    and recomputes these whenever entries are set."""

    value_type: type = np.complex128

    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.matrix = np.zeros((dim, dim), dtype=self.value_type)
        self.eigenvals = np.zeros((dim,), dtype=self.value_type)
        self.eigenvectors = np.eye(dim, dtype=self.value_type)
        super().__init__(**kwargs)

    def _recompute_eig(self):
        """Recomputes the eigenvalues and eigenvectors from scratch, e.g. when the matrix is changed
        in an arbitrary way."""
        eig_result = np.linalg.eig(
            self.uniforms["matrix"].reshape((self.dim, self.dim))
        )
        self.eigenvals = eig_result.eigenvalues
        self.eigenvectors = eig_result.eigenvectors
        return self

    def _recompute_eig_local(self):
        """Recomputes the eigenvalues and eigenvectors by locally perturbing the current eigenvalues and eigenvectors
        according to Newton's method, thus avoiding reproducing work"""
        # TODO Implement this.
        return self._recompute_eig()

    def init_uniforms(self) -> None:
        super().init_uniforms()
        self.uniforms["matrix"] = self.matrix.flatten()

    def get_matrix(self) -> np.ndarray:
        # return self.matrix
        return self.uniforms["matrix"].reshape((self.dim, self.dim))

    def set_matrix(self, matrix: np.ndarray):
        self.matrix = matrix
        self.uniforms["matrix"] = matrix.flatten()
        self._recompute_eig()
        return self

    def get_column(self, i: int) -> np.ndarray:
        return self.uniforms["matrix"][i :: self.dim]

    def get_all_eigenvalues(self):
        """Returns a shape (D,) array containing the eigenvalues."""
        return self.eigenvals

    def get_all_eigenvectors(self):
        """Returns a D-by-D matrix whose columns are the eigenvectors."""
        return self.eigenvectors

    def get_eigenvalue(self, i: int):
        """Returns the i-th eigenvalue"""
        return self.eigenvals[i]

    def get_eigenvector(self, i: int):
        """Returns the i-th eigenvector"""
        return self.eigenvectors[:, i]

    def get_value(self, i: int, j: int) -> np.ndarray:
        # return self.matrix[i, j]
        return self.uniforms["matrix"][i * self.dim + j]

    def set_value(self, i: int, j: int, val: float | complex):
        self.matrix[i, j] = val
        self.uniforms["matrix"][i * self.dim + j] = val
        self._recompute_eig()
        return self

    def slide_value(self, i: int, j: int, val: float | complex):
        """Continuously changes a single entry of the matrix. This is the result which *should* be produced by
        self.animate.set_value(i, j, val), but is not because that method doesn't recompute the eigenvals/vecs
        at all intermediate steps."""
        self.clear_updaters()
        init_val = self.get_value(i, j)
        v = m.ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_value(
                i, j, init_val + (val - init_val) * v.get_value()
            )
        )
        return v.animate.set_value(1.0)

    def slide_matrix(self, mat: np.ndarray):
        """Continuously vary the entire matrix in a linear fashion towards the given target matrix."""
        self.clear_updaters()
        init_mat = self.get_matrix()

        v = m.ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_matrix(init_mat + (mat - init_mat) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def increment_value(self, i: int, j: int, d_value: float | complex) -> None:
        self.set_value(i, j, self.get_value(i, j) + d_value)
        self._recompute_eig_local()
        return self

    def matmul(self, vec: np.ndarray):
        """Multiplies by an input vector v."""
        return np.matmul(self.get_matrix(), vec)

    def inner_product(self, vec: np.ndarray):
        """Uses this matrix A to compute the inner product <v, Av>."""
        return np.dot(vec, np.matmul(self.get_matrix(), vec))


class SymmetricMatrixTracker(MatrixTracker):
    """Subclass of MatrixWithEig where the matrix is constrained to have real entries and be symmetric.
    Uses more efficient eigencalculation subroutines."""

    value_type: type = np.float64

    def set_matrix(self, matrix: np.ndarray):
        if not np.array_equal(matrix, matrix.T):
            print("Cannot set a non-symmetric matrix!")
            return self
        super().set_matrix(matrix)

    def set_value(self, i: int, j: int, val: float | complex):
        self.uniforms["matrix"][i * self.dim + j] = val
        self.uniforms["matrix"][j * self.dim + i] = val
        self.matrix[i, j] = val
        self.matrix[j, i] = val
        self._recompute_eig()
        return self

    def _recompute_eig(self):
        """Recomputes the eigenvalues and eigenvectors from scratch, e.g. when the matrix is changed
        in an arbitrary way."""
        eig_result = np.linalg.eigh(
            self.uniforms["matrix"].reshape((self.dim, self.dim))
        )
        self.eigenvals = eig_result.eigenvalues
        self.eigenvectors = eig_result.eigenvectors
        return self

    def _recompute_eig_local(self):
        """Recomputes the eigenvalues and eigenvectors by locally perturbing the current eigenvalues and eigenvectors
        according to Newton's method, thus avoiding reproducing work"""
        # TODO Implement this.
        return self._recompute_eig()


def balanced_sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RadialFn(m.Sphere):
    """Depicting a real-valued function on the sphere as a surface where each point on the unit sphere is
    extended outwards (for positive values) or inwards (for negative values) using a sigmoid function"""

    def __init__(self, f: Callable[[np.ndarray], float], axes: m.ThreeDAxes, **kwargs):
        self.f = f
        self.axes = axes
        super().__init__(**kwargs)

    def set_f(self, f):
        self.f = f
        self.init_points()

    def uv_func(self, u: float, v: float) -> np.ndarray:
        sign = -1 if self.clockwise else +1
        unit_vec = np.array(
            [
                math.cos(sign * u) * math.sin(v),
                math.sin(sign * u) * math.sin(v),
                -math.cos(v),
            ]
        )
        return self.axes.c2p(*(sigmoid(self.f(unit_vec)) * self.radius * unit_vec))


def make_random_symmetric_matrix(dim: int):
    x = np.random.randn(dim, dim)
    return x + x.T


class LinAlgMatrix(m.Scene):
    """We can allow the parameters of a linear transformation to be controlled by ValueTrackers.
    What I want is an efficient way to vary e.g. the eigenvalues and eigenvectors, without having
    to recompute them from scratch every time. Probably what it will come down to is:
        - Update the matrix itself according to the new value tracker values.
        - Directly recompute the characteristic polynomial.
        - Using Newton's method to retrieve the roots, starting from the roots at the previous method.
        - Shift the vectors"""

    def construct(self):
        d = 3

        # Create the underlying matrix which varies
        mat = SymmetricMatrixTracker(d)

        # Instantiate a displayed decimal matrix object which tracks with mat
        mat_mobj = m.DecimalMatrix(mat.get_matrix()).move_to((-5, 0, 0))

        def mat_updater(mobj: m.DecimalMatrix):
            for i in range(d):
                for j in range(d):
                    mobj.mob_matrix[i][j].set_value(mat.get_value(i, j))

        mat_mobj.add_updater(mat_updater)

        ## 3D visualization
        three_d_vis = m.Group()

        # Axes on which everything will be placed
        t_axes = m.ThreeDAxes(
            (-2, 2),
            (-2, 2),
            (-2, 2),
            axis_config={"include_ticks": False, "include_tip": True},
        )
        three_d_vis.add(t_axes)

        def add_column_vectors():
            """Column vectors of the matrix, i.e. the images of the three unit basis vectors"""
            e0 = (
                m.Arrow()
                .set_color(m.BLUE)
                .add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        t_axes.c2p(*m.ORIGIN),
                        t_axes.c2p(*mat.get_column(0).real),
                    )
                )
            )

            e1 = (
                m.Arrow()
                .set_color(m.BLUE)
                .add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        t_axes.c2p(*m.ORIGIN),
                        t_axes.c2p(*mat.get_column(1).real),
                    )
                )
            )
            e2 = (
                m.Arrow()
                .set_color(m.BLUE)
                .add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        t_axes.c2p(*m.ORIGIN),
                        t_axes.c2p(*mat.get_column(2).real),
                    )
                )
            )
            column_vecs = m.VGroup(e0, e1, e2)
            three_d_vis.add(column_vecs)

        def add_heatmap_real():
            # Heatmap of function values
            heatmap = m.SphereHeatMap(radius=1.0).move_to(t_axes.c2p(0, 0, 0))
            heatmap.init_heatmap(m.HeatMapType.REAL)

            def heatmap_fn(arr: np.ndarray):
                return np.apply_along_axis(
                    lambda p: mat.inner_product(heatmap.uv_func(*p)),
                    axis=-1,
                    arr=arr,
                )

            heatmap.add_updater(lambda mobj: mobj.set_f(heatmap_fn))
            three_d_vis.add(heatmap)

        def add_heatmap_complex():
            """Constructs a spherical heatmap whose values track with the (complex)"""
            heatmap = m.SphereHeatMap(radius=1.0).move_to(t_axes.c2p(0, 0, 0))
            heatmap.init_heatmap(m.HeatMapType.COMPLEX)

            def heatmap_fn(arr: np.ndarray):
                return np.apply_along_axis(
                    lambda p: mat.inner_product(heatmap.uv_func(*p)),
                    axis=-1,
                    arr=np.stack((arr.real, arr.imag), axis=-1),
                )

            heatmap.add_updater(lambda mobj: mobj.set_f(heatmap_fn))
            three_d_vis.add(heatmap)

        def add_eigenvectors():
            # Eigenvectors
            v0 = m.Line(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(0))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(0))),
                color=m.GREEN,
            )
            v0.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(0))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(0))),
                )
            )

            v1 = m.Line(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(1))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(1))),
                color=m.GREEN,
            )
            v1.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(1))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(1))),
                )
            )

            v2 = m.Line(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(2))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(2))),
                color=m.GREEN,
            )
            v2.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(2))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(2))),
                )
            )
            three_d_vis.add(v0, v1, v2)

        def add_radial_surface():
            # Radial depicting of function values
            radial_fn = RadialFn(
                f=lambda vec: mat.inner_product(vec), axes=t_axes, radius=1.0
            ).move_to(t_axes.c2p(0, 0, 0))

            def radial_fn_updater(mobj):
                mobj.set_f(lambda vec: mat.inner_product(vec))

            radial_fn.add_updater(radial_fn_updater)
            three_d_vis.add(radial_fn)

        def add_flow_vector():
            """Add a vector v which moves around the unit sphere, along with the vector Av as well as the vector
            Av - F(v)v sitting tangent to v"""
            v = VectorTracker(d).set_vector(np.array([1.0, 0.0, 0.0]))

            def w_updater(mobj):
                v_vec = v.get_vector()
                mobj.set_vector(
                    mat.matmul(v_vec)
                    - mat.inner_product(v_vec) * v_vec / (np.linalg.norm(v_vec) ** 2)
                )

            w = VectorTracker(d).add_updater(w_updater)

            v_arrow = (
                m.Arrow()
                .set_color(m.RED)
                .add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        t_axes.c2p(*m.ORIGIN), t_axes.c2p(*v.get_vector())
                    )
                )
            )

            w_arrow = (
                m.Arrow()
                .set_color(m.YELLOW)
                .add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        t_axes.c2p(*v.get_vector()),
                        t_axes.c2p(*(v.get_vector() + w.get_vector())),
                    )
                )
            )
            three_d_vis.add(v, w, v_arrow, w_arrow)
            return v, w

        add_column_vectors()
        add_eigenvectors()
        v, w = add_flow_vector()
        # Setting orientation
        three_d_vis.rotate(-20 * m.DEGREES, (1, 0, 0))
        three_d_vis.set_width(8.0)
        three_d_vis.move_to((4, 0, 0))

        self.add(mat, mat_mobj)
        self.add(three_d_vis)

        self.embed()

        # Switch to a rnadom symmetric matrix
        self.play(mat.slide_matrix(make_random_symmetric_matrix(3)))

        # Rotate along flow direction
        eps = 0.02
        self.play(
            v.rotate_vector(v.get_vector() + eps * w.get_vector()),
            rate_func=m.linear,
            run_time=5.0 * eps,
        )

        # # Make the dictionary of value trackers which control the matrix
        # entries_trackers: dict[tuple[int, int], m.ValueTracker] = {}

        # for j in range(d):
        #     for i in range(j + 1):
        #         entries_trackers[(i, j)] = m.ValueTracker()

        # # Make the TeX representation of the matrix
        # entries_text: list[list[m.DecimalNumber]] = []
        # for i in range(d):
        #     row = []
        #     for j in range(d):
        #         tup = tuple(sorted((i, j)))
        #         num = ControlledNumber(entries_trackers[tup])
        #         num.add_updater(lambda mobj: mobj.set_value(mobj.tracker.get_value()))
        #         row.append(num)
        #     entries_text.append(row)

        # mat_text = m.Matrix(entries_text)

        # # Draw it
        # self.add(mat_text)

        # ## Calculate the eigenvalues and eigenvectors
        # def get_matrix():
        #     """Method which instantaneously retrieves the matrix."""
        #     return np.array(
        #         [
        #             [
        #                 entries_trackers[tuple(sorted((i, j)))].get_value()
        #                 for j in range(d)
        #             ]
        #             for i in range(d)
        #         ]
        #     )

        def inner_product(vec: np.ndarray, m: np.ndarray):
            """Maps an incoming vector v of shape (3,) and an inner product matrix M of shape (3, 3)
            to the dot product <v, Mv>."""
            return np.dot(vec, np.matmul(m, vec))

        def heatmap_fn(mat: np.ndarray, arr: np.ndarray):
            return np.apply_along_axis(
                lambda p: inner_product(heatmap.uv_func(p[0], p[1]), mat),
                axis=-1,
                arr=arr,
            )

        def heatmap_updater(mobj):
            mat = get_matrix()
            mobj.set_f(lambda arr: heatmap_fn(mat, arr))

        # Heatmap which shifts over time based on the matrix values
        t_axes = m.ThreeDAxes((-2, 2), (-2, 2), (-2, 2)).move_to((6, 0, 0))
        heatmap = m.SphereHeatMap(radius=1.0).move_to(t_axes.c2p(0, 0, 0))
        heatmap.init_heatmap(m.HeatMapType.REAL)

        heatmap.add_updater(heatmap_updater)

        self.add(t_axes, heatmap)

        # Roots which shift over time according to Newton's method on the characteristic polynomial
        # roots =
        # TODO Make these into ValueTracker's which themselves change according to the main value trackers.

        # Eigenvectors as a function of the matrix
        self.embed()


class FunctionSpaceOnS1(m.Scene):
    """Depict the space of functions on S^1 (or any other compact manifold with metric), which has an inner
    product given by multiplying pointwise and then integration. Under this inner product,
    - the linear map given by multiplying pointwise by a function is symmetric/hermitian;
    - the linear map given by differentiation (needs to be properly defined in higher dimensions) is symplectic."""

    pass
