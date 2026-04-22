# Dumping ground for new ideas: particularly the most challenging animations which will make their
# way into other files.

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


class ControlledNumber(m.DecimalNumber):
    """A decimal number controlled by a value tracker"""

    tracker: m.ValueTracker

    def __init__(self, tracker: m.ValueTracker, **kwargs):
        self.tracker = tracker
        super().__init__(**kwargs)


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

    def increment_value(self, i: int, j: int, d_value: float | complex) -> None:
        self.set_value(i, j, self.get_value(i, j) + d_value)
        self._recompute_eig_local()
        return self

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
        t_axes = m.ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        three_d_vis.add(t_axes)

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

        three_d_vis.rotate(-20 * m.DEGREES, (1, 0, 0))
        three_d_vis.set_width(8.0)
        three_d_vis.move_to((4, 0, 0))

        self.add(mat_mobj)
        self.add(mat)
        self.add(three_d_vis)

        self.embed()

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
