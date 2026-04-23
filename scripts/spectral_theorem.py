# Visual proof of the spectral theorem

import math
from typing import Callable

import manimlib as m
import numpy as np


### For general linear algebra
def make_random_symmetric_matrix(dim: int):
    """Random symmetric matrix whose diagonal entries are chosen from N(0, 1) and off-diagonal entries from N(0, 0.5)."""
    x = np.random.randn(dim, dim)
    return 0.5 * (x + x.T)


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


### For visualization of the Spectral theorem in R^3
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


### Dumping ground
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
        mat_mobj = m.DecimalMatrix(mat.get_matrix()).move_to((-5, 2, 0))

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
        t_axes.add_axis_labels(font_size=36)
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
            return column_vecs

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
            return heatmap

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
            return heatmap

        def add_eigenvalues():
            eigenvals = VectorTracker(3).add_updater(
                lambda mobj: mobj.set_vector(mat.eigenvals)
            )
            eigenvals_mobj = m.DecimalMatrix(eigenvals.get_vector().reshape((1, d)))

            def vec_updater(mobj: m.DecimalMatrix):
                for i in range(d):
                    mobj.mob_matrix[0][i].set_value(eigenvals.get_value(i))

            eigenvals_mobj.add_updater(vec_updater)
            self.add(eigenvals, eigenvals_mobj)
            return eigenvals, eigenvals_mobj

        def add_eigenvectors():
            # Eigenvectors
            v0 = m.Line().set_color(m.GREEN).set_opacity(0.5)
            v0.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(0))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(0))),
                )
            )

            v1 = m.Line().set_color(m.GREEN).set_opacity(0.5)
            v1.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(1))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(1))),
                )
            )

            v2 = m.Line().set_color(m.GREEN).set_opacity(0.5)
            v2.add_updater(
                lambda arr: arr.put_start_and_end_on(
                    t_axes.c2p(*(-1.5 * mat.get_eigenvector(2))),
                    t_axes.c2p(*(1.5 * mat.get_eigenvector(2))),
                )
            )
            eigenvecs = m.VGroup(v0, v1, v2)
            three_d_vis.add(eigenvecs)
            return eigenvecs

        def add_radial_surface():
            # Radial depicting of function values
            radial_fn = RadialFn(
                f=lambda vec: mat.inner_product(vec), axes=t_axes, radius=1.0
            ).move_to(t_axes.c2p(0, 0, 0))

            def radial_fn_updater(mobj):
                mobj.set_f(lambda vec: mat.inner_product(vec))

            radial_fn.add_updater(radial_fn_updater)
            three_d_vis.add(radial_fn)
            return radial_fn

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

        column_vecs = add_column_vectors()
        eigenvecs = add_eigenvectors()
        eigenvals, eigenvals_mobj = add_eigenvalues()
        eigenvals_mobj.move_to((-5, -2, 0))

        v, w = add_flow_vector()
        # Setting orientation
        three_d_vis.rotate(20 * m.DEGREES, (1, 0, 0))
        three_d_vis.set_width(8.0)
        three_d_vis.move_to((4, 0, 0))

        self.add(mat, mat_mobj)
        self.add(three_d_vis)

        self.embed()

        # Switch to a random symmetric matrix
        self.play(mat.slide_matrix(make_random_symmetric_matrix(3)))

        # Rotate v along flow direction
        eps = 0.005
        for _ in range(200):
            self.play(
                v.rotate_vector(v.get_vector() + eps * w.get_vector()),
                rate_func=m.linear,
                run_time=5.0 * eps,
            )
            three_d_vis.rotate(50 * eps * m.DEGREES, (1, 0, 0))


"""
There are several versions of the spectral theorem:

    - Symmetric matrices over R, for which the eigenvalues are real;
    - Self-adjoint (Hermitian) matrices over C, for which the eigenvalues are real;
    - Symplectic matrices (equal to the negative of the adjoint) over C, for which
    the eigenvalues are purely imaginary;
    - More generally, normal matrices (matrices which commute with the adjoint) over C,
    for which the eigenvalues are arbitrary but the eigenvectors are still orthonormal.

Understanding this theorem starts with understanding why `symmetric` or `unitary` or `symplectic` are
natural conditions which arise.

Symmetric means that <v, Aw> = <w, Av>.

Case study: The space of real-valued functions on R, with inner product <f, g> = int_{R}f(t)g(t)dt.
Any function A(t) defines a linear transformation Af(t) = A(t)f(t). This is symmetric.

Case study: The space of complex-valued functions on S^1, with inner product <f, g> = int_{S^1}f*(t)g(t)dt.
Any function A(t) defines a linear transformation Af(t) = A(t)f(t). This is symmetric.
The differentiation operator d/dt defines a linear transformation. This is symplectic.
Convolutionw with A(t) defines yet another linear transformation. I think this one commutes with differentiation.
    """


### Scenes
class WhatIsSymmetric(m.ThreeDScene):
    """Depict the meaning of `symmetric` in two dimensions."""

    def construct(self):
        pass


class SpectralTheorem(m.ThreeDScene):
    """Let S be a symmetric matrix, i.e. where S = S^T. We want to show that S has an orthonormal basis of eigenvectors.

    Q: How do we visualize what "symmetric" means?

    A formal proof relies on the following very short lemma.

    Lemma: If V is a subspace which is invariant under S, then it orthogonal complement is invariant under S.
    Pf: For any vector v in V and w in the orthogonal complement of V, <Sw, v> = <w, Sv> = 0.

    So if you can find *any* eigenvector v, you can split your space into the span of v and its orthogonal complement
    (which has dimension n-1), each of which is invariant under S. Then you can restrict your attention to said orthogonal complement,
    and inductively go down the line. There's a non-constructive argument that any linear operator over R or C has an eigenvector (possibly
    complex) so this produces a non-constructive proof.

    However, I want to show a constructive proof which I find more illuminating.

    Consider the function f: v -> <v, Sv> on the unit sphere. Because the unit sphere is compact, this has a *maximum value*
    at some vector v. We claim v is an eigenvector (and so its eigenvalue is the largest one). Let's suppose that Sv = Lv + aw for some
    other unit vector w which is orthogonal to v, (where by definition a = <w, Sv>), and say that a > 0.
    Then consider d_w(f(v)), i.e. the derivative of f at v along the direction w.
    We claim it's positive, contradicting the maximality of v. Indeed, you can calculate this derivative by calculating

    <v + ew, S(v + ew)> = <v, Sv> + e<w, Sv> + e<v, Sw> + e^2<w, Sw> = L + 2ea + O(e^2)

    so the derivative is 2a. This last equality relies on the fact that S is symmetric, which told us that <w, Sv> = <v, Sw>.
    The only way"""

    def construct(self):
        # Draw axes and sphere
        axes = ThreeDAxes()
        heatmap = SphereHeatMap(radius=1.0)  # Sphere heatmap
        heatmap.init_heatmap(HeatMapType.REAL)

        # Matrix which is under consideration
        c_11 = ValueTracker(0.0)
        c_22 = ValueTracker(0.0)
        c_33 = ValueTracker(0.0)
        c_12 = ValueTracker(0.0)
        c_13 = ValueTracker(0.0)
        c_23 = ValueTracker(0.0)

        def make_matrix() -> np.ndarray:
            return np.array(
                [
                    [c_11.get_value(), c_12.get_value(), c_13.get_value()],
                    [c_12.get_value(), c_22.get_value(), c_23.get_value()],
                    [c_13.get_value(), c_23.get_value(), c_33.get_value()],
                ]
            )

        # Maps a
        def _heatmap_fn(pt: np.ndarray, m: np.ndarray):
            """Maps an incoming chart vector p of shape (2,) and an inner product matrix M of shape (3, 3)
            to the dot product <f(p), Mf(p)>."""
            vec = heatmap.uv_func(pt[0], pt[1])
            return np.dot(vec, np.matmul(m, vec))

        def heatmap_fn(arr: np.ndarray):
            """Maps the array of points in R^2 (parametrizing a surface) to the self-inner-products under the
            matrix parametrized by the value trackers"""
            # Initialize matrix from parameters
            mat = make_matrix()
            # Apply underlying
            return np.apply_along_axis(lambda p: _heatmap_fn(p, mat), axis=-1, arr=arr)

        heatmap.set_f(heatmap_fn)

        # Add an updater which sets the function as parameters change
        # TODO Check if anything changed before recomputing
        heatmap.add_updater(
            lambda mobj: mobj.set_f(
                lambda z: np.apply_along_axis(
                    lambda p: _heatmap_fn(p, make_matrix()), axis=-1, arr=z
                )
            )
        )

        # heatmap.mesh = SurfaceMesh(heatmap).set_stroke(BLUE, 1, opacity=0.5)
        self.play(
            ShowCreation(axes),
            ShowCreation(heatmap),
        )

        self.embed()
