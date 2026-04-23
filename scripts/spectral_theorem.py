# Visual proof of the spectral theorem

import math
from typing import Callable

import manimlib as m
import numpy as np


### For general linear algebra
#
def make_random_symmetric_matrix(dim: int):
    """Random symmetric matrix whose diagonal entries are chosen from N(0, 1) and off-diagonal entries from N(0, 0.5)."""
    # TODO Fix the mean and variance
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
        # Sort in descending order
        self.eigenvals = eig_result.eigenvalues[::-1]
        self.eigenvectors = eig_result.eigenvectors[:, ::-1]
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
        # Sort in descending order
        self.eigenvals = eig_result.eigenvalues[::-1]
        self.eigenvectors = eig_result.eigenvectors[:, ::-1]
        return self

    def _recompute_eig_local(self):
        """Recomputes the eigenvalues and eigenvectors by locally perturbing the current eigenvalues and eigenvectors
        according to Newton's method, thus avoiding reproducing work"""
        # TODO Implement this.
        return self._recompute_eig()


### For visualization of the Spectral theorem in R^3


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
        return self.axes.c2p(*(m.sigmoid(self.f(unit_vec)) * self.radius * unit_vec))


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
class WhatIsSymmetric(m.Scene):
    """Simple explanation of what `symmetric` means for a 2x2 matrix."""

    def construct(self):
        xmax = 6.0
        plane = m.NumberPlane(x_range=(-xmax, xmax, 1.0), y_range=(-xmax, xmax, 1.0))
        self.play(m.ShowCreation(plane))

        basis = m.VGroup()
        e0 = m.Arrow(plane.c2p(0.0, 0.0), plane.c2p(1.0, 0.0), buff=0.0).set_color(
            m.WHITE
        )
        e1 = m.Arrow(plane.c2p(0.0, 0.0), plane.c2p(0.0, 1.0), buff=0.0).set_color(
            m.WHITE
        )
        basis.add(e0, e1)
        self.play(m.ShowCreation(basis))

        mat = MatrixTracker(2)
        mat.set_matrix(np.eye(2))
        mat_mobj = m.DecimalMatrix(mat.get_matrix()).move_to((4.5, 4.5, 0))

        def mat_updater(mobj: m.DecimalMatrix):
            for i in range(2):
                for j in range(2):
                    mobj.mob_matrix[i][j].set_value(mat.get_value(i, j))

        mat_mobj.add_updater(mat_updater)

        self.add(mat)
        self.play(m.ShowCreation(mat_mobj))

        column_vecs = m.VGroup()
        v0 = m.Arrow(plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(0))).set_color(
            m.BLUE
        )
        v0.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(0))
            )
        )
        v1 = m.Arrow(plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(1))).set_color(
            m.BLUE
        )
        v1.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(1))
            )
        )
        column_vecs.add(v0, v1)
        self.play(m.ShowCreation(column_vecs))

        # Slide to a random symmetric matrix
        self.play(mat.slide_matrix(make_random_symmetric_matrix(2)))

        # Indicate the pair e0, v1 and compute the dot product
        self.play(
            m.Indicate(e1),
            m.Indicate(v0),
            m.Indicate(mat_mobj.mob_matrix[1][0]),
            run_time=1.0,
        )

        self.play(
            m.Indicate(e0),
            m.Indicate(v1),
            m.Indicate(mat_mobj.mob_matrix[0][1]),
            run_time=1.0,
        )

        # Draw dashed lines to compute these
        d0 = (
            m.DashedLine(dash_length=0.1)
            .set_color(m.YELLOW)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    plane.c2p(*mat.get_column(0)), plane.c2p(mat.get_value(0, 0), 0)
                )
            )
        )
        d1 = (
            m.DashedLine(dash_length=0.1)
            .set_color(m.YELLOW)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    plane.c2p(*mat.get_column(1)), plane.c2p(0, mat.get_value(1, 1))
                )
            )
        )

        self.play(m.ShowCreation(d0), m.ShowCreation(d1), run_time=1.0)

        # Slide to a few more random symmetric matrices
        self.play(mat.slide_matrix(2 * make_random_symmetric_matrix(2)))
        self.embed()


class RepresentationsOfVectors(m.Scene):
    """It's worth thinking about the three different ways we `visualize` a vector and being ready to
    jump between these.

    (1) A list of D numbers.
    (2) A function on a domain of size D.
    (3) A point in D-dimensional space."""

    def construct(self):
        pass


class GramSchmidt(m.Scene):
    """Animate the Gram-Schmidt procedure in three dimensions, while depicting symbolic manipulation of the matrix
    on the side."""

    def construct(self):
        pass


class ComputeLegendrePolynomials(m.Scene):
    """Compute the Legendre polynomials over [-1, 1] using Gram-Schmidt, showing the symbolic manipulation of the matrix
    on the side."""

    def construct(self):
        pass


class WhereDoSymmetricMatricesAppear(m.Scene):
    """Motivate the concept of symmetric matrices by showing a few example settings:
    - Consider a finite undirected graph, and a random walk on the vertices of the graph.
    Our vector space consists of probability distributions on the vertices of the graph, with dot product
    given by the probability of being at the same location, and our linear operator is expressed in terms
    of the adjacency matrix.

    The linear operator is symmetric precisely when P(i -> j) == P(j -> i), which occurs when the graph
    is *undirected* and *regular*, meaning every vertex has the same out-degree. The Spectral theorem
    tells us that there is an eigendecomposition

    - Consider the interval I = [0, 1], and the vector space of all real-valued functions on I, with an
    inner product given by <f, g> = \int_0^1 f(t)g(t)dt. This is an infinite-dimensional space with
    many possible linear operators on it. One such operator is f(t) -> f'(t) (this is skew-symmetric if we
    require f(0) = f(1)). Another such operator is f(t) -> tf(t) (this is symmetric). In this case, the spectral
    theorem doesn't hold because the operator in question is not *compact* (we will see exactly why the proof breaks down).
    """

    def construct(self):
        pass


# Below: to be refined.
# - The proof essentially relies on the fact that the unit sphere in V is *compact*, which implies
# that any continuous scalar-valued function on V has a maximum value. This is why the proof breaks
# down in general when V is infinite-dimensional.
# - The proof also relies on the statement that for any subspace W < V, V = W \oplus W^{\perp}. This is
# why the proof breaks down for vector spaces over fields of nonzero characteristic.
class SpectralTheorem(m.Scene):
    """Let S be a symmetric linear operator on a finite-dimensional space, i.e. where S = S^T. We want to show that S has an orthonormal basis of eigenvectors.

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

    def make_eigenvalues(self, mat: MatrixTracker):
        eigenvals = VectorTracker(mat.dim).add_updater(
            lambda mobj: mobj.set_vector(mat.eigenvals)
        )
        eigenvals_mobj = m.DecimalMatrix(eigenvals.get_vector().reshape((1, mat.dim)))

        def vec_updater(mobj: m.DecimalMatrix):
            for i in range(mat.dim):
                mobj.mob_matrix[0][i].set_value(eigenvals.get_value(i))

        eigenvals_mobj.add_updater(vec_updater)
        return eigenvals, eigenvals_mobj

    def make_eigenvectors(self, mat: MatrixTracker, t_axes: m.ThreeDAxes):
        """Constructs the eigenvectors of a matrix as geometric objects."""
        assert mat.dim <= 3
        v0 = m.Line().set_color(m.GREEN).set_opacity(1.0)
        v0.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(0))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(0))),
            )
        )

        v1 = m.Line().set_color(m.GREEN).set_opacity(0.3)
        v1.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(1))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(1))),
            )
        )

        v2 = m.Line().set_color(m.GREEN).set_opacity(0.3)
        v2.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(2))),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(2))),
            )
        )
        return m.VGroup(v0, v1, v2)

    def make_column_vectors(self, mat: MatrixTracker, t_axes: m.ThreeDAxes):
        """Construct column vectors of a matrix, i.e. the images of the three unit basis vectors"""
        assert mat.dim <= 3
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
        return m.VGroup(e0, e1, e2)

    def make_heatmap(self, mat: MatrixTracker, t_axes: m.ThreeDAxes):
        """Makes a heatmap over the unit sphere depicting the value of f(v) = <v, Av>"""
        assert mat.dim <= 3
        heatmap = m.SphereHeatMap(radius=1.0, resolution=(101, 51)).move_to(
            t_axes.c2p(0, 0, 0)
        )
        heatmap.init_heatmap(m.HeatMapType.REAL)

        def rgba_real(arr: np.ndarray):
            """Maps an array of real numbers (shape (*,)) to an RGBA array (shape (*, 4))."""
            return np.stack(
                (
                    (arr > 0).astype(float)
                    * (2 * m.sigmoid(arr) - 1),  # Red for positive positions
                    np.zeros_like(arr),
                    (arr < 0).astype(float)
                    * (1 - 2 * m.sigmoid(arr)),  # Blue for negative positions
                    0.8 * np.ones_like(arr),
                ),
                axis=-1,
            )

        heatmap.set_rgba_fn(rgba_real)

        def heatmap_fn(arr: np.ndarray):
            return np.apply_along_axis(
                lambda p: mat.inner_product(heatmap.uv_func(*p)),
                axis=-1,
                arr=arr,
            )

        heatmap.add_updater(lambda mobj: mobj.set_f(heatmap_fn))
        return heatmap

    def make_flow_vector(self, mat: MatrixTracker, t_axes: m.ThreeDAxes):
        """Add a vector v which moves around the unit sphere, along with the vector Av as well as the vector
        Av - F(v)v sitting tangent to v. Key to the proof of the Spectral Theorem."""
        assert mat.dim <= 3
        v = VectorTracker(mat.dim).set_vector(np.array([1.0, 0.0, 0.0]))

        def w_updater(mobj: VectorTracker):
            v_vec = v.get_vector()
            mobj.set_vector(
                mat.matmul(v_vec)
                - mat.inner_product(v_vec) * v_vec / (np.linalg.norm(v_vec) ** 2)
            )

        w = VectorTracker(mat.dim).add_updater(w_updater)

        v_arrow = (
            m.Arrow()
            .set_color(m.RED)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    t_axes.c2p(*m.ORIGIN), t_axes.c2p(*v.get_vector())
                )
            )
        )

        def w_arrow_updater(mobj: m.Arrow):
            mobj.put_start_and_end_on(
                t_axes.c2p(*v.get_vector()),
                t_axes.c2p(*(v.get_vector() + w.get_vector())),
            )

        w_arrow = m.Arrow().set_color(m.YELLOW).add_updater(w_arrow_updater)
        return v, w, v_arrow, w_arrow

    def randomize(self, mat: MatrixTracker):
        """Randomize the matrix"""
        self.play(mat.slide_matrix(make_random_symmetric_matrix(mat.dim)))

    def construct(self):
        # Dimension 3
        d = 3

        # Create the underlying matrix which varies
        mat = SymmetricMatrixTracker(d)
        mat.set_matrix(make_random_symmetric_matrix(3))

        ### Symbolic half of the picture
        symbolic_part = m.Group()

        # Instantiate a displayed decimal matrix object which tracks with mat
        def mat_updater(mobj: m.DecimalMatrix):
            for i in range(d):
                for j in range(d):
                    mobj.mob_matrix[i][j].set_value(mat.get_value(i, j))

        mat_mobj = m.DecimalMatrix(mat.get_matrix()).add_updater(mat_updater)
        symbolic_part.add(mat_mobj)

        eigenvals, eigenvals_mobj = self.make_eigenvalues(mat)
        eigenvals_mobj.next_to(mat_mobj, m.DOWN, buff=1.5)
        symbolic_part.add(eigenvals, eigenvals_mobj)
        symbolic_part.move_to((-5, 0, 0))

        self.play(m.FadeIn(mat_mobj), run_time=1.0)
        self.play(m.FadeIn(eigenvals_mobj), run_time=1.0)

        ### Geometric half of the picture
        geometric_part = m.Group()

        # Axes on which everything will be placed
        axes = m.ThreeDAxes(
            (-2, 2),
            (-2, 2),
            (-2, 2),
            axis_config={"include_ticks": False, "include_tip": True},
        )
        axes.add_axis_labels(font_size=36)
        geometric_part.add(axes)

        column_vecs = self.make_column_vectors(mat, axes)
        eigenvecs = self.make_eigenvectors(mat, axes)
        heatmap = self.make_heatmap(mat, axes)
        v, w, v_arrow, w_arrow = self.make_flow_vector(mat, axes)
        self.add(v, w)
        geometric_part.add(heatmap, column_vecs, eigenvecs, v, w, v_arrow, w_arrow)
        geometric_part.rotate(-20 * m.DEGREES, (0, 1, 0))
        geometric_part.set_width(8.0)
        geometric_part.move_to((4, 0, 0))

        self.play(m.ShowCreation(axes))
        self.play(m.ShowCreation(column_vecs))
        self.play(m.ShowCreation(eigenvecs))
        self.play(m.ShowCreation(heatmap))
        self.play(m.ShowCreation(v_arrow), m.ShowCreation(w_arrow))

        self.embed()

        # TODO Flow vector
        # Rotate v along flow direction
        eps = 1 / 30
        for _ in range(30):
            self.play(
                three_d_vis.animate.rotate(-50 * eps * m.DEGREES, (0, 1, 0)),
                v.rotate_vector(v.get_vector() + eps * w.get_vector()),
                rate_func=m.linear,
                run_time=3.0 * eps,
            )
