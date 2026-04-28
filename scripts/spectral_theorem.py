# Visual proof of the spectral theorem

import math
from typing import Callable

from manimlib import *
import numpy as np

class TestScene(Scene):
    def construct(self):
        self.embed()

### Borrowed from 3b1b
def get_vector_field(
    axes, v_func, n_points, start_point=None, mvltss=1.0, random_order=False
):
    points = fibonacci_sphere(n_points)

    if start_point is not None:
        points = points[
            np.argsort(np.linalg.norm(points - start_point.reshape(-1, 3), axis=1))
        ]
    if random_order:
        indices = list(range(len(points)))
        random.shuffle(indices)
        points = points[indices]

    alpha = clip(inverse_interpolate(10_000, 1000, n_points), 0, 1)
    stroke_width = interpolate(1, 3, alpha**2)

    return get_sphereical_vector_field(
        v_func, axes, points, mvltss=mvltss, stroke_width=stroke_width
    )

def get_sphereical_vector_field(
    v_func,
    axes,
    points,
    color=BLUE,
    stroke_width=1,
    mvltss=1.0,
    tip_width_ratio=4,
    tip_len_to_width=0.01,
):
    field = VectorField(
        v_func,
        axes,
        sample_coords=1.01 * points,
        max_vect_len_to_step_size=mvltss,
        density=1,
        stroke_width=stroke_width,
        tip_width_ratio=tip_width_ratio,
        tip_len_to_width=tip_len_to_width,
    )
    field.apply_depth_test()
    field.set_stroke(color, opacity=0.8)
    field.set_scale_stroke_with_zoom(True)
    return field

def fibonacci_sphere(samples=1000):
    """
    Create uniform-ish points on a sphere

    Parameters
    ----------
    samples : int
        Number of points to create. The default is 1000.

    Returns
    -------
    points : NumPy array
        Points on the unit sphere.

    """

    # Define the golden angle
    phi = np.pi * (np.sqrt(5) - 1)

    # Define y-values of points
    pos = np.array(range(samples), ndmin=2)
    y = 1 - (pos / (samples - 1)) * 2

    # Define radius of cross-section at y
    radius = np.sqrt(1 - y * y)

    # Define the golden angle increment
    theta = phi * pos

    # Define x- and z- values of poitns
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    # Merge together x,y,z
    points = np.concatenate((x, y, z))

    # Transpose to get coordinates in right place
    points = np.transpose(points)

    return points

### For general linear algebra
#
def make_random_symmetric_matrix(dim: int):
    """Random symmetric matrix whose diagonal entries are chosen from N(0, 1) and off-diagonal entries from N(0, 0.5)."""
    # TODO Fix the mean and variance
    x = np.random.randn(dim, dim)
    return 0.5 * (x + x.T)


def make_random_integer_symmetric_matrix(dim: int, low: int = -9, high: int = 9):
    x = np.random.randint(low, high, (dim, dim))
    # TODO Fix
    return 0.5 * (x + x.T)


class VectorTracker(Mobject):
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
        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_value(i, init_val + (val - init_val) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def slide_vector(self, vec: np.ndarray):
        """Continuously vary the entire vector in a linear fashion towards the given target vector."""
        self.clear_updaters()
        init_vec = self.get_vector()

        tracker = ValueTracker(0)
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

        v = ValueTracker(0)
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


class MatrixTracker(Mobject):
    """Stores a finite-dimensional square matrix with complex entries, where the individual entries act
    as value trackers. Contains convenience functions for computing, e.g., eigenvalues and eigenvectors,
    and recomputes these whenever entries are set."""

    value_type: type
    recompute_eig: bool = False
    matrix: np.ndarray
    eigenvals: np.ndarray
    eigenvectors: np.ndarray

    def __init__(self, dim: int, value_type: type = np.complex128, **kwargs):
        self.dim = dim
        self.value_type = value_type
        self.matrix = np.zeros((dim, dim), dtype=self.value_type)
        self.eigenvals = np.zeros((dim,), dtype=self.value_type)
        self.eigenvectors = np.eye(dim, dtype=self.value_type)
        super().__init__(**kwargs)

    def set_recompute_eig(self, v: bool):
        self.recompute_eig = v
        return self

    def _do_recompute_eig(self):
        """Recomputes the eigenvalues and eigenvectors from scratch, e.g. when the matrix is changed
        in an arbitrary way."""
        eig_result = np.linalg.eig(
            self.uniforms["matrix"].reshape((self.dim, self.dim))
        )
        # Sort in descending order
        self.eigenvals = eig_result.eigenvalues[::-1]
        self.eigenvectors = eig_result.eigenvectors[:, ::-1]
        return self

    def _do_recompute_eig_local(self):
        """Recomputes the eigenvalues and eigenvectors by locally perturbing the current eigenvalues and eigenvectors
        according to Newton's method, thus avoiding reproducing work"""
        # TODO Implement this.
        return self._do_recompute_eig()

    def init_uniforms(self) -> None:
        super().init_uniforms()
        self.uniforms["matrix"] = self.matrix.flatten()

    def get_matrix(self) -> np.ndarray:
        # return self.matrix
        return self.uniforms["matrix"].reshape((self.dim, self.dim))

    def set_matrix(self, matrix: np.ndarray):
        self.matrix = matrix
        self.uniforms["matrix"] = matrix.flatten()
        if self.recompute_eig:
            self._do_recompute_eig()
        return self

    def get_column(self, i: int) -> np.ndarray:
        return self.uniforms["matrix"][i :: self.dim]

    def set_column(self, j: int, col: np.ndarray):
        self.matrix[:, j] = col
        self.uniforms["matrix"][j :: self.dim] = col
        if self.recompute_eig:
            self._do_recompute_eig()
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
        if self.recompute_eig:
            self._do_recompute_eig()
        return self

    def slide_value(self, i: int, j: int, val: float | complex):
        """Continuously changes a single entry of the matrix. This is the result which *should* be produced by
        self.animate.set_value(i, j, val), but is not because that method doesn't recompute the eigenvals/vecs
        at all intermediate steps."""
        self.clear_updaters()
        init_val = self.get_value(i, j)
        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_value(
                i, j, init_val + (val - init_val) * v.get_value()
            )
        )
        return v.animate.set_value(1.0)

    def slide_column(self, j: int, col: np.ndarray):
        """Continuously changes a single column of the matrix."""
        self.clear_updaters()
        init_col = self.get_column(j)
        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_column(j, init_col + (col - init_col) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def slide_matrix(self, mat: np.ndarray):
        """Continuously vary the entire matrix in a linear fashion towards the given target matrix."""
        self.clear_updaters()
        init_mat = self.get_matrix()

        v = ValueTracker(0)
        self.add_updater(
            lambda mobj: mobj.set_matrix(init_mat + (mat - init_mat) * v.get_value())
        )
        return v.animate.set_value(1.0)

    def increment_value(self, i: int, j: int, d_value: float | complex) -> None:
        self.set_value(i, j, self.get_value(i, j) + d_value)
        if self.recompute_eig:
            self._do_recompute_eig_local()
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
        if self.recompute_eig:
            self._do_recompute_eig()
        return self

    def _do_recompute_eig(self):
        """Recomputes the eigenvalues and eigenvectors from scratch, e.g. when the matrix is changed
        in an arbitrary way."""
        eig_result = np.linalg.eigh(
            self.uniforms["matrix"].reshape((self.dim, self.dim))
        )
        # Sort in descending order
        self.eigenvals = eig_result.eigenvalues[::-1]
        self.eigenvectors = eig_result.eigenvectors[:, ::-1]
        return self

    def _do_recompute_eig_local(self):
        """Recomputes the eigenvalues and eigenvectors by locally perturbing the current eigenvalues and eigenvectors
        according to Newton's method, thus avoiding reproducing work"""
        # TODO Implement this.
        return self._do_recompute_eig()


### For visualization of the Spectral theorem in R^3


class RadialFn(Sphere):
    """Depicting a real-valued function on the sphere as a surface where each point on the unit sphere is
    extended outwards (for positive values) or inwards (for negative values) using a sigmoid function"""

    def __init__(self, f: Callable[[np.ndarray], float], axes: ThreeDAxes, **kwargs):
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


def make_column_vectors(mat: MatrixTracker, t_axes: ThreeDAxes) -> VGroup:
    """Construct column vectors of a matrix, i.e. the images of the three unit basis vectors"""
    assert mat.dim <= 3
    e0 = (
        Arrow(t_axes.c2p(*ORIGIN), t_axes.c2p(*mat.get_column(0).real), buff=0.0)
        .set_color(BLUE)
        .add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                t_axes.c2p(*ORIGIN),
                t_axes.c2p(*mat.get_column(0).real),
            )
        )
    )

    e1 = (
        Arrow(t_axes.c2p(*ORIGIN), t_axes.c2p(*mat.get_column(1).real), buff=0.0)
        .set_color(BLUE)
        .add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                t_axes.c2p(*ORIGIN),
                t_axes.c2p(*mat.get_column(1).real),
            )
        )
    )
    e2 = (
        Arrow(t_axes.c2p(*ORIGIN), t_axes.c2p(*mat.get_column(2).real), buff=0.0)
        .set_color(BLUE)
        .add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                t_axes.c2p(*ORIGIN),
                t_axes.c2p(*mat.get_column(2).real),
            )
        )
    )
    return VGroup(e0, e1, e2)


### Scenes

def make_eigenvectors(mat: SymmetricMatrixTracker, axes: ThreeDAxes):
    """Constructs the eigenvectors of a matrix as geometric objects."""
    assert mat.dim <= 3
    v0 = Line().set_color(GREEN).set_opacity(1.0) # Highlight the one with max eigenvalue
    v0.add_updater(
        lambda arr: arr.put_start_and_end_on(
            axes.c2p(*(-1.5 * mat.get_eigenvector(0).real)),
            axes.c2p(*(1.5 * mat.get_eigenvector(0).real)),
        )
    )

    v1 = Line().set_color(GREEN).set_opacity(0.3)
    v1.add_updater(
        lambda arr: arr.put_start_and_end_on(
            axes.c2p(*(-1.5 * mat.get_eigenvector(1).real)),
            axes.c2p(*(1.5 * mat.get_eigenvector(1).real)),
        )
    )

    v2 = Line().set_color(GREEN).set_opacity(0.3)
    v2.add_updater(
        lambda arr: arr.put_start_and_end_on(
            axes.c2p(*(-1.5 * mat.get_eigenvector(2).real)),
            axes.c2p(*(1.5 * mat.get_eigenvector(2).real)),
        )
    )
    return VGroup(v0, v1, v2)

class PlaneSpannedBy(Surface):
    def __init__(
        self, uvec: np.ndarray, vvec: np.ndarray, origin: np.ndarray, **kwargs
    ):
        self.origin = origin
        self.uvec = uvec
        self.vvec = vvec
        super().__init__(**kwargs)

    def uv_func(self, u: float, v: float) -> tuple[float, float, float]:
        return tuple(u * self.uvec + v * self.vvec + self.origin)


class WhatIsSymmetric(Scene):
    """Simple explanation of what `symmetric` means for a 2x2 matrix."""

    def construct(self):
        # Set up the 2D axes
        xmax = 6.0
        plane = NumberPlane(x_range=(-xmax, xmax, 1.0), y_range=(-xmax, xmax, 1.0))
        self.play(ShowCreation(plane))

        basis = VGroup()
        e0 = Arrow(plane.c2p(0.0, 0.0), plane.c2p(1.0, 0.0), buff=0.0).set_color(
            WHITE
        )
        e1 = Arrow(plane.c2p(0.0, 0.0), plane.c2p(0.0, 1.0), buff=0.0).set_color(
            WHITE
        )
        basis.add(e0, e1)
        self.play(ShowCreation(basis))

        # Construct the matrix object
        mat = MatrixTracker(2)
        mat.set_matrix(np.eye(2))
        mat_mobj = DecimalMatrix(mat.get_matrix()).move_to((4.5, 4.5, 0))

        def mat_updater(mobj: DecimalMatrix):
            for i in range(2):
                for j in range(2):
                    mobj.mob_matrix[i][j].set_value(mat.get_value(i, j))

        mat_mobj.add_updater(mat_updater)

        self.add(mat)
        self.play(ShowCreation(mat_mobj))

        # Construct the associated column vectors
        column_vecs = VGroup()
        v0 = Arrow(plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(0))).set_color(
            BLUE
        )
        v0.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(0))
            )
        )
        v1 = Arrow(plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(1))).set_color(
            BLUE
        )
        v1.add_updater(
            lambda mobj: mobj.put_start_and_end_on(
                plane.c2p(0.0, 0.0), plane.c2p(*mat.get_column(1))
            )
        )
        column_vecs.add(v0, v1)
        self.play(ShowCreation(column_vecs))

        # Slide to a random symmetric matrix
        self.play(mat.slide_matrix(make_random_symmetric_matrix(2)))

        # Indicate the pair e0, v1 and compute the dot product
        self.play(
            Indicate(e1),
            Indicate(v0),
            Indicate(mat_mobj.mob_matrix[1][0]),
            run_time=1.0,
        )

        self.play(
            Indicate(e0),
            Indicate(v1),
            Indicate(mat_mobj.mob_matrix[0][1]),
            run_time=1.0,
        )

        # Draw dashed lines to compute these
        d0 = (
            DashedLine(dash_length=0.1)
            .set_color(YELLOW)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    plane.c2p(*mat.get_column(0)), plane.c2p(mat.get_value(0, 0), 0)
                )
            )
        )
        d1 = (
            DashedLine(dash_length=0.1)
            .set_color(YELLOW)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    plane.c2p(*mat.get_column(1)), plane.c2p(0, mat.get_value(1, 1))
                )
            )
        )

        self.play(ShowCreation(d0), ShowCreation(d1), run_time=1.0)

        # Slide to a few more random symmetric matrices
        self.play(mat.slide_matrix(2 * make_random_symmetric_matrix(2)))

        #
        self.embed()


class RepresentationsOfVectors(Scene):
    """It's worth thinking about the three different ways we `visualize` a vector and being ready to
    jump between these.

    (1) A list of D numbers.
    (2) A function on a domain of size D.
    (3) A point in D-dimensional space."""

    def construct(self):
        pass


class GramSchmidt(Scene):
    """Animate the Gram-Schmidt procedure in three dimensions, while depicting symbolic manipulation of the matrix
    on the side."""
    def setup_vars(self):
        # Components which should be rotated in-frame
        self.geometric_part = Group()
        self.camera.frame.move_to((0, 0, 8))

        # Create the underlying matrix whose columns are the vectors we're operating on
        self.mat = MatrixTracker(3)
        mat = self.mat

        # mat.set_matrix(3 * np.random.randn(3, 3))
        mat.set_matrix(np.array([[3, 0.5, 1], [1, 2, 1], [0.5, 1.5, 3]]))
        self.add(mat)

        # Instantiate a displayed decimal matrix object which tracks with mat
        def mat_updater(mmobj: DecimalMatrix):
            for i in range(3):
                for j in range(3):
                    mmobj.mob_matrix[i][j].set_value(self.mat.get_value(i, j))

        self.mat_mobj = DecimalMatrix(mat.get_matrix())
        mat_mobj = self.mat_mobj
        mat_mobj.move_to((-5, 3, 0))
        mat_mobj.fix_in_frame()
        mat_mobj.scale(0.8)
        mat_mobj.add_updater(mat_updater)
        self.add(mat_mobj)

        # Axes on which everything will be placed. Will be centered at the scene origin.
        self.axes = ThreeDAxes(
            (-4, 4),
            (-4, 4),
            (-4, 4),
            axis_config={"include_ticks": False, "include_tip": True},
        )
        axes = self.axes
        axes.add_axis_labels(font_size=36)
        axes.move_to((0, 0, 0))
        self.add(axes)

        self.geometric_part.add(axes)

        # Column vectors which track with matrix
        self.column_vecs = make_column_vectors(mat, axes)
        self.initial_vecs = VGroup()
        for vec in self.column_vecs:
            self.initial_vecs.add(
                Arrow(vec.start.copy(), vec.end.copy(), buff=0.0)
                .set_color(BLUE)
                .set_opacity(0.3)
            )

        self.geometric_part.add(self.column_vecs, self.initial_vecs)
        return self

    def do_gram_schmidt(self):
        # Do Gram-Schmidt to produce an orthogonal basis
        orth_vectors = [self.mat.get_column(0).copy()]
        projections_to_subspaces = [np.zeros(3)]
        for i in range(1, 3):
            w = self.mat.get_column(i).copy()
            proj = np.zeros(3)
            for j in range(i):
                x = orth_vectors[j]
                proj += x * w.dot(x) / x.dot(x)
            orth_vectors.append(w - proj)
            projections_to_subspaces.append(proj)

        return orth_vectors, projections_to_subspaces


    def construct(self):
        self.setup_vars()
        mat = self.mat
        axes = self.axes
        mat_mobj = self.mat_mobj
        column_vecs = self.column_vecs
        initial_vecs = self.initial_vecs
        orth_vectors, projections_to_subspaces = self.do_gram_schmidt()
        frame = self.frame


        # Updater which tells the camera to spin around the axes
        # angular_speed = -4 * DEGREES
        # frame.add_updater(lambda m, dt: m.increment_gamma(angular_speed * dt))

        angular_speed = 4 * DEGREES
        self.geometric_part.add_updater(lambda mobj, dt: mobj.rotate(angular_speed * dt, np.array([0,1,0]), about_point=np.array([0,0,0])))
        # frame.add_ambient_rotation(angular_speed)

        # dist = np.linalg.norm(self.camera.frame.get_center() - axes.c2p(*ORIGIN))
        # self.camera.frame.add_updater(
        #     lambda m, dt: m.rotate(angular_speed * dt, np.array([0, 1, 0])).shift(
        #         self.camera.frame.view_matrix[:-1, 0] * dist * dt * angular_speed
        #     )
        # )
        # self.wait(10)

        # Draw the three vectors one-by-one
        for i in range(3):
            self.play(
                FadeIn(column_vecs[i]),
                FadeIn(initial_vecs[i]),
                mat_mobj.get_column(i).animate.set_color(BLUE),
            )
            self.play(
                mat_mobj.get_column(i).animate.set_color(WHITE),
            )

        self.wait()

        self.save_state()

        def mat_updater(mmobj: DecimalMatrix):
            for i in range(3):
                for j in range(3):
                    mmobj.mob_matrix[i][j].set_value(self.mat.get_value(i, j))
        mat_mobj.add_updater(mat_updater)

        # Animate it

        for i in range(1, 3):
            # Draw a dotted line from the i-th vector to the subspace spanned by vectors
            # 0, 1, ..., (i-1), and the projected vector
            p = projections_to_subspaces[i]
            v = mat.get_column(i)
            w = orth_vectors[i]
            proj_pt = Dot(axes.c2p(*p)).set_color(RED).set_opacity(0.8)
            dashed_line = (
                DashedLine(axes.c2p(*mat.get_column(i)), axes.c2p(*p))
                .set_color(YELLOW)
                .set_opacity(0.8)
            )

            self.geometric_part.add(proj_pt, dashed_line)

            if i == 1:
                # Indicate the orthogonal line
                orth_space = Line(axes.c2p(*(-1.5 * p)), axes.c2p(*(1.5 * p)))
                orth_space.set_opacity(0.3)
                orth_space.set_color(GREEN)
                orth_space.add_updater(
                    lambda mobj: mobj.put_start_and_end_on(
                        axes.c2p(*(-1.5 * p)), axes.c2p(*(1.5 * p))
                    )
                )
                self.play(
                    ShowCreation(proj_pt),
                    ShowCreation(dashed_line),
                    VFadeIn(orth_space),
                    # rate_func=linear,
                )
            elif i == 2:
                # Indicate the orthogonal plane
                orth_space = PlaneSpannedBy(
                    uvec=axes.c2p(*orth_vectors[0]),
                    vvec=axes.c2p(*p),
                    origin=axes.c2p(*ORIGIN),
                    u_range=(-1.0, 1.0),
                    v_range=(-1.0, 1.0),
                ).set_opacity(0.2)
                mesh = SurfaceMesh(orth_space).set_stroke(BLUE, 1, opacity=0.5)
                self.play(
                    ShowCreation(proj_pt),
                    ShowCreation(dashed_line),
                    FadeIn(orth_space),
                    FadeIn(mesh),
                    # rate_func=linear,
                )

            # Slide to eliminate the projection
            tracker = ValueTracker(1.0)
            proj_pt.add_updater(
                lambda mobj: mobj.move_to(axes.c2p(*(p * tracker.get_value())))
            )
            dashed_line.add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    axes.c2p(*(w + p * tracker.get_value())),
                    axes.c2p(*(p * tracker.get_value())),
                )
            )

            mat.add_updater(
                lambda mobj: mobj.set_column(
                    i,
                    orth_vectors[i] + projections_to_subspaces[i] * tracker.get_value(),
                )
            )
            mat_mobj.add_updater(lambda mobj: mobj.mob_matrix[0][i].set_value(mat.get_value(0, i)))
            mat_mobj.add_updater(lambda mobj: mobj.mob_matrix[1][i].set_value(mat.get_value(1, i)))
            mat_mobj.add_updater(lambda mobj: mobj.mob_matrix[2][i].set_value(mat.get_value(2, i)))

            self.play(
                tracker.animate.set_value(0.0),
                run_time=2.0,
            )
            self.remove(proj_pt, dashed_line, orth_space)
            self.geometric_part.remove(proj_pt, dashed_line, orth_space)
            if i == 2:
                self.remove(mesh)
                self.geometric_part.remove(mesh)
            # TODO Add an elbow

        self.save_state()
        self.embed()


class GradientField(Scene):
    """Vector field construction in proof of Spectral Theorem"""
    def construct(self):
        # Initial setup
        frame = self.frame
        frame.reorient(120, 55, 0)
        self.camera.light_source.move_to([0, -10, 10])

        radius = 3
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2), axis_config={"include_ticks": True, "include_tip": False},)
        axes.scale(radius)
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE, 0.3)
        sphere.always_sort_to_camera(self.camera) # Sorts the faces of the sphere back-to-front w.r.t the camera
        mesh = SurfaceMesh(sphere, (51, 25))
        mesh.set_stroke(WHITE, 1, 0.15) # color, width, opacity

        self.play(ShowCreation(axes), run_time=2.0)
        self.play(ShowCreation(sphere), ShowCreation(mesh))

        ## Pick a point on the sphere, parametrized by polar coordinates
        v_dot = DotCloud([axes.c2p(0,0,1)], color=RED, radius=0.1)
        v_line = Line(axes.c2p(*ORIGIN), axes.c2p(0,0,1)).set_stroke(color=RED, opacity=0.5)
        self.play(ShowCreation(v_dot), ShowCreation(v_line))

        theta = ValueTracker()
        phi = ValueTracker()
        def v_func(theta, phi):
            return np.array([
                np.sin(theta.get_value()) * np.cos(phi.get_value()),
                np.sin(theta.get_value()) * np.sin(phi.get_value()),
                np.cos(theta.get_value())
            ])

        v_dot.add_updater(lambda mobj: mobj.move_to(
            axes.c2p(*v_func(theta, phi))
        ))
        v_line.add_updater(lambda mobj: mobj.put_start_and_end_on(
            axes.c2p(*ORIGIN),
            axes.c2p(*v_func(theta, phi))
        ))

        v_tex = Tex("v").set_color(RED)
        v_tex.fix_in_frame()
        v_tex.add_updater(lambda mobj: mobj.move_to(frame.to_fixed_frame_point(v_line.get_center()) + LEFT * 0.5))
        v_tex.scale(0.8)

        ## Move it around a bit
        phi.set_value(-PI/2)
        self.play(theta.animate.set_value(PI/2))
        self.play(theta.animate.set_value(3 * PI/2), phi.animate.set_value(0))
        self.play(theta.animate.set_value(PI/2), phi.animate.set_value(PI/2))
        self.play(theta.animate.set_value(0), phi.animate.set_value(0))

        ## Add a tangent vector, parametrized by a direction coordinate
        # TODO Make this vary as v varies. Will probably need to use quaternions here to define
        # a local coordinate system that's consistent.
        def w_func(gamma, arrow_length):
            return arrow_length.get_value() * np.array([
                np.cos(gamma.get_value()),
                np.sin(gamma.get_value()),
                0.
            ])

        arrow_length = ValueTracker(0.2)
        gamma = ValueTracker()
        w_arrow = Arrow(
            axes.c2p(*v_func(theta, phi)),
            axes.c2p(*(v_func(theta, phi) + w_func(gamma, arrow_length))),
            buff=0
        ).set_color(YELLOW)
        w_arrow.add_updater(lambda mobj: mobj.put_start_and_end_on(
            axes.c2p(*v_func(theta, phi)),
            axes.c2p(*(v_func(theta, phi) + w_func(gamma, arrow_length))),
        ))

        w_tex = Tex("w").set_color(YELLOW)
        w_tex.fix_in_frame()
        w_tex.add_updater(lambda mobj: mobj.move_to(frame.to_fixed_frame_point(w_arrow.get_center()) + LEFT * 0.5))
        w_tex.scale(0.8)

        # Spin it around
        self.play(ShowCreation(w_arrow), FadeIn(w_tex), FadeIn(v_tex))
        self.play(gamma.animate.set_value(TAU))
        gamma.set_value(0)


        ## Initialize matrix
        mat = SymmetricMatrixTracker(3)
        mat.set_recompute_eig(True)
        mat.set_matrix(np.array([[2,1,1],[1,2,1],[1,1,2]]))

        eigenvecs = make_eigenvectors(mat, axes)

        # Define relevant functions
        def A(vec: np.ndarray):
            return mat.matmul(vec.T).T

        def scalar_f(vec: np.ndarray):
            return np.sum(vec * mat.matmul(vec.T).T, axis=1) / np.sum(np.pow(vec, 2), axis=1)

        def grad(vec: np.ndarray):
            return mat.matmul(vec.T).T - vec * (np.sum(vec * mat.matmul(vec.T).T, axis=1) / np.sum(np.pow(vec, 2), axis=1))[:, None]

        # Make Av and Aw
        av_dot = DotCloud(axes.c2p(*A(v_func(theta, phi)))).set_color(RED)
        av_dot.add_updater(lambda mobj: mobj.move_to(
            axes.c2p(*A(v_func(theta, phi)))
        ))

        av_line = Line(
            axes.c2p(*ORIGIN), axes.c2p(*A(v_func(theta, phi)))
        ).set_stroke(color=RED, opacity=0.5)
        av_line.add_updater(lambda mobj: mobj.put_start_and_end_on(
            axes.c2p(*ORIGIN), axes.c2p(*A(v_func(theta, phi)))
        ))
        aw_arrow = Arrow(
            axes.c2p(*A(v_func(theta, phi))),
            axes.c2p(*A(v_func(theta, phi) + w_func(gamma, arrow_length))),
            buff=0
        ).set_color(YELLOW)
        aw_arrow.add_updater(lambda mobj: mobj.put_start_and_end_on(
            axes.c2p(*A(v_func(theta, phi))),
            axes.c2p(*A(v_func(theta, phi) + w_func(gamma, arrow_length))),
        ))

        av_tex = Tex("Av").set_color(RED)
        av_tex.fix_in_frame()
        av_tex.add_updater(lambda mobj: mobj.move_to(frame.to_fixed_frame_point(av_line.get_center()) + RIGHT * 0.5))
        av_tex.scale(0.8)

        aw_tex = Tex("Aw").set_color(YELLOW)
        aw_tex.fix_in_frame()
        aw_tex.add_updater(lambda mobj: mobj.move_to(frame.to_fixed_frame_point(aw_arrow.get_center()) + RIGHT * 0.5))
        aw_tex.scale(0.8)

        # Draw them in (consider transforming over from the original objects, depicting an "A" there)
        self.play(
            ShowCreation(av_dot),
            ShowCreation(av_line),
            ShowCreation(aw_arrow),
            FadeIn(av_tex),
            FadeIn(aw_tex),
            self.camera.frame.animate.move_to(0.5 * self.camera.get_location())
        )

        deriv_eq = Tex("(\partial_w f)(v) = \\frac{\\langle v, Aw\\rangle + \\langle w, Av\\rangle }{\\langle v, v\\rangle ^2}")
        deriv_eq.fix_in_frame()
        deriv_eq.move_to((-3.5, 3, 0))
        self.play(FadeIn(deriv_eq))


        self.embed()

        # Slide a new copy of v along w to make v + w
        # Slide a new copy of Av along Aw to make A(v+w)

        # Make the gradient field Av - F(v)v
        n_points = 200
        vecfld = get_vector_field(
            axes,
            lambda v: mat.matmul(v.T).T - v * (np.sum(v * mat.matmul(v.T).T, axis=1) / np.sum(np.pow(v, 2), axis=1))[:, None],
            n_points
        ).set_color(ORANGE)
        vecfld.add_updater(lambda mobj: mobj.become(
            get_vector_field(
                axes,
                lambda v: mat.matmul(v.T).T - v * (np.sum(v * mat.matmul(v.T).T, axis=1) / np.sum(np.pow(v, 2), axis=1))[:, None],
                n_points
            ).set_color(ORANGE)
        ))

        # Pick a random unit vector
        v_pt = Dot(axes.c2p(1,0,0))
        self.add(v_pt)

        # Flow it along the vector field

        self.embed()
        self.camera.frame.move_to((0, 0, 8))

        mat = SymmetricMatrixTracker(3)
        mat.set_recompute_eig(True)
        mat.set_matrix(make_random_symmetric_matrix(3))
        self.add(mat)

        axes = ThreeDAxes(
            (-2, 2),
            (-2, 2),
            (-2, 2),
            axis_config={"include_ticks": False, "include_tip": True},
        )
        axes.add_axis_labels(font_size=36)
        axes.move_to((0, 0, 0))
        axes.set_width(6.0)

        sphere = Sphere(radius=1.0)
        self.embed()

        vector.set_perpendicular_to_camera(frame)
        pass

class ComputeLegendrePolynomials(Scene):
    """Compute the Legendre polynomials over [-1, 1] using Gram-Schmidt, showing the symbolic manipulation of the matrix
    on the side."""

    def construct(self):
        pass


class WhereDoSymmetricMatricesAppear(Scene):
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


class Playground(Scene):
    def construct(self):
        # Set up for the spectral theorem
        self.embed()


# Below: to be refined.
# - The proof essentially relies on the fact that the unit sphere in V is *compact*, which implies
# that any continuous scalar-valued function on V has a maximum value. This is why the proof breaks
# down in general when V is infinite-dimensional.
# - The proof also relies on the statement that for any subspace W < V, V = W \oplus W^{\perp}. This is
# why the proof breaks down for vector spaces over fields of nonzero characteristic.
class SpectralTheorem(Scene):
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
        eigenvals_mobj = DecimalMatrix(eigenvals.get_vector().reshape((1, mat.dim)))

        def vec_updater(mobj: DecimalMatrix):
            for i in range(mat.dim):
                mobj.mob_matrix[0][i].set_value(eigenvals.get_value(i))

        eigenvals_mobj.add_updater(vec_updater)
        return eigenvals, eigenvals_mobj

    def make_eigenvectors(self, mat: SymmetricMatrixTracker, t_axes: ThreeDAxes):
        """Constructs the eigenvectors of a matrix as geometric objects."""
        assert mat.dim <= 3
        v0 = Line().set_color(GREEN).set_opacity(1.0)
        v0.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(0).real)),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(0).real)),
            )
        )

        v1 = Line().set_color(GREEN).set_opacity(0.3)
        v1.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(1).real)),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(1).real)),
            )
        )

        v2 = Line().set_color(GREEN).set_opacity(0.3)
        v2.add_updater(
            lambda arr: arr.put_start_and_end_on(
                t_axes.c2p(*(-1.5 * mat.get_eigenvector(2).real)),
                t_axes.c2p(*(1.5 * mat.get_eigenvector(2).real)),
            )
        )
        return VGroup(v0, v1, v2)

    def make_heatmap(self, mat: MatrixTracker, t_axes: ThreeDAxes):
        """Makes a heatmap over the unit sphere depicting the value of f(v) = <v, Av>"""
        assert mat.dim <= 3
        heatmap = SphereHeatMap(radius=1.0, resolution=(101, 51)).move_to(
            t_axes.c2p(0, 0, 0)
        )
        heatmap.init_heatmap(HeatMapType.REAL)

        def rgba_real(arr: np.ndarray):
            """Maps an array of real numbers (shape (*,)) to an RGBA array (shape (*, 4))."""
            return np.stack(
                (
                    (arr > 0).astype(float)
                    * (2 * sigmoid(arr) - 1),  # Red for positive positions
                    np.zeros_like(arr),
                    (arr < 0).astype(float)
                    * (1 - 2 * sigmoid(arr)),  # Blue for negative positions
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

    def make_flow_vector(self, mat: MatrixTracker, t_axes: ThreeDAxes):
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
            Arrow()
            .set_color(RED)
            .add_updater(
                lambda mobj: mobj.put_start_and_end_on(
                    t_axes.c2p(*ORIGIN), t_axes.c2p(*v.get_vector())
                )
            )
        )

        def w_arrow_updater(mobj: Arrow):
            mobj.put_start_and_end_on(
                t_axes.c2p(*v.get_vector()),
                t_axes.c2p(*(v.get_vector() + w.get_vector())),
            )

        w_arrow = Arrow().set_color(YELLOW).add_updater(w_arrow_updater)
        return v, w, v_arrow, w_arrow

    def randomize(self, mat: MatrixTracker):
        """Randomize the matrix"""
        self.play(mat.slide_matrix(make_random_symmetric_matrix(mat.dim)))

    def construct(self):
        # Dimension 3
        d = 3

        self.camera.frame.move_to((0, 0, 8))

        # Create the underlying matrix which varies
        mat = SymmetricMatrixTracker(d)
        mat.set_recompute_eig(True)
        mat.set_matrix(make_random_symmetric_matrix(3))
        self.add(mat)

        ### Symbolic half of the picture
        symbolic_part = Group()

        # Instantiate a displayed decimal matrix object which tracks with mat
        def mat_updater(mobj: DecimalMatrix):
            for i in range(d):
                for j in range(d):
                    mobj.mob_matrix[i][j].set_value(mat.get_value(i, j))

        mat_mobj = DecimalMatrix(mat.get_matrix()).add_updater(mat_updater)
        symbolic_part.add(mat_mobj)

        eigenvals, eigenvals_mobj = self.make_eigenvalues(mat)
        self.add(eigenvals)
        eigenvals_mobj.next_to(mat_mobj, DOWN, buff=1.0)
        symbolic_part.add(eigenvals, eigenvals_mobj)
        symbolic_part.move_to((-5, 1, 0))
        symbolic_part.fix_in_frame()
        symbolic_part.scale(0.7)

        self.add(mat_mobj, eigenvals_mobj)

        # self.play(FadeIn(mat_mobj), run_time=1.0)
        # self.play(FadeIn(eigenvals_mobj), run_time=1.0)

        ### Geometric half of the picture
        geometric_part = Group()

        # Axes on which everything will be placed
        axes = ThreeDAxes(
            (-2, 2),
            (-2, 2),
            (-2, 2),
            axis_config={"include_ticks": False, "include_tip": True},
        )
        axes.add_axis_labels(font_size=36)
        axes.move_to((0, 0, 0))
        axes.set_width(6.0)
        geometric_part.add(axes)

        column_vecs = make_column_vectors(mat, axes)
        eigenvecs = self.make_eigenvectors(mat, axes)
        heatmap = self.make_heatmap(mat, axes)
        v, w, v_arrow, w_arrow = self.make_flow_vector(mat, axes)
        self.add(v, w)

        geometric_part.add(heatmap, column_vecs, eigenvecs, v, w, v_arrow, w_arrow)
        geometric_part.rotate(-20 * DEGREES, (0, 1, 0))
        geometric_part.set_width(8.0)
        geometric_part.move_to((4, 0, 0))

        self.play(ShowCreation(axes))
        self.play(ShowCreation(column_vecs))
        self.play(ShowCreation(eigenvecs))
        self.play(ShowCreation(heatmap))
        self.play(ShowCreation(v_arrow), ShowCreation(w_arrow))

        self.embed()

        # TODO Flow vector
        # Rotate v along flow direction
        eps = 1 / 30
        for _ in range(30):
            self.play(
                three_d_vis.animate.rotate(-50 * eps * DEGREES, (0, 1, 0)),
                v.rotate_vector(v.get_vector() + eps * w.get_vector()),
                rate_func=linear,
                run_time=3.0 * eps,
            )
