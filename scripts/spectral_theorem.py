# Visual proof of the spectral theorem

import math

import numpy as np
from manimlib import *

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


class SpectralTheorem(ThreeDScene):
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
