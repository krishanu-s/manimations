"""
This is intended as an addition to the manim library.
"""

from typing import Callable, Any
from typing_extensions import Self
import numpy as np
import manim as m


class SmoothOpenPathBezierHandleCalculator:
    n: int
    below_diag: np.ndarray
    diag: np.ndarray
    above_diag: np.ndarray
    two_above_diag: np.ndarray
    """
    Given that an open path consists of n parts (splines), this object can take
    the anchor points P_0, P_1, ..., P_n as inputs and produces the Bezier handles
    B_1, B_2, B_3, ..., B_{2n} as output, such that the sequence

    P_0, B_1, B_2, P_1, B_3, B_4, P_2, B_5, B_6, ...

    is a smooth Bezier path
    """
    def __init__(self, n: int):
        """
        Calculates and stores the n-by-(n+1) transformation matrix used for computing
        the first Bezier handles of a sequence of n+1 anchor points, as a function of
        the n+1 anchor points. This is computed as A^{-1}B, where A is an n-by-n
        tridiagonal matrix and B is an n-by-(n+1) matrix."""
        self.n = n

        # Tridiagonal matrix which is to be inverted
        below_diag = np.array([1.0]*(n-2) + [2.0]) # Below diagonal
        diag = np.array([2.0] + [4.0]*(n-2) + [7.0]) # Main diagonal
        above_diag = np.array([1.0]*(n-1)) # Above diagonal

        # n-by-(n+1) matrix
        self.result = np.zeros(shape=(n, n+1))
        self.result[0, 0], self.result[0, 1] = 1, 2
        self.result[n-1, n-1], self.result[n-1, n] = 8, 1
        for i in range(1, n-1):
            self.result[i, i], self.result[i, i+1] = 4, 2
        self.result[n-1, n-1], self.result[n-1, n] = 8, 1
        
        # Eliminate lower-triangular entries in tridiagonal matrix
        for i in range(n-1):
            scale = below_diag[i] / diag[i]
            diag[i+1] -= above_diag[i] * scale
            # below_diag[i] -= diag[i] * scale
            self.result[i+1] -= self.result[i] * scale


        # Eliminate upper-triangular entries in tridiagonal matrix
        for i in range(n-2, -1, -1):
            scale = above_diag[i] / diag[i+1]
            # above_diag[i] -= diag[i+1] * scale
            self.result[i] -= self.result[i+1] * scale
            
        # Normalize by diagonal entries in tridiagonal matrix
        for i in range(n):
            scale = 1 / diag[i]
            # diag[i] *= scale
            self.result[i] *= scale

        # Assertions
        for i in range(n):
            assert np.isclose(np.sum(self.result[i]), 1.0)


    def get_bezier_handles(self, A: np.ndarray,) -> np.ndarray:
        """Given a sequence of n+1 anchors, produces the corresponding handles,
        using the pre-computed transformation matrix"""
        assert A.shape[0] == self.n + 1
        H1 = np.matmul(self.result, A)

        H2 = np.zeros(shape=(self.n, *A.shape[1:]))
        H2[0 : self.n - 1] = 2 * A[1:self.n] - H1[1:self.n]
        H2[self.n - 1] = 0.5 * (A[self.n] + H1[self.n - 1])

        handles = np.empty(shape=(2*self.n, *A.shape[1:]))
        handles[::2] = H1
        handles[1::2] = H2
        return handles

class ParametrizedHomotopy(m.Animation):
    """
    A function H: [0, L] x [0, 1] -> R^2.

    At the time this object is created, the input curve which is being homotoped
    already has a known number of anchors, P_0, P_1, ..., P_N with
    
    P_i(t) = H(L * i / N, t)

    Hence, the Bezier handle functions can themselves be defined, each one as a
    linear functional on these N+1 functions. Each Bezier handle is thus stored
    as an array of shape (N+1,) with sum of entries equal to 1.

    This is done by following the computation of bezier handles in 
    `utils.bezier.py:get_smooth_open_cubic_bezier_handle_points()'
    i.e. use Thomas's algorithm to fully invert the matrix.
    """
    def _make_calc(self, mobj: m.ParametricFunction):
        """Make the Bezier Handle calculator associated to this object. We assume
        the mobject's points form a single path consisting of n intervals,
        where each interval is a Bezier curve."""
        nppcc = mobj.n_points_per_cubic_curve
        n_steps = len(mobj.points) // nppcc
        self.calc = SmoothOpenPathBezierHandleCalculator(n_steps)
    
    def _interpolate_mobject_points(self, t: float, mobject: m.ParametricFunction) -> np.ndarray:
        """Homotopes the points forward to time t.
        We assume that the mobject's points form a single path consisting of n intervals,
        where each interval is a Bezier curve."""
        n = self.calc.n
        # TODO See if we can make this creation of anchors a bit more general and controlled
        # by the mobject instead.
        anchors = np.stack(
            [self.homotopy((mobject.t_max - mobject.t_min) * i / n, t) for i in range(n+1)],
            axis=0
            )
        handles = self.calc.get_bezier_handles(anchors)
        points = np.empty(shape=(4 * n, 3))
        points[::4] = anchors[:-1]
        points[1::4] = handles[::2]
        points[2::4] = handles[1::2]
        points[3::4] = anchors[1:]
        return points

    def __init__(
        self,
        # First coordinate is parametric function variable, second is time evolution
        homotopy: Callable[[float, float], np.ndarray],
        mobject: m.ParametricFunction,
        run_time: float = 3,
        apply_function_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.homotopy = homotopy
        self.mobject = mobject
        self._make_calc(mobject)
        self.apply_function_kwargs = (
            apply_function_kwargs if apply_function_kwargs is not None else {}
        )
        super().__init__(mobject, run_time=run_time, **kwargs)

    def function_at_time_t(self, t: float) -> tuple[float, float, float]:
        return lambda p: self.homotopy(*p, t)

    def interpolate_submobject(
        self,
        submobject: m.ParametricFunction,
        starting_submobject: m.ParametricFunction,
        alpha: float,
    ) -> None:
        submobject.points = starting_submobject.points
        interpolated_points = self._interpolate_mobject_points(alpha, submobject)
        submobject.points = interpolated_points


# class TestScene(m.Scene):
#     def construct(self):

#         def circle(theta: float):
#             return np.array([np.cos(theta), np.sin(theta), 0])

#         curve = m.ParametricFunction(circle, (0, 1, 0.05))
#         def htpy(theta: float, t: float):
#             return np.array([(1+t) * np.cos(theta), (1+t) * np.sin(theta), 0])
        
#         homotopy = ParametrizedHomotopy(htpy, curve, run_time=1.0)
#         self.add(curve)
#         self.play(homotopy)

# if __name__ == "__main__":
#     # Test Bezier handle calculator
#     n = 100
#     calc = SmoothOpenPathBezierHandleCalculator(n)

#     anchors = np.array([[np.cos(i / n), np.sin(i / n)] for i in range(n+1)])
#     handles = calc.get_bezier_handles(anchors)
#     print(anchors)
#     print(handles)
#     points = np.empty(shape=(3 * n + 1, 2))
#     points[::3] = anchors
#     points[1::3] = handles[::2]
#     points[2::3] = handles[1::2]
#     import matplotlib.pyplot as plt
#     plt.scatter(points[:, 0], points[:, 1])
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.show()