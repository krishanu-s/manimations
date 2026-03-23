"""
This is intended as an addition to the manim library.
"""

from typing import Callable, Any
from typing_extensions import Self
import numpy as np
import manim as m


# TODO Move these into their own file called "bezier_utils.py"

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
    
    def get_smooth_global_bezier_function(self, anchors: np.ndarray) -> Callable[[float], np.ndarray]:
        """TODO Similar to the function for SmoothClosedPathBezierHandleCalculator"""
        raise NotImplementedError

class SmoothClosedPathBezierHandleCalculator:
    """The same as SmoothOpenPathBezierHandleCalculator but for closed loops."""
    def __init__(self, n: int):
        """
        Calculates and stores the n-by-(n+1) transformation matrix used for computing
        the first Bezier handles of a sequence of n+1 anchor points, as a function of
        the n+1 anchor points. This is computed as A^{-1}B, where A is an n-by-n
        tridiagonal matrix and B is an n-by-(n+1) matrix."""
        self.n = n

        # Tridiagonal matrix which is to be inverted
        below_diag = np.array([1.]*(n-1)) # Below diagonal
        diag = np.array([3.] + [4.]*(n-2) + [3.]) # Main diagonal
        above_diag = np.array([1.]*(n-1)) # Above diagonal

        # n-by-(n+1) matrix to be computed
        self.result = np.zeros(shape=(n, n+1))
        for i in range(n):
            self.result[i, i], self.result[i, i+1] = 4, 2

        # Computation of q, described in extra step below
        v = np.array([1.] + [0.] * (n-2) + [1.])
        q = np.array([1.] + [0.] * (n-2) + [1.])
        
        # Eliminate lower-triangular entries in tridiagonal matrix
        for i in range(n-1):
            scale = below_diag[i] / diag[i]
            diag[i+1] -= above_diag[i] * scale
            self.result[i+1] -= self.result[i] * scale
            q[i+1] -= q[i] * scale


        # Eliminate upper-triangular entries in tridiagonal matrix
        for i in range(n-2, -1, -1):
            scale = above_diag[i] / diag[i+1]
            self.result[i] -= self.result[i+1] * scale
            q[i] -= q[i+1] * scale
            
        # Normalize by diagonal entries in tridiagonal matrix
        for i in range(n):
            scale = 1 / diag[i]
            self.result[i] *= scale
            q[i] *= scale

        # Extra step: left-multiply the result by (I + qv^t)^{-1} = I - \frac{1}{1 + v^tq} qv^t, where v = [1 0 0 ... 0 1] and q = T^{-1}v
        m = np.eye(n) - (1 / (1 + np.dot(v, q))) * np.outer(q, v)
        self.result = np.matmul(m, self.result)

    def get_bezier_handles(self, anchors: np.ndarray,) -> np.ndarray:
        """Given a sequence of n anchors, produces the corresponding handles,
        using the pre-computed transformation matrix."""
        if anchors.shape[0] == self.n:
            A = np.concatenate((anchors, np.expand_dims(anchors[0], axis=0)), axis=0)
        elif anchors.shape[0] == self.n + 1:
            assert np.allclose(anchors[0], anchors[-1])
            A = anchors
        else:
            raise NotImplementedError("Wrong number of anchors")
        H1 = np.matmul(self.result, A)

        H2 = np.zeros(shape=(self.n, *A.shape[1:]))
        H2[0 : self.n - 1] = 2 * A[1:self.n] - H1[1:self.n]
        H2[self.n - 1] = 2 * A[0] - H1[0]

        handles = np.empty(shape=(2*self.n, *A.shape[1:]))
        handles[::2] = H1
        handles[1::2] = H2
        return handles
    
    def get_anchors_and_handles(self, anchors: np.ndarray) -> np.ndarray:
        assert len(anchors.shape) == 2
        if anchors.shape[0] == self.n:
            A = np.concatenate((anchors, np.expand_dims(anchors[0], axis=0)), axis=0)
        elif anchors.shape[0] == self.n + 1:
            assert np.allclose(anchors[0], anchors[-1])
            A = anchors
        else:
            raise NotImplementedError("Wrong number of anchors")


        # Generate the Bezier handles H_1, H_2, ... and stack into an array of shape (4, N, 3)
        # [P_0, H_1, H_2, P_1]
        # [P_1, H_3, H_4, P_2]
        # [P_2, H_5, H_6, P_3]
        # [...]
        handles = self.get_bezier_handles(A)
        anchors_and_handles = np.zeros((4, self.n, 3))
        anchors_and_handles[0, :, :] = A[:-1]
        anchors_and_handles[1, :, :] = handles[::2]
        anchors_and_handles[2, :, :] = handles[1::2]
        anchors_and_handles[3, :, :] = A[1:]

        return anchors_and_handles


    def get_smooth_global_bezier_function(self, anchors: np.ndarray) -> Callable[[float], np.ndarray]:
        """Given n points P_0, P_1, ..., P_{N-1} in the plane, draws a smooth closed Bezier curve through them.
        Outputs the function f: [0, 1] -> R^2 which outputs this curve such that f(k / N) = P_k."""
        anchors_and_handles = self.get_anchors_and_handles(anchors)

        # Given all of the anchors and handles for the closed curve, defines the global Bezier function f: [0, 1] -> R^3.
        # along with its tangent vector
        # This is similar to the function m.bezier.bezier in the cubic case.
        def global_bezier(a: float) -> float:
            k = (int(a * self.n)) % self.n
            t = (a * self.n) % 1

            p = anchors_and_handles[0, k, :]
            h1 = anchors_and_handles[1, k, :]
            h2 = anchors_and_handles[2, k, :]
            q = anchors_and_handles[3, k, :]

            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt

            return mt3 * p + 3 * t * mt2 * h1 + 3 * t2 * mt * h2 + t3 * q
        
        def global_bezier_derivative(a: float) -> float:
            k = (int(a * self.n)) % self.n
            t = (a * self.n) % 1

            p = anchors_and_handles[0, k, :]
            h1 = anchors_and_handles[1, k, :]
            h2 = anchors_and_handles[2, k, :]
            q = anchors_and_handles[3, k, :]

            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt

            return (3 * (-mt2) * p + 3 * (mt2 - 2 * t * mt) * h1 + 3 * (2 * t * mt - t2) * h2 + 3 * t2 * q) * self.n
            
        
        return global_bezier, global_bezier_derivative

class ParametrizedHomotopy(m.Animation):
    """
    A function H: [tmin, tmax] x [0, 1] -> R^2.

    At the time this object is created, the input curve which is being homotoped
    already has a known number of anchors, P_0, P_1, ..., P_N with
    
    P_i(a) = H(tmin + (tmax - tmin) * (i / N), a)

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
            [self.homotopy(mobject.t_min + (mobject.t_max - mobject.t_min) * i / n, t) for i in range(n+1)],
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