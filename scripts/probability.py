"""Visuals related to probability theory"""

from __future__ import annotations

import cmath
import math
from typing import Annotated, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
from manimlib import *
from scipy.special import laguerre


def gaussian_pdf(mean: float, std: float, t: float):
    """The probability density function for a Gaussian distribution with a given mean and standard deviation."""
    return math.exp(-((t - mean) ** 2) / (2 * std**2)) / (std * math.sqrt(TAU))


class PDFDiscreteScene(Scene):
    """
    Base class for scenes involving the visualization of a probability distribution on a finite set, using a bar graph
    or placing them.
    """

    def init_params(self):
        # Number of bins
        self.num_x = 7

        self.ax = Axes((0, self.num_x), (0, 1), width=5.0, x_axis_config={})
        self.ax.add(
            *[
                Tex("n", font_size=30).next_to(self.ax.x_axis, RIGHT, 0.2),
                Tex("p_n", font_size=30).next_to(self.ax.y_axis, UP, 0.2),
            ]
        )
        pass

    def construct(self):
        self.init_params()
        self.embed()


class PDFPositiveRealsScene(Scene):
    """
    Base class for scenes involving the visualization of a probability density function on the positive real numbers.
    Convolution of density functions is generically performed using the Laplace transform.
    """

    def construct(self):
        self.init_params()
        self.embed()

    def init_params(self):
        # Visualization of the probability distribution
        self.xmax = 5.0
        self.num_x = 200
        self.del_x = self.xmax / self.num_x
        self.xmin = self.del_x
        self.xspace = np.linspace(self.del_x, self.xmax, self.num_x)

        # Parameters for the Laplace transform
        self.smax = 10.0
        self.num_s = 400
        self.del_s = self.smax / self.num_s
        self.smin = self.del_s
        self.sspace = np.linspace(self.del_s, self.smax, self.num_s)

        self.laplace_matrix = np.array(
            [[math.exp(-x * s) for x in self.xspace] for s in self.sspace]
        )

        # Set up the axes for probability density functions.
        self.ymin = 0.0
        self.ymax = 8.0
        self.ax = Axes(
            (0, self.xmax), (self.ymin, self.ymax), width=10.0, x_axis_config={}
        )
        self.ax.add(
            *[
                Tex("x", font_size=30).next_to(self.ax.x_axis, RIGHT, 0.2),
                Tex("p(x)", font_size=30).next_to(self.ax.y_axis, UP, 0.2),
            ]
        )

        # Set up the axes for the Laplace transform
        self.symin = 0.0
        self.symax = 4.0
        self.s_ax = Axes(
            (0, self.smax),
            (self.symin, self.symax),
            width=10.0,
            x_axis_config={"tick_offset": 10.0, "include_ticks": False},
        ).next_to(self.ax, RIGHT, 2.0)

        self.s_ax.add(
            *[
                Tex("s", font_size=30).next_to(self.s_ax.x_axis, RIGHT, 0.2),
                Tex("\\hat{p}(s)", font_size=30).next_to(self.s_ax.y_axis, UP, 0.2),
            ]
        )

    def calculate_mean(self, fn_vals: np.ndarray) -> float:
        """Calculates the mean of a given probability density function based on input values.
        This is given by the formula int p(x) * x dx."""
        return np.sum(fn_vals * self.xspace) * self.del_x

    def calculate_variance(self, fn_vals: np.ndarray) -> float:
        """Calculates the variance of a given probability density function based on input values.
        This is given by the formula int p(x) * x^2 dx - (int p(x) * x dx)^2."""
        return (
            np.sum(fn_vals * np.pow(self.xspace, 2)) * self.del_x
            - self.calculate_mean(fn_vals) ** 2
        )

    def calculate_entropy(self, fn_vals: np.ndarray) -> float:
        """Calculates the entropy of a given probability density function based on input values.
        This is given by the formula -int p(x) * log(p(x)) dx."""
        return -np.sum(fn_vals * np.log(np.abs(fn_vals))) * self.del_x

    def fn_vals_to_pdf(self, fn_vals: np.ndarray, color: Color = BLUE) -> Polygon:
        """Given function values defined on xspace, produces a visual MObject representing the probability density function."""
        ax = self.ax
        return Polygon(
            *list(map(lambda t: ax.c2p(*t), zip(self.xspace, fn_vals))),
            ax.c2p(self.xmax, 0),
            ax.c2p(self.xmin, 0),
        ).set_style(stroke_width=1.0, fill_opacity=0.5, fill_color=color)

    def make_gibbs_curve(self):
        raise NotImplementedError


class PDFRealsScene(Scene):
    """
    Base class for scenes involving the visualization of a probability density function on the real numbers.
    Convolution of density functions is performed using the Fourier transform.
    """

    def construct(self):
        """
        A sample scene which demonstrates that
        - The Fourier transform of a Gaussian is again a Gaussian.
        - Applying the inverse Fourier transform returns us to the original distribution
        - You can calculate the convolution of two distributions by directly multiplying the Fourier coefficients.
        """
        self.init_params()
        self.add(self.ax, self.w_ax)
        std_1 = 1.0
        std_2 = 0.5

        g1_vals = np.array([gaussian_pdf(0, std_1, x) for x in self.xspace])
        g2_vals = np.array([gaussian_pdf(0, std_2, x) for x in self.xspace])
        g3_vals = np.array(
            [gaussian_pdf(0, math.sqrt(std_1**2 + std_2**2), x) for x in self.xspace]
        )

        g1_coeffs = self.fourier_transform(g1_vals)
        g2_coeffs = self.fourier_transform(g2_vals)
        g3_coeffs = g1_coeffs * g2_coeffs

        g1_vals_rec = self.inv_fourier_transform(g1_coeffs)
        g2_vals_rec = self.inv_fourier_transform(g2_coeffs)
        g3_vals_rec = self.inv_fourier_transform(g3_coeffs)

        c1 = self.fn_vals_to_pdf(g1_vals)
        c2 = self.fn_vals_to_pdf(g2_vals)
        c3 = self.fn_vals_to_pdf(g3_vals)
        c1_rec = self.fn_vals_to_pdf(g1_vals_rec)
        c2_rec = self.fn_vals_to_pdf(g2_vals_rec)
        c3_rec = self.fn_vals_to_pdf(g3_vals_rec)

        self.embed()
        fn_vals = np.array([gaussian_pdf(0, 1, x) for x in self.xspace])
        fn_coeffs = self.fourier_transform(fn_vals)
        fn_vals_2 = self.inv_fourier_transform(fn_coeffs)

        pdf_1 = self.fn_vals_to_pdf(fn_vals)

        pdf_fourier = self.fn_coeffs_to_pdf(fn_coeffs)
        pdf_2 = self.fn_vals_to_pdf(fn_vals_2)

        # Updaters
        std = ValueTracker(1.0)
        # pdf_1.add_updater(
        #     lambda mobj: mobj.become(
        #         self.fn_vals_to_pdf(
        #             np.array([gaussian_pdf(0, std.get_value(), x) for x in self.xspace])
        #         )
        #     )
        # )

        # pdf_fourier.add_updater(
        #     lambda mobj: mobj.become(
        #         self.fn_coeffs_to_pdf(
        #             self.fourier_transform(
        #                 np.array(
        #                     [gaussian_pdf(0, std.get_value(), x) for x in self.xspace]
        #                 )
        #             )
        #         )
        #     )
        # )
        self.embed()

    def init_params(self):
        # Domain over which probability density functions are defined
        # We assume input functions are (mostly) supported on this domain
        # and therefore this is a reasonable interval over which to integrate.
        self.xmin = -5.0
        self.xmax = 5.0
        self.num_x = 400
        self.xspace = np.linspace(self.xmin, self.xmax, self.num_x + 1)

        # Fourier domain
        # TODO Find a way to make the normalizations work out when the number of points is not equal to (xmax - xmin) * (wmax - wmin)
        self.del_x = (self.xmax - self.xmin) / self.num_x
        self.wmax = 20.0
        self.wmin = -20.0
        self.num_w = 400
        self.wspace = np.linspace(self.wmin, self.wmax, self.num_w + 1)
        self.del_w = (self.wmax - self.wmin) / self.num_w

        # Fourier transform matrix. First axis is w, second axis is x.
        # In theory, we should have adj(F) * F ~ I. In practice, these two matrices differ
        # because the integral of two different Fourier modes against each other is computed
        # over a finite domain here (i.e., (xmin, xmax)), and is therefore not over an
        # integer number of periods -- but because *enough* periods are taken, the difference is small.
        self.fourier_matrix = np.array(
            [
                [cmath.exp(TAU * complex(0, 1) * w * x) for x in self.xspace]
                for w in self.wspace
            ]
        )

        # Set up the axes for probability density functions.
        self.ymin = 0.0
        self.ymax = 4.0
        self.ax = Axes(
            (self.xmin, self.xmax), (self.ymin, self.ymax), width=10.0, x_axis_config={}
        )
        self.ax.add(
            *[
                Tex("x", font_size=30).next_to(self.ax.x_axis, RIGHT, 0.2),
                Tex("p(x)", font_size=30).next_to(self.ax.y_axis, UP, 0.2),
            ]
        )

        # Set up the axes for the Fourier transform
        self.wymin = 0.0
        self.wymax = 4.0
        self.w_ax = Axes(
            (self.wmin, self.wmax),
            (self.wymin, self.wymax),
            width=10.0,
            x_axis_config={"tick_offset": 10.0, "include_ticks": False},
        ).next_to(self.ax, RIGHT, 2.0)

        self.w_ax.add(
            *[
                Tex("w", font_size=30).next_to(self.w_ax.x_axis, RIGHT, 0.2),
                Tex("\\hat{p}(x)", font_size=30).next_to(self.w_ax.y_axis, UP, 0.2),
            ]
        )

    def calculate_entropy(self, fn_vals: np.ndarray) -> float:
        """Calculates the entropy of a given probability density function based on input values.
        This is given by the formula -int p(x) * log(p(x)) dx."""
        return -np.sum(fn_vals * np.log(np.abs(fn_vals))) * self.del_x

    def calculate_fisher_information(self, fn_vals: np.ndarray) -> float:
        """Calculates the fisher information of a given probability density function based on input values.
        This is given by the formula int p'(x)^2 / p(x) dx."""
        abs_vals = np.abs(fn_vals)
        d_vals = (abs_vals[1:] - abs_vals[:-1]) / self.del_x
        vals = (abs_vals[1:] + abs_vals[:-1]) / 2
        return np.sum(np.pow(d_vals, 2) / vals) * self.del_x

    def calculate_mean(self, fn_vals: np.ndarray) -> float:
        """Calculates the mean of a given probability density function based on input values.
        This is given by the formula int p(x) * x dx."""
        return np.sum(fn_vals * self.xspace) * self.del_x

    def calculate_variance(self, fn_vals: np.ndarray) -> float:
        """Calculates the variance of a given probability density function based on input values.
        This is given by the formula int p(x) * x^2 dx - (int p(x) * x dx)^2."""
        return (
            np.sum(fn_vals * np.pow(self.xspace, 2)) * self.del_x
            - self.calculate_mean(fn_vals) ** 2
        )

    def fn_vals_to_pdf(self, fn_vals: np.ndarray, color: Color = BLUE) -> Polygon:
        """Given function values defined on xspace, produces a visual MObject representing the probability density function."""
        ax = self.ax
        return Polygon(
            *list(map(lambda t: ax.c2p(*t), zip(self.xspace, fn_vals))),
            ax.c2p(self.xmax, 0),
            ax.c2p(self.xmin, 0),
        ).set_style(stroke_width=1.0, fill_opacity=0.5, fill_color=color)

    def fn_coeffs_to_pdf(self, fn_coeffs: np.ndarray, color: Color = RED) -> Polygon:
        """Given function Fourier coefficients defined on wspace, produces a visual MObject representing the graph."""
        ax = self.w_ax
        return Polygon(
            *list(map(lambda t: ax.c2p(*t), zip(self.wspace, fn_coeffs))),
            ax.c2p(self.wmax, 0),
            ax.c2p(self.wmin, 0),
        ).set_style(stroke_width=1.0, fill_opacity=0.5, fill_color=color)

    def fourier_transform(self, fn_vals: np.ndarray) -> np.ndarray:
        """Given function values defined on xspace, returns fourier coefficients defined on wspace.
        Computes each Fourier coefficient via a Riemann sum."""
        return (self.fourier_matrix @ fn_vals) * self.del_x

    def inv_fourier_transform(self, fn_coeffs: np.ndarray) -> np.ndarray:
        """Given fourier coefficients defined on wspace, returns fourier coefficients defined on wspace."""
        return (np.conj(self.fourier_matrix).T @ (fn_coeffs / self.del_x)) / (
            self.num_x
        )

    def make_gaussian_curve(
        self, mean: float, std: float, num_pts: int = 100
    ) -> ParametricCurve:
        """Constructs the curve representing the PDF of a Gaussian distribution."""
        ax = self.ax
        return ParametricCurve(
            lambda t: ax.c2p(t, gaussian_pdf(mean, std, t)),
            (self.xmin, self.xmax, (self.xmax - self.xmin) / num_pts),
        )

    def make_pdf_curve(
        self, fn: Callable[[float], float], num_pts: int = 100
    ) -> Polygon:
        """Constructs the curve for a given function. Makes jagged."""
        ax = self.ax
        return Polygon(
            *[ax.c2p(t, fn(t)) for t in np.linspace(self.xmin, self.xmax, num_pts)]
        )

    def make_gaussian_area(
        self, mean: float, std: float, num_pts: int = 100
    ) -> Polygon:
        """Constructs the area under the PDF of a Gaussian distribution."""
        ax = self.ax
        return Polygon(
            *[
                ax.c2p(t, gaussian_pdf(mean, std, t))
                for t in np.linspace(self.xmin, self.xmax, num_pts)
            ],
            ax.c2p(self.xmax, 0),
            ax.c2p(self.xmin, 0),
        ).set_style(stroke_opacity=0.0, fill_opacity=0.5, fill_color=BLUE)

    def make_pdf_area(
        self, fn: Callable[[float], float], num_pts: int = 100
    ) -> Polygon:
        """Constructs the area under the PDF of a Gaussian distribution."""
        ax = self.ax
        return Polygon(
            *[ax.c2p(t, fn(t)) for t in np.linspace(self.xmin, self.xmax, num_pts)],
            ax.c2p(self.xmax, 0),
            ax.c2p(self.xmin, 0),
        ).set_style(stroke_opacity=0.0, fill_opacity=0.5, fill_color=BLUE)


class OUProcess(PDFRealsScene):
    """
    Animates the flow of (the PDF of) an arbitrary mean-zero random variable X on R
    along the Ornstein-Uhlenbeck evolution

    X^(t) = e^{-t}X + sqrt(1 - e^{-2t})N

    towards a normal distribution N with mean zero and the same variance as X.
    The entropy H as a function of time increases along the way via

    d/dt H(X^(t)) = J(X^(t))

    where J is the standardized Fisher information.
    """

    def construct(self):
        self.init_params()
        ax = self.ax
        self.add(ax)
        # self.add(self.w_ax)

        # Example probability density function: two spikes at +1 and -1.
        width = 0.18

        def fn(t):
            if abs(abs(t) - 1.0) <= width / 2:
                return 0.5 / width
            return 0.0

        # Define Fourier coefficient function, i.e. its integral against e^{2πiwx}.
        def fn_fourier(w):
            x4 = 1.0 + width / 2
            x3 = 1.0 - width / 2
            x2 = -1.0 + width / 2
            x1 = -1.0 - width / 2

            if w == 0:
                return 1.0
            return (
                (0.5 / width)
                * (complex(0, 1) / (TAU * w))
                * (
                    cmath.exp(TAU * complex(0, 1) * w * x1)
                    - cmath.exp(TAU * complex(0, 1) * w * x2)
                    + cmath.exp(TAU * complex(0, 1) * w * x3)
                    - cmath.exp(TAU * complex(0, 1) * w * x4)
                )
            )

        # Make arrays of function values, and the Fourier coefficients.
        fn_vals = np.array([fn(x) for x in self.xspace])
        fn_coeffs = np.array([fn_fourier(w) for w in self.wspace])

        curve = self.fn_vals_to_pdf(fn_vals)
        curve_fourier = self.fn_coeffs_to_pdf(fn_coeffs)
        self.add(curve)
        self.wait()

        def convolve_with_gaussian(s: float):
            """Produces the function values for sqrt(1 - s^2) * fn + s * G, where G is
            a normal distribution with mean 0 and variance 1."""
            if s <= 0.0:
                return fn_vals

            elif s >= 1.0:
                return np.array([gaussian_pdf(0, 1.0, x) for x in self.xspace])

            # Make the Gaussian
            gaussian_vals = np.array([gaussian_pdf(0, s, x) for x in self.xspace])
            gaussian_coeffs = self.fourier_transform(gaussian_vals)

            # Do the appropriate scaling on fn_coeffs, by stretching out by a factor of sqrt(1 - s^2).
            # Issue: when s gets close to 1, this function is
            stretched_fn_coeffs = np.array(
                [fn_fourier(w * math.sqrt(1 - s**2)) for w in self.wspace]
            )

            conv_coeffs = stretched_fn_coeffs * gaussian_coeffs
            conv_vals = self.inv_fourier_transform(conv_coeffs)
            return conv_vals

        # Define evolving objects
        t_tracker = ValueTracker(0.0001)
        conv_vals = ValueTracker(
            convolve_with_gaussian(math.sqrt(1 - math.exp(-2 * t_tracker.get_value())))
        )
        conv_vals.add_updater(
            lambda mobj: mobj.set_value(
                convolve_with_gaussian(
                    math.sqrt(1 - math.exp(-2 * t_tracker.get_value()))
                )
            )
        )
        conv_curve = self.fn_vals_to_pdf(
            convolve_with_gaussian(math.sqrt(1 - math.exp(-2 * t_tracker.get_value()))),
            GREEN,
        )
        self.add(conv_vals)

        # t-value
        t_label = VGroup()
        t_label.add(Tex("t="))
        t_label.add(
            DecimalNumber(num_decimal_places=4)
            .add_updater(lambda mobj: mobj.set_value(t_tracker.get_value()))
            .next_to(t_label[0], RIGHT, 0.3)
        )
        t_label.set_height(0.3).next_to(self.ax, UR, -1.5)

        # entropy value
        h_label = VGroup()
        h_label.add(Tex("H(X^{(t)})="))
        h_label.add(
            DecimalNumber(num_decimal_places=4)
            .add_updater(
                lambda mobj: mobj.set_value(
                    self.calculate_entropy(conv_vals.get_value())
                )
            )
            .next_to(h_label[0], RIGHT, 0.3)
        )
        h_label.set_height(0.3).next_to(t_label, DOWN, 0.5)

        # fisher information
        j_label = VGroup()
        j_label.add(Tex("J(X^{(t)})="))
        j_label.add(
            DecimalNumber(num_decimal_places=4)
            .add_updater(
                lambda mobj: mobj.set_value(
                    self.calculate_fisher_information(conv_vals.get_value())
                )
            )
            .next_to(j_label[0], RIGHT, 0.3)
        )
        j_label.set_height(0.3).next_to(h_label, DOWN, 0.5)

        # Graphic showing the idea in the real vector space of probability distributions
        # (i.e. functions whose domain is the event space and codomain is the outcome space.)
        imag_ax = Axes(
            (-0.1, 1.1), (-0.1, 1.1), axis_config={"include_ticks": False}
        ).set_width(1.5)
        imag_ax_components = [
            Tex("\\mathcal{N}(0, \\sigma)", font_size=24).next_to(
                imag_ax.x_axis, RIGHT, 0.3
            ),
            Tex("X", font_size=24).next_to(imag_ax.y_axis, UP, 0.3),
            Arrow(imag_ax.c2p(0, 0), imag_ax.c2p(0, 1))
            .set_color(GREEN)
            .add_updater(
                lambda arr: arr.put_start_and_end_on(
                    imag_ax.c2p(0, 0),
                    imag_ax.c2p(
                        math.sqrt(1 - math.exp(-2 * t_tracker.get_value())),
                        math.exp(-t_tracker.get_value()),
                    ),
                )
            ),
            Tex("X^{(t)}", font_size=24).add_updater(
                lambda mobj: mobj.next_to(
                    imag_ax.c2p(
                        math.sqrt(1 - math.exp(-2 * t_tracker.get_value())),
                        math.exp(-t_tracker.get_value()),
                    ),
                    RIGHT,
                    0.1,
                )
            ),
        ]
        imag_ax.add(*imag_ax_components)
        imag_ax.next_to(self.ax, UL, -1.0)

        self.play(
            FadeIn(conv_curve),
            FadeOut(curve),
            FadeIn(t_label),
            FadeIn(h_label),
            FadeIn(j_label),
            FadeIn(imag_ax),
        )

        self.wait()

        conv_curve.add_updater(
            lambda mobj: mobj.become(self.fn_vals_to_pdf(conv_vals.get_value(), GREEN))
        )
        self.play(
            t_tracker.animate.set_value(1.5),
            run_time=20.0,
            rate_func=linear,
        )
        self.embed()


class CIRProcess(PDFPositiveRealsScene):
    """
    Given an initial probability distribution p_0(x) for x > 0, models the Cox-Ingersoll-Ross (CIR)
    process (AKA Feller diffusion) to evolve the probability distribution over time

    d/dt p_t(x) = d/dx ((x/λ - 1)p_t(x)) + (d/dx)^2 p_t(x)

    where λ is the target mean. The long-term steady state is the Gibbs/exponential distribution
    g(x) = (1/λ) * exp(-x/λ), which is the maximal-entropy distribution with mean λ. The rate
    of change of the entropy at time t is equal to the relative Fisher information I_rel(p_t || g).
    The process can also be viewed as the gradient flow of the KL-divergence D_{KL}(p || g), weighted by x.

    This simulation shows the entropy steadily increasing.
    """

    def init_params(self):
        super().init_params()

        # Stable finite-difference timestep
        self.del_t = 0.4 * (self.del_x**2) / self.xmax  # stability condition

    def evolve_vals_finite_difference(
        self,
        mean: float,
        fn_vals: np.ndarray,
        substeps: int = 20,
        dt: Optional[float] = None,
    ):
        """Simulates the CIR process using finite-difference method."""
        if dt is None:
            dt = self.del_t

        for _ in range(substeps):
            dp = np.zeros(self.num_x)
            dp[0] = fn_vals[0] / mean + (fn_vals[1] - fn_vals[0]) / self.del_x
            dp[-1] = 0.0
            d_vals = (fn_vals[2:] - fn_vals[:-2]) / (2 * self.del_x)
            d2_vals = (fn_vals[2:] + fn_vals[:-2] - 2 * fn_vals[1:-1]) / (self.del_x**2)
            dp[1:-1] = (
                fn_vals[1:-1] / mean
                + (1 + self.xspace[1:-1] / mean) * d_vals
                + self.xspace[1:-1] * d2_vals
            )
            fn_vals += dt * dp
            fn_vals = fn_vals / (self.del_x * np.sum(fn_vals))  # renormalize
        return fn_vals

    def decompose_into_laguerre(
        self, fn_vals: np.ndarray, mean: float, max_deg: int = 200
    ):
        """Decomposes a function p(x) as a linear combination
        p(x) = exp(-x/λ) * sum_n (a_n * L_n(x/λ))

        and then evolve with time"""
        laguerre_array = np.stack(
            [laguerre(n)(self.xspace / mean) for n in range(max_deg + 1)],
            axis=0,
        )

        laguerre_coeffs = np.array(
            [
                np.sum(laguerre_array[n] * fn_vals) * self.del_x / mean
                for n in range(max_deg + 1)
            ]
        )
        timer = ValueTracker(0.02)

        fn_curve = self.fn_vals_to_pdf(
            np.exp(-self.xspace / mean)
            * sum(
                np.exp(-n * timer.get_value() / mean)
                * laguerre_coeffs[n]
                * laguerre_array[n]
                for n in range(max_deg + 1)
            ),
            GREEN,
        )

        fn_curve.add_updater(
            lambda mobj: mobj.become(
                self.fn_vals_to_pdf(
                    np.exp(-self.xspace / mean)
                    * sum(
                        np.exp(-n * timer.get_value() / mean)
                        * laguerre_coeffs[n]
                        * laguerre_array[n]
                        for n in range(max_deg + 1)
                    ),
                    GREEN,
                )
            )
        )

    def construct(self):
        """Defines an initial"""
        self.init_params()
        self.add(self.ax)

        # Define starting function as a rectangular bar
        mean = ValueTracker(2.0)
        width = 0.4

        def initial_function(x: float):
            if abs(x - mean.get_value()) < width / 2:
                return 1 / width
            return 0.0

        # Calculate initial PDF values
        init_vals = np.array([initial_function(x) for x in self.xspace])
        init_curve = self.fn_vals_to_pdf(init_vals)
        self.add(init_curve)

        # OPTION 1: Finite-difference according to the PDE, with a specified mean
        fn_vals = np.array([initial_function(x) for x in self.xspace])
        curve = self.fn_vals_to_pdf(fn_vals, GREEN)
        self.add(curve)
        self.wait()

        for _ in range(100):
            fn_vals = self.evolve_vals_finite_difference(mean.get_value(), fn_vals)
            curve.become(self.fn_vals_to_pdf(fn_vals, GREEN))
            self.wait(0.005)
        self.embed()


class Entropy(Scene):
    """Depicting the entropy of a distribution."""

    pass


class RelativeEntropy(Scene):
    """Depicting the relative entropy D(P|Q) (also known as the KL-divergence)."""

    pass
