"""Visuals related to probability theory"""

from __future__ import annotations

import cmath
import math
from typing import Annotated, Dict, Iterable, Literal, Tuple, Union

import numpy as np
from manimlib import *


def gaussian_pdf(mean: float, std: float, t: float):
    return math.exp(-((t - mean) ** 2) / (2 * std**2)) / (std * math.sqrt(TAU))


class PDFScene(Scene):
    """Base class for scenes involving the visualization of a probability density function"""

    def construct(self):
        # Sample scene, showing that the fourier, and then inverse fourier transform of
        # a Gaussian produces a Gaussian.
        self.init_pdf()
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

    def init_pdf(self):
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

        # Set up the axes for the Fourier transform
        self.wymin = 0.0
        self.wymax = 4.0
        self.w_ax = Axes(
            (self.wmin, self.wmax),
            (self.wymin, self.wymax),
            width=10.0,
            x_axis_config={"tick_offset": 10.0, "include_ticks": False},
        ).next_to(self.ax, RIGHT, 2.0)

    def fn_vals_to_pdf(self, fn_vals: np.ndarray, color: Color = BLUE) -> Polygon:
        """Given function values defined on xspace, produces a visual MObject representing the probability density function."""
        ax = self.ax
        return Polygon(
            *list(map(lambda t: ax.c2p(*t), zip(self.xspace, fn_vals))),
            ax.c2p(self.xmax, 0),
            ax.c2p(self.xmin, 0),
        ).set_style(stroke_width=1.0, fill_opacity=0.5, fill_color=color)

    def fn_coeffs_to_pdf(self, fn_coeffs: np.ndarray, color: Color = RED) -> Polygon:
        """Given function fourier coefficients defined on wspace, produces a visual MObject representing the graph."""
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


class OrnsteinUhlenbeck(PDFScene):
    """
    Animates the flow of (the PDF of) an arbitrary mean-zero random variable X on R
    along the Ornstein-Uhlenbeck evolution

    X^(t) = e^{-t}X + sqrt(1 - e^{-2t})N

    towards a normal distribution N with mean zero and the same variance as X.
    The entropy H as a function of time increases along the way via

    d/dt H(X^(t)) = J(X^(t))

    where J is the standardized Fisher information.
    """

    def convolve_with_gaussian(
        self,
        fn: Callable[[float], float],
        bd: float,  # Assumes the function is supported on [-bd, bd]
        mean: float,
        std: float,
        step_size: float = 1e-2,
    ):
        """Convolves the probability distribution for the given function with compact support
        against the PDF of a Gaussian with the given mean and standard deviation. Does so by
        computing a grid of values for each functionand computing the convolution."""
        xmin = self.xmin
        xmax = self.xmax
        fx_min, fx_max = -bd, bd

        # Calculate values of the two functions being convoluted within this range
        f_imin = math.floor(fx_min / step_size)
        f_imax = math.ceil(fx_max / step_size)
        num_fn_vals = f_imax - f_imin + 1
        fn_array = np.array([fn(i * step_size) for i in range(f_imin, f_imax + 1)])

        gaussian_imin = math.floor((xmin + fx_min) / step_size)
        gaussian_imax = math.ceil((xmax + fx_max) / step_size)
        num_gaussian_vals = gaussian_imax - gaussian_imin + 1
        gaussian_array = np.array(
            [
                gaussian_pdf(mean, std, i * step_size)
                for i in range(gaussian_imin, gaussian_imax + 1)
            ]
        )

        # Calculate the function values of the convolution.
        # Zero-th entry is the value of the function at step_size * (gaussian_imin + f_imax)
        # while the last entry is the value of the function at step_size * (gaussian_imax + f_imin)
        conv_array = step_size * np.array(
            [
                np.sum(fn_array[::-1] * gaussian_array[i : i + num_fn_vals])
                for i in range(num_gaussian_vals - num_fn_vals + 1)
            ]
        )
        num_conv_vals = num_gaussian_vals - num_fn_vals
        conv_xmin = step_size * (gaussian_imin + f_imax)
        conv_xmax = step_size * (gaussian_imax + f_imin)

        # Turn it into an actual PDF curve
        x_vals = np.linspace(conv_xmin, conv_xmax, num_conv_vals)
        ax = self.ax
        conv_curve_and_area = Polygon(
            *[ax.c2p(x_vals[i], conv_array[i]) for i in range(num_conv_vals)],
            ax.c2p(x_vals[-1], 0),
            ax.c2p(x_vals[0], 0),
        ).set_style(
            stroke_opacity=1.0, stroke_width=1.5, fill_opacity=0.5, fill_color=BLUE
        )

        return conv_curve_and_area

    def construct(self):
        self.init_pdf()
        ax = self.ax
        self.add(ax)
        # self.add(self.w_ax)

        # Generate a probability density function of choice.
        width = 0.18

        def fn(t):
            if abs(abs(t) - 1.0) <= width / 2:
                return 0.5 / width
            return 0.0

        # Define Fourier coefficient function for the function, i.e. its integral against e^{TAU * iwx}.
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

        # Do the evolution
        theta_tracker = ValueTracker(0.03)
        s_tracker = ValueTracker(0.03)
        conv_curve = self.fn_vals_to_pdf(
            convolve_with_gaussian(math.sin(theta_tracker.get_value())), GREEN
        )

        s_label = VGroup()
        s_label.add(Tex("s="))
        s_label.add(
            DecimalNumber()
            .add_updater(lambda mobj: mobj.set_value(math.sin(theta_tracker.get_value())))
            .next_to(s_label[0], RIGHT, 0.3)
        )
        s_label.set_height(0.3).next_to(self.ax, UR, -1.5)

        self.play(FadeIn(conv_curve), FadeOut(curve), FadeIn(s_label))

        # TODO Make a number line evoking the idea of interpolation between the
        # base distribution and a Gaussian.

        self.wait()

        conv_curve.add_updater(
            lambda mobj: mobj.become(
                self.fn_vals_to_pdf(
                    convolve_with_gaussian(math.sin(theta_tracker.get_value())), GREEN
                )
            )
        )
        self.play(theta_tracker.animate.set_value(PI/2 - 0.01), run_time=10.0, rate_func=linear)
        self.embed()

        # # Do the same for a Gaussian
        # std = 0.1
        # gaussian_vals = np.array([gaussian_pdf(0, std, x) for x in self.xspace])
        # gaussian_coeffs = self.fourier_transform(gaussian_vals)

        # # Optional: Add graph of the gaussian
        # gaussian_curve = self.fn_vals_to_pdf(gaussian_vals, GREEN)
        # gaussian_curve_fourier = self.fn_coeffs_to_pdf(gaussian_coeffs, ORANGE)

        # # Do the convolution
        # conv_coeffs = fn_coeffs * gaussian_coeffs
        # conv_vals = self.inv_fourier_transform(conv_coeffs)
        # conv_curve = self.fn_vals_to_pdf(conv_vals, GREEN)
        # conv_curve_fourier = self.fn_coeffs_to_pdf(conv_coeffs, ORANGE)

        # self.add(conv_curve)

        # self.embed()
        # # We might need to take a wider Fourier domain to detect the function itself.
        # wmin, wmax = -200.0, 200.0
        # xmin, xmax = self.xmin, self.xmax
        # num_pts = 1000
        # fourier_domain = np.linspace(wmin, wmax, num_pts)
        # pos_domain = np.linspace(xmin, xmax, num_pts)

        # # Fourier transform matrix. First axis is w, second axis is x
        # fourier_transform = np.array(
        #     [
        #         [cmath.exp(complex(0, 1) * w * x) for x in pos_domain]
        #         for w in fourier_domain
        #     ]
        # )

        # def rotate_to_gaussian(s: float):
        #     """Computes the PDF of the function sf + sqrt(1-s^2)N(0, 1)
        #     using the Fourier transform."""
        #     # Calculate the Fourier coefficients of the function
        #     fn_fourier_coeffs = np.array(
        #         [fn_fourier(w / s) / s for w in fourier_domain]
        #     ).astype(np.float64)

        #     # Compute the Fourier coefficients of a Gaussian of variance sqrt(1-s^2).
        #     std = math.sqrt(1 - s**2)
        #     gaussian_fourier_coeffs = np.array(
        #         [gaussian_pdf(0, 1 / s, w) for w in fourier_domain]
        #     )

        #     # Directly multiply them to get the Fourier coefficients of the convolution
        #     conv_fourier_coeffs = fn_fourier_coeffs * gaussian_fourier_coeffs
        #     conv_values = np.sum(
        #         fourier_transform * conv_fourier_coeffs[:, np.newaxis], axis=0
        #     )
        #     return conv_values.astype(np.float64)

        # conv_values = rotate_to_gaussian(0.9)
        # conv_curve = Polygon(
        #     *[
        #         ax.c2p(pos_domain[i], conv_values[i].astype(np.float64))
        #         for i in range(num_pts)
        #     ],
        #     ax.c2p(xmax, 0),
        #     ax.c2p(xmin, 0),
        # ).set_style(stroke_width=1.0, fill_opacity=0.5, fill_color=BLUE)

        # self.embed()

        # # Set the range within which the function is supported
        # bd = 1.0 + 0.5 * width

        # # pdf_curve = self.make_pdf_curve(fn, 400)
        # # pdf_area = self.make_pdf_area(fn, 400)

        # # Tracker which is exponential of time value
        # exp_time = ValueTracker(1.0001)
        # exp_time_label = VGroup()
        # exp_time_label.add(Tex("t = "))
        # exp_time_label.add(
        #     DecimalNumber(num_decimal_places=5)
        #     .add_updater(lambda mobj: mobj.set_value(math.log(exp_time.get_value())))
        #     .next_to(exp_time_label[0], RIGHT, 0.3)
        # )
        # exp_time_label.set_height(0.3)
        # exp_time_label.next_to(ax, UR, -1.0)
        # self.add(exp_time_label)

        # # Calculate the convolution with a Gaussian according to the Ornstein-Uhlenbeck process
        # conv_curve_and_area = self.convolve_with_gaussian(
        #     lambda x: 1.0001 * fn(x * 1.0001),
        #     bd / 1.0001,
        #     0.0,
        #     1 / math.sqrt(1 - 1 / math.pow(1.0001, 2)),
        #     step_size=0.002 / 1.0001,
        # )

        # self.add(conv_curve_and_area)
        # self.wait()

        # # Change it to the evolute X^(t) = e^{-t}X + sqrt(1 - e^{-2t})N

        # # Option 1: Add an updater, and tweak that
        # conv_curve_and_area.add_updater(
        #     lambda mobj: mobj.become(
        #         self.convolve_with_gaussian(
        #             lambda x: exp_time.get_value() * fn(x * exp_time.get_value()),
        #             bd / exp_time.get_value(),
        #             0.0,
        #             math.sqrt(1 - 1 / math.pow(exp_time.get_value(), 2)),
        #             step_size=0.002 / exp_time.get_value(),
        #         )
        #     )
        # )
        # self.play(exp_time.animate.set_value(1.001), rate_func=linear, run_time=3.0)
        # self.play(exp_time.animate.set_value(1.01), rate_func=linear, run_time=3.0)
        # self.play(exp_time.animate.set_value(1.1), rate_func=linear, run_time=3.0)
        # self.play(exp_time.animate.set_value(2.0), rate_func=linear, run_time=3.0)
        # self.play(exp_time.animate.set_value(5.0), rate_func=linear, run_time=3.0)

        # Option 2: Do animated transformations step-by-step
        # num_steps = 100
        # for et in np.logspace(math.log10(1.0001), math.log10(5.0), num_steps):
        #     self.play(
        #         exp_time.animate.set_value(et),
        #         conv_curve_and_area.animate.become(
        #             self.convolve_with_gaussian(
        #                 lambda x: et * fn(x * et),
        #                 bd / et,
        #                 0.0,
        #                 math.sqrt(1 - 1 / math.pow(et, 2)),
        #                 step_size=0.01 / et,
        #             )
        #         ),
        #         rate_func=linear,
        #         run_time=15.0 / num_steps,
        #     )

        # self.embed()


class GaussianCurve(PDFScene):
    def construct(self):
        self.init_pdf()
        ax = self.ax
        self.add(ax)

        pdf_curve = self.make_gaussian_curve(0.0, 1.0)
        pdf_area = self.make_gaussian_area(0.0, 1.0)

        mean = ValueTracker(0.0)
        std = ValueTracker(1.0)

        mean_label = VGroup()
        mean_label.add(Tex("\\mu ="))
        mean_val_label = DecimalNumber()
        mean_val_label.add_updater(lambda mobj: mobj.set_value(mean.get_value()))
        mean_val_label.next_to(mean_label[0], RIGHT, 0.3)
        mean_label.add(mean_val_label)
        mean_label.set_height(0.3)

        std_label = VGroup()
        std_label.add(Tex("\\sigma ="))
        std_val_label = DecimalNumber()
        std_val_label.add_updater(lambda mobj: mobj.set_value(std.get_value()))
        std_val_label.next_to(std_label[0], RIGHT, 0.3)
        std_label.add(std_val_label)
        std_label.set_height(0.3)
        std_label.next_to(mean_label, DOWN, 0.4)

        pdf_label = VGroup(mean_label, std_label)
        pdf_label.next_to(ax, UR, -1.0)

        pdf_curve.add_updater(
            lambda mobj: mobj.become(
                self.make_gaussian_curve(mean.get_value(), std.get_value())
            )
        )
        pdf_area.add_updater(
            lambda mobj: mobj.become(
                self.make_gaussian_area(mean.get_value(), std.get_value())
            )
        )

        self.add(pdf_label)
        self.add(pdf_curve, pdf_area)

        self.embed()
