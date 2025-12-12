"""Introducing ergodic theory through simple examples"""

import math
import numpy as np
import manim as m

class BarChart:
    def __init__(
        self,
        x_values: list,
        y_values: list,
        # Display size
        chart_w: float = 5.0,
        chart_h: float = 4.0,
        offset: np.ndarray = np.array([0., 0., 0])
    ):
        self.x_values = x_values
        self.y_values = y_values
        self.chart_w = chart_w
        self.chart_h = chart_h
        self.offset = offset
        self.bar_width = chart_w / len(x_values)

        # Make the axes
        self.x_axis = m.Line(
            np.array([0., 0., 0.]),
            np.array([chart_w, 0., 0.]),
            stroke_width=3.0
            ).shift(offset)
        self.y_axis = m.Line(
            np.array([0., 0., 0.]),
            np.array([0., chart_h, 0.]),
            stroke_width=3.0
            ).shift(offset)
        
        # Set the x values on the x axis
        num_x = len(x_values)
        x_tick_length = 0.2
        x_ticks = []
        x_labels = []
        for i, x in enumerate(x_values):
            # TODO Not sure if it's better to put the ticks in the middle of the bars vs between them.
            x_pos = offset + ((i + 0.5) / num_x) * np.array([chart_w, 0., 0.])
            x_ticks.append(m.Line(
                x_pos - np.array([0., x_tick_length / 2, 0.]),
                x_pos + np.array([0., x_tick_length / 2, 0.])
                ))
            # TODO Either set the font size adaptively, or curtail the number of them
            x_labels.append(m.Tex(f"{x}", font_size=24).move_to(x_pos - np.array([0., 0.4, 0])))

        self.x_labels = x_labels
        self.x_ticks = x_ticks

        num_y = len(y_values)
        y_tick_length = 0.2
        y_ticks = []
        y_lines = []
        y_labels = []
        for i, y in enumerate(y_values):
            y_pos = offset + ((i + 1) / num_y) * np.array([0., chart_h, 0.])
            y_ticks.append(m.Line(
                y_pos - np.array([y_tick_length / 2, 0., 0.]),
                y_pos + np.array([y_tick_length / 2, 0., 0.])
                ))
            y_lines.append(m.Line(y_pos, y_pos + np.array([chart_w, 0., 0.]), stroke_width=3.0, stroke_color=m.GRAY, stroke_opacity=0.5))
            # TODO Either set the font size adaptively, or curtail the number of them
            y_labels.append(m.Tex(f"{y:.1f}", font_size=24).move_to(y_pos - np.array([0.3, 0., 0])))

        self.y_labels = y_labels
        self.y_ticks = y_ticks
        self.y_lines = y_lines

    def make_bar_chart(self) -> m.VGroup():
        bar_chart = m.VGroup()
        bar_chart.add(
            self.x_axis, self.y_axis,
            *self.x_labels, *self.x_ticks,
            *self.y_labels, *self.y_ticks, *self.y_lines)
        return bar_chart
        
class LeadingDigitScene(m.ThreeDScene):
    def lead_digit_hist(self, n: int, base: float = 2.0) -> dict[int, int]:
        """Computes a histogram of the ratios of  leading digits for the first n powers of the given base."""
        hist = {i: 0 for i in range(1, 10)}
        p = 1.0
        hist[1] += 1
        for i in range(n - 1):
            p *= base
            if p >= 10:
                p /= 10
            hist[int(p)] += 1
        for i in range(1, 10):
            hist[i] /= n
        return hist

    def leading_digit_puzzle(self):
        """Scene for posing the question:
        
        `If you pick a random power of 2 and write its digits out in base-10, how likely is it that the *leading digit* of that number is 1? 2?'
        
        Along with numerical data."""
        self.clear()
        ## Posing the question as a teaser, then showing numerical data

        # Write out the question.
        # "Consider the powers of 2 from 1 to 2^{N-1} for some large N, and note down all of the leading digits when written in base-10. What's the percentage breakdown?"
        m1 = m.MathTex("1, 2, 4, 8, 16, 32, 64, 128, \ldots, 2^{N-1}").move_to(np.array([0, 3, 0]))
        m2 = m.Text("Consider the first N powers of two, written in base-10.\nHow often does each digit 1-9 appear as the leading digit in this list??", font_size=24).move_to(np.array([0, 2, 0]))
        self.play(m.FadeIn(m1), run_time=1.0)
        self.play(m.FadeIn(m2), run_time=1.0)

        # Fade out text above.
        self.play(m.FadeOut(m1), m.FadeOut(m2), run_time=1.0)

        # Calculate the desired histograms for several different values of N
        histograms = {}
        n_values = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 500000, 100000)
        for n in n_values:
            histograms[n] = self.lead_digit_hist(n)

        ## Make a bar chart.
        x_values = list(range(1, 10))
        y_values = list(np.linspace(0.2, 1.0, 5))
        chart_w = 5.0
        chart_h = 4.0
        offset = np.array([1.0, -1.0, 0])
        b = BarChart(x_values, y_values, chart_w, chart_h, offset)

        bar_chart = b.make_bar_chart()

        # Add bars to it
        # TODO Make this a method of the bar chart
        num_x = len(x_values)
        bar_width = chart_w / num_x
        def bar_center(height: float, x: int):
            return offset + np.array([0., height / 2, -5.]) + ((x-0.5) / num_x) * np.array([chart_w, 0., 0.])

        def make_bar(x: int):
            val_tracker = m.ValueTracker(histograms[20][x])
            bar = m.Rectangle(height = max(chart_h * val_tracker.get_value(), 1e-3), width = 0.9 * bar_width, color = m.ORANGE)
            bar.set_opacity(1.0)
            bar.move_to(bar_center(chart_h * val_tracker.get_value(), x))
            bar.add_updater(lambda mobj: mobj.stretch_to_fit_height(max(chart_h * val_tracker.get_value(), 1e-3)).move_to(
                        bar_center(chart_h * val_tracker.get_value(), x)))
            return bar, val_tracker
            
        x_bars_and_trackers = []
        for x in range(1, 10):
            x_bars_and_trackers.append(make_bar(x))
        x_bars, hist_val_trackers = zip(*x_bars_and_trackers)
        bar_chart.add(*x_bars)


        n_value_text = m.Tex(f"N = 10").move_to(np.array([3.8, -2.8, 0.]))



        # Define Mobject for displaying the numbers in a table
        numbers = m.VGroup()
        for i in range(10):
            numbers.add(m.Tex(f"{2**i}", font_size=20).move_to((-5.5, 3.3 - 0.35 * i, 0)))

        # TODO Fade in data for N=10, with the leading digits highlighted.
        self.play(
            m.FadeIn(bar_chart), m.FadeIn(n_value_text), m.FadeIn(numbers), run_time=2.0
        )
        self.play(m.Wait(1.0))

        # Animate transitions
        # TODO Do this further down, in steps
        for n in n_values[1:]:
            self.play(
                *[v.animate.set_value(histograms[n][i + 1]) for i, v in enumerate(hist_val_trackers)],
                m.Transform(n_value_text, m.Tex(f"N = {n}").move_to(np.array([3.8, -2.8, 0.])))
                )
        # TODO Fade in up to N = 100 and zoom out to display a 20-by-5 table (each up to 4 digits accuracy)
        
        # TODO Convert each number into a dot of the corresponding color, and then animate to N=1000 in a rectangular table, while animating the change in the histogram
        # TODO Fade out and then fade in for N=10,000 as a simple grid of dots, while animating the change in the histogram
        # TODO Fade out and then fade in for N=100,000 as a pixelated image, while animating the change in the histogram
        pass
    
    def history_of_ergodic_theory(self):
        """Some history of ergodic theory."""

    def gauss_map_scene(self):
        """Dumping ground for a scene about the Gauss map T: Y -> Y, where
        
        - S = [0, 1] \ Q
        - f(y) = 1 / y - floor(1 / y) is the first continued fraction coefficient of any number y in S

        - Y = \{(y, z): y \in S, 0 <= z <= 1 / (1 + y)\}
        - T(y, z) = (f(y), y(1 - yz))
        """

        self.clear()

        # Define the map f: S -> S and its iterates
        def f(y: float):
            if y <= 0:
                # Deal with boundary case which is undefined
                return -1
            else:
                assert y > 0 and y <= 1
                return 1 / y - math.floor(1 / y)
        
        def f_iter(y: float, n: int = 1):
            if n == 0:
                return y
            elif n == 1:
                return f(y)
            else:
                return f_iter(f(y), n - 1)
        
        # Define the map T: Y -> Y and its iterates
        def T(y: float, z: float, n: int = 1):
            return f(y), y * (1 - y * z)
        
        def T_iter(y: float, z: float, n: int = 1):
            if n == 0:
                return y, z
            elif n == 1:
                return T(y, z)
            else:
                a, b = T(y, z)
                return T_iter(a, b, n - 1)

        # Define the derivative of T with respect to each coordinate
        def dy_T(y: float, z: float):
            return -1 / (y**2), 1 - 2 * y * z
        def dz_T(y: float, z: float):
            return 0, -y ** 2
        

        # Construct the Gauss set Y as an object
        ax = m.Axes(x_range=[0,1], y_range=[0,1], x_length=3, y_length=3, tips=False, axis_config={"include_ticks": False})
        fmax_graph = ax.plot(lambda x: 1 / (1 + x), x_range=[0, 1])
        gauss_set = ax.get_area(fmax_graph, [0, 1], color=m.BLUE, opacity=0.5)
        outline = m.Polygon(
            *[ax.c2p(x, 1 / (1 + x)) for x in np.linspace(0, 1, 10)],
            ax.c2p(1, 0.25), ax.c2p(1, 0), ax.c2p(0, 0), ax.c2p(0, 0.5),
            stroke_color=m.WHITE
        )
        
        self.add(ax, gauss_set, outline)
        self.wait(1.0)

        ax_2 = ax.copy()
        gauss_set_2 = gauss_set.copy()
        outline_2 = outline.copy()
        self.add(ax_2, gauss_set_2, outline_2)
        
        for mobj in [ax, outline, gauss_set]:
            mobj.generate_target()
            mobj.target.shift(np.array([-2, 0, 0]))
        for mobj in [ax_2, outline_2, gauss_set_2]:
            mobj.generate_target()
            mobj.target.shift(np.array([2, 0, 0]))

        self.play(
            *[m.MoveToTarget(mobj) for mobj in [ax, outline, gauss_set, ax_2, outline_2, gauss_set_2]],
            run_time=1.0
            )

        num_slices = 100

        # Construct vertical slices
        fmax_graph = ax.plot(lambda x: 1 / (1 + x), x_range=[0, 1])
        vertical_slices = [
            ax.get_area(fmax_graph, [1/(k+1), 1/k], color=m.BLUE, opacity=0.5)
            for k in range(1, num_slices + 1)
        ]

        # Construct horizontal slices
        fmax_graphs_2 = [ax_2.plot(lambda x: 1 / (1 + x), x_range=[0, 1])]
        horizontal_slices = []
        for k in range(1, num_slices + 1):
            fmax_graphs_2.append(ax_2.plot(lambda x: 1 / (k + 1 + x), x_range=[0, 1]))
            horizontal_slices.append(ax_2.get_area(
                fmax_graphs_2[-2], x_range=[0, 1], bounded_graph=fmax_graphs_2[-1]
            ))
        

        self.play(m.FadeOut(gauss_set), m.FadeOut(gauss_set_2))
        for k in range(num_slices):
            self.play(m.FadeIn(vertical_slices[k]), m.FadeIn(horizontal_slices[k]), run_time=(8.0 / ((k + 1) * math.log(num_slices))))
        self.wait(1.0)


        # Clear for a new part to the scene
        self.clear()
        self.wait(1.0)
        self.play(m.FadeIn(gauss_set), run_time=1.0)

        # Construct a moving point
        py = m.ValueTracker(1.0)
        pz = m.ValueTracker(0.3)
        p = m.Dot(ax.c2p(py.get_value(), pz.get_value()), radius=0.08)
        p.add_updater(lambda mobj: mobj.move_to(
            ax.c2p(py.get_value(), pz.get_value())
            ))
        
        # Make the iterates of the point with respect to T
        def make_iterate(k: int):
            p_it = m.Dot(ax.c2p(*T_iter(py.get_value(), pz.get_value(), k)), radius=0.08)
            p_it.add_updater(lambda mobj: mobj.move_to(
                ax.c2p(*T_iter(py.get_value(), pz.get_value(), k))
            ))
            return p_it
        
        p_iterates = []
        num_iterates = 50
        for k in range(1, num_iterates):
            p_iterates.append(make_iterate(k))

        # Construct the tangent vectors
        # TODO This may animate better as an actual plane object
        tangent_scale = 0.3
        def dy_updater(mobj: m.VMobject):
            tg = dy_T(py.get_value(), pz.get_value())
            mobj.become(m.Line(
                start=ax.c2p(py.get_value(), pz.get_value()),
                end=ax.c2p(
                py.get_value() + tangent_scale * tg[0],
                pz.get_value() + tangent_scale * tg[1]
            ),
            stroke_width=2.0
            ))
        tangent_wrt_y = m.VMobject().add_updater(dy_updater)


        def dz_updater(mobj: m.VMobject):
            tg = dz_T(py.get_value(), pz.get_value())
            mobj.become(m.Line(
                start=ax.c2p(py.get_value(), pz.get_value()),
                end=ax.c2p(
                py.get_value() + tangent_scale * tg[0],
                pz.get_value() + tangent_scale * tg[1]
            ),
            stroke_width=2.0
            ))
        tangent_wrt_z = m.VMobject().add_updater(dz_updater)

        self.add(p)
        for k in range(1, num_iterates):
            self.play(m.FadeIn(p_iterates[k - 1]), run_time = 5.0 / num_iterates)
        self.add(*p_iterates)
        # self.add(tangent_wrt_y, tangent_wrt_z)


        # Animate the point moving within a fundamental region
        self.play(py.animate.set_value(0.1))
        self.play(pz.animate.set_value(0.4))
        self.play(py.animate.set_value(0.3))
        self.play(pz.animate.set_value(0.1))

    def test(self):
        """Dumping ground for new ideas"""
        # NOTES ON DRAWING: See camera.set_cairo_context_path. This is an example of where ctx drawing is actually done.
        self.clear()
        camera_phi = 90
        camera_theta = 0
        self.set_camera_orientation(phi=camera_phi * m.DEGREES, theta=camera_theta * m.DEGREES, zoom=1.0)
        # Make a ray, then bent it into a cosine wave, then turn it into a spiral and pan the camera over

        # Make a number line. Then curl it into a spring in 3D and collapse it
        number_line = m.VGroup()
        # line = m.ParametricFunction(lambda t: np.array([0, t, 0]), t_range=(0, 10))
        line = m.ParametricFunction(lambda t: np.array([np.cos(t), t, np.sin(t)]), t_range=(0, 10))
        number_line.add(line)
        ticks = []
        # TODO Add tick marks in log scale, making it into a VGroup

        self.add(number_line)

        # # Animate the number line curling into a spring in 3D, and then collapsing as a circle
        # time = m.ValueTracker(0)
        # line.add_updater(lambda mobj: mobj.become(
        #     m.ParametricFunction(lambda t: np.array([time.get_value() * np.cos(t), t, time.get_value() * np.sin(t)]), t_range=(0, 10))
        # ))
        # self.begin_ambient_camera_rotation(-1.5 / 2.0)
        # self.play(time.animate.set_value(1.0), run_time=2.0)

        #

    def leading_digit_puzzle_solution(self):
        """Present a solution to the leading digit puzzle by rephrasing it in terms of the distribution of \{T^n(x)\}_{n <= N} where T: R/Z -> R/Z is rotation by the irrational number alpha = log_{10}(2)."""
        self.clear()
        # Write powers of 2 on number line, color-coded by leading digit
        # number_line = m.NumberLine(
        #     [0, 20],
        #     # scaling=m.LinearBase(scale_factor = 0.1)
        #     # scaling=m.LogBase(2)
        #     ).move_to(np.array([8., 0., 0.]))
        # self.add(number_line)

        # points = [
        #     m.Dot(number_line.number_to_point(2**i))
        #     for i in range(4)
        #     ]
        # for p in points:
        #     self.play(m.FadeIn(p), run_time = 0.5)
        # self.add(*points)

        # Zoom out TODO


        # Switch to log_10 scale. Here we observe the regularity in spacing. Comment: "That's exactly how logarithms are defined."
        # Fade in equation \log_{10}(2^n) = n\log_{10}(2)
        # Zoom out.
        # Curl up the number line into a circle. See the dots clump together

        number_circle = m.Circle(radius=2.0, color=m.WHITE)
        tick_lengths = {i: 0.1 for i in np.linspace(1, 9.75, 36)}
        for i in range(1, 10):
            tick_lengths[i] = 0.2
        tick_angles = []
        for r, l in tick_lengths.items():
            theta = m.TAU * (math.log(r) / math.log(10))
            tick_angles.append((r, l, theta))
        
        ticks = []
        labels = []
        def cis(theta: float):
            return np.array([np.cos(theta), np.sin(theta), 0])
        
        for r, l, theta in tick_angles:
            p = number_circle.point_at_angle(theta)
            ticks.append(m.Line(
                p + (l / 2) * cis(theta),
                p - (l / 2) * cis(theta),
                color=m.WHITE
            ))
            if l >= 0.2:
                labels.append(m.Tex(f"{r:.0f}", font_size=24).move_to(p + 0.4 * cis(theta)))
        self.add(number_circle, *ticks, *labels)
        
        for n in range(100):
            s = n * math.log(2) / math.log(10)
            s -= int(s)
            self.play(m.FadeIn(m.Dot(
                number_circle.point_at_angle(m.TAU * s),
                radius=0.05,
                # TODO Classify color based on leading digit
                color=m.BLUE
            )), run_time=0.1)


        # Fade in equation: "leading digit = 1 if log_{10}(2^n) has fractional part between 0 and \log_{10}(2)."
        pass

    def construct(self):
        # self.leading_digit_puzzle()
        # self.gauss_map_scene()
        # self.leading_digit_puzzle_solution()
        self.test()
        