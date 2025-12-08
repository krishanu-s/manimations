"""Introducing ergodic theory through simple examples"""

import numpy as np
import manim as m

class LeadingDigitScene(m.Scene):
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

    def scene_1(self):
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

        ## Make a histogram.
        ## TODO Move this into its own class
        histogram = m.VGroup()
        
        # Set up the axes
        hist_w = 5.0
        hist_h = 4.0
        hist_offset = np.array([1.0, -1.0, 0])
        x_axis = m.Line(
            hist_offset,
            hist_offset + np.array([hist_w, 0., 0.]),
            stroke_width=3.0)
        y_axis = m.Line(
            hist_offset,
            hist_offset + np.array([0., hist_h, 0.]),
            stroke_width=3.0)

        # Set the x values on the x axis
        x_values = list(range(1, 10))
        num_x = len(x_values)
        x_tick_length = 0.2
        x_ticks = []
        x_labels = []
        for i, x in enumerate(x_values):
            # TODO Not sure if it's better to put the ticks in the middle of the bars vs between them.
            x_pos = hist_offset + ((i + 0.5) / num_x) * np.array([hist_w, 0., 0.])
            x_ticks.append(m.Line(
                x_pos - np.array([0., x_tick_length / 2, 0.]),
                x_pos + np.array([0., x_tick_length / 2, 0.])
                ))
            # TODO Either set the font size adaptively, or curtail the number of them
            x_labels.append(m.Tex(f"{x}", font_size=24).move_to(x_pos - np.array([0., 0.4, 0])))

        # Set the y values on the y-axis
        y_values = np.linspace(0.2, 1.0, 5)
        num_y = len(y_values)
        y_tick_length = 0.2
        y_ticks = []
        y_lines = []
        y_labels = []
        for i, y in enumerate(y_values):
            y_pos = hist_offset + ((i + 1) / num_y) * np.array([0., hist_h, 0.])
            y_ticks.append(m.Line(
                y_pos - np.array([y_tick_length / 2, 0., 0.]),
                y_pos + np.array([y_tick_length / 2, 0., 0.])
                ))
            y_lines.append(m.Line(y_pos, y_pos + np.array([hist_w, 0., 0.]), stroke_width=3.0, stroke_color=m.GRAY, stroke_opacity=0.5))
            # TODO Either set the font size adaptively, or curtail the number of them
            y_labels.append(m.Tex(f"{y:.1f}", font_size=24).move_to(y_pos - np.array([0.3, 0., 0])))


        # Make the bars using the data
        bar_width = hist_w / num_x
        def bar_center(height: float, x: int):
            return hist_offset + np.array([0., height / 2, -5.]) + ((x-0.5) / num_x) * np.array([hist_w, 0., 0.])

        def make_bar(x: int):
            val_tracker = m.ValueTracker(histograms[20][x])

            bar = m.Rectangle(height = max(hist_h * val_tracker.get_value(), 1e-3), width = 0.9 * bar_width, color = m.ORANGE)
            bar.set_opacity(1.0)
            bar.move_to(bar_center(hist_h * val_tracker.get_value(), x))
            bar.add_updater(lambda mobj: mobj.stretch_to_fit_height(max(hist_h * val_tracker.get_value(), 1e-3)).move_to(
                        bar_center(hist_h * val_tracker.get_value(), x)))
            return bar, val_tracker
            
        x_bars_and_trackers = []
        for x in range(1, 10):
            x_bars_and_trackers.append(make_bar(x))
        x_bars, hist_val_trackers = zip(*x_bars_and_trackers)

        histogram.add(x_axis, y_axis)
        histogram.add(*x_ticks, *x_labels)
        histogram.add(*y_ticks, y_lines, *y_labels)
        histogram.add(*x_bars)

        n_value_text = m.Tex(f"N = 10").move_to(np.array([3.8, -2.8, 0.]))



        # Define Mobject for displaying the numbers in a table
        numbers = m.VGroup()
        for i in range(10):
            numbers.add(m.Tex(f"{2**i}", font_size=20).move_to((-5.5, 3.3 - 0.35 * i, 0)))

        # TODO Fade in data for N=10, with the leading digits highlighted.
        self.play(
            m.FadeIn(histogram), m.FadeIn(n_value_text), m.FadeIn(numbers), run_time=2.0
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

    def scene_2(self):
        ## Talking a bit about ergodic theory ideas.
        pass

    def scene_3(self):
        # Write powers of 2 on number line, color-coded by leading digit
        # Switch to log_10 scale. Here we observe the regularity in spacing. Comment: "That's exactly how logarithms are defined."
        # Fade in equation \log_{10}(2^n) = n\log_{10}(2)
        # Zoom out.
        # Fade in a "curled-up" version where it's wrapped around a circle. See the colors clump together
        # Fade in equation: "leading digit = 1 if log_{10}(2^n) has fractional part between 0 and \log_{10}(2)."
        pass

    def construct(self):
        self.scene_1()