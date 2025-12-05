"""Introducing ergodic theory through simple examples"""

import numpy as np
import manim as m

class LeadingDigitScene(m.Scene):
    def scene_1(self):
        ## Posing the question as a teaser, then showing numerical data

        # Write out the question.
        # "Consider the powers of 2 from 1 to 2^{N-1} for some large N, and note down all of the leading digits when written in base-10. What's the percentage breakdown?"
        m1 = m.MathTex("1, 2, 4, 8, 16, 32, 64, 128, \ldots, 2^{N-1}").move_to(np.array([0, 3, 0]))
        m2 = m.Text("Consider the first N powers of two, written in base-10.\nHow often does each digit 1-9 appear as the leading digit in this list??", font_size=24).move_to(np.array([0, 2, 0]))
        self.play(m.FadeIn(m1), run_time=1.0)
        self.play(m.FadeIn(m2), run_time=1.0)

        # Fade out text above.
        # Fade in data for N=20, with the leading digits highlighted.
        # Fade in a histogram showing the data.
        
        # Zoom out the data into a rectangular table up to N=100 (up to 4 digits accuracy)
        # Convert each number into a dot of the corresponding color, and then animate to N=1000 in a rectangular table, while animating the change in the histogram
        # Fade out and then fade in for N=10,000 as a simple grid of dots, while animating the change in the histogram
        # Fade out and then fade in for N=100,000 as a pixelated image, while animating the change in the histogram
        pass

    def scene_2(self):
        ## Talking a bit about ergodic theory ideas.
        pass

    def scene_3(self):
        # Write powers of 2 on number line, color-coded by leading digit
        # Switch to log_10 scale. Here we observe the regularity in spacing.
        # Fade in equation \log_{10}(2^n) = n\log_{10}(2)
        # Zoom out.
        # Fade in a "curled-up" version where it's wrapped around a circle. See the colors clump together
        # Fade in equation: "leading digit = 1 if log_{10}(2^n) has fractional part between 0 and \log_{10}(2)."
        pass

    def construct(self):
        self.scene_1()