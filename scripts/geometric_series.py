"""Simple animation depicting the sum of a geometric series"""

import manim as m

class GeometricSeries(m.Scene):
    def set_parameters(self):
        self.ratio = 0.3
        self.sidelength = 4

    def construct(self):
        self.set_parameters()
        r = self.ratio
        w = self.sidelength
        # Draw a unit square
        square = m.Square(side_length=w)
        self.add(square)

        # Draw the parts of the sum
        rect = m.Rectangle(height=w*(1-r), width=w, color=m.ORANGE)
        rect.set_opacity(0.5)
        rect.move_to((0, -w*r/2, -10))
        
        self.play(m.FadeIn(rect, m.MathTex("1-r")))