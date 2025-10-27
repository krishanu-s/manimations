"""Geometric proof of the series for sin/cosine, by `unwrapping'.

Definition of sine and cosine
- Draw axes, and unit circle.
- Draw a segment of some length x, in color.
- Map the segment onto an arc of that length.
- Draw dotted altitudes to the x-axis (y-axis) and label it sin(x) (cos(x)).
- Let the length x vary.

Applications of sine and cosine
- ... (draw from spherical trigonometry, astronomy, etc)

Series for sine and cosine
- Set x to some nice value.
-
"""

from math import factorial
import numpy as np
import manim as m

CENTER = np.array([-1.5, -1.5, 0])


class UnwrappingScene(m.Scene):
    def set_parameters(self):
        """Set parameters for the scene."""
        self.center = CENTER
        self.radius = 2.5
        self.theta = 1.8

    def draw_fixed_elements(self):
        """Draw the axes and unit circle."""
        self.add(
            # x-axis
            m.Line(
                start=CENTER + np.array([-1.5 * self.radius, 0, 0]),
                end=CENTER + np.array([1.5 * self.radius, 0, 0]),
                stroke_width=2.0,
            ),
            # y-axis
            m.Line(
                start=CENTER + np.array([0, -1.5 * self.radius, 0]),
                end=CENTER + np.array([0, 1.5 * self.radius, 0]),
                stroke_width=2.0,
            ),
            # circle
            m.Circle(
                radius=self.radius, arc_center=CENTER, color=m.WHITE, stroke_width=2.0
            ),
        )

    def draw_point(self):
        """
        Draw the point at theta radians along the circle, and label the defining arc.
        Also draw segments representing sine and cosine.
        """
        # Point and segment connecting to the center of the circle
        theta = self.theta
        self.point = np.array([np.cos(theta), np.sin(theta), 0])
        self.add(
            m.Line(start=CENTER, end=CENTER + self.radius * self.point, stroke_width=1.0),
            m.Dot(point=CENTER + self.radius * self.point, radius=0.05),
        )

        # Arc
        arc = m.Arc(
            arc_center=CENTER,
            radius=self.radius,
            start_angle=0,
            angle=theta,
            stroke_color=m.RED,
        )
        arc_label = m.MathTex("\\theta", color=m.RED, font_size=30).move_to(
            CENTER
            + self.radius * 0.8 * np.array([np.cos(theta / 2), np.sin(theta / 2), 0])
        )
        self.add(arc, arc_label)

        # Sine and cosine
        sin_def = m.Line(start=CENTER + self.radius * np.array([np.cos(theta), 0, 0]), end=CENTER + self.radius * self.point, stroke_width=1.0)
        cos_def = m.Line(start=CENTER + self.radius * np.array([0, np.sin(theta), 0]), end=CENTER + self.radius * self.point, stroke_width=1.0)
        self.add(sin_def, cos_def)



    def unwrapping_arc(self, n: int):
        """Constructs the function defining the n-th unwrapping arc."""
        assert n >= 0
        def parametrization(t):
            # Even part
            even_part = np.array([
                sum((t ** k) * (1 / factorial(k)) * ((-1) ** (k//2)) * np.cos(self.theta - t) for k in range(0, n + 1, 2)),
                sum((t ** k) * (1 / factorial(k)) * ((-1) ** (k//2)) * np.sin(self.theta - t) for k in range(0, n + 1, 2)),
                0
            ])
            # Odd part
            odd_part = np.array([
                sum((t ** k) * (1 / factorial(k)) * ((-1) ** ((k + 1)//2)) * np.sin(self.theta - t) for k in range(1, n + 1, 2)),
                sum((t ** k) * (1 / factorial(k)) * ((-1) ** (k//2)) * np.cos(self.theta - t) for k in range(1, n + 1, 2)),
                0
            ])
            return even_part + odd_part
        return parametrization

    def add_convergent(self, n: int, add_labels: bool = True):
        """Adds the n-th convergent"""
        assert n > 0
        f = self.unwrapping_arc(n - 1)
        g = self.unwrapping_arc(n)
        line = m.DashedLine(
            start=CENTER + self.radius * f(self.theta),
            end=CENTER + self.radius * g(self.theta),
            stroke_width=1.0,
            stroke_color=m.PURPLE
        )
        if n % 4 == 1:
            offset = np.array([0.35, 0, 0])
        elif n % 4 == 2:
            offset = np.array([0, 0.35, 0])
        elif n % 4 == 3:
            offset = np.array([-0.35, 0, 0])
        else:
            offset = np.array([0, -0.35, 0])
        line_label = m.MathTex(f"\\theta^{n} / {n}!", color=m.PURPLE, font_size=20).move_to(
            CENTER + 0.5 * self.radius * (f(self.theta) + g(self.theta) + offset)
        )
        arc = m.ParametricFunction(
            t_range=[0, self.theta],
            function=lambda t: CENTER + self.radius * g(t),
            stroke_width=1.0,
            stroke_color=m.BLUE,
        )
        self.add(line, arc)
        if add_labels:
            self.add(line_label)

    def construct(self):
        # Set the parameters for the scene
        self.set_parameters()

        # Draw the axes and unit circle.
        self.draw_fixed_elements()

        # Add point and defining arc
        self.draw_point()

        # Add convergents
        for i in range(1, 5):
            self.add_convergent(i)
        for i in range(5, 9):
            self.add_convergent(i, add_labels=False)