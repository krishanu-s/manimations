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

# TODO Animate "unwrapping"

from math import factorial
import numpy as np
import manim as m

CENTER = np.array([-1.5, -1.5, 0])

### PARAMETRIZING THE UNWRAPPING ARCS
# The arc along the unit circle defining the point is the 0-th arc.
# It is parametrized by the function A_0(φ) = e^{i(θ-φ)} for φ in [0, θ].
#
# The k-th arc is defined by unwrapping the (k-1)st arc. It is parametrized
# by a function A_k(φ) which satisfies
# - A_k(0) = e^{iθ}
# - A_k'(φ) = i^{k-1} * φ^k/k! * e^{i(θ-φ)}
#
# Integration by parts yields the recurrence
# A_k(φ) = A_{k-1}(φ) + (iφ)^k/k! * e^{i(θ-φ)}
# 
# Thus, A_k(φ) = (sum_k (iφ)^k/k!) * e^{i(θ-φ)}

### PARAMETRIZING THE UNWRAPPING PROCESS
# We can describe the unwrapping of the k-th arc as a time-depending function
# at each point of the arc, A_k(φ, t). It can be defined by the differential equation
# - A_k(φ, t) = A_k(φ) for 0 <= t <= φ
# - dA_k(φ, t)/dt = i^{k-1} * (t^k - φ^k)/k! * e^{i(θ-φ)} for φ <= t <= θ
#
# Integration by parts yields the recurrence
# A_k(φ, t) = A_k(φ) + (A_{k-1}(t) - A_{k-1}(φ)) + ((it)^k - (iφ)^k)/k! * e^{i(θ-φ)}

### CONSTRUCTING THE m.Homotopy
# Given coordinates (x, y), we will first have to determine the angle φ
# so that A_k(φ) = x + iy. This is the hard part.
# Next, we will have to figure out how to advance this point forward by time t. This is
# the easy part, as we just use the equation for A_k(φ, t).

class UnwrappingScene(m.Scene):

    def construct(self):
        # Set the parameters for the scene
        self.set_parameters()

        # Draw the axes and unit circle.
        self.draw_fixed_elements()

        # Add point
        self.draw_point()

        # Make arc
        arc = m.Arc(
            arc_center=CENTER,
            radius=self.radius,
            start_angle=0,
            angle=self.theta,
            stroke_color=m.RED,
        )
        arc_label = m.MathTex("\\theta", color=m.RED, font_size=30).move_to(
            CENTER
            + self.radius * 0.8 * np.array([np.cos(self.theta / 2), np.sin(self.theta / 2), 0])
        )
        self.play(m.FadeIn(arc, arc_label))
        self.play(m.FadeOut(arc_label))

        # Unwrap arc
        self.play(m.Homotopy())

        # # Sine and cosine
        # sin_def = m.Line(start=self.shift_coords(np.array([np.cos(theta), 0, 0])), end=self.shift_coords(self.point), stroke_width=1.0)
        # cos_def = m.Line(start=self.shift_coords(np.array([0, np.sin(theta), 0])), end=self.shift_coords(self.point), stroke_width=1.0)
        # self.add(sin_def, cos_def)

        # Add convergents
        for i in range(0, 5):
            self.add_convergent(i)
        for i in range(5, 9):
            self.add_convergent(i, add_labels=False)

    def set_parameters(self):
        """Set parameters for the scene."""
        self.center = CENTER
        self.radius = 2.5
        self.theta = 1.8

    def shift_coords(self, pt: np.ndarray):
        """Shifts coordinates to make them appropriate for drawing in the scene"""
        return self.center + self.radius * pt

    def draw_fixed_elements(self):
        """Draw the axes and unit circle."""
        self.add(
            # x-axis
            m.Line(
                start=self.shift_coords(np.array([-1.5, 0, 0])),
                end=self.shift_coords(np.array([1.5, 0, 0])),
                stroke_width=2.0,
            ),
            # y-axis
            m.Line(
                start=self.shift_coords(np.array([0, -1.5, 0])),
                end=self.shift_coords(np.array([0, 1.5, 0])),
                stroke_width=2.0,
            ),
            # circle
            m.Circle(
                radius=self.radius, arc_center=CENTER, color=m.WHITE, stroke_width=2.0
            ),
        )

    def draw_point(self):
        """
        Draw the point at theta radians along the circle
        """
        # Point and segment connecting to the center of the circle
        theta = self.theta
        self.point = np.array([np.cos(theta), np.sin(theta), 0])
        self.add(
            m.Line(start=self.shift_coords(np.array([0, 0, 0])), end=self.shift_coords(self.point), stroke_width=1.0),
            m.Dot(point=self.shift_coords(self.point), radius=0.05),
        )

    def unwrapping_arc(self, n: int):
        """Constructs the function defining the n-th unwrapping arc."""
        assert n >= -1
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
        """Adds the n-th convergent as a static element."""
        assert n >= 0
        f = self.unwrapping_arc(n - 1)
        g = self.unwrapping_arc(n)
        line = m.DashedLine(
            start=self.shift_coords(f(self.theta)),
            end=self.shift_coords(g(self.theta)),
            stroke_width=1.5,
            stroke_color=m.GREEN
        )

        if n % 4 == 1:
            offset = np.array([0.35, 0, 0])
        elif n % 4 == 2:
            offset = np.array([0, 0.35, 0])
        elif n % 4 == 3:
            offset = np.array([-0.35, 0, 0])
        else:
            offset = np.array([0, -0.35, 0])
        
        if n == 0:
            line_label = m.MathTex(f"1", color=m.GREEN, font_size=20)
        elif n == 1:
            line_label = m.MathTex(f"\\theta", color=m.GREEN, font_size=20)
        else:
            line_label = m.MathTex(f"\\theta^{n} / {n}!", color=m.GREEN, font_size=20)
        
        line_label = line_label.move_to(
            self.shift_coords(0.5 * (f(self.theta) + g(self.theta))) + offset
        )
        arc = m.ParametricFunction(
            t_range=[0, self.theta],
            function=lambda t: self.shift_coords(g(t)),
            stroke_width=1.5,
            stroke_color=m.BLUE,
        )
        self.add(line, arc)
        if add_labels:
            self.add(line_label)
