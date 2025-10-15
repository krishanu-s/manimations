# Animation of rays bouncing within an ellipse.

# Command line build instructions: manim -pql filename.py SceneName

from __future__ import annotations
from typing import List
import manim as m
import numpy as np
import math
from ray import Point, Vector
from trail import make_trail
from wavefront import Wavefront
from polyfunction import Conic
from symphony import (Symphony, Sequence, AnimationEvent, Add, Remove)

RAY_WIDTH = 1.0
RAY_OPACITY = 1.0
RAY_COLOR = m.WHITE


class Parabola:
    focus: np.ndarray
    vertex: np.ndarray
    x_init: float
    x_final: float

    def __init__(
        self,
        focus: np.ndarray,
        vertex: np.ndarray,
        x_init: float,
        x_final: float,
    ):
        # Assume the parabola is symmetric around a vertical line
        assert focus[0] == vertex[0]
        self.focus = focus
        self.vertex = vertex
        self.x_init = x_init
        self.x_final = x_final

        # Coefficients as a polynomial y = ax^2 + bx + c
        self.a = 1 / (4 * (focus[1] - vertex[1]))
        self.b = -2 * self.a * vertex[0]
        self.c = self.vertex[1]

    def make_curve(self) -> m.CubicBezier:
        """Makes a parabola with a given focus, vertex, and endpoints."""

        # First get the polynomial defining the curve
        f = lambda x: self.a * x ** 2 + self.b * x + self.c
        f_prime = lambda x: 2 * self.a * x + self.b
        
        # Plug in the two endpoints
        y_init = f(self.x_init)
        y_final = f(self.x_final)

        # Then construct quadratic bezier curve
        x_mid = (self.x_init + self.x_final) / 2
        y_mid = f(self.x_init) + f_prime(self.x_init) * (self.x_final - self.x_init) / 2

        # Then get the cubic one
        x_1 = (self.x_init + 2 * x_mid) / 3
        y_1 = (y_init + 2 * y_mid) / 3

        x_2 = (self.x_final + 2 * x_mid) / 3
        y_2 = (y_final + 2 * y_mid) / 3
        return m.CubicBezier(
            np.array([self.x_init, y_init, 0]),
            np.array([x_1, y_1, 0]),
            np.array([x_2, y_2, 0]),
            np.array([self.x_final, y_final, 0]),
            )
    
    def get_angle_bounds(self, radius: float) -> tuple[float, float]:
        """Calculates the angles at which a circle centered at the focus of a given
        radius intersects the parabola."""
        min_angle = m.TAU / 4 - np.arctan((2 * radius * self.a - 1) / np.sqrt(4 * radius * self.a - 1))
        max_angle = m.TAU / 4 + np.arctan((2 * radius * self.a - 1) / np.sqrt(4 * radius * self.a - 1))
        return min_angle, max_angle


class ParabolaScene(m.Scene):
    def add_fixed_elements(self):
        """Build the fixed elements of the scene."""
        # Construct a parabola y = ax^2 + c
        a = 0.25
        c = -3.0
        self.focus = Point(0, c + 1 / (4 * a))
        self.vertex = Point(0, c)
        self.conic = Conic(a, 0, 0, 0, -1.0, c)

        # Drawing
        parabola = Parabola(self.focus.to_array(), self.vertex.to_array(), -6, 6).make_curve()
        focus_dot = m.Dot(self.focus.coords(), radius=0.08, color=m.RED)
        self.add(parabola, focus_dot)

    def construct(self):
        """Construct the scene."""
        self.add_fixed_elements()

        # Parameters for the set of trajectories to build
        num_directions = 10
        w = 1 - 1 / num_directions
        directions = [
            Vector(np.sin(theta), -np.cos(theta))
            for theta in np.linspace(-w * math.pi, w * math.pi, num_directions)
        ]
        speed = 4.0

        ### For each direction, create a sequence of animations, which are packaged in series into a Succession:
        sequences = []
        for direction in directions:
            # Make the sequence of collision points, including the starting one
            points = self.conic.make_trajectory(
                start=self.focus, direction=direction, total_dist=10.0
            )
            # Turn it into animations
            sequences.append(animate_trajectory(points, speed))
        # ... then play the Successions all in parallel
        symphony = Symphony(sequences)
        symphony.animate(self)

class WavefrontScene(m.Scene):
    def construct(self):
        """Homotope one arc to another"""
        # Construct a parabola y = ax^2 + c
        a = 0.5
        c = -3.0
        self.focus = Point(0, c + 1 / (4 * a))
        self.vertex = Point(0, c)
        self.conic = Conic(a, 0, 0, 0, -1.0, c)

        # Drawing
        parabola = Parabola(self.focus.to_array(), self.vertex.to_array(), -6, 6).make_curve()
        focus_dot = m.Dot(self.focus.coords(), radius=0.08, color=m.RED)
        self.add(parabola, focus_dot)

        # Construct wavefront
        center = self.focus.to_array()
        w = Wavefront(center=center, a=a)
        # for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        #     self.add(w.make_arc(r))

        # Make a series of cascading wavefronts
        def make_anim(delay):
            arc = w.make_arc(0.1)
            return [
                AnimationEvent.wait(delay),
                AnimationEvent(
                    header=[Add(arc)],
                    middle=w.isotopy(arc, old_radius=0.1, new_radius=4.1, run_time=3.6),
                    footer=[Remove(arc)]
                ),
                AnimationEvent.wait(2.0 - delay)
            ]
        
        sequences = []
        for delay in np.linspace(0.1, 1.9, 10):
            sequences.append(make_anim(delay))
        symphony = Symphony(sequences)
        symphony.animate(self)

def animate_trajectory(
    points: List[Point], speed: float = 4.0
) -> Sequence:
    """Given a sequence of points, creates a Sequence of AnimationEvents for the
    linear motions connecting them. Creates a trail behind."""

    # Make the starting point
    now_point = points[0]
    now_coords = now_point.coords()
    dot = m.Dot(now_coords, radius=0.03, color=m.RED)

    sequence: Sequence = []

    for i, pt in enumerate(points[1:]):
        next_point = pt.copy()
        next_coords = next_point.coords()
        distance = now_point.distance_to(next_point)

        # Define the trail behind
        trail = make_trail(dot, starting_point=now_coords, stroke_width=RAY_WIDTH, stroke_opacity=RAY_OPACITY)
        
        # Define movement along the line segment
        segment = m.Line(now_coords, next_coords, stroke_width=RAY_WIDTH, stroke_opacity=RAY_OPACITY)
        mover_animation = m.MoveAlongPath(
            dot,
            segment,
            run_time=distance/speed,
            rate_func=lambda t: t
        )

        # Construct the animation event
        sequence.append(
            AnimationEvent(
                header=[Add(dot), Add(trail)] if i == 0 else [Add(trail)],
                middle=mover_animation,
                # Remove the trail and convert to a line segment
                footer=[
                    Remove(trail),
                    Add(segment)
                ])
        )

        # Update the current point
        now_point = pt.copy()
        now_coords = now_point.coords()

    return sequence