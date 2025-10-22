# Animation of rays bouncing within an ellipse.
# TODO Replace this with an updated version

# Command line build instructions: manim -pql filename.py SceneName

from __future__ import annotations
from typing import List
import manim as m
import numpy as np
import math
from lib.ray import Point, Vector, Ray
from lib.trail import make_trail, Wavefront
from lib.polyfunction import Conic
from lib.symphony import (Sequence, AnimationEvent, Add, Remove)

RAY_WIDTH = 1.0
RAY_OPACITY = 1.0
RAY_COLOR = m.WHITE


# TODO Deprecate this
def calculate_collisions(conic: Conic, ray: Ray, num_points: int = 10) -> List[Point]:
    """Constructs the sequence of collision points for an initial ray."""
    collision_points = []
    last_ray = ray.copy()
    next_ray = ray.copy()
    for i in range(num_points):
        match conic.bounce(last_ray):
            case None:
                raise InterruptedError("No collision")
            case t:
                next_ray = t
                collision_points.append(t.start)
                last_ray = next_ray

    return collision_points


class EllipseScene(m.Scene):
    def construct(self):
        # Construct an ellipse
        a = 5.0  # Major axis length
        b = 3.0  # Minor axis length
        c = np.sqrt(a**2 - b**2)  # Focal length
        conic = Conic(1 / a**2, 0, 1 / b**2, 0, 0, -1.0)

        # Set a starting point
        start = Point(c, 0.0)

        # Drawing
        ellipse = m.Ellipse(width=2 * a, height=2 * b, color=m.BLUE)
        foci = [
            m.Dot(np.array([c, 0.0, 0.0]), radius=0.08),
            m.Dot(np.array([-c, 0.0, 0.0]), radius=0.08),
        ]
        self.add(ellipse, *foci)

        num_directions = 20
        directions = [
            Vector(np.cos(theta), np.sin(theta))
            for theta in np.linspace(0, 2 * math.pi, num_directions)
        ]
        # For each direction, ...
        successions = []
        # TODO Leave a trail behind.
        for direction in directions:
            # ... create a sequence of animations, which are packaged in series into a Succession
            point = m.Dot(np.array([start.x, start.y, 0.0]), radius=0.03, color=m.RED)
            now_coords = point.get_center()
            self.add(point)
            animations = []
            for p in calculate_collisions(conic, Ray(start, direction), num_points=10):
                next_coords = np.array([p.x, p.y, 0])
                dist = np.linalg.norm(now_coords - next_coords)
                animations.append(
                    m.MoveAlongPath(
                        point,
                        m.Line(now_coords, next_coords),
                        run_time=dist * 0.3,
                        rate_func=lambda t: t,
                    )
                )
                now_coords = next_coords
            successions.append(m.Succession(*animations))
        # ... then play the Successions all in parallel
        self.play(m.AnimationGroup(*successions))




class WavefrontScene(m.Scene):
    def construct(self):
        """Homotope one arc to another"""

        a = 1.0
        w = Wavefront(a=a)

        arc = w.make_arc(1.0)
        homotopy = w.homotopy(arc, 3.0)
        # self.add(arc)
        self.play(homotopy)
        # square = m.Square()

        # def homotopy(x, y, z, t):
        #     if t <= 0.25:
        #         progress = t / 0.25
        #         return (x, y + progress * 0.2 * np.sin(x), z)
        #     else:
        #         wave_progress = (t - 0.25) / 0.75
        #         return (x, y + 0.2 * np.sin(x + 10 * wave_progress), z)

        # self.play(m.Homotopy(homotopy, square, rate_func= m.linear, run_time=2))


        # circle_1 = m.Arc(
        #     radius=r_init,
        #     arc_center=(0, 1 / (4 * a), 0),
        #     start_angle=m.TAU / 4,
        #     angle=m.TAU,
        # )

        # def htpy(x_0, y_0, z_0, t):
        #     # If the point a distance t * r_final from the arc center in the given direction lies
        #     # - above the parabola, then go to r_final * t + r_init * (1-t)
        #     # - below the parabola, then
        #     #   - calculate the intersection x-value
        #     #   - go to the point vertically directly above the intersection point at y-coordinate r_final - (1/4a).

        #     # Expand out t of the way to the final circle in the given direction.
        #     c = t * (r_final / r_init) + (1 - t)
        #     x_1, y_1, z_1 = c * x_0, c * (y_0 - 1 / (4 * a)) + 1 / (4 * a), c * z_0
        #     if y_1 < a * x_1**2:
        #         # If the result lies above the parabola, then that's our point
        #         return x_1, y_1, z_1
        #     else:
        #         # Otherwise, we first find the x-coordinate of the intersection with the parabola, and then manually set the y-coordinate.
        #         x_1 = (
        #             a * y_0
        #             - 0.25
        #             + np.sqrt((a**2) * (x_0**2 + y_0**2) - 0.5 * a * y_0 + 0.0625)
        #         ) / (2 * x_0 * a**2)
        #         y_1 = t * r_final + (1 - t) * r_init - 0.25
        #         return x_1, y_1, z_1

        # # circle_2 = m.Arc(radius = 2.0, start_angle = 0.0, angle = m.TAU / 2, arc_center = (0, 0, 0))

        # self.play(
        #     m.Homotopy(
        #         homotopy=htpy, mobject=circle_1, run_time=1.0, rate_func=m.linear
        #     )
        # )

        # # ## Construct a wavefront as a circle
        # # # Set the initial wavefront as a circle
        # # focus = [0, 0.25, 0]
        # # initial_wavefront = m.Circle(radius=0.001, arc_center=focus, color=m.BLUE)

        # # # Set the final wavefront at time z, meaning we expanded for distance z
        # z = 2.5
        # target_wavefront = make_wavefront(z)

        # # Define the path function connecting them by outputting an array of intermediate points
        # # Arrays of shape (M, 3)
        # def path_func(init_array: np.ndarray, final_array: np.ndarray, t):
        # 	y = t * z - 0.25
        # 	x = np.sqrt(y)
        # 	theta_cutoff: float = 2 * np.arctan(0.5 / x)
        # 	array_focus = np.stack([focus] * init_array.shape[0], axis=0)

        # 	# For each point in the initial array, calculate its angle around the focus.
        # 	# - If the angle from the vertical is less than theta_cutoff, expand straight outwards by distance t * z.
        # 	# - If it is greater, bounce in two steps, ending up at y-coordinate t * z - 0.25 and x-coordinate equal to the intersection with the parabola.
        # 	init_vecs = init_array - array_focus
        # 	init_angles = np.arctan2(init_vecs[:, 0], init_vecs[:, 1])

        # 	less_than_cutoff = init_angles < theta_cutoff

        # 	# Convert a numpy array to one of the same shape with all booleans depending on whether the angle is less than theta_cutoff
        # 	less_than_cutoff = (init_angles < theta_cutoff).astype(int)
        # 	greater_than_cutoff = 1 - less_than_cutoff

        # 	a = np.stack([less_than_cutoff] * 3, axis=-1) * (np.stack([np.sin(init_angles), np.cos(init_angles), np.zeros_like(init_angles)], axis=-1) + array_focus)

        # 	b = np.stack([greater_than_cutoff] * 3, axis=-1) * np.stack([1 / 2 * np.tan(0.5 * init_angles), t * z * np.ones_like(init_angles), np.zeros_like(init_angles)], axis=-1)

        # 	return a + b
        # 	# def foo(theta):
        # 	# 	if theta < theta_cutoff:
        # 	# 		return np.array([np.sin(theta), np.cos(theta), 0]) * t * z + np.array(focus)
        # 	# 	else:
        # 	# 		return np.array([1 / 2 * np.tan(0.5 * theta), t * z - 0.25, 0])

        # 	# f = np.vectorize(foo)
        # 	# return f(init_angles)
        # 	# intersection_xs = 1 / 2 * np.tan(0.5 * init_angles)
        # 	# ys = (t * z - 0.25) * np.ones_like(intersection_xs)

        # # Add the initial one
        # self.add(initial_wavefront)

        # # Transform it
        # transform = m.Transform(
        # 	initial_wavefront,
        # 	target_wavefront,
        # 	run_time=2,
        # 	path_func=path_func
        # )
        # self.play(transform)

        # # self.add(make_wavefront(y))

        # self.wait(2)


def animate_trajectory(
    points: List[Point], ax: m.Axes, speed: float = 4.0
) -> Sequence:
    """Given a sequence of points, creates a Sequence of AnimationEvents for the
    linear motions connecting them. Creates a trail behind."""

    # Make the starting point
    now_point = points[0]
    dot = m.Dot(
        ax.coords_to_point(*now_point.coords()), radius=0.03, color=m.RED
    )
    now_coords = ax.coords_to_point(*now_point.coords())

    sequence: Sequence = []

    for i, pt in enumerate(points[1:]):
        next_point = pt.copy()
        next_coords = ax.coords_to_point(*next_point.coords())
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
        now_coords = ax.coords_to_point(*now_point.coords())

    return sequence