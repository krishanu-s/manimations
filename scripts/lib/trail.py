from typing import List
import numpy as np
import manim as m
from .ray import Point3D
from .symphony import Sequence, AnimationEvent, Add, Remove

RAY_WIDTH = 1.0
RAY_OPACITY = 1.0

class RayObject(m.VMobject):
	"""A named class of VMobject for debugging purposes"""


def make_trail(dot: m.Dot, starting_point: tuple[float, float, float] | np.ndarray, **kwargs) -> RayObject:
	"""Make a trail tracking the given dot beginning at the starting point"""
	trail = RayObject()
	trail.add_updater(lambda x: x.become(
		m.Line(starting_point, dot, **kwargs)
	))
	return trail

## TODO Change the above so that the visual alpha of the line varies depending on distance from
## the current location -- i.e., a trajectory leaves a faded "trail".

def animate_trajectory(
    points: List[Point3D], speed: float = 4.0
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
            rate_func=m.linear
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