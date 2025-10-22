### Types which specify wavefronts

from typing import Callable
import numpy as np
from .point import Point2D
from .isotopy import IsotopyFn

"""A function which takes a radius r as input, and returns the minimum and maximum
angles θmin and θmax on the arc at distance r from the focus and in the same plane region
as the focus. The angles should satisfy 0 < θmax - θmin < 2 * π and should vary continuously
with the radius."""
BoundsFn = Callable[[float], tuple[float, float]]

class Arc:
    """An Arc is specified by a center point, radius, starting angle, and ending angle."""
    center: tuple[float, float]
    radius: float
    angle_bounds: tuple[float, float]

class Segment:
    """A Segment is specified by two endpoints."""
    start: tuple[float, float]
    end: tuple[float, float]

class ArcEnvelope:
    """An ArcEnvelope is a function which takes a radius value as input and produces angle bounds
    as output. It defines a region containing the center point as the union of arcs over
    all positive radii."""
    center: Point2D
    bounds: BoundsFn
    def __init__(self, center: Point2D, bounds: BoundsFn):
        self.center = center
        self.bounds = bounds

    def interpolate_arcs(self, radius_1: float, radius_2: float) -> Callable[[float, float], float]:
        """Given an initial and final radius, defines a function which isotopes a point
        at angle θ on the first arc forward in time by t.
        """
        # TODO This function is a problem if the initial arc is a full circle.
        # We can solve this by setting the initial arc to (ε, 2π-ε)
        min_angle, max_angle = self.bounds(radius_1)
        def f(theta, t):
            alpha = (theta - min_angle) / (max_angle - min_angle)
            new_radius = t * radius_2 + (1 - t) * radius_1
            new_min_angle, new_max_angle = self.bounds(new_radius)
            return alpha * new_max_angle + (1 - alpha) * new_min_angle
        return f
    
    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        """Converts the function interpolate_arcs into an isotopy in Cartesian coordinates."""
        def istpy(x: float, y: float, z: float, t: float, dt: float):
            # Convert to polar coordinates (r, θ)
            vec = np.array([x - self.center.x, y - self.center.y])
            r = np.linalg.norm(vec)

            # Arcsin always gives a value in [-π/2, π/2], so we must reflect over the y-axis
            # if the original vector had negative x-coordinate.
            if vec[0] < 0:
                theta = (np.pi - np.arcsin(vec[1] / r))
            else:
                theta = np.arcsin(vec[1] / r)

            # Get current arc and distance along it. May need to shift theta to lie in the angle bounds.
            min_angle, max_angle = self.bounds(r)
            while theta < min_angle:
                theta += 2 * np.pi
            while theta > max_angle:
                theta -= 2 * np.pi
            alpha = (theta - min_angle) / (max_angle - min_angle)
            
            # Get the corresponding point on the target arc
            target_radius = radius_1 * (1 - (t + dt)) + radius_2 * (t + dt)
            new_min_angle, new_max_angle = self.bounds(target_radius)
            new_theta = alpha * new_max_angle + (1 - alpha) * new_min_angle

            # Convert back to Cartesian coordinates
            return (
                target_radius * np.cos(new_theta) + self.center.x,
                target_radius * np.sin(new_theta) + self.center.y,
                z,
                t + dt
            )
        return istpy
    
class SegmentEnvelope:
    """Same as an ArcEnvelope but defined by two orthogonal lines L1 and L2, where
    the radius is replaced by distance from L1, and angles are replaced by distance from L2."""
    def interpolate_segments(self, radius_1: float, radius_2: float) -> Callable[[float, float], float]:
        pass
    
    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        pass

if __name__ == "__main__":
	# TODO Add tests.
	pass