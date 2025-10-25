### Types which specify wavefronts

from typing import Callable
import numpy as np
from .point import Point2D
from .isotopy import IsotopyFn
from .tolerances import ANGLE_TOLERANCE

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

    def from_cartesian(self, x: float, y: float) -> tuple[float, float]:
        """Convert to polar coordinates (r, θ) centered on the center."""
        vec = np.array([x - self.center.x, y - self.center.y])
        r = np.linalg.norm(vec)

        # Arcsin always gives a value in [-π/2, π/2], so we must reflect over the y-axis
        # if the original vector had negative x-coordinate.
        if vec[0] < 0:
            theta = (np.pi - np.arcsin(vec[1] / r))
        else:
            theta = np.arcsin(vec[1] / r)
        
        return 

    def to_cartesian(self, r: float, theta: float) -> np.ndarray:
        pass
    
    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        """Converts the function interpolate_arcs into an isotopy in Cartesian coordinates."""
        # TODO This function is a problem if the initial arc is a full circle.
        # We can solve this by setting the initial arc to (ε, 2π-ε)
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
            while theta < min_angle - ANGLE_TOLERANCE:
                theta += 2 * np.pi
            while theta > max_angle + ANGLE_TOLERANCE:
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
    """Same as an ArcEnvelope but defined by a point P and a unit normal vector N.
    Given a radius r, we proceed to the point P + rN, and the resulting two bounding
    points (a, b) indicate P + rN + aM, P + rN + bM, where M is the rotation of N clockwise
    by 90 degrees."""
    n_vector: np.ndarray
    m_vector: np.ndarray
    center: Point2D
    bounds: BoundsFn
    def __init__(self, center: Point2D, bounds: BoundsFn, normal_vector: Point2D):
        self.center = center
        self.bounds = bounds
        self.n_vector = normal_vector.to_array() * (1 / normal_vector.length())
        self.m_vector = np.array([normal_vector.y, -normal_vector.x])
    
    def from_cartesian(self, x: float, y: float) -> tuple[float, float]:
        """Converts a pair (x, y) into envelope coordinates"""
        vec = np.array([x - self.center.x, y - self.center.y])
        return np.dot(vec, self.n_vector), np.dot(vec, self.m_vector)

    def to_cartesian(self, r: float, s: float) -> np.ndarray:
        """Converts a pair (r, s) into Cartesian coordinates"""
        return r * self.n_vector + s * self.m_vector + self.center.to_array()

    def isotopy(self, radius_1: float, radius_2: float) -> IsotopyFn:
        def istpy(x: float, y: float, z: float, t: float, dt: float):
            # Convert to coordinates (r, s) where r is the projection onto the normal
            # vector and s is the orthogonal component
            r, s = self.from_cartesian(x, y)
            min_s, max_s = self.bounds(r)
            alpha = (s - min_s) / (max_s - min_s)

            # Get the corresponding point on the target segment
            target_radius = radius_1 * (1 - (t + dt)) + radius_2 * (t + dt)
            new_min_s, new_max_s = self.bounds(target_radius)
            new_s = alpha * new_max_s + (1 - alpha) * new_min_s

            # Convert back to Cartesian coordinates
            new_vec = self.to_cartesian(target_radius, new_s)

            return (
                new_vec[0],
                new_vec[1],
                z,
                t + dt
            )
        return istpy

if __name__ == "__main__":
	# TODO Add tests.
	pass