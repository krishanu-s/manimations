
from enum import Enum
from __future__ import annotations
from typing import Literal, Callable, List
from dataclasses import dataclass
import numpy as np
from ray import Point, Vector, Hyperplane, Ray
from polyfunction import PolyFunction

ROOT_TOLERANCE = 1e-4
MAX_ROOT = 2 ** 32

@dataclass
class ProjectivePoint:
    """A 3-tuple [x:y:z] representing a point (x/z, y/z) in the projective plane."""
    x: float
    y: float
    z: float

### Types which specify wavefronts

"""An isotopy is a function I: (x, y, z, t, dt) -> (x', y', z', t + dt) for all
	-t < dt < 1 - t. It restricts to a homotopy H(x, y, z, t) = I(x, y, z, 0, t)."""
IsotopyFn = Callable[[float, float, float, float, float], tuple[float, float, float]]


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
    center: tuple[float, float]
    bounds: Callable[[float], tuple[float, float]]

    def interpolate_arcs(self, radius_1: float, radius_2: float) -> Callable[[float, float], float]:
        """Given an initial and final radius, defines a function which isotopes a point
        at angle θ on the first arc forward in time by t.
        """
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
            vec = np.array([x - self.center[0], y - self.center[1]])
            r = np.linalg.norm(vec)

            # Arcsin gives a value in [-π/2, π/2], so we must reflect over the y-axis
            # if the original vector had negative x-coordinate.
            theta = (np.pi - np.arcsin(vec[1] / r)) if vec[0] < 0 else np.arcsin(vec[1] / r)

            # Get current arc and distance along it
            min_angle, max_angle = self.bounds(r)
            alpha = (theta - min_angle) / (max_angle - min_angle)
            
            # Get the corresponding point on the target arc
            target_radius = radius_1 * (1 - (t + dt)) + radius_2 * (t + dt)
            new_min_angle, new_max_angle = self.bounds(target_radius)
            new_theta = alpha * new_max_angle + (1 - alpha) * new_min_angle

            # Convert back to Cartesian coordinates
            return (
                target_radius * np.cos(new_theta) + self.center[0],
                target_radius * np.sin(new_theta) + self.center[1],
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

### Types for specifying a conic section

class PolarConicEquation:
    """An equation for r in terms of θ of the form r = C / (1/E + cos(θ - θ_0)).
    Invariant under sending E -> -E, C -> -C, and θ_0 -> θ_0 + π, so we assume
    that E > 0."""
    focus: ProjectivePoint
    e: float
    c: float
    theta_0: float

    def __init__(
        self,
        focus: ProjectivePoint,
        e: float,
        c: float,
        theta_0: float
    ):
        assert(
            e > 0,
            "Eccentricity must be positive."
        )
        self.focus = focus
        self.e = e
        self.c = c
        self.theta_0 = theta_0

    def other_focus(self) -> ProjectivePoint:
        """Calculates the coordinates of the other focus."""
        # TODO
        pass

    def to_cartesian(self) -> CartesianConicEquation:
        """Generates the Cartesian form."""
        return CartesianConicEquation(
            c_xx = (1 / self.e ** 2) - np.cos(self.theta_0) ** 2,
            c_yy = (1 / self.e ** 2) - np.sin(self.theta_0) ** 2,
            c_xy = - np.sin(2 * self.theta_0),
            c_x = 2 * self.c * np.cos(self.theta_0),
            c_y = 2 * self.c * np.sin(self.theta_0),
            c = -self.c ** 2
        )

class CartesianConicEquation:
    """An equation relating x and y in the form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
    Invariant under scaling all parameters, so we assume that self.c_xx + self.c_yy >= 0."""
    c_xx: float
    c_xy: float
    c_yy: float
    c_x: float
    c_y: float
    c_0: float

    def __init__(self, c_xx: float, c_xy: float, c_yy: float, c_x: float, c_y: float, c_0: float):
        assert(
            any(coeff != 0 for coeff in (c_xx, c_xy, c_yy, c_x, c_y)),
            "Must have at least one non-zero non-constant coefficient"
        )
        scale = -1.0 if self.c_xx + self.c_yy < 0 else 1.0
        self.c_xx = c_xx * scale
        self.c_xy = c_xy * scale
        self.c_yy = c_yy * scale
        self.c_x = c_x * scale
        self.c_y = c_y * scale
        self.c_0 = c_0 * scale

    def to_polar(self) -> PolarConicEquation:
        """Generates the polar form."""
        # TODO
        # First, translate so that one of the foci is at (0, 0), i.e. so that 
        # c_x * c_y = 2 * c * c_xy.
        pass

    ### For animating trajectories

    def restrict(self, ray: Ray) -> PolyFunction:
        """
        Constructs to a one-variable quadratic function on the given ray. Since points
        on the ray are parametrized by a new variable t, the quadratic function has
        the form At^2 + Bt + C = 0
        """
        x0 = ray.start.x
        y0 = ray.start.y
        vx = ray.direction.x
        vy = ray.direction.y

        ylin = PolyFunction([y0, vy])
        xlin = PolyFunction([x0, vx])

		# Start with constant term
        polynomial = PolyFunction([self.c_0])

		# Add linear terms
        polynomial += ylin.scale(self.c_y)
        polynomial += xlin.scale(self.c_x)

		# Add quadratic terms
        polynomial += (ylin * ylin).scale(self.c_yy)
        polynomial += (xlin * ylin).scale(self.c_xy)
        polynomial += (xlin * xlin).scale(self.c_xx)

        return polynomial

    def intersection(self, ray: Ray) -> float | None:
        """Calculates the first t-value where the ray intersects the conic. If there is no intersection,
		returns np.inf. If the nearest intersection is too small for numerical precision purposes,
		returns None"""
		# Restrict the polynomial to the ray
        polynomial = self.restrict(ray)

		# Solve the quadratic equation
        roots = polynomial.roots()

		# Find the smallest positive root within tolerance
        min_positive_root = np.inf
        min_tolerated_root = np.inf
        for root in roots:
            if (root > ROOT_TOLERANCE) and (root < min_tolerated_root):
                min_tolerated_root = root
            if root > 0 and root < min_positive_root:
                min_positive_root = root
        match (min_tolerated_root, min_positive_root):
            case (np.inf, np.inf):
                return np.inf
            case (np.inf, min_positive_root):
                return None
            case (min_tolerated_root, _):
                return min_tolerated_root

    def tangent(self, point: Point) -> Hyperplane:
        """Produces the tangent hyperplane at the given point on the curve."""
		# 0 = dP(x, y) = (2Ax + By + D)dx + (Bx + 2Cy + E)dy
        normal = Vector(
			2.0 * self.c_xx * point.x + self.c_xy * point.y * self.c_x,
			self.c_xy * point.x + 2.0 * self.c_yy * point.y + self.c_y,
		)
        return Hyperplane(normal, point)

    def make_trajectory(self, start: Point, direction: Vector, total_dist: float):
        """
        Given a starting point and direction, produces a sequence of points such that the connecting line
        segments are the trajectory of the point as it bounces off the fixed conic.
        """
        remaining_dist = total_dist
        points = [start.copy()]
        ray = Ray(start, direction)
        while True:
            next_t = self.intersection(ray)

			# The next point is too small for numerical precision -- end the process here.
            if next_t is None:
                break

			# If there is no intersection, go out the remaining distance in that direction.
            if next_t is np.inf:
                points.append(ray.point_at_dist(remaining_dist))
                break

            vec = ray.direction * next_t
            dist = vec.magnitude()

			# If there is no intersection within the remaining distance, go out the remaining distance in that direction.
            if dist > remaining_dist:
                points.append(ray.point_at_dist(remaining_dist))
                break

			# Otherwise, go to the next point and update the ray
            next_point = ray.point_at(next_t)
            tangent = self.tangent(next_point)
            reflected_vector = tangent.reflect(ray.direction)
            ray = Ray(ray.point_at(next_t), reflected_vector)

            points.append(next_point)
            remaining_dist -= dist


        return points

class ConicType(Enum):
    """Types of conic based on eccentricity."""
    Ellipse = 1
    Parabola = 2
    Hyperbola = 3

class Conic:
    """
    A conic section can be defined either from its Cartesian form Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0,
    or from its polar-coordinates form r = C / (1/E + cos(θ - θ_0)) around one focus.

    We initialize a Conic from one of these two forms, rather than calling __init__ directly.
    """
    def __init__(
        self,
        polar_eq: PolarConicEquation,
        cart_eq: CartesianConicEquation
    ):
        self.polar_eq: PolarConicEquation = polar_eq
        self.cart_eq: CartesianConicEquation = cart_eq

        # Define the foci
        self.focus = polar_eq.focus
        self.other_focus = polar_eq.other_focus()

        # Define the eccentricity
        self.eccentricity = polar_eq.e
        assert self.eccentricity > 0

        # Calculate the focal length
        # TODO
        self.focal_length = 0
    
    def _type(self) -> ConicType:
        """Returns the type of conic."""
        if self.eccentricity == 1:
            return ConicType.Parabola
        elif self.eccentricity < 1:
            return ConicType.Ellipse
        elif self.eccentricity > 1:
            return ConicType.Hyperbola

    @classmethod
    def from_cartesian(cls, cart_eq: CartesianConicEquation) -> Conic:
        """Initializes a conic section from a two-variable quadratic equation."""
        Conic(polar_eq=cart_eq.to_polar(), cart_eq=cart_eq)

    @classmethod
    def from_polar(cls, polar_eq: PolarConicEquation) -> Conic:
        """Given one focus, the conic may be defined by a function of polar coordinates
        (r, θ) centered around that focus of the form
        r = C / (1/E + cos(θ - θ_0))"""
        Conic(polar_eq=polar_eq, cart_eq=polar_eq.to_cartesian())

    ### For animating wavefronts

    def make_arc_from_main_focus(self, r: float) -> Arc:
        # TODO Move this to PolarConicEquation?
        """Given a distance r from the main focus, calculates angles (θ1, θ2) such that
        the arc {(r, θ): θ1 < θ < θ2} has endpoints on the conic and lies inside the
        section containing the focus. Then generates the Arc
        
        Used to animate spherical wavefronts centered around the main focus."""
        # TODO
        pass

    def make_arc_from_other_focus(self, r: float) -> Arc | Segment:
        # TODO Move this to PolarConicEquation?
        """A spherical wavefront emanating from the main focus reflects off the
        conic section and turns into a wavefront centered on the other focus.
        Given a total travel distance r from the main focus, returns a specification of
        the corresponding arc centered on the other focus with endpoints on the conic.
        
        - If the conic is an ellipse with distance sum d, this can be described by
          polar coordinates {(d - r, θ): θ1 < θ < θ2} centered at the other focus.
        - If the conic is a hyperbola with distance difference d, this can be described by
          polar coordinates {(d + r, θ): θ1 < θ < θ2} centered at the other focus.
        - If the conic is a parabola, this can be described by a line segment at
          distance r from the directrix and endpoints on the parabola.

        Used to animate spherical wavefronts centered around the other focus."""
        # TODO
        pass
